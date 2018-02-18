import hometrainer.core as core
from hometrainer.config import Configuration
import time
import concurrent.futures
import numpy as np


class Agent:
    """Agent that is able to play a game. This means it can take an game state and give it's best move.

    The agent get's some more lifecycle callbacks if it needs to for example note moves of the enemy to
    improve its decisions. An agent is usually an AI program that calculates a good move,
    but can also be an interface for an human player to compete in an match.

    Note: Agents must be able to be copied, as they might be initialized multiple times for one match
          with more than two players. This helps you if your agent holds player specific state."""
    def game_start(self, game_state, agents_player):
        """Called when the new game starts. Can be used for setup code."""
        pass

    def find_move_with_time_limit(self, game_state, move_time):
        """Implement your logic to find a move with a given time limit in here."""
        raise NotImplementedError('Move finding (time limit) method not implemented!')

    def find_move_with_iteration_limit(self, game_state, move_iterations):
        """Implement your logic to find a move with a given iteration limit in here."""
        raise NotImplementedError('Move finding (iteration limit) method not implemented!')

    def move_executed(self, old_game_state, move, new_game_state):
        """Called after a move was executed in the current game."""
        pass

    def game_ended(self, game_state):
        """Called when the game ended. Can be used for teardown code."""
        pass


class NeuralNetworkAgent(Agent):
    """Agent using a neural network and the alpha zero search for finding moves."""
    def __init__(self, nn_client, config: Configuration, collect_evaluations=False, temperature=None):
        """
        :param nn_client: The neural network client used to get evaluations
        :param collect_evaluations: If true a evaluations will be created using the search trees from each executed move
        :param temperature: If set to a value moves will be chosen probabilistic using this temperature
        """
        self.nn_client = nn_client

        self.collect_evaluations = collect_evaluations
        self.collected_evaluations = []

        self.temperature = temperature
        self.config = config

        self.current_mcts_node = None
        self.thread_pool = None

    def game_start(self, game_state, agents_player):
        self.current_mcts_node = core.MCTSNode(1.0, game_state, self.config)

        n_threads = self.config.n_search_threads_selfplay()
        self.thread_pool = None
        if n_threads > 1:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(n_threads)

        self.nn_client.start(self.config)

    def find_move_with_time_limit(self, game_state, move_time):
        move_end_time = time.time() + move_time

        while True:
            self._run_mcts_simulations(16)
            if move_end_time < time.time():
                break

        self._create_evaluation()
        return self._find_best_move()

    def find_move_with_iteration_limit(self, game_state, move_iterations):
        self._run_mcts_simulations(move_iterations)
        self._create_evaluation()
        return self._find_best_move()

    def _run_mcts_simulations(self, n_simulations):
        # We can run serial or parallel in a thread pool
        if not self.thread_pool:
            for i in range(n_simulations):
                self.current_mcts_node.run_simulation_step(self.nn_client)
        else:
            futures = []
            for i in range(n_simulations):
                futures.append(self.thread_pool.submit(self.current_mcts_node.run_simulation_step, self.nn_client))
            concurrent.futures.wait(futures)

    def _find_best_move(self):
        # Either select probabilistic or simply take the best move.
        if self.temperature:
            move_probabilities = self.current_mcts_node.move_probabilities(self.temperature).items()
            moves = [item[0] for item in move_probabilities]
            probabilities = [item[1] for item in move_probabilities]

            # Select the move according to the probability distribution
            index = np.random.choice(len(moves), p=probabilities)
            return moves[index]
        else:
            move_probabilities = self.current_mcts_node.move_probabilities(1.0)
            best_move = None
            best_prob = -1

            for move, prob in move_probabilities.items():
                if prob > best_prob:
                    best_prob = prob
                    best_move = move

            return best_move

    def _create_evaluation(self):
        """Creates an evaluation of the current MCTSExecutor and adds it to the collected evaluations for this run"""
        if not self.collect_evaluations:
            return

        evaluation = self.current_mcts_node.game_state.wrap_in_evaluation()
        evaluation.set_move_probabilities(self.current_mcts_node.move_probabilities(1.0))

        self.collected_evaluations.append(evaluation)

    def move_executed(self, _, move, new_game_state):
        # Try to keep parts of the tree if possible
        if self.current_mcts_node and self.current_mcts_node.children and self.current_mcts_node.children[move]:
            self.current_mcts_node = self.current_mcts_node.children[move]
        else:
            self.current_mcts_node = core.MCTSNode(1.0, new_game_state, self.config)

    def game_ended(self, game_state):
        # If we did collect evaluations update them according to the actual results.
        actual_game_result = game_state.calculate_scores()
        for evaluation in self.collected_evaluations:
            evaluation.set_expected_result(actual_game_result)

        # Tear down any resources left
        self.thread_pool.shutdown(wait=False)

        # Stop NN client
        self.nn_client.stop()
