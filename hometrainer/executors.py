import hometrainer.agents
from hometrainer.config import Configuration
import threading
import numpy as np
import hometrainer.util as util
import os
import pickle
import random
import functools
import time
import hometrainer.core as core
import logging
import hometrainer.agents as agents


class GameExecutor:
    """Handles the simulation of one game between different opponents.

    This includes mapping these opponents to players in the game,
    progressing the game according to the opponents move decisions
    and evaluating the game outcome.

    Effectively this can be used as:
    Play a game with this starting game state between these participants and tell me the average score of each agent."""
    def __init__(self, start_game_state, agents, add_up_time=False):
        self.start_game_state = start_game_state
        self.external_agents = agents

        # If set to true players can save up time from past moves.
        self.add_up_time = add_up_time

    def play_game(self, time_limit=None, iteration_limit=None):
        """Plays one game and returns the average scores of each agent."""
        if not time_limit and not iteration_limit:
            raise ValueError("At least one of 'time_limit' or 'iteration_limit' must be provided!")

        self._shuffle_player_mapping()

        time_left = dict()
        for player in self.start_game_state.get_player_list():
            time_left[player] = 0
            agent = self.player_executor_mappings[player]
            agent.game_start(self.start_game_state, player)

        current_game_state = self.start_game_state
        while not current_game_state.is_finished():
            next_player = current_game_state.get_next_player()
            current_agent = self.player_executor_mappings[next_player]

            if time_limit:
                time_left[next_player] = time_left[next_player] + time_limit
                if self.add_up_time:
                    final_time_limit = time_left[next_player]
                else:
                    final_time_limit = time_limit

                before_move = time.time()
                move = current_agent.find_move_with_time_limit(current_game_state, final_time_limit)
                after_move = time.time()

                time_left[next_player] = time_left[next_player] - (after_move - before_move)
            else:
                move = current_agent.find_move_with_iteration_limit(current_game_state, iteration_limit)

            next_game_state = current_game_state.execute_move(move)
            for agent in self.internal_agents:
                agent.move_executed(current_game_state, move, next_game_state)

            current_game_state = next_game_state

        for agent in self.internal_agents:
            agent.game_ended(current_game_state)

        player_scores = current_game_state.calculate_scores()
        agent_scores = {agent: 0 for agent in self.external_agents}
        for player, score in player_scores.items():
            current_agent = self.player_result_mappings[player]
            agent_scores[current_agent] = agent_scores[current_agent] + score

        avg_agent_scores = dict()
        for agent, score in agent_scores.items():
            avg_agent_scores[agent] = score / self.agent_counts[agent]

        return avg_agent_scores

    def _shuffle_player_mapping(self):
        players = self.start_game_state.get_player_list()

        # Shuffle the agents to give no one an advantage
        np.random.shuffle(self.external_agents)
        self.agent_counts = {agent: 0 for agent in self.external_agents}
        self.player_result_mappings = dict()
        self.player_executor_mappings = dict()

        # Assign agents to each player in this game
        for i in range(len(players)):
            agent = self.external_agents[i % len(self.external_agents)]
            self.player_result_mappings[players[i]] = agent
            self.player_executor_mappings[players[i]] = util.deepcopy(agent)
            self.agent_counts[agent] = self.agent_counts[agent] + 1

        # Keep track of them as our 'internal' players
        self.internal_agents = [agent for player, agent in self.player_executor_mappings.items()]


class SelfplayExecutor:
    """Handles the simulation one selfplay game.

    This should run one game of selfplay and return a list of all states and all
    corresponding probability/value targets that can then be used as training data."""
    def __init__(self, game_state, nn_clients, n_simulations_per_move, config: Configuration, temperature=1.0):
        self.config = config
        self.start_game_state = game_state
        self.n_simulations_per_move = n_simulations_per_move
        self.nn_clients = nn_clients
        self.temperature = temperature

    def run(self):
        """Actually run the selfplay game.

        This will run the one game played against itself."""
        agents = [hometrainer.agents.NeuralNetworkAgent(nn_client, self.config, True, self.temperature)
                  for nn_client in self.nn_clients]
        game_executor = GameExecutor(self.start_game_state, agents)
        game_executor.play_game(iteration_limit=self.n_simulations_per_move)

        # Return all collected evaluation by each neural network agent.
        # This allows to have multiple agents play against each other and collect different stats, helping
        # with searching a diverse set of game states.
        return [evaluation for agent in game_executor.internal_agents for evaluation in agent.collected_evaluations]


class TrainingExecutor:
    """Manages the training process of a neural network.

    This is managing the training data and the training process.
    The class is given neural network client to work with.

    The training history size indicates how many of the last games to consider
    for training (e.g. use the 500 most recent games of training data)."""
    def __init__(self, nn_client, data_dir, training_history_size, apply_transformations=True, config=None):
        super().__init__()
        self.nn_client = nn_client
        self.training_history_size = training_history_size

        self.apply_transformations = apply_transformations

        if config:
            self.config = config
        else:
            self.config = Configuration()

        # We will keep the training and test data in a local folder.
        # This class is only responsible for somehow doing the training,
        # this does not constrain it to run only on this machine,
        # but its a good start to have all training data somewhere for whatever training method.
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.test_data_dir = data_dir + '-test'
        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

        self.lock = threading.Lock()
        self._current_number = util.count_files(self.data_dir)
        self._current_test_number = util.count_files(self.test_data_dir)

    def add_examples(self, evaluations):
        with self.lock:
            total_number = self._current_number + self._current_test_number

            # Keep some of the played games as test data to evaluate the neural network overfitting
            if total_number % self.config.nn_test_set_step_size() == 0:
                data_dir = self.test_data_dir
                self._current_test_number += 1
                number = self._current_test_number
            else:
                data_dir = self.data_dir
                self._current_number += 1
                number = self._current_number

            with open(os.path.join(data_dir, "{0:010d}.pickle".format(number)), 'wb') as file:
                pickle.dump(evaluations, file)

    def get_examples(self, n_examples, test_example=False):
        with self.lock:
            if test_example:
                number = self._current_test_number
            else:
                number = self._current_number

            evaluations = []
            while len(evaluations) < n_examples:
                oldest_index = max(1, number - self.training_history_size)
                number = random.randint(oldest_index, number)

                try:
                    loaded_evaluations = self._load_evaluations(number, test_example)
                except IOError:
                    continue

                random.shuffle(loaded_evaluations)
                end_index = min(round(n_examples / 8 + 1), len(loaded_evaluations))
                evaluations = evaluations + loaded_evaluations[:end_index]

            return evaluations

    # TODO: Make cache size adjustable
    @functools.lru_cache(maxsize=512)
    def _load_evaluations(self, example_number, test_example):
        if test_example:
            data_dir = self.test_data_dir
        else:
            data_dir = self.data_dir

        with open(os.path.join(data_dir, "{0:010d}.pickle".format(example_number)), 'rb') as file:
            loaded_evaluations = pickle.load(file)

            # Add every possible transformation to our samples
            if self.apply_transformations:
                results = []
                for evaluation in loaded_evaluations:
                    for i in evaluation.get_possible_transformations():
                        transformed_evaluation = evaluation.apply_transformation(i)
                        results.append(transformed_evaluation)

                return results
            else:
                return loaded_evaluations

    def load_weights(self, filename):
        with open(filename, 'rb') as file:
            weights_zip_binary = file.read()
            self.nn_client.load_weights(weights_zip_binary)

    def save_weights(self, filename):
        weights_zip_binary = self.nn_client.save_weights()
        with open(filename, 'wb') as file:
            file.write(weights_zip_binary)

    def run_training_batch(self, batch_size=32):
        # Skip if there is no data
        if self._current_number <= 0:
            time.sleep(10)
            return

        evaluations = self.get_examples(batch_size)
        self.nn_client.execute_training_batch(evaluations)

    def run_logging_batch(self, batch_size=32):
        # Skip if there is no data
        if self._current_number <= 0:
            time.sleep(10)
            return

        evaluations = self.get_examples(batch_size)
        test_evaluations = self.get_examples(batch_size, test_example=True)
        self.nn_client.execute_logging_batch(evaluations, test_evaluations)

    def log_loss(self, epoch, batch_size=32):
        raise NotImplementedError()


class ModelEvaluator:
    """Compares two neural network configurations by playing out a game between them.

    This is a optional step in the training process where we play with our current best network weights
    against our currently trained weights to see if they are better and therefore our new best weights.
    This step in the training is optional."""
    def __init__(self, nn_client_one, nn_client_two, start_game_state: core.GameState, config: Configuration):
        self.nn_client_one = nn_client_one
        self.nn_client_two = nn_client_two
        self.start_game_state = start_game_state
        self.config = config

    def run(self, n_simulations):
        """Executes the match between the two neural networks.

        Returns an array with [avg_score_nn_one, avg_score_nn_two]."""
        try:
            agent_one = hometrainer.agents.NeuralNetworkAgent(self.nn_client_one, self.config)
            agent_two = hometrainer.agents.NeuralNetworkAgent(self.nn_client_two, self.config)
            agents = [agent_one, agent_two]

            game_executor = GameExecutor(self.start_game_state, agents)
            agent_scores = game_executor.play_game(iteration_limit=n_simulations)

            return [agent_scores[agent_one], agent_scores[agent_two]]
        except Exception:
            import traceback
            traceback.print_exc()


class ExternalEvaluator:
    """Compares a neural network to an external AI program by playing out a small tournament."""
    def __init__(self, nn_client, external_agent, start_game_state, config: Configuration):
        self.nn_agent = agents.NeuralNetworkAgent(nn_client, config)
        self.external_agent = external_agent
        self.start_game_state = start_game_state
        self.config = config

    def run(self, turn_time):
        try:
            return self._run_internal(turn_time)
        except Exception as e:
            logging.error("Exception during external evaluation match! {}".format(e))
            import traceback
            traceback.print_exc()
            # We do not care too much if there was an error.
            # This is just a metric used to see 'general' progress, one game wont matter too much.
            # We are so careful here, as this is a place with potential code that accesses external network
            # resources, so stuff can go really wrong (for example the ReversiXT client can not resume at the current
            # point in the match after a crash
            return [0, 0]

    def _run_internal(self, turn_time):
        game_executor = GameExecutor(self.start_game_state, [self.nn_agent, self.external_agent])
        agent_scores = game_executor.play_game(time_limit=turn_time)

        return [agent_scores[self.nn_agent], agent_scores[self.external_agent]]
