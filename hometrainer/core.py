"""Holds all code needed for the actual alpha zero search and training.

This is the 'core' code that executes the actual parts needed/described in the alpha zero
papers. The actual search is executed in the MCTSNode, all other classes coordinate the process.

This core could be used by itself to execute the training, but usually it will be used by the
distribution module to handle the 'large scale' coordination of the process. Also this implementation
is fully generic, it needs an actual game implementation to work."""
import threading
import math
import hometrainer.util as util
import concurrent.futures
import numpy as np
import functools
import pickle
import time
import random
import logging
import os
from hometrainer.config import Configuration


class Move:
    """Represents an executed move.

    This is some sort of 'short' description
    on what the move was. In chess this would for example be 'white, Ke4'.
    This will be used to refer to individual moves in all context, e.g. when telling how good a move is."""
    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        return not (self == other)


class GameState:
    """Holds all information needed to describe one specific game state.

    This includes information on turn limits, active player and so on.
    Must also remember the last executed move (when using the execution logic)."""

    def get_next_game_states(self):
        """Returns an array of game states that follow this one after one move.

        The returned array is empty if the game is finished.
        The self object must not be changed by this call! (at least not publicly visible, caching is fine)"""
        raise NotImplementedError()

    def get_next_player(self):
        """Return the next player that will perform an move."""
        raise NotImplementedError()

    def is_finished(self):
        """Return if the game is finished. Overwrite with a faster method if needed."""
        return len(self.get_next_game_states()) == 0

    def get_last_move(self):
        """Returns the last executed move as a Move object."""
        raise NotImplementedError()

    def execute_move(self, move: Move):
        """Return the game state following from the given move."""
        raise NotImplementedError()

    def get_player_list(self):
        """Returns a list of all players.

        This is usually a list of enums or numbers. These will be used to associate statistics to players.
        The players used here must be the same as used in the Move class!

        Example for two player game: [1, 2] or ['player_one', 'player_two']"""
        raise NotImplementedError()

    def get_average_score(self):
        """The average outcome of a game.

        This is needed as a starting value for not processed nodes in the search tree.
        0 is a sane default scores are between -1 and 1."""
        return 0

    def get_virtual_loss(self):
        """The value used for virtual loss in the search tree.

        Keep this reasonable considering your scoring scheme.
        -1 is a sane default for scores between -1 and 1"""
        return -1

    def calculate_scores(self):
        """Score the game state according to the game rules.

        This will be called on terminal game states. Scores should be somehow normalized."""
        raise NotImplementedError()

    def wrap_in_evaluation(self):
        """Return an evaluation that wraps this game state.

        Simply call the constructor of your custom evaluation for this."""
        raise NotImplementedError()


class Evaluation:
    """The 'glue' between GameState and the neural Network.

    Holds move probabilities and the expected game outcome returned by the neural network.
    This will work for both directions: Training the neural network and using it to evaluate game states.
    Think of it as one training sample (input: BoardState, output: move probs & expected outcome).

    To work properly this will also encapsulate the logic needed to transform a board state, so that the
    current active play is always player one (to have a simpler neural network) and to apply rotations/mirrors
    to the game state if appropriate in the given game."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def get_expected_result(self):
        """Returns an floating point value on the expected result for all players in an dictionary.

        This is the guess of the neural network for the game outcome for this player.
        The output should be some sort of normalized score, higher is better,
        best values between 0 and 1 or -1 and 1 from worst to best player."""
        raise NotImplementedError()

    def set_expected_result(self, expected_results):
        """Same as getter, simple the matching setter"""
        raise NotImplementedError()

    def get_move_probabilities(self):
        """Returns a directory with the form {move: move_probability} of all possible moves in the game state."""
        raise NotImplementedError()

    def set_move_probabilities(self, move_probabilities):
        """Set a directory with the form {move: move_probability} of all possible moves in the game state."""
        raise NotImplementedError()

    def convert_to_normal(self):
        """Converts the evaluation to a form where the next active player is player one.

        Returns the converted evaluation and does not change the original data!
        DO NOT change self or self.game_state in the process. This is to avoid any side effects!
        If the evaluation is already in normal form it is simply returned."""
        raise NotImplementedError()

    def convert_from_normal(self):
        """Converts the evaluation from normal form to its original form.

        Returns the converted evaluation and does not change the original data!
        DO NOT change self or self.game_state in the process. This is to avoid any side effects!
        If the evaluation is already in normal form it is simply returned."""
        raise NotImplementedError()

    def get_possible_transformations(self):
        """List of all possible transformations (mirror, rotate, ...).

        This will be used to alter the game_state for more training samples."""
        # By default we do not support any transformations.
        # If possible implement this as soon as the 'basics' work, as it helps a lot.
        return []

    def apply_transformation(self, transformation):
        """Returns the transformed evaluation and does not change the original data!

        The transformed evaluation should keep track of all applied transformations, so it can undo them later on.
        This should really transform all internal data, including the move probabilities and expected game result"""
        # By default we do not support any transformations.
        # If possible implement this as soon as the 'basics' work, as it helps a lot.
        return self

    def undo_transformations(self):
        """Undos all previously applied transformations.

        Returns the converted evaluation and does not change the original data!"""
        # By default we do not support any transformations.
        # If possible implement this as soon as the 'basics' work, as it helps a lot.
        return self


class MCTSNode:
    """A single node in the constructed search tree.

    Each node holds its visit count, its value, its direct child nodes and the probability of the move
    that lead to them. This is 'the other way around' of what the paper states (values per edge), but was
    simpler to implement and follow, as most programmers are used to hold information in the actual nodes
    and model connections as references.
    It also simplifies where values belong, as the expected outcome of a game state is in the node holding it.

    This is the 'heart' of the actual alpha zero search. It is actually quite simple and easy to follow."""

    def __init__(self, probability, game_state: GameState, config: Configuration):
        """ Create a new node for the search tree.

        :param probability: The initial probability of the move leading to this node.
                            This is the 'intuition' value of the neural network.
        :param game_state: The game state of this node in the game search tree.
        """
        self.game_state = game_state
        self.config = config

        # Keep Statistics
        # P(s_prev, a)      Probability of the move leading to this node
        self.probability = probability
        # N(s)              Visit count of this node
        self.visits = 0
        # W(s) and Q(s)     Expected game outcome (total and mean)
        average_score = game_state.get_average_score()  # sane default as long as this node was not evaluated
        self.total_action_value = {player: average_score for player in game_state.get_player_list()}
        self.mean_action_value = {player: average_score for player in game_state.get_player_list()}

        # Children will be added when the node is expanded
        self.children = None

        # Keep track if we are a leave node, in that case we need to treat the game outcome different
        self.is_terminal_state = False

        # Locks for multithreaded search.
        # This is mainly to avoid 'waiting' for the neural network.
        # It also helps with exploration thanks to virtual loss.
        self.expand_lock = threading.Lock()
        self.setter_lock = threading.Lock()

        # Helpers for applying virtual losses.
        virtual_loss = game_state.get_virtual_loss()
        self._add_virtual_loss = {player: virtual_loss for player in game_state.get_player_list()}
        self._subtract_virtual_loss = {player: -virtual_loss for player in game_state.get_player_list()}

    def run_simulation_step(self, nn_client):
        """One full simulation step is selection, expansion and backup.

        This runs the three steps of the alpha zero tree search.
        It's straight forward an easy to follow with the comments."""
        self._apply_virtual_loss()

        try:
            if self.is_terminal_state:
                return self._process_terminal_state()

            if self.is_expanded():
                return self._process_normal(nn_client)

            return self._process_leave(nn_client)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        finally:
            self._undo_virtual_loss()

    def _process_terminal_state(self):
        # Processing terminal states is quite simple as their action value is clear.
        self._increase_visits()
        self._update_action_value(self.game_state.calculate_scores())

        return self.game_state.calculate_scores()

    def _process_normal(self, nn_client):
        # Processing 'normal' nodes involves...
        # ...first selection of the path to the leave node...
        move = self._select_move()
        child = self.children[move]

        # ...and then backing up the result from the leave.
        result = child.run_simulation_step(nn_client)
        self._increase_visits()
        self._update_action_value(result)

        return result

    def _process_leave(self, nn_client):
        # We only want to expand in one thread at a time
        self.expand_lock.acquire()

        # We already expanded in another thread, simply release the lock and re-run.
        if self.is_expanded():
            self.expand_lock.release()
            return self.run_simulation_step(nn_client)

        # We actually need to expand here, lets go.
        next_states = self.game_state.get_next_game_states()
        if len(next_states) <= 0:
            # We found a terminal game state. The action value of this node is the actual score of the game state.
            self.is_terminal_state = True
            self.expand_lock.release()
            self._update_action_value(self.game_state.calculate_scores())
        else:
            # A non terminal game state will be evaluated by the neural network.
            # This gives us the expected action value for this node and predictions for move probabilities.
            evaluation = nn_client.evaluate_game_state(self.game_state)
            self._execute_expansion(evaluation, next_states)
            self.expand_lock.release()
            self._update_action_value(evaluation.get_expected_result())

        self._increase_visits()
        self.mean_action_value = util.deepcopy(self.total_action_value)

        # We have a expected action value for this node to back up into the tree.
        return self.mean_action_value

    def _execute_expansion(self, evaluation: Evaluation, next_states):
        if self.children:
            return

        move_probabilities = evaluation.get_move_probabilities()
        self.children = dict()

        # Add one Node to the tree for each possible move.
        # It can be initialized with the initial move probabilities from the neural network.
        for next_state in next_states:
            move = next_state.get_last_move()
            self.children[move] = MCTSNode(move_probabilities[move], next_state, self.config)

    def _apply_virtual_loss(self):
        self._increase_visits()
        self._update_action_value(self._add_virtual_loss)

    def _undo_virtual_loss(self):
        self._decrease_visits()
        self._update_action_value(self._subtract_virtual_loss)

    def _select_move(self):
        # Select a move using the variant of the PUCT algorithm.
        # See the alpha zero paper for a description.

        # this equals 'sqrt(sum(visit count of children))'
        sqrt_total_child_visits = math.sqrt(self.visits)
        # constant determining exploration
        c_puct = self.config.c_puct()

        next_player = self.game_state.get_next_player()  # We need the best value for this player!
        best_move_value = -10000.0
        best_move = None
        for move, child in self.children.items():
            u = c_puct * child.probability * (sqrt_total_child_visits/(1 + child.visits))
            q = child.mean_action_value[next_player]

            move_value = u + q
            if move_value > best_move_value:
                best_move_value = move_value
                best_move = move

        return best_move

    def _update_action_value(self, new_action_value):
        # Add a action value to the total value and keep the mean action value up to date
        with self.setter_lock:
            for player, value in new_action_value.items():
                self.total_action_value[player] = self.total_action_value[player] + value
                self.mean_action_value[player] = self.total_action_value[player] / max(self.visits, 1)

    def _increase_visits(self):
        with self.setter_lock:
            self.visits = self.visits + 1

    def _decrease_visits(self):
        with self.setter_lock:
            self.visits = self.visits - 1

    def is_expanded(self):
        return not not self.children


class MCTSExecutor:
    """Handles the simulation of MCTS in one specific game state.

    This excludes actually progressing the game. The sole purpose of this class
    is to build up a search tree to get the next move to play."""
    def __init__(self, game_state: GameState, nn_client, config: Configuration,
                 root_node: MCTSNode=None, thread_pool=None):
        self.start_game_state = game_state
        self.nn_client = nn_client
        self.config = config

        # Optionally use the sub-tree from the last iteration.
        self.root_node = root_node
        # Optionally run mulithreaded using a given thread pool
        self.thread_pool = thread_pool

    def run(self, n_simulations):
        """Expands the search tree by n_simulations nodes.

        :param n_simulations: The number of iterations/nodes added to the search tree
        """
        if not self.root_node:
            self.root_node = MCTSNode(1.0, self.start_game_state, self.config)

        # We can run serial or parallel in a thread pool
        if not self.thread_pool:
            for i in range(n_simulations):
                self.root_node.run_simulation_step(self.nn_client)
        else:
            futures = []
            for i in range(n_simulations):
                futures.append(self.thread_pool.submit(self.root_node.run_simulation_step, self.nn_client))
            concurrent.futures.wait(futures)

    def move_probabilities(self, temperature):
        """Returns each move and its probability. Temperature controls how much extreme values are damped.

        See the Alpha Go Zero paper for more details. Usually temperature is simply 1 to return the
        number of visits for each subtree."""
        exponent = 1.0 / temperature

        visit_sum = 0
        exponentiated_visit_counts = dict()
        for move, child in self.root_node.children.items():
            exponentiated_visit_count = child.visits ** exponent
            visit_sum = visit_sum + exponentiated_visit_count

            exponentiated_visit_counts[move] = exponentiated_visit_count

        if visit_sum > 0:
            # Usually we go the normal way and look at the visit counts
            return {move: count / visit_sum for move, count in exponentiated_visit_counts.items()}
        else:
            raise Exception('Must run simulations before getting move probabilities.')


class SelfplayExecutor:
    """Handles the simulation one selfplay game.

    This should run one game of selfplay and return a list of all states and all
    corresponding probability/value targets that can then be used as training data."""
    def __init__(self, game_state, nn_client, n_simulations_per_move, config: Configuration, temperature=1.0):
        self.config = config
        self.current_executor = MCTSExecutor(game_state, nn_client, config)
        self.current_game_state = util.deepcopy(game_state)
        self.nn_client = nn_client
        self.n_simulations_per_move = n_simulations_per_move
        self.evaluations = []
        self.temperature = temperature

        # TODO: Maybe add the ability to let different neural networks play against each other.
        #       The most recent alpha zero paper states that this helps with avoiding overfitting
        #       to one specific strategy.

    def run(self):
        """Actually run the selfplay game.

        This will run the one game played against itself."""
        n_threads = self.config.n_search_threads_selfplay()
        thread_pool = None
        if n_threads > 1:
            thread_pool = concurrent.futures.ThreadPoolExecutor(n_threads)

        while True:
            # Attach our thread pool to the current executor
            self.current_executor.thread_pool = thread_pool

            # Make sure the game is not finished
            next_states = self.current_game_state.get_next_game_states()
            if len(next_states) == 0:
                break

            # Run the simulation
            self.current_executor.run(self.n_simulations_per_move)

            # Take a snapshot for training
            self._create_evaluation()

            # Select next move
            move_probabilities = self.current_executor.move_probabilities(self.temperature).items()
            moves = [item[0] for item in move_probabilities]
            probabilities = [item[1] for item in move_probabilities]

            # Execute the move
            index = np.random.choice(len(moves), p=probabilities)
            move = moves[index]
            self.current_game_state = self.current_game_state.execute_move(move)

            # Update our executor. We keep the part of the search tree that was selected.
            selected_child = self.current_executor.root_node.children[move]
            self.current_executor = MCTSExecutor(self.current_game_state, self.nn_client, self.config, root_node=selected_child)

        actual_results = self.current_game_state.calculate_scores()

        old_evaluations = self.evaluations
        self.evaluations = []
        for evaluation in old_evaluations:
            # Add results the next possible moves
            evaluation.set_expected_result(actual_results)

            # Convert to normal...
            evaluation = evaluation.convert_to_normal()
            self.evaluations.append(evaluation)

        thread_pool.shutdown(wait=False)
        return self.evaluations

    def _create_evaluation(self):
        """Creates an evaluation of the current MCTSExecutor and adds it to the collected evaluations for this run"""
        evaluation = self.current_game_state.wrap_in_evaluation()
        evaluation.set_move_probabilities(self.current_executor.move_probabilities(1.0))

        self.evaluations.append(evaluation)


class TrainingExecutor:
    """Manages the training process of a neural network.

    This is managing the training data and the training process.
    The class is given neural network client to work with.

    The training history size indicates how many of the last games to consider
    for training (e.g. use the 500 most recent games of training data)."""
    def __init__(self, nn_client, data_dir, training_history_size, apply_transformations=True):
        super().__init__()
        self.nn_client = nn_client
        self.training_history_size = training_history_size

        self.apply_transformations = True

        # We will keep the training and test data in a local folder.
        # This class is only responsible for somehow doing the training,
        # this does not constrain it to run only on this machine,
        # but its a good start to have all training data somewhere for whatever training method.
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.lock = threading.Lock()
        self._current_number = util.count_files(self.data_dir)

    def add_examples(self, evaluations):
        with self.lock:
            self._current_number = self._current_number + 1

            with open(os.path.join(self.data_dir, "{0:010d}.pickle".format(self._current_number)), 'wb') as file:
                pickle.dump(evaluations, file)

    def get_examples(self, n_examples):
        with self.lock:
            evaluations = []
            while len(evaluations) < n_examples:
                oldest_index = max(1, self._current_number - self.training_history_size)
                number = random.randint(oldest_index, self._current_number)

                try:
                    loaded_evaluations = self._load_evaluations(number)
                except IOError:
                    continue

                random.shuffle(loaded_evaluations)
                end_index = min(round(n_examples / 8 + 1), len(loaded_evaluations))
                evaluations = evaluations + loaded_evaluations[:end_index]

            return evaluations

    # TODO: Make cache size adjustable
    @functools.lru_cache(maxsize=512)
    def _load_evaluations(self, example_number):
        with open(os.path.join(self.data_dir, "{0:010d}.pickle".format(example_number)), 'rb') as file:
            loaded_evaluations = pickle.load(file)

            # Add every possible transformation to our samples
            for evaluation in loaded_evaluations:
                for i in evaluation.get_possible_transformations():
                    transformed_evaluation = evaluation.apply_transformation(i)
                    loaded_evaluations.append(transformed_evaluation)

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

    def log_loss(self, epoch, batch_size=32):
        raise NotImplementedError()


class ModelEvaluator:
    """Compares two neural network configurations by playing out a game between them.

    This is a optional step in the training process where we play with our current best network weights
    against our currently trained weights to see if they are better and therefore our new best weights.
    This step in the training is optional."""
    def __init__(self, nn_client_one, nn_client_two, start_game_state: GameState, config: Configuration):
        self.nn_client_one = nn_client_one
        self.nn_client_two = nn_client_two
        self.start_game_state = start_game_state
        self.config = config

    def run(self, n_simulations):
        """Executes the match between the two neural networks.

        Returns an array with [avg_score_nn_one, avg_score_nn_two]."""
        n_threads = self.config.n_search_threads_self_eval()
        thread_pool = None
        if n_threads > 1:
            thread_pool = concurrent.futures.ThreadPoolExecutor(n_threads)

        # Randomly assign neural network configs to different players
        n_player_one, n_player_two, player_mappings = self._shuffle_player_mapping()

        current_game_state = self.start_game_state
        while True:
            # Be sure the game is not ended already
            if current_game_state.is_finished():
                break

            # Find the correct nn to execute this move
            current_player = current_game_state.get_next_player()
            if player_mappings[current_player] == 'nn_one':
                nn_client = self.nn_client_one
            else:
                nn_client = self.nn_client_two

            # Run the actual simulation to find a move
            mcts_executor = MCTSExecutor(current_game_state, nn_client, self.config, thread_pool=thread_pool)
            mcts_executor.run(n_simulations)

            # Find the best move
            selected_move = None
            best_probability = -1.0
            for move, probability in mcts_executor.move_probabilities(1).items():
                if probability > best_probability:
                    best_probability = probability
                    selected_move = move

            # Execute the move
            current_game_state = current_game_state.execute_move(selected_move)

        # Return the scores (average score per neural network for games with more than two players).
        # This should still be fair, even for uneven player numbers.
        scores = current_game_state.calculate_scores()
        result = [0, 0]
        for player, client in player_mappings.items():
            if client == 'nn_one':
                result[0] += scores[player]
            else:
                result[1] += scores[player]

        result[0] /= n_player_one
        result[1] /= n_player_two

        thread_pool.shutdown(wait=False)
        return result

    def _shuffle_player_mapping(self):
        players = self.start_game_state.get_player_list()

        n_client_one = round(len(players)/2)
        n_client_two = len(players) - n_client_one
        if np.random.choice([False, True]):
            # This makes sure we do not add one neural network more then the other in the long run
            n_client_one, n_client_two = n_client_two, n_client_one

        clients = (['nn_one'] * n_client_one) + (['nn_two'] * n_client_two)

        # Shuffle what player is played by what client
        np.random.shuffle(clients)

        player_mappings = dict()
        for i in range(len(clients)):
            player_mappings[players[i]] = clients[i]

        return n_client_one, n_client_two, player_mappings


class ExternalEvaluator:
    """Compares a neural network to an external AI program by playing out a small tournament."""
    def __init__(self, nn_client, start_game_state, config: Configuration):
        self.nn_client = nn_client
        self.start_game_state = start_game_state
        self.config = config

    def external_ai_select_move(self, current_game_state, turn_time):
        raise NotImplementedError()

    def setup_external_ai(self, start_game_state, player_mappings):
        """Callback before the game starts.

        Use this to setup the external ai."""
        pass

    def move_selected(self, old_game_state, move, new_game_state):
        """Callback after a move was actually executed.

        Use this if you need to provide feedback about moves played by the internal ai."""
        pass

    def shutdown_external_ai(self, end_game_state):
        """Callback after the game was finished.

        Use this to clean up any resources used by the external ai."""
        pass

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
        n_threads = self.config.n_search_threads_external_eval()
        thread_pool = None
        if n_threads > 1:
            thread_pool = concurrent.futures.ThreadPoolExecutor(n_threads)

        # Randomly assign the neural network and external ai to different players
        n_player_one, n_player_two, player_mappings = self._shuffle_player_mapping()

        # Chance to setup external resources/programs
        self.setup_external_ai(self.start_game_state, player_mappings)

        current_game_state = self.start_game_state
        while True:
            # Be sure the game is not over
            if current_game_state.is_finished():
                break

            selected_move = None
            current_player = current_game_state.get_next_player()
            # Find the correct nn to execute this move
            if player_mappings[current_player] == 'neural_network':
                end_time = time.time() + turn_time

                mcts_executor = MCTSExecutor(current_game_state, self.nn_client, self.config, thread_pool=thread_pool)
                while True:
                    mcts_executor.run(16)
                    if time.time() >= end_time:
                        break

                # Find the best move
                best_probability = -100.0
                for move, probability in mcts_executor.move_probabilities(1).items():
                    if probability > best_probability:
                        best_probability = probability
                        selected_move = move
            else:
                selected_move = self.external_ai_select_move(current_game_state, turn_time)

            current_game_state = current_game_state.execute_move(selected_move)

        # Chance to clean up any resources used by the external ai client
        self.shutdown_external_ai(current_game_state)

        # Return the scores (average score per neural network for games with more than two players).
        # This should still be fair, even for uneven player numbers.
        scores = current_game_state.calculate_scores()
        result = [0, 0]
        for player, client in player_mappings.items():
            if client == 'neural_network':
                result[0] += scores[player]
            else:
                result[1] += scores[player]

        result[0] /= n_player_one
        result[1] /= n_player_two

        thread_pool.shutdown(wait=False)
        return result

    def _shuffle_player_mapping(self):
        players = self.start_game_state.get_player_list()

        n_client_one = round(len(players)/2)
        n_client_two = len(players) - n_client_one
        if np.random.choice([False, True]):
            # This makes sure we do not add one neural network more then the other in the long run
            n_client_one, n_client_two = n_client_two, n_client_one

        clients = (['neural_network'] * n_client_one) + (['external_ai'] * n_client_two)

        # Shuffle what player is played by what client
        np.random.shuffle(clients)

        player_mappings = dict()
        for i in range(len(clients)):
            player_mappings[players[i]] = clients[i]

        return n_client_one, n_client_two, player_mappings
