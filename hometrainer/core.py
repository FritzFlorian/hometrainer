"""Holds all code needed for the actual alpha zero search and training.

This is the 'core' code that executes the actual parts needed/described in the alpha zero
papers. The actual search is executed in the MCTSNode, all other classes coordinate the process.

This core could be used by itself to execute the training, but usually it will be used by the
distribution module to handle the 'large scale' coordination of the process. Also this implementation
is fully generic, it needs an actual game implementation to work."""
import threading
import math
import hometrainer.util as util
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
        return ['identity']

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
        It's straight forward an easy to follow with the comments.

        :return: The expected game outcome from this simulation run."""
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
        """Runs simulation step in an end game state/terminal game state.

        :return: The expected game outcome from this simulation run. As it is a terminal node this is simply the score.
        """
        self._increase_visits()
        self._update_action_value(self.game_state.calculate_scores())

        return self.game_state.calculate_scores()

    def _process_normal(self, nn_client):
        """Runs the simulation Step for a node mid tree (neither terminal, nor leave)

        :param nn_client: This client will be used to evaluate the current game state on expansion.
        :return: The expected game outcome from this simulation run.
                 This one simply returns the result backed up from lower nodes.
        """
        # Select the path to a leave node (selection step)
        move = self._select_move()
        child = self.children[move]

        # Return the result from the leave node and update own Q values (backup step)
        result = child.run_simulation_step(nn_client)
        self._increase_visits()
        self._update_action_value(result)

        return result

    def _process_leave(self, nn_client):
        """Runs simulation step in a leave node. This involves expanding the tree.

        :return: The expected game outcome from this simulation run. This is given by the neural network."""
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
            self._execute_expansion(evaluation.get_move_probabilities(), next_states)
            self.expand_lock.release()
            self._update_action_value(evaluation.get_expected_result())

        self._increase_visits()
        self.mean_action_value = util.deepcopy(self.total_action_value)

        # We have a expected action value for this node to back up into the tree.
        return self.mean_action_value

    def _execute_expansion(self, move_probabilities, next_states):
        if self.children:
            return

        self.children = dict()
        # Add one Node to the tree for each possible move.
        # It can be initialized with the initial move probabilities from the neural network.
        for next_state in next_states:
            move = next_state.get_last_move()
            prob = move_probabilities[move]
            self.children[move] = MCTSNode(prob, next_state, self.config)

    def _apply_virtual_loss(self):
        """Adds 'one instance' of virtual loss to this node."""
        self._increase_visits()
        self._update_action_value(self._add_virtual_loss)

    def _undo_virtual_loss(self):
        """Removes 'one instance' of virtual loss form this node."""
        self._decrease_visits()
        self._update_action_value(self._subtract_virtual_loss)

    def _select_move(self):
        """Select a move using the variant of the PUCT algorithm.
        See the alpha zero paper for a description."""

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
            if not best_move or move_value > best_move_value:
                best_move_value = move_value
                best_move = move

        return best_move

    def _update_action_value(self, new_action_value):
        """Add the 'new_action_value' to the total and keep the average action value up to date."""
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

    def move_probabilities(self, temperature):
        """Returns each possible move and its probability. Temperature controls how much extreme values are damped.

        See the Alpha Go Zero paper for more details. Usually temperature is simply 1 to return the
        number of visits for each subtree."""
        exponent = 1.0 / temperature

        visit_sum = 0
        exponentiated_visit_counts = dict()
        for move, child in self.children.items():
            exponentiated_visit_count = child.visits ** exponent
            visit_sum = visit_sum + exponentiated_visit_count

            exponentiated_visit_counts[move] = exponentiated_visit_count

        if visit_sum > 0:
            return {move: count / visit_sum for move, count in exponentiated_visit_counts.items()}
        else:
            raise Exception('Must run simulations before getting move probabilities.')
