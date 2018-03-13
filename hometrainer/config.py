"""This hold's the default config object used in the projet.

Overwrite it and pass it to the moving parts that depend on it.
The Config is a 'programmer config' intended for projects that want to implement
alpha zero for some game. It allows to do most stuff at runtime to allow for own policies
on resource use and own runtime config passed by the user.

If you use this library for a project you can and should still do a user config
that is fed into this object to allow for easy experiments."""
import time
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# NN Server Settings
SELFPLAY_NN_SERVER_PORT = 5100
SELFEVAL_NN_SERVER_PORT = 5101
TRAINING_NN_SERVER_PORT = 5102

# Distribution settings
TRAINING_MASTER_PORT = 5200


class Configuration:
    def __init__(self):
        self._training_master_port = TRAINING_MASTER_PORT

        # Default Settings for Search/Training
        # Number of game states for one training batch
        self._training_batch_size = 64
        # Number of last games used for training
        self._training_history_size = 128
        # Simulations per selfplay/selfeval turn
        self._simulations_per_turn = 128
        # Turn time for each player during ai evaluation
        self._external_evaluation_turn_time = 1.0

        # Number of selfplay games for each iteration
        self._n_self_play = 42
        # Number of self evaluation games for each iteration, 0 will skip it
        self._n_self_eval = 0
        # Number of evaluation games against the ai-trivial client for each client, 0 will skip it
        self._n_external_eval = 0

        # The self evaluation avg. score needed to see this iteration as new best.
        # This means if the new weights scored >= this value they will be chosen as best weights.
        # 0.05 is a sane default for scores between -1 and 1.
        self._needed_avg_self_eval_score = 0.05

        # C-PUCT used in Tree search
        self._c_puct = 3

        # Configure the number of concurrent search threads, 1 means no multithreading
        self._n_search_threads_external_eval = 8
        self._n_search_threads_self_eval = 8
        self._n_search_threads_selfplay = 8

        # Config for stats on the neural network
        self._nn_test_set_step_size = 10  # Every X executed game will be used to log a test error of the nn
        self._nn_log_step_size = 100  # Every X executed training batch a log of the losses will be performed

    def zmq_use_secure_connection(self):
        """:returns True if the zeromq connections should be secured with certificates."""
        return False

    def zmq_client_secret(self):
        """:returns The path to the client secret file."""
        raise NotImplementedError('Implement Config.zmq_client_secret to use a secure zmq zeromq connection.')

    def zmq_server_secret(self):
        """:returns The path to the server secret file."""
        raise NotImplementedError('Implement Config.zmq_server_secret to use a secure zmq zeromq connection.')

    def zmq_server_public(self):
        """:returns The path to the server public file."""
        raise NotImplementedError('Implement Config.zmq_server_public to use a secure zmq zeromq connection.')

    def zmq_public_keys_dir(self):
        """:returs The path to the directory containing both client and server public keys."""
        raise NotImplementedError('Implement Config.zmq_public_keys_dir to use a secure zmq zeromq connection.')

    def nn_server_tensorboard_logdir(self, port, neural_network):
        """Default logging directory for tensorboard output from neural network.

        Usually you wont need this, as in the 'fully managed training' everything is self contained in
        one folder for the current experiment."""
        return os.path.join(os.path.curdir, 'nn_logs/{}-{}'.format(port, round(time.time() * 1000)))

    def training_master_port(self):
        """:returns The default port of the training master."""
        return self._training_master_port

    def nn_server_training_port(self):
        """:returns The port of the local NN Server running the training on the training master."""
        return TRAINING_NN_SERVER_PORT

    def nn_server_selfplay_port(self):
        """:returns The port of the local NN processing selfplay games."""
        return SELFPLAY_NN_SERVER_PORT

    def nn_server_selfeval_port(self):
        """:returns The port of the local NN processing the 'last weights' in the selfeval games."""
        return SELFEVAL_NN_SERVER_PORT

    def training_batch_size(self):
        """:returns The size of a single training batch."""
        return self._training_batch_size

    def training_history_size(self):
        """:returns The number of last selfplay games considered for training."""
        return self._training_history_size

    def simulations_per_turn(self):
        """:returns The number of treesearch simulation per turn in a selfplay game."""
        return self._simulations_per_turn

    def external_evaluation_turn_time(self):
        """:returns The time each agent has per move in the external evaluation game (in seconds)."""
        return self._external_evaluation_turn_time

    def n_self_play(self):
        """:returns The number of selfplay games per iteration."""
        return self._n_self_play

    def n_self_eval(self):
        """:returns The number of selfeval games per iteration."""
        return self._n_self_eval

    def n_external_eval(self):
        """:returns The number of comparison games against an external ai per iteration."""
        return self._n_external_eval

    def needed_avg_self_eval_score(self):
        """:returns: The minimum average game outcome from selfeval games to choose the new weights as current best."""
        return self._needed_avg_self_eval_score

    def external_ai_agent(self, start_game_state):
        """Returns an instance of an Agent class that plays for the external AI to compare to."""
        raise NotImplementedError()

    def external_evaluation_possible(self):
        """Specifies if the given machine can play games against the external ai client.
        This can come in handy if the external client only runs on specific machines, for example
        the client could only be available on Linux, but one machine runs Windows."""
        return True  # Usually this should work, maybe configure this at runtime

    def c_puct(self):
        """:returns The c_puct constant used in the alphazero treesearch."""
        return self._c_puct

    def n_search_threads_external_eval(self):
        """:returns The number of threads searching in the external eval games."""
        return self._n_search_threads_external_eval

    def n_search_threads_self_eval(self):
        """:returns The number of threads searching in the self eval games."""
        return self._n_search_threads_self_eval

    def n_search_threads_selfplay(self):
        """:returns The number of threads searching in the selfpaly games."""
        return self._n_search_threads_selfplay

    def nn_test_set_step_size(self):
        """:returns Every x game of the selfpaly games will be used to measure test error."""
        return self._nn_test_set_step_size

    def nn_log_step_size(self):
        """:returns Every x training batch will generate a step in the neural network logs."""
        return self._nn_log_step_size
