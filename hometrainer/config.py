"""This hold's the default config object used in the projet.

Overwrite it and pass it to the moving parts that depend on it.
The Config is a 'programmer config' intended for projects that want to implement
alpha zero for some game. It allows to do most stuff at runtime to allow for own policies
on resource use and own runtime config passed by the user."""
import hometrainer.definitions as definitions
import zmq.auth
import time
import os


class Configuration:
    def __init__(self):
        self._training_master_port = definitions.TRAINING_MASTER_PORT

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

    def zmq_use_secure_connection(self):
        return False

    def zmq_client_secret(self):
        return zmq.auth.load_certificate(definitions.CLIENT_SECRET)

    def zmq_server_secret(self):
        return zmq.auth.load_certificate(definitions.SERVER_SECRET)

    def zmq_server_public(self):
        return zmq.auth.load_certificate(definitions.SERVER_PUBLIC)

    def zmq_public_keys_dir(self):
        return definitions.PUBLIC_KEYS_DIR

    def nn_server_tensorboard_logdir(self, port, neural_network):
        return os.path.join(os.path.curdir, 'nn_logs/{}-{}'.format(port, round(time.time() * 1000)))

    def training_master_port(self):
        return self._training_master_port

    def nn_server_training_port(self):
        return definitions.TRAINING_NN_SERVER_PORT

    def nn_server_selfplay_port(self):
        return definitions.SELFPLAY_NN_SERVER_PORT

    def nn_server_selfeval_port(self):
        return definitions.SELFEVAL_NN_SERVER_PORT

    def training_batch_size(self):
        return self._training_batch_size

    def training_history_size(self):
        return self._training_history_size

    def simulations_per_turn(self):
        return self._simulations_per_turn

    def external_evaluation_turn_time(self):
        return self._external_evaluation_turn_time

    def n_self_play(self):
        return self._n_self_play

    def n_self_eval(self):
        return self._n_self_eval

    def n_external_eval(self):
        return self._n_external_eval

    def needed_avg_self_eval_score(self):
        return self._needed_avg_self_eval_score

    def external_ai_evaluator(self, nn_client, start_game_state):
        """Returns an instance of the ExternalAIEvaluator subclass specific to your project."""
        raise NotImplementedError()

    def external_evaluation_possible(self):
        """Specifies if the given machine can play games against the external ai client.
        This can come in handy if the external client only runs on specific machines, for example
        the client could only be available on Linux, but one machine runs Windows."""
        return True  # Usually this should work, maybe configure this at runtime

    def c_puct(self):
        return self._c_puct

    def n_search_threads_external_eval(self):
        return self._n_search_threads_external_eval

    def n_search_threads_self_eval(self):
        return self._n_search_threads_self_eval

    def n_search_threads_selfplay(self):
        return self._n_search_threads_selfplay
