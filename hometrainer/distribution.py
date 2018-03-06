"""Code that coordinates different steps defined in the core module.

This is coordinating the training process and managing the results.
The distribution module allows one training master and many playing slaves to generate data.

This is the highest level interface to use the """
from hometrainer.config import Configuration
import multiprocessing
import zmq
import zmq.error
import pickle
import logging
import time
import numpy as np
import enum
import os
import threading
import kim
import json
import signal
import traceback
import hometrainer.util as util
import hometrainer.executors as executors
import hometrainer.neural_network as neural_network


class PlayingSlave:
    """Runs selfplay games and evaluations using a given neural network configuration.

    The selfplay server reports it's results back to a master node.
    The master node coordinates workloads, stores results and configures the worker."""
    TIMEOUT = 30

    class WorkResult:
        pass

    class EmptyWorkResult(WorkResult):
        pass

    class SelfplayWorkResult(WorkResult):
        def __init__(self, evaluation_lists):
            self.evaluation_lists = evaluation_lists

    class SelfEvaluationWorkResult(WorkResult):
        def __init__(self, nn_one_score, nn_two_score, n_games):
            self.nn_one_score = nn_one_score
            self.nn_two_score = nn_two_score
            self.n_games = n_games

    class AIEvaluationWorkResult(WorkResult):
        def __init__(self, nn_score, ai_score, n_games):
            self.nn_score = nn_score
            self.ai_score = ai_score
            self.n_games = n_games

    class WorkRequest:
        def __init__(self, work_result):
            self.work_result = work_result

    class SelfplayWorkResponse:
        def __init__(self, n_games, nn_name, weights_zip_binary, game_states, simulations_per_turn):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_zip_binary = weights_zip_binary
            self.game_states = game_states
            self.simulations_per_turn = simulations_per_turn

    class SelfEvaluationWorkResponse:
        def __init__(self, n_games, nn_name, weights_binary_one, weights_binary_two,
                     game_states, simulations_per_turn, epoch):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_binary_one = weights_binary_one
            self.weights_binary_two = weights_binary_two
            self.game_states = game_states
            self.simulations_per_turn = simulations_per_turn
            self.epoch = epoch

    class AIEvaluationWorkResponse:
        def __init__(self, n_games, nn_name, weights_zip_binary, game_states, turn_time, epoch):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_zip_binary = weights_zip_binary
            self.game_states = game_states
            self.turn_time = turn_time
            self.epoch = epoch

    class WaitResponse:
        def __init__(self, wait_time):
            self.wait_time = wait_time

    def __init__(self, master_address, config: Configuration=None):
        if not config:
            # Use default configuration...
            config = Configuration()
        self.config = config

        self.master_address = master_address

        self.nn_client_one = None
        self.nn_client_host_one = ''
        self.nn_class_name_one = ''

        self.nn_client_two = None
        self.nn_client_host_two = ''
        self.nn_class_name_two = ''

        self.context = None
        self.zmq_client = None
        self.poll = None

        # Disable Interrupts in workers, we shutdown gracefully by our self
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.process_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        signal.signal(signal.SIGINT, original_sigint_handler)

    def run(self):
        self.context = zmq.Context()
        self.poll = zmq.Poller()

        try:
            self._handle_connections()
        except KeyboardInterrupt:
            logging.error('Keyboard Interrupt, shutting down Playing Slave...')
        finally:
            self.process_pool.terminate()
            if self.nn_client_one:
                self.nn_client_one.stop()
            if self.nn_client_two:
                self.nn_client_two.stop()

            self._disconnect_client()

            logging.info('Terminating process in 5 seconds...')
            time.sleep(5)
            os._exit(1)

    def _handle_connections(self):
        last_work_result = self.EmptyWorkResult()

        while True:
            self._connect_client()

            try:
                logging.info('Sending work request to master server...')
                self.zmq_client.send_pyobj(self.WorkRequest(last_work_result), flags=zmq.NOBLOCK)

                socks = dict(self.poll.poll(round(self.TIMEOUT * 1000)))
                if socks.get(self.zmq_client) == zmq.POLLIN:
                    response = self.zmq_client.recv()
                    if response:
                        last_work_result = self._handle_response(response)
                    else:
                        raise zmq.ZMQError()
                else:
                    raise zmq.ZMQError()
            except zmq.ZMQError:
                last_work_result = self.EmptyWorkResult()
                self._disconnect_client()
                logging.info('Server connection closed. Waiting {} seconds, then reconnect...'.format(self.TIMEOUT))
                time.sleep(self.TIMEOUT)

    def _handle_response(self, response):
        message = pickle.loads(response)
        if isinstance(message, self.WaitResponse):
            time.sleep(message.wait_time)
            return self.EmptyWorkResult()
        if isinstance(message, self.SelfplayWorkResponse):
            logging.info('Start processing Selfplay work request from master...')
            self._prepare_networks(message.nn_name, message.weights_zip_binary)
            evaluation_lists = self._play_games(message.n_games, message.game_states, message.simulations_per_turn)
            return self.SelfplayWorkResult(evaluation_lists)
        if isinstance(message, self.SelfEvaluationWorkResponse):
            logging.info('Start processing SelfEvaluation work request from master...')
            self._prepare_networks(message.nn_name, message.weights_binary_one, message.weights_binary_two)
            scores = self._self_evaluate(message.n_games, message.game_states, message.simulations_per_turn)
            return self.SelfEvaluationWorkResult(scores[0], scores[1], message.n_games)
        if isinstance(message, self.AIEvaluationWorkResponse):
            logging.info('Start processing AIEvaluation work request from master...')
            if self.config.external_evaluation_possible():
                self._prepare_networks(message.nn_name, message.weights_zip_binary)
                scores = self._ai_evaluate(message.n_games, message.game_states, message.turn_time)
                return self.AIEvaluationWorkResult(scores[0], scores[1], message.n_games)
            else:
                logging.warning('Can not process work request, AITrivial executable is missing!')
                logging.warning('Waiting 30 seconds and retry work request...')
                time.sleep(30)
                return self.EmptyWorkResult()

        return self.EmptyWorkResult()

    def _prepare_networks(self, nn_class_name, weights_zip_binary, weights_zip_binary_two=None):
        if self.nn_class_name_one != nn_class_name:
            self._restart_network_one(nn_class_name)
        if self.nn_class_name_two != nn_class_name:
            self._restart_network_two(nn_class_name)

        if weights_zip_binary:
            self.nn_client_one.load_weights(weights_zip_binary)
        if weights_zip_binary_two:
            self.nn_client_two.load_weights(weights_zip_binary_two)

    def _restart_network_one(self, nn_class_name):
        self.nn_class_name_one = nn_class_name
        if self.nn_client_one:
            self.nn_client_one.shutdown_server()
            self.nn_client_one.stop()
            time.sleep(15)

        neural_network.start_nn_server(self.config.nn_server_selfplay_port(), self.nn_class_name_one, self.config)
        self.nn_client_host_one = 'tcp://localhost:{}'.format(self.config.nn_server_selfplay_port())
        self.nn_client_one = neural_network.NeuralNetworkClient(self.nn_client_host_one)
        self.nn_client_one.start(self.config)

    def _restart_network_two(self, nn_class_name):
        self.nn_class_name_two = nn_class_name
        if self.nn_client_two:
            self.nn_client_two.shutdown_server()
            self.nn_client_two.stop()
            time.sleep(15)

        neural_network.start_nn_server(self.config.nn_server_selfeval_port(), self.nn_class_name_two, self.config)
        self.nn_client_host_two = 'tcp://localhost:{}'.format(self.config.nn_server_selfeval_port())
        self.nn_client_two = neural_network.NeuralNetworkClient(self.nn_client_host_two)
        self.nn_client_two.start(self.config)

    def _play_games(self, n_games, game_states, simulations_per_turn):
        results = []
        for _ in range(n_games):
            nn_executor_client = neural_network.NeuralNetworkClient(self.nn_client_host_one)
            game_state = np.random.choice(game_states)

            params = (game_state, nn_executor_client, simulations_per_turn, self.config)
            result = self.process_pool.apply_async(PlayingSlave._play_game, params,
                                                   callback=PlayingSlave.selfplay_callback)
            results.append(result)

        evaluation_lists = []
        for result in results:
            evaluation_lists.append(result.get())

        return evaluation_lists

    @staticmethod
    def _play_game(game_state, nn_executor_client, n_simulations, config):
        try:
            selfplay_executor = executors.SelfplayExecutor(game_state, [nn_executor_client], n_simulations, config)
            result = selfplay_executor.run()
        except Exception as e:
            # Print and re-raise the exception, as python will ignore it otherwise
            print(traceback.format_exc())
            raise e

        return result

    @staticmethod
    def selfplay_callback(result):
        logging.info('Selfplay-Game finished.')

    def _self_evaluate(self, n_games, game_states, simulations_per_turn):
        results = []
        for _ in range(n_games):
            nn_executor_client_one = neural_network.NeuralNetworkClient(self.nn_client_host_one)
            nn_executor_client_two = neural_network.NeuralNetworkClient(self.nn_client_host_two)

            game_state = np.random.choice(game_states)

            params = (game_state, nn_executor_client_one, nn_executor_client_two, simulations_per_turn, self.config)
            result = self.process_pool.apply_async(PlayingSlave._play_evaluation_game, params,
                                                   callback=PlayingSlave.eval_callback)
            results.append(result)

        total_score = [0, 0]
        for result in results:
            score = result.get()
            total_score[0] += score[0]
            total_score[1] += score[1]

        return total_score

    @staticmethod
    def _play_evaluation_game(game_state, nn_executor_one, nn_executor_two, n_simulations, config):
        model_evaluator = executors.ModelEvaluator(nn_executor_one, nn_executor_two, game_state, config)
        result = model_evaluator.run(n_simulations)

        return result

    @staticmethod
    def eval_callback(result):
        logging.info('Evaluation-Game finished. ({}:{})'.format(result[0], result[1]))

    def _ai_evaluate(self, n_games, game_states, turn_time):
        results = []
        for _ in range(n_games):
            nn_executor_client_one = neural_network.NeuralNetworkClient(self.nn_client_host_one)

            game_state = np.random.choice(game_states)

            params = (game_state, nn_executor_client_one, turn_time, self.config)
            result = self.process_pool.apply_async(PlayingSlave._play_ai_evaluation_game, params,
                                                   callback=PlayingSlave.ai_eval_callback)
            results.append(result)

        total_score = [0, 0]
        for result in results:
            score = result.get()
            for i in range(2):
                total_score[i] += score[i]

        return total_score

    @staticmethod
    def _play_ai_evaluation_game(game_state, nn_executor_one, turn_time, config: Configuration):
        ai_evaluator = executors.ExternalEvaluator(nn_executor_one, config.external_ai_agent(game_state),
                                                   game_state, config)
        result = ai_evaluator.run(turn_time)

        return result

    @staticmethod
    def ai_eval_callback(result):
        score = result
        logging.info('AI-Evaluation-Game finished. ({}:{})'.format(score[0], score[1]))

    def _connect_client(self):
        if not self.zmq_client:
            self.zmq_client = self.context.socket(zmq.REQ)
            util.secure_client_connection(self.zmq_client, self.context, self.config)
            self.zmq_client.connect(self.master_address)

            self.poll.register(self.zmq_client, zmq.POLLIN)

    def _disconnect_client(self):
        if self.zmq_client:
            self.zmq_client.setsockopt(zmq.LINGER, 0)
            self.zmq_client.close()
            self.poll.unregister(self.zmq_client)
            self.zmq_client = None


class TrainingMaster:
    """Master Node that coordinates the training process.

    This will order slaves to execute selfplay, selfevaluation and aievaluation games,
    collect the results and use them to train a neural network instance."""
    class State(enum.Enum):
        SELFPLAY = 'SELFPLAY'
        SELFEVAL = 'SELFEVAL'
        AIEVAL = 'AIEVAL'

    DATA_DIR = 'selfplay-data'
    WEIGHTS_DIR = 'weights-history'
    BEST_WEIGHTS = 'best-weights.zip'
    LOG_DIR = 'tensorboard-logs'

    def __init__(self, work_dir, nn_name, start_game_states, config: Configuration=None):
        if not config:
            # Use default configuration...
            config = Configuration()
        self.config = config

        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.weights_dir = os.path.join(self.work_dir, self.WEIGHTS_DIR)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        self.log_dir = os.path.join(self.work_dir, self.LOG_DIR, 'current-run')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.nn_name = nn_name
        self.port = config.training_master_port()

        self.start_game_states = start_game_states

        self.server = None
        self.context = None

        self.training_executor = None
        self.nn_client = None

        # weights cache
        self.current_weights_binary = None
        self.best_weights_binary = None

        # training threads
        self.stopped = False
        self.training_thread_one = None
        self.training_thread_two = None
        self.training_progress_lock = threading.Lock()

        # keep some stats about our progress
        self.progress = TrainingRunProgress(os.path.join(work_dir, 'stats.json'))
        self.progress.load_stats()

        # We could customize settings here, but defaults are fine
        self._copy_stats(config)

        self.progress.save_stats()

    def _copy_stats(self, config: Configuration):
        self.progress.stats.settings.turn_time = config.external_evaluation_turn_time()
        self.progress.stats.settings.needed_avg_score = config.needed_avg_self_eval_score()
        self.progress.stats.settings.n_ai_eval = config.n_external_eval()
        self.progress.stats.settings.simulations_per_turn = config.simulations_per_turn()
        self.progress.stats.settings.n_self_eval = config.n_self_eval()
        self.progress.stats.settings.n_self_play = config.n_self_play()
        self.progress.stats.settings.batch_size = config.training_batch_size()
        self.progress.stats.settings.training_history_size = config.training_history_size()

    def run(self):
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        util.secure_server_connection(self.server, self.context, self.config)
        self.server.bind('tcp://*:{}'.format(self.port))

        self._setup_nn()
        self._setup_training_executor()
        self._setup_training_threads()
        try:
            self._handle_messages()
        except KeyboardInterrupt:
            logging.error('Keyboard Interrupt, shutting down Training Master...')
        finally:
            self.progress.save_stats()

            self.stopped = True
            self.nn_client.stop()
            self.server.setsockopt(zmq.LINGER, 0)
            self.server.close()

            logging.info('Terminating process in 5 seconds...')
            time.sleep(5)
            os._exit(1)

    def _setup_nn(self):
        neural_network.start_nn_server(self.config.nn_server_training_port(), self.nn_name, self.config,
                                  log_dir=self.log_dir, start_batch=self.progress.stats.progress.current_batch)
        self.nn_client = neural_network.NeuralNetworkClient('tcp://localhost:{}'.format(self.config.nn_server_training_port()))
        self.nn_client.start(self.config)

        if self.progress.stats.progress.iteration > 0:
            checkpoint_name = 'checkpoint-{0:05d}.zip'.format(self.progress.stats.progress.iteration)
            with open(os.path.join(self.weights_dir, checkpoint_name), 'rb') as file:
                weights_binary = file.read()
                self.nn_client.load_weights(weights_binary)

            with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'rb') as file:
                self.best_weights_binary = file.read()
        else:
            self.progress.stats.progress.iteration = 1
            self.best_weights_binary = self.nn_client.save_weights()
            self._save_best_weights()

        self.current_weights_binary = self.nn_client.save_weights()
        self._save_current_weights()
        self.progress.save_stats()

    def _setup_training_executor(self):
        self.training_executor = executors.TrainingExecutor(self.nn_client, os.path.join(self.work_dir, self.DATA_DIR),
                                                            self.progress.stats.settings.training_history_size)

    def _setup_training_threads(self):
        self.training_thread_one = threading.Thread(target=self._run_training_loop)
        self.training_thread_one.start()
        self.training_thread_two = threading.Thread(target=self._run_training_loop)
        self.training_thread_two.start()

    def _run_training_loop(self):
        while not self.stopped:
            self._add_training_progress()
            try:
                self.training_executor.run_training_batch(self.progress.stats.settings.batch_size)

                if self.progress.stats.progress.current_batch % self.config.nn_log_step_size() == 0:
                    self.training_executor.run_logging_batch(self.progress.stats.settings.batch_size)
            except zmq.error.ContextTerminated:
                return

    def _add_training_progress(self):
        with self.training_progress_lock:
            self.progress.stats.progress.current_batch += 1
            if self.progress.stats.progress.current_batch % 100 == 0:
                self.progress.save_stats()

    def _handle_messages(self):
        while True:
            request = self.server.recv_pyobj()

            if isinstance(request, PlayingSlave.WorkRequest):
                logging.debug('Work request {}.'.format(request))
                self._handle_work_request(request)
            else:
                self.server.send('unsupported message type')

    def _handle_work_request(self, request: PlayingSlave.WorkRequest):
        work_result = request.work_result

        if isinstance(work_result, PlayingSlave.SelfplayWorkResult):
            self._handle_selfplay_result(work_result)
        if isinstance(work_result, PlayingSlave.SelfEvaluationWorkResult):
            self._handle_selfeval_result(work_result)
        if isinstance(work_result, PlayingSlave.AIEvaluationWorkResult):
            self._handle_aieval_result(work_result)

        self._send_work_response()

    def _send_work_response(self):
        n_games = 7
        simulations_per_turn = self.progress.stats.settings.simulations_per_turn
        epoch = self.progress.stats.progress.iteration
        turn_time = self.progress.stats.settings.turn_time
        game_states = self.start_game_states

        if self.progress.stats.progress.state == self.State.SELFPLAY:
            logging.info('Sending Selfplay Work Request to Playing Slave')
            self.server.send_pyobj(
                PlayingSlave.SelfplayWorkResponse(n_games, self.nn_name, self.best_weights_binary, game_states,
                                                  simulations_per_turn))
        if self.progress.stats.progress.state == self.State.SELFEVAL:
            logging.info('Sending Selfeval Work Request to Playing Slave')
            self.server.send_pyobj(
                PlayingSlave.SelfEvaluationWorkResponse(n_games, self.nn_name, self.current_weights_binary,
                                                        self.best_weights_binary, game_states, simulations_per_turn,
                                                        epoch))
        if self.progress.stats.progress.state == self.State.AIEVAL:
            logging.info('Sending AI Eval Work Request to Playing Slave')
            self.server.send_pyobj(
                PlayingSlave.AIEvaluationWorkResponse(n_games, self.nn_name, self.best_weights_binary, game_states,
                                                      turn_time, epoch))

    def _handle_selfplay_result(self, work_result):
        logging.info('Getting Selfplay Results from Playing Slave')
        n_evaluations = len(work_result.evaluation_lists)

        self.progress.stats.current_epoch().self_play.n_games += n_evaluations
        if self.progress.stats.progress.add_samples(self.State.SELFPLAY, n_evaluations):
            self._progress_to_selfeval()

            # Save a snapshot of the current weights
            self.current_weights_binary = self.nn_client.save_weights()
            self._save_current_weights()

        for evaluations in work_result.evaluation_lists:
            self.training_executor.add_examples(evaluations)

    def _progress_to_selfeval(self):
        # Progress to the next step of the training
        self.progress.stats.progress.n_remaining = self.progress.stats.settings.n_self_eval
        self.progress.stats.progress.state = self.State.SELFEVAL

        # Collect some stats on the current training step
        self.progress.stats.current_epoch().self_play.end_batch = self.progress.stats.progress.current_batch
        self.progress.stats.current_epoch().self_eval.start_batch = self.progress.stats.progress.current_batch

        if self.config.n_self_eval() <= 0:
            # Skip SelvEval
            logging.info('Skipping SelfEval...')

            # Be sure to still select the new weights as best ones
            self._select_new_weights_as_best()

            self._progress_to_ai_eval()
        else:
            logging.info('Start Self-Evaluation for Iteration {}...'.format(self.progress.stats.progress.iteration))

    def _progress_to_ai_eval(self):
        # Progress to the next step of the training
        self.progress.stats.progress.n_remaining = self.progress.stats.settings.n_ai_eval
        self.progress.stats.progress.state = self.State.AIEVAL

        # Collect some stats on the current training step
        self.progress.stats.current_epoch().self_eval.end_batch = self.progress.stats.progress.current_batch
        self.progress.stats.current_epoch().ai_eval.start_batch = self.progress.stats.progress.current_batch

        if self.config.n_external_eval() <= 0:
            # Skip external ai eval
            logging.info('Skipping External AI Eval...')
            self._progress_to_selfplay()
        else:
            logging.info('Start AI-Evaluation for Iteration {}...'.format(self.progress.stats.progress.iteration))

    def _progress_to_selfplay(self):
        # Progress to the next step of the training
        self.progress.stats.progress.n_remaining = self.progress.stats.settings.n_self_play
        self.progress.stats.progress.state = self.State.SELFPLAY
        self.progress.stats.progress.iteration += 1

        # Collect some stats on the current training step
        self.progress.stats.current_epoch().ai_eval.end_batch = self.progress.stats.progress.current_batch
        self.progress.stats.current_epoch().self_play.start_batch = self.progress.stats.progress.current_batch

    def _save_current_weights(self):
        checkpoint_name = 'checkpoint-{0:05d}.zip'.format(self.progress.stats.progress.iteration)
        with open(os.path.join(self.weights_dir, checkpoint_name), 'wb') as file:
            file.write(self.current_weights_binary)

    def _save_best_weights(self):
        with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'wb') as file:
            file.write(self.best_weights_binary)

    def _handle_selfeval_result(self, work_result):
        logging.info('Getting Selfeval Results from Playing Slave (New {} vs. Old {})'.format(work_result.nn_one_score, work_result.nn_two_score))

        self.progress.stats.current_epoch().self_eval.n_games += work_result.n_games
        self.progress.stats.current_epoch().self_eval.old_score += work_result.nn_two_score
        self.progress.stats.current_epoch().self_eval.new_score += work_result.nn_one_score

        if self.progress.stats.progress.add_samples(self.State.SELFEVAL, work_result.n_games):
            self._progress_to_ai_eval()

            # See if the new nn was better
            old_score = self.progress.stats.current_epoch().self_eval.old_score
            new_score = self.progress.stats.current_epoch().self_eval.new_score
            n_games = self.progress.stats.current_epoch().self_eval.n_games

            logging.info('Finishing selfplay with result {} vs. {}'
                         .format(new_score, old_score))
            if new_score / n_games > self.progress.stats.settings.needed_avg_score:
                logging.info('Choosing new weights, as it scored better then the current best.')
                self.progress.stats.current_epoch().self_eval.new_better = True
                self.best_weights_binary = self.current_weights_binary

                self._save_best_weights()
            else:
                logging.info('Choosing old weights, as the new ones where not better.')
                self.progress.stats.current_epoch().self_eval.new_better = False

    def _select_new_weights_as_best(self):
        self.progress.stats.current_epoch().self_eval.new_better = True
        self.best_weights_binary = self.current_weights_binary

        self._save_best_weights()

    def _handle_aieval_result(self, work_result):
        logging.info('Getting external AI evaluation Results from Playing Slave (NN {} vs. AI {})'.format(work_result.nn_score, work_result.ai_score))

        self.progress.stats.current_epoch().ai_eval.n_games += work_result.n_games
        self.progress.stats.current_epoch().ai_eval.ai_score += work_result.ai_score
        self.progress.stats.current_epoch().ai_eval.nn_score += work_result.nn_score

        if self.progress.stats.progress.add_samples(self.State.AIEVAL, work_result.n_games):
            self._progress_to_selfplay()

            logging.info('Start Selfplay for Iteration {}...'.format(self.progress.stats.progress.iteration))
            self._save_current_weights()


class TrainingRunProgress:
    """Captures all important information about one training run.

    This includes the settings for the run as well as it's progress.
    This can also be used to coordinate the run itself, including continuing a run."""
    def __init__(self, stats_file_name):
        self.stats_file_name = stats_file_name
        self.stats = TrainingRunStats()
        self.stats.progress.n_remaining = self.stats.settings.n_self_play

    def save_stats(self):
        # Store the enum as an string.
        # Could be cleaned up by using an appropriate mapper.
        to_save = util.deepcopy(self.stats)
        to_save.progress.state = self.stats.progress.state.value

        mapper = TrainingRunStatsMapper(obj=to_save)
        json_string = json.dumps(mapper.serialize(), indent=4)

        with open(self.stats_file_name + '-copy', 'w') as stats_file:
            stats_file.write(json_string)
        with open(self.stats_file_name, 'w') as stats_file:
            stats_file.write(json_string)

    def load_stats(self):
        if os.path.isfile(self.stats_file_name):
            with open(self.stats_file_name, 'r') as stats_file:
                json_data = json.loads(stats_file.read())
                mapper = TrainingRunStatsMapper(data=json_data)
                loaded = mapper.marshal()
                loaded.progress.state = TrainingMaster.State(loaded.progress.state)

                self.stats = loaded


class TrainingRunStats:
    """Data container for the actual training run stats."""
    class Settings:
        def __init__(self):
            """Init's with default settings. Overwrite if needed."""
            # Number of game states for one training batch
            self.batch_size = 64
            # Number of last games used for training
            self.training_history_size = 128
            # Simulations per selfplay/selfeval turn
            self.simulations_per_turn = 128
            # Turn time for each player during ai evaluation
            self.turn_time = 1.0

            # Number of selfplay games for each iteration
            self.n_self_play = 42
            # Number of self evaluation games for each iteration
            self.n_self_eval = 21
            # Number of evaluation games against the ai-trivial client for each client
            self.n_ai_eval = 14

            # The self evaluation avg. score needed to see this iteration as new best
            self.needed_avg_score = 0.05

    class Progress:
        def __init__(self):
            """Progress statistics of one training run."""
            # Our current state
            self.state = TrainingMaster.State.SELFPLAY
            # The number of samples of the current state needed to progress to the next state
            self.n_remaining = 0
            # The current iteration
            self.iteration = 0
            # The current batch
            self.current_batch = 0

        def add_samples(self, state, n_samples):
            """Call when new samples arrive. Returns True if progress to the next state should be made."""
            if self.state == state:
                self.n_remaining -= n_samples
                if self.n_remaining <= 0:
                    return True

            return False

    class Iteration:
        class SelfEval:
            pass
        class SelfPlay:
            pass
        class AIEval:
            pass

        def __init__(self):
            self.self_eval = self.SelfEval()
            self.self_eval.n_games = 0
            self.self_eval.old_score = 0
            self.self_eval.new_score = 0
            self.self_eval.start_batch = 0
            self.self_eval.end_batch = 0
            self.self_eval.new_better = False
            self.self_eval.avg_score = 0

            self.self_play = self.SelfPlay()
            self.self_play.start_batch = 0
            self.self_play.end_batch = 0
            self.self_play.n_games = 0

            self.ai_eval = self.AIEval()
            self.ai_eval.n_games = 0
            self.ai_eval.start_batch = 0
            self.ai_eval.end_batch = 0
            self.ai_eval.nn_score = 0
            self.ai_eval.ai_score = 0

    def __init__(self):
        self.settings = self.Settings()
        self.progress = self.Progress()
        self.iterations = []

    def current_epoch(self):
        while len(self.iterations) < self.progress.iteration:
            self.iterations.append(self.Iteration())

        return self.iterations[self.progress.iteration - 1]


# Python has NO good json serializer, so we need some boilerplate
class SettingsMapper(kim.Mapper):
    __type__ = TrainingRunStats.Settings
    batch_size = kim.field.Integer()
    training_history_size = kim.field.Integer()
    simulations_per_turn = kim.field.Integer()
    turn_time = kim.field.Float()
    n_self_play = kim.field.Integer()
    n_self_eval = kim.field.Integer()
    n_ai_eval = kim.field.Integer()
    needed_avg_score = kim.field.Float()


class ProgressMapper(kim.Mapper):
    __type__ = TrainingRunStats.Progress
    state = kim.field.String()
    n_remaining = kim.field.Integer()
    iteration = kim.field.Integer()
    current_batch = kim.field.Integer()


class IterationSelfEvalMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration.SelfEval
    n_games = kim.field.Integer()
    old_score = kim.field.Integer()
    new_score = kim.field.Integer()
    start_batch = kim.field.Integer()
    end_batch = kim.field.Integer()
    new_better = kim.field.Boolean()

class IterationSelfPlayMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration.SelfPlay
    n_games = kim.field.Integer()
    start_batch = kim.field.Integer()
    end_batch = kim.field.Integer()


class IterationAIEvalMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration.AIEval
    n_games = kim.field.Integer()
    start_batch = kim.field.Integer()
    end_batch = kim.field.Integer()
    nn_score = kim.field.Integer()
    ai_score = kim.field.Integer()


class IterationMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration
    self_eval = kim.field.Nested(IterationSelfEvalMapper, allow_create=True)
    ai_eval = kim.field.Nested(IterationAIEvalMapper, allow_create=True)
    self_play = kim.field.Nested(IterationSelfPlayMapper, allow_create=True)


class TrainingRunStatsMapper(kim.Mapper):
    __type__ = TrainingRunStats
    settings = kim.field.Nested(SettingsMapper, allow_create=True)
    progress = kim.field.Nested(ProgressMapper, allow_create=True)
    iterations = kim.field.Collection(kim.field.Nested(IterationMapper, allow_create=True), allow_create=True)
