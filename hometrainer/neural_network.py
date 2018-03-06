import pickle
import multiprocessing
import numpy as np
import logging
import importlib
import zmq
import time
import random
import threading
import hometrainer.core as core
import hometrainer.util as util
from hometrainer.config import Configuration
import tensorflow as tf



class NeuralNetwork:
    """Wrapper that represents a single neural network instance.

    This is intended to abstract away the actual creation, training and execution of the neural network.
    This should hopefully also allow to re-use major parts of the code for different network structures.

    The network is not responsible for managing its scope/tensorflow graph, this is managed in the
    nn_server module and abstracts away the actual tf session and the interaction with it.

    See examples for how to implement these, but mostly all functions follow their brief descriptions."""
    def construct_network(self, sess, graph):
        raise NotImplementedError("Add the construction of your custom graph structure.")

    def init_network(self):
        raise NotImplementedError("Run initialisation code for your network.")

    def input_conversion_function(self):
        raise NotImplementedError("Return a reference to the function converting evaluations to input for thin NN.")

    def output_conversion_function(self):
        raise NotImplementedError("Return a reference to the function filling outputs into evaluations.")

    def execute_batch(self, sess, input_arrays):
        raise NotImplementedError("Add implementation that takes prepared input arrays and executes them as a batch.")

    def train_batch(self, sess, input_arrays, output_arrays):
        raise NotImplementedError("Add implementation that executes one batch training step.")

    def save_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that saves the weights of this network to a checkpoint.")

    def load_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that loads the weights of this network to a checkpoint.")

    def log_training_progress(self, sess, tf_file_writer, input_arrays, target_arrays, training_batch):
        raise NotImplementedError("Add implementation to write stats on the current training progress.")


class NeuralNetworkClient:
    """Client Functionality for our neural network abstraction.

    The neural network is abstracted into an client and server, in the way that the actual network can run as
    a 'server' and that the client acts as a simple proxy/interface to use the neural network.
    This helps, as we can then simply call functions on the neural network in our alpha zero code,
    without worrying how it is actually executed, it's a plain function call.

    The biggest drawback is, that we have to serialize/deserialize requests to the network.
    This might be a problem, but can be solved by using foster deserialization like ray.
    One extra bonus of decoupling the neural network is that we have no synchronisation issues and
    that the network runs in it's own process, so it can use extra cpu cores without dragging down the
    main program."""
    def __init__(self, address):
        self.address = address

        self.context = None
        self.nn_server_socket = None
        self.internal_router = None

        self.input_conversion = None
        self.output_conversion = None

        # Make sure we will never get collisions. The name is arbitrary anyways,
        # as it is only used to handle concurrent use of the neural network client.
        self.internal_address = 'inproc://nn_client_router_{}_{}'.format(time.time(), random.random)

    def evaluate_game_state(self, game_state: core.GameState) -> core.Evaluation:
        """Executes a neural network to evaluate a given game state.

        Returns the evaluation generated by the network.
        Might block for a while because other evaluations are being performed."""
        # Create the evaluation to send to the NN.
        # We will apply rotations on random instances.
        evaluation = game_state.wrap_in_evaluation()
        evaluation = evaluation.convert_to_normal()
        evaluation = evaluation.apply_transformation(random.randint(0, 6))

        # Execute it using the neural network process.
        # We hand it through our internal proxy to be able to
        # map parallel execution results properly.
        input_array, target_array = self.input_conversion(evaluation)
        response = self.send_to_server(ExecutionRequest(input_array))
        evaluation = self.output_conversion(evaluation, response.output_array)

        # Undo our random rotation on the instance.
        evaluation = evaluation.undo_transformations()
        evaluation = evaluation.convert_from_normal()

        return evaluation

    def _evaluations_to_inputs(self, evaluations):
        inputs = []
        targets = []
        for evaluation in evaluations:
            input_array, target_array = self.input_conversion(evaluation, calculate_target=True)
            inputs.append(input_array)
            targets.append(target_array)

        return inputs, targets

    def execute_training_batch(self, evaluations):
        """Executes a batch of training using the given evaluations."""
        inputs, targets = self._evaluations_to_inputs(evaluations)

        return self.send_to_server(TrainingRequest(np.asarray(inputs), np.asarray(targets)))

    def execute_logging_batch(self, evaluations, test_evaluations):
        inputs, targets = self._evaluations_to_inputs(evaluations)
        test_inputs, test_targets = self._evaluations_to_inputs(test_evaluations)

        return self.send_to_server(LoggingRequest(inputs, targets, test_inputs, test_targets))

    def save_weights(self):
        """Saves the current weights to a binary of it's checkpoint."""
        response = self.send_to_server(SaveWeightsRequest())
        return response.checkpoint_data

    def load_weights(self, checkpoint_zip_binary):
        """Loads a checkpoint from the binary of it's checkpoint."""
        self.send_to_server(LoadWeightsRequest(checkpoint_zip_binary))

    def shutdown_server(self):
        return self.send_to_server(ShutdownRequest())

    def send_to_server(self, message):
        """Sends a message to the NN Server and returns its response message.

        This method is thread save and will block each thread until the individual answer
        has been returned by the server (e.g. for execution requests this can take until
        a full batch was executed)"""
        client = self.context.socket(zmq.REQ)
        client.connect(self.internal_address)

        client.send_pyobj(message)
        response = client.recv_pyobj()

        client.close()

        return response

    def start(self, config: Configuration):
        self.context = zmq.Context()

        self.nn_server_socket = self.context.socket(zmq.DEALER)
        util.secure_client_connection(self.nn_server_socket, self.context, config)
        self.nn_server_socket.connect(self.address)

        self.internal_router = self.context.socket(zmq.ROUTER)
        self.internal_router.bind(self.internal_address)

        def run_proxy():
            try:
                zmq.proxy(self.internal_router, self.nn_server_socket)
            except zmq.ZMQError:
                pass  # The closed proxy is on purpose, so ignore the error.

        threading.Thread(target=run_proxy).start()

        # Get the conversion methods
        response = self.send_to_server(ConversionFunctionRequest())
        self.input_conversion = response.input_conversions
        self.output_conversion = response.output_conversions

    def stop(self):
        self.internal_router.setsockopt(zmq.LINGER, 0)
        self.internal_router.close()
        self.nn_server_socket.setsockopt(zmq.LINGER, 0)
        self.nn_server_socket.close()
        self.context.term()


def start_nn_server(port, nn_class_name, config: Configuration, nn_init_args=(), batch_size=32, log_dir=None, start_batch=0):
    """Starts a neural network server on the given port.

    The server is started using the venv cpython executable.
    The nn_class_name must be the fully qualified name of the neural network class.
    This method can be used from any python interpreter, for example also from
    the pypy interpreter, as all tensorflow components are loaded in the new process."""
    logging.info('Starting NN Server on port {}.'.format(port))
    p = multiprocessing.Process(target=_start_nn_server_internal,
                                args=(port, nn_class_name, batch_size, log_dir, start_batch, nn_init_args, config))
    p.start()
    return p


def _start_nn_server_internal(port, nn_class_name, batch_size, log_dir, start_batch,
                              nn_init_args, config: Configuration):
    module_name, class_name = nn_class_name.rsplit('.', 1)
    nn_module = importlib.import_module(module_name)
    nn_class = getattr(nn_module, class_name)

    server = NeuralNetworkServer(port, nn_class(*nn_init_args), batch_size,
                                 config=config, start_batch=start_batch, log_dir=log_dir)
    server.run()


# Message types used to communicate with the NNServer
class AbstractMessage:
    pass


class Response(AbstractMessage):
    def __init__(self, response_ids):
        self.response_ids = response_ids

    def to_multipart(self):
        return self.response_ids + [b'', pickle.dumps(self)]


class ShutdownRequest(AbstractMessage):
    pass


class SaveWeightsRequest(AbstractMessage):
    pass


class SaveWeightsResponse(Response):
    def __init__(self, response_ids, checkpoint_data):
        super().__init__(response_ids)
        self.checkpoint_data = checkpoint_data


class LoadWeightsRequest(AbstractMessage):
    def __init__(self, checkpoint_data):
        self.checkpoint_data = checkpoint_data


class ConversionFunctionRequest(AbstractMessage):
    pass


class ConversionFunctionResponse(Response):
    def __init__(self, response_ids, input_conversion, output_conversion):
        super().__init__(response_ids)
        self.input_conversions = input_conversion
        self.output_conversions = output_conversion


class ExecutionRequest(AbstractMessage):
    def __init__(self, input_array):
        self.input_array = input_array


class ExecutionResponse(Response):
    def __init__(self, response_ids, input_array):
        super().__init__(response_ids)
        self.input_array = input_array
        self.output_array = None

    def set_output(self, output_array):
        self.output_array = output_array
        self.input_array = None


class TrainingRequest(AbstractMessage):
    def __init__(self, input_arrays, target_arrays):
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays


class LoggingRequest(AbstractMessage):
    def __init__(self, input_arrays, target_arrays, test_input_arrays, test_target_arrays):
        # Training Set
        self.input_arrays = input_arrays
        self.target_arrays = target_arrays

        # Test Set
        self.test_input_arrays = test_input_arrays
        self.test_target_arrays = test_target_arrays


class NeuralNetworkServer:
    """The server component of our decoupled neural network.

    This actually runs the neural network in a separate process.
    See the nn_client for details."""
    def __init__(self, port, neural_network, batch_size, log_dir: None, config: Configuration, start_batch=0):
        self.port = port
        self.neural_network = neural_network
        self.input_conversion = neural_network.input_conversion_function()
        self.output_conversion = neural_network.output_conversion_function()
        self.stopped = False
        self.socket = None
        self.execution_responses = []
        self.batch_size = batch_size
        self.config = config

        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = config.nn_server_tensorboard_logdir(port, neural_network)
        self.test_log_dir = self.log_dir + '-test'

        self.current_training_batch = start_batch
        self.log_file_writer = None
        self.test_log_file_writer = None
        self.graph = None

    def run(self):
        # Init network code. Router is used because we want to synchronize
        # all request in in a request-reply fashion.
        context = zmq.Context()
        self.socket = context.socket(zmq.ROUTER)
        util.secure_server_connection(self.socket, context, self.config)
        self.socket.bind('tcp://*:{}'.format(self.port))

        # Shutdown gracefully in case of interrupts
        try:
            # Setup a tensorflow session to be used for the whole run.
            self.graph = tf.Graph()
            with self.graph.as_default():
                # GPU Memory is allocated only as needed, this allows to run multiple
                # sessions on one machine.
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    self.neural_network.construct_network(sess, self.graph)
                    self.neural_network.init_network()

                    while not self.stopped:
                        try:
                            # Multipart messages are needed to correctly map the response
                            message = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                            response_ids = message[0:-2]
                            message_content = pickle.loads(message[-1])

                            # Process the incoming message...
                            if isinstance(message_content, ShutdownRequest):
                                self._process_shutdown_request(response_ids, message_content)
                            elif isinstance(message_content, ExecutionRequest):
                                self._process_execution_request(response_ids, message_content, sess)
                            elif isinstance(message_content, TrainingRequest):
                                self._process_training_request(response_ids, message_content, sess)
                            elif isinstance(message_content, LoggingRequest):
                                self._process_logging_request(response_ids, message_content, sess)
                            elif isinstance(message_content, SaveWeightsRequest):
                                self._process_save_weights_request(response_ids, message_content, sess)
                            elif isinstance(message_content, LoadWeightsRequest):
                                self._process_load_weights_request(response_ids, message_content, sess)
                            elif isinstance(message_content, ConversionFunctionRequest):
                                self._process_conversion_function_request(response_ids, message_content)
                            else:
                                print("Unknown message '{}' received!".format(message_content))

                        except zmq.ZMQError:
                            # Also execute not full batches if no new data arrived in time
                            if len(self.execution_responses) >= 1:
                                self._execute_batch(sess)
                            else:
                                # Don't  busy wait all the time
                                time.sleep(0.01)
                if self.log_file_writer:
                    self.log_file_writer.close()
        except KeyboardInterrupt:
            logging.error('Keyboard Interrupt, shutting down NN server...')
        finally:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            context.term()

    def _process_execution_request(self, response_ids, message_content, sess):
        response = ExecutionResponse(response_ids, message_content.input_array)
        self.execution_responses.append(response)

        if len(self.execution_responses) > self.batch_size:
            self._execute_batch(sess)

    def _execute_batch(self, sess):
        inputs = [execution_response.input_array for execution_response in self.execution_responses]
        outputs = self.neural_network.execute_batch(sess, inputs)

        for i in range(len(outputs)):
            self.execution_responses[i].set_output(outputs[i])
            self.socket.send_multipart(self.execution_responses[i].to_multipart())

        self.execution_responses = []

    def _process_training_request(self, response_ids, message_content, sess):
        # FIXME: Research what causes this call to leak memory if the inputs get too big.
        #        The leak happens at a fixed threshold on the input size (about 200).
        self.neural_network.train_batch(sess, message_content.input_arrays, message_content.target_arrays)
        self.current_training_batch += 1

        response = Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_logging_request(self, response_ids, message_content, sess):
        if not self.log_file_writer:
            self.log_file_writer = tf.summary.FileWriter(self.log_dir, self.graph)
        if not self.test_log_file_writer:
            self.test_log_file_writer = tf.summary.FileWriter(self.test_log_dir, self.graph)

        self.neural_network.log_training_progress(sess, self.log_file_writer, message_content.input_arrays,
                                                  message_content.target_arrays, self.current_training_batch)
        self.neural_network.log_training_progress(sess, self.test_log_file_writer, message_content.test_input_arrays,
                                                  message_content.test_target_arrays, self.current_training_batch)

        response = Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_save_weights_request(self, response_ids, message_content, sess):
        checkpoint_content = util.save_neural_net_to_zip_binary(self.neural_network, sess)
        response = SaveWeightsResponse(response_ids, checkpoint_content)
        self.socket.send_multipart(response.to_multipart())

    def _process_load_weights_request(self, response_ids, message_content, sess):
        util.load_neural_net_from_zip_binary(message_content.checkpoint_data, self.neural_network, sess)

        response = Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_conversion_function_request(self, response_ids, message_content):
        response = ConversionFunctionResponse(response_ids, self.input_conversion, self.output_conversion)
        self.socket.send_multipart(response.to_multipart())

    def _process_shutdown_request(self, response_ids, message_content):
        self.stopped = True

        response = Response(response_ids)
        self.socket.send_multipart(response.to_multipart())
