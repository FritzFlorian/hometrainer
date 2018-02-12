"""The server component of our decoupled neural network.

This actually runs the neural network in a separate process.
See the nn_client for details."""
import hometrainer.nn_client as nn_client
import hometrainer.util as util
from hometrainer.config import Configuration
import zmq
import tensorflow as tf
import os
import logging
import time
import pickle


class NeuralNetworkServer:
    def __init__(self, port, neural_network, batch_size, config: Configuration, start_batch=0):
        self.port = port
        self.neural_network = neural_network
        self.input_conversion = neural_network.input_conversion_function()
        self.output_conversion = neural_network.output_conversion_function()
        self.stopped = False
        self.socket = None
        self.execution_responses = []
        self.batch_size = batch_size
        self.config = config

        self.log_dir = config.nn_server_tensorboard_logdir(port, neural_network)

        self.n_batches_for_log = 100
        self.current_training_batch = start_batch
        self.log_file_writer = None
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
                            if isinstance(message_content, nn_client.ShutdownRequest):
                                self._process_shutdown_request(response_ids, message_content)
                            elif isinstance(message_content, nn_client.ExecutionRequest):
                                self._process_execution_request(response_ids, message_content, sess)
                            elif isinstance(message_content, nn_client.TrainingRequest):
                                self._process_training_request(response_ids, message_content, sess)
                            elif isinstance(message_content, nn_client.SaveWeightsRequest):
                                self._process_save_weights_request(response_ids, message_content, sess)
                            elif isinstance(message_content, nn_client.LoadWeightsRequest):
                                self._process_load_weights_request(response_ids, message_content, sess)
                            elif isinstance(message_content, nn_client.ConversionFunctionRequest):
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
        response = nn_client.ExecutionResponse(response_ids, message_content.input_array)
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
        if not self.log_file_writer:
            self.log_file_writer = tf.summary.FileWriter(self.log_dir, self.graph)

        if self.current_training_batch % self.n_batches_for_log == 0:
            # It's not smart to test with the training data.
            # But as we do not really have a test set here and we
            # will only train on this exact data after the summary data it seems 'ok' to do for now.
            self.neural_network.log_training_progress(sess, self.log_file_writer, message_content.input_arrays,
                                                      message_content.target_arrays, self.current_training_batch)

        # FIXME: Research what causes this call to leak memory if the inputs get too big.
        #        The leak happens at a fixed threshold on the input size (about 200).
        self.neural_network.train_batch(sess, message_content.input_arrays, message_content.target_arrays)
        self.current_training_batch += 1

        response = nn_client.Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_save_weights_request(self, response_ids, message_content, sess):
        checkpoint_content = util.save_neural_net_to_zip_binary(self.neural_network, sess)
        response = nn_client.SaveWeightsResponse(response_ids, checkpoint_content)
        self.socket.send_multipart(response.to_multipart())

    def _process_load_weights_request(self, response_ids, message_content, sess):
        util.load_neural_net_from_zip_binary(message_content.checkpoint_data, self.neural_network, sess)

        response = nn_client.Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_conversion_function_request(self, response_ids, message_content):
        response = nn_client.ConversionFunctionResponse(response_ids, self.input_conversion, self.output_conversion)
        self.socket.send_multipart(response.to_multipart())

    def _process_shutdown_request(self, response_ids, message_content):
        self.stopped = True

        response = nn_client.Response(response_ids)
        self.socket.send_multipart(response.to_multipart())
