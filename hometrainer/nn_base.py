"""Base Class for actual neural network implementations used for training/playing.

Subclasses of this class can be used as neural networks for training/playing by
passing them into the main module. This ensures that you can easily define a network
structure that matches your problem case.
The input- and output conversion only returns a reference to a function instead of directly
transforming the input/output. This is a legacy of the codebase, but also serves the neat
addition that you can define the functions in an external file without tensorflow import.
This will prevent all 'non-tensorflow' tasks from performing the quite heavy tensorflow init code."""


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
