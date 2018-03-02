# Hometrainer

Hometrainer is a library that implements the AlphaZero algorithm for any turn based game.
It enables developers to use the algorithm to train agents for different games.


The goal of the library is mainly to enable runs of the algorithm with spare hardware at home,
as it allows to easily connect multiple clients to contribute to the training process.
That way you can try out training an AI using the AlphaZero algorithm at home. A decent setup is a
main training machine with a GPU and one or more secondary machines to execute games.


The library was used to train an AI for the Reversi (also called Othello) that
plays stronger than an classical, hand optimized Alpha-Beta-Agent. This training
took about 100 hours on a typical gaming pc plus an mid tier laptop. This shows that
you can train a decent AI without breaking the bank on an extreme amount of cloud resources.


## Usage

The following section summarizes the usage of the library. The library is by no means perfect, so
feel free to submit issues and pull requests for missing features.
For an example implementation of an draughts AI using this library see https://github.com/FritzFlorian/englishdraughts.

### Installation

Add the `hometrainer` dependency to your project.
This can be done with pip using 'pip install hometrainer'.

On windows you might want to use anaconda or manully install numpy.
If you have a GPU in your system you should `pip uninstall tensorflow` and install
the GPU version using `pip install tensorflow-gpu`. The library was tested with
tensorflow 1.5.0, but should also work fine with newer versions.
Please consult the tensorflow documentation for further installation instructions.


### Extending the required classes

To use the library for your own game you have to extend some abstract base classes and add the logic
of the game you are working on.


The required classes are `GameState`, `Move` and `Evaluation`. These can be found in the
`hometrainer.core` module. Please see their documentation and the sample draughs project for details.
Both `GameState` and `Move` should be constant, all data changes should happen on a copy to avoid side
effects.


Optionally you can implement an external AI Client to evaluate against during training.
This is highly recommended, as it is the most reliable metric on how your training proceeds.
Please see the sample implementation in english draughts for a simple alpha-beta-agent.


To configure all this you can customize the config object. Here you can set your external evaluator
and all other variables for the training. All config is handled as runtime config via methods.
This seems a little odd at first, but will allow configuration for dynaic values, like different
exploration factors in different training stages. In here you can also configure secure network communication.
This should be activated in any non private networks, as the transmitted pickle messages allow
attackers to run arbitrary python code on your system!


### Running the training

To run the training you will need exactly one training master and at least one playing slave.
Both can be found in the `distribution` module.
A usual setup is to run the training master and a playing slave an a desktop pc with powerful gpu.
This pc will also collect all training stats and results. You can then add any number of other pcs
as playing slaves, to generate more selfplay games. All components of this system can be shut down at
any time, without loosing progress. Simply restart them at a later point to continue training.
This makes it easy to for example run training over night, but still use the PC during the day for work.


### Analyzing the results

The `util` module has an function to plot the average game outcome versus the external opponent during training.
This should be your main way to see how the training goes. Besides that you can use the usual tensorboard
statistics to see how losses change. Please note that it is typical for the losses to stop lowering at some
point, this does not mean that there is no more training process, as new data can have a new strategy, thus
an initially high loss.


Currently there are no tools to analyze the tree search phase and the selfplay samples. Feel free to submit
code to record an analyze these, as this would probably be the next step to better understand the training process.
