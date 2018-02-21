import pickle
import copy as old_copy
import zmq.auth
from hometrainer.config import Configuration
from zmq.auth.thread import ThreadAuthenticator
import tempfile
import os
import shutil
import zipfile
import io
import numpy as np


def copy(obj):
    return old_copy.copy(obj)


def deepcopy(obj):
    """Way faster then normal deep copy."""
    return pickle.loads(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def secure_client_connection(client, ctx, config: Configuration, only_localhost=False):
    """Configures certificates for zeromq client connection."""
    if not config.zmq_use_secure_connection():
        return

    _init_auth(ctx, only_localhost, config)

    client_public, client_secret = zmq.auth.load_certificate(config.zmq_client_secret())
    client.curve_secretkey = client_secret
    client.curve_publickey = client_public

    server_public, _ = zmq.auth.load_certificate(config.zmq_server_public())
    # The client must know the server's public key to make a CURVE connection.
    client.curve_serverkey = server_public


def secure_server_connection(server, ctx, config: Configuration, only_localhost=False):
    """Configures certificates for zeromq server connection."""
    if not config.zmq_use_secure_connection():
        return

    _init_auth(ctx, only_localhost, config)

    server_public, server_secret = zmq.auth.load_certificate(config.zmq_server_secret())
    server.curve_secretkey = server_secret
    server.curve_publickey = server_public
    server.curve_server = True  # must come before bind


def save_neural_net_to_zip_binary(neural_network, session):
    with tempfile.TemporaryDirectory() as base_dir:
        # Prepare Saving Path's
        checkpoint_dir = os.path.join(base_dir, 'checkpoint')
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
        os.makedirs(checkpoint_dir)

        # Save it
        neural_network.save_weights(session, checkpoint_file)

        # Pack it up to send it through the network
        checkpoint_zip = os.path.join(base_dir, 'checkpoint')
        shutil.make_archive(checkpoint_zip, 'zip', checkpoint_dir)
        with open(checkpoint_zip + '.zip', 'rb') as file:
            return file.read()


def load_neural_net_from_zip_binary(zip_binary, neural_network, session):
    with tempfile.TemporaryDirectory() as base_dir:
        # Prepare Saving Path's
        checkpoint_dir = os.path.join(base_dir, 'checkpoint')
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
        checkpoint_zip = os.path.join(base_dir, 'checkpoint.zip')
        os.makedirs(checkpoint_dir)

        # Prepare the checkpoint data for loading
        with open(checkpoint_zip, 'wb') as file:
            file.write(zip_binary)

        with zipfile.ZipFile(checkpoint_zip, "r") as zip_ref:
            zip_ref.extractall(checkpoint_dir)

        neural_network.load_weights(session, checkpoint_file)


def _init_auth(ctx, only_localhost, config: Configuration):
    auth = ThreadAuthenticator(ctx)
    auth.start()

    if only_localhost:
        auth.allow('127.0.0.1')
    else:
        auth.allow('*')

    auth.configure_curve(domain='*', location=config.zmq_public_keys_dir())


def count_files(dir_path):
    return len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])


def plot_external_eval_avg_score(work_dir, lower_bound, upper_bound, show_progress_lines,
                                 smoothing=0.1, min=-1, max=1, middle=0,
                                 x_label='Batch', y_label='Average Game Outcome',
                                 smoothed_score_lable='Smoothed Score', score_label='Score',
                                 x_scaling=1, new_was_better_label='New weights better than previous'):
    """Returns a binary containing a plot of the current winrate."""
    import matplotlib.pyplot as plt
    import hometrainer.distribution as distribution
    import matplotlib.patches as mpatches

    x_scaling = x_scaling

    progress = distribution.TrainingRunProgress(os.path.join(work_dir, 'stats.json'))
    progress.load_stats()

    avg_score = []
    x_steps = []

    new_was_better = []
    for iteration in progress.stats.iterations[lower_bound:upper_bound]:
        if iteration.ai_eval.n_games < 14:
            break

        avg_score.append(iteration.ai_eval.nn_score / iteration.ai_eval.n_games)
        x_steps.append(iteration.ai_eval.end_batch * x_scaling)

        if iteration.self_eval.new_better:
            new_was_better.append(iteration.ai_eval.end_batch * x_scaling)

    smoothed_wins = smooth_array(avg_score, smoothing)

    label_1, = plt.plot(x_steps, avg_score, linestyle='--', label=score_label)
    label_2, = plt.plot(x_steps, smoothed_wins, label=smoothed_score_lable)
    plt.ylim(min, max)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(y=middle, color='r')
    if show_progress_lines:
        label_3 = None
        for better_x in new_was_better:
            label_3 = plt.axvline(x=better_x, linestyle=':', label=new_was_better_label)
        if label_3:
            plt.legend(handles=[label_1, label_2, label_3])
        else:
            plt.legend(handles=[label_1, label_2])
    else:
        plt.legend(handles=[label_1, label_2])

    result = io.BytesIO()
    plt.savefig(result, dpi=400)
    plt.clf()

    return result


def smooth_array(array, smoothing):
    smoothed_array = []
    max_steps = max(1, round(len(array) * smoothing))
    for i in range(len(array)):
        values = [array[i]]
        for j in range(1, max_steps + 1):
            if i - j < 0:
                break
            values.append(array[i - j])
        for j in range(1, max_steps + 1):
            if i + j >= len(array):
                break
            values.append(array[i + j])

        smoothed_array.append(np.average(values))

    return smoothed_array

