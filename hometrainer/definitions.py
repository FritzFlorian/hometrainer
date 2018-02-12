import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# NN Server Settings
SELFPLAY_NN_SERVER_PORT = 5100
SELFEVAL_NN_SERVER_PORT = 5101
TRAINING_NN_SERVER_PORT = 5102

# Distribution settings
TRAINING_MASTER_PORT = 5200

# Settings for ZeroMQ security
KEYS_DIR = os.path.join(ROOT_DIR, 'keys')
PUBLIC_KEYS_DIR = os.path.join(KEYS_DIR, 'public_keys')
PRIVATE_KEYS_DIR = os.path.join(KEYS_DIR, 'private_keys')
SERVER_SECRET = os.path.join(PRIVATE_KEYS_DIR, 'server.key_secret')
SERVER_PUBLIC = os.path.join(PUBLIC_KEYS_DIR, 'server.key')
CLIENT_SECRET = os.path.join(PRIVATE_KEYS_DIR, 'client.key_secret')
CLIENT_PUBLIC = os.path.join(PUBLIC_KEYS_DIR, 'client.key')
