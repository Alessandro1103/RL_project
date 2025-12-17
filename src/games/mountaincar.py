import datetime
import os
import pathlib

import gymnasium as gym
import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. By default muzero uses every GPUs available

        # Game
        # MountainCar Observation: [position, velocity] -> (1, 1, 2)
        self.observation_shape = (1, 1, 2)
        # MountainCar Actions: 0 (push left), 1 (no push), 2 (push right)
        self.action_space = list(range(3))
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0
        self.opponent = None

        # Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 200  # Standard limit for MountainCar-v0
        self.num_simulations = 50
        self.discount = 0.999
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "fullyconnected"
        self.support_size = 10

        # Residual Network
        self.downsample = False
        self.blocks = 1
        self.channels = 16
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.resnet_fc_reward_layers = []
        self.resnet_fc_value_layers = []
        self.resnet_fc_policy_layers = []

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = [16]
        self.fc_reconstruction_layers = [16]
        self.fc_dynamics_layers = [16]
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]

        # Training
        self.results_path = pathlib.Path(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results",
            os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        ))
        self.save_model = True
        self.training_steps = 20000 # Un po' di più di CartPole perché l'esplorazione è difficile
        self.self_supervised_steps = 0
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 1
        self.reconstruction_loss_weight = 1
        self.consistency_loss_weight = 1
        self.train_on_gpu = True if torch.cuda.is_available() else False

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.005
        self.lr_decay_rate = 1
        self.lr_decay_steps = 1000

        # Replay Buffer
        self.replay_buffer_size = 2000
        self.num_unroll_steps = 10
        self.td_steps = 50
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None

    def visit_softmax_temperature_fn(self, trained_steps):
        return 0.35


class Game(AbstractGame):
    """
    Game wrapper for MountainCar-v0
    """

    def __init__(self, seed=None):
        self.env = gym.make("MountainCar-v0", render_mode="rgb_array")
        self.env_seed = seed

    def step(self, action):
        """
        Apply action to the game.
        """
        observation, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        # MountainCar dà reward -1 ad ogni step.
        return numpy.array([[observation]]), reward, done

    def legal_actions(self):
        return list(range(3))

    def reset(self):
        """
        Reset the game for a new game.
        """
        observation, _ = self.env.reset(seed=self.env_seed)
        return numpy.array([[observation]])

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        actions = {
            0: "Accelerate to the Left",
            1: "Don't accelerate",
            2: "Accelerate to the Right",
        }
        return f"{action_number}. {actions[action_number]}"