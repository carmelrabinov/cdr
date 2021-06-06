import cv2
import gym
import numpy as np


class PushPandaEnv(gym.Env):
    """Panda gym Environment"""
    metadata = {"render.modes": ["regular", "alternative", "segmentation"]}

    def __init__(self):
        super().__init__()  # Define action and observation space

        # observation parameters
        self._image_hights = 128
        self._image_width = 128

        # push parameters
        self._min_push_magnitude = 0.05
        self._max_push_magnitude = 0.15
        self._ee_push_hight = 0.03

        # define the distance between the action source coordinate and the pushed point on the object coordinate
        self._source_push_magnitude = 0.06
        assert self._source_push_magnitude <= self._min_push_magnitude + 0.1

        # in real_coordinates
        self._action_space = gym.spaces.Box(low=np.array([0.3, -0.3, 0.36, -0.2]), high=np.array([0.7, 0.3, 0.64, 0.2]), dtype=np.float)  # Example for using image as input:
        self._observation_space = gym.spaces.Box(low=0, high=255, shape=(self._image_hights, self._image_width, 3), dtype=np.uint8)
        self._state_space = gym.spaces.Box(low=np.array([0.32, -0.23]), high=np.array([0.68, 0.23]), dtype=np.float)

    def step(self, action: np.ndarray) -> [np. ndarray, float, bool, dict]:
        raise NotImplemented

    def reset(self) -> None:
        raise NotImplemented

    @property
    def state(self) -> np.ndarray:
        return NotImplementedError

    def out_of_state_space(self, state: np.ndarray) -> bool:
        return np.any(state > self._state_space.high) or np.any(state < self._state_space.low)

    def sample_states(self, n_states: int) -> np.ndarray:
        return np.stack([self._state_space.sample() for _ in range(n_states)])

    def sample_actions(self, n_actions: int) -> np.ndarray:
        assert n_actions > 0, "n_actions must be positive number"
        return np.stack([self._action_space.sample() for _ in range(n_actions)])

    def sample_push_actions(self, pushed_coordinate: np.ndarray, n_actions: int) -> np.ndarray:
        assert n_actions > 0, "n_actions must be positive number"
        # sample random direction and normalized it to have magnitude of 1
        random_directions = np.random.rand(n_actions, 2) - 0.5
        magnitudes = np.stack([np.linalg.norm(random_directions, axis=-1), np.linalg.norm(random_directions, axis=-1)]).transpose()
        normalized_random_directions = random_directions / magnitudes

        # sample random magnitude
        random_magnitudes = np.random.uniform(size=n_actions, low=self._min_push_magnitude, high=self._max_push_magnitude)

        # sample random magnitude
        pushed_coordinate = np.tile(pushed_coordinate, (n_actions, 1))
        source_coordinate = pushed_coordinate - normalized_random_directions * self._source_push_magnitude
        target_coordinate = pushed_coordinate + normalized_random_directions * (random_magnitudes - self._source_push_magnitude)[:, None]

        # concat source and target coordinates and clip them to action space
        actions = np.concatenate((source_coordinate, target_coordinate), axis=-1)
        return self._clip_to_action_space(actions)

    def _clip_to_action_space(self, actions: np.array) -> np.array:
        return np.clip(actions, a_min=self._action_space.low, a_max=self._action_space.high)

    def render(self, mode: str = "regular") -> np.ndarray:
        return NotImplementedError

    def close(self) -> None:
        raise NotImplemented

    def convert_real_2_sim_spaces(self):
        action_space = np.stack([self.convert_real_2_sim_action(self._action_space.low),
                                 self.convert_real_2_sim_action(self._action_space.high)])
        self._action_space = gym.spaces.Box(low=np.min(action_space, axis=0), high=np.max(action_space, axis=0),
                                            dtype=np.float)  # Example for using image as input:
        state_space = np.stack([self.convert_real_2_sim_coordinate(self._state_space.low),
                                self.convert_real_2_sim_coordinate(self._state_space.high)])
        self._state_space = gym.spaces.Box(low=np.min(state_space, axis=0), high=np.max(state_space, axis=0),
                                           dtype=np.float)

    @staticmethod
    def convert_sim_2_real_action(action: np.ndarray) -> np.ndarray:
        action[:2] = -np.flip(action[:2])
        action[2:4] = -np.flip(action[2:4])
        return action

    @staticmethod
    def convert_real_2_sim_action(action: np.ndarray) -> np.ndarray:
        action[:2] = -np.flip(action[:2])
        action[2:4] = -np.flip(action[2:4])
        return action

    @staticmethod
    def convert_sim_2_real_coordinate(state: np.ndarray) -> np.ndarray:
        return -np.flip(state)

    @staticmethod
    def convert_real_2_sim_coordinate(state: np.ndarray) -> np.ndarray:
        return -np.flip(state)

    @staticmethod
    def convert_sim_2_real_actions(actions: np.ndarray) -> np.ndarray:
        actions[:, :2] = -np.flip(actions[:, :2], axis=-1)
        actions[:, 2:4] = -np.flip(actions[:, 2:4], axis=-1)
        return actions

    @staticmethod
    def convert_real_2_sim_actions(actions: np.ndarray) -> np.ndarray:
        actions[:, :2] = -np.flip(actions[:, :2], axis=-1)
        actions[:, 2:4] = -np.flip(actions[:, 2:4], axis=-1)
        return actions

    @staticmethod
    def convert_sim_2_real_coordinates(states: np.ndarray) -> np.ndarray:
        return -np.flip(states, axis=-1)

    @staticmethod
    def convert_real_2_sim_coordinates(states: np.ndarray) -> np.ndarray:
        return -np.flip(states, axis=-1)

    def image2world(self, image_coordinates: np.ndarray) -> np.ndarray:
        return NotImplementedError

    def world2image(self, world_coordinates: np.ndarray) -> np.ndarray:
        return NotImplementedError

    def visualize_action(self, action: np.ndarray, image: np.ndarray = None) -> np.ndarray:
        """add action arrow to image"""
        if image is None:
            image = self.render()
        source = self.world2image(action[:2])
        target = self.world2image(action[2:])
        # visual params
        color = (0, 255, 0)
        thickness = 2
        return cv2.arrowedLine(img=image.astype(np.uint8),
                               pt1=(source[0], source[1]),
                               pt2=(target[0], target[1]),
                               color=color,
                               thickness=thickness)
