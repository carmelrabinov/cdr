from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from franka_panda.environment import PushPandaEnv
from franka_panda.pybullet_simulation.environment import PybulletPushCubePandaEnv


def visualize_mpc_trajectory(images, goal_image, save_path=None, title=None):
    """plots nearest neighbours images"""

    n_cols = len(images)+2
    n_rows = 1
    fig = plt.figure(figsize=(int(12*n_cols/4.), 12))
    if title is not None:
        fig.suptitle(title, fontsize=15)

    # goal
    fig.add_subplot(n_rows, n_cols, 1).set_title(f"Goal", fontsize=20)
    plt.imshow(goal_image)
    plt.axis('off')

    for col, img in enumerate(images):

        # state
        fig.add_subplot(n_rows, n_cols, col+2).set_title(f"{col}", fontsize=20)
        plt.imshow(img)
        plt.axis('off')

    # goal again
    fig.add_subplot(n_rows, n_cols, n_cols).set_title(f"Goal", fontsize=20)
    plt.imshow(goal_image)
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path + ".png")
        print(f"results saved to {save_path}.jpg")


class MPC:
    """ base class for model predictive control planner"""

    @abstractmethod
    def sample_actions(self, n_actions: int = 1) -> np.ndarray:
        """sample random n_actions"""
        return NotImplemented

    @abstractmethod
    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """predict next state"""
        return NotImplemented

    @abstractmethod
    def score(self, state_1: np.ndarray, state_2: np.ndarray) -> np.ndarray:
        """distance score between 2 state, higher is better/closer"""
        return NotImplemented

    @abstractmethod
    def step(self, action: np.ndarray) -> np.ndarray:
        """step and return next state/observation"""
        return NotImplemented

    @abstractmethod
    def plan(self, start_state: np.ndarray, goal_state: np.ndarray, n_trials: int = 100, n_steps: int = 1) -> [np.ndarray, np.ndarray, bool]:
        """run n steps mpc and returns the [selected action, predicted state, is_better]"""
        curr_states = np.expand_dims(start_state, axis=0).repeat(n_trials, axis=0)
        trial_states = []
        trial_actions = []

        # check for current distance from goal
        init_dist = self.score(np.expand_dims(start_state, axis=0), np.expand_dims(goal_state, axis=0))[0]

        # run prediction
        for i in range(n_steps):

            # sample random actions
            actions = self.sample_actions(n_actions=n_trials)

            trial_actions.append(actions)
            next_states = self.predict(states=curr_states, actions=actions)
            trial_states.append(np.copy(next_states))
            curr_states = next_states

        trial_states = np.stack(trial_states)
        trial_actions = trial_actions[0]

        # calculate the distance for every train in every step (n_steps X n_trails X 1)
        goal_states = np.expand_dims(goal_state, axis=[0,1]).repeat(n_steps, axis=0).repeat(n_trials, axis=1)
        dists = self.score(trial_states, goal_states)

        # select the best step in each trail
        steps_ind = np.argmax(dists, axis=0)
        steps_scores = np.max(dists, axis=0)

        # select the trail with the highest score in some step
        action_idx = np.argmax(steps_scores)
        selected_action = trial_actions[action_idx]
        selected_action_next_state_pred = trial_states[0, action_idx]

        # check if the best score for all trails and for all steps is smaller then the initial score
        is_better_then_current = steps_scores[action_idx] > init_dist

        return selected_action, selected_action_next_state_pred, is_better_then_current


class PushPandaMPC(MPC):
    """
    Model predictive control planner for Franka-Panda robotic arm
    with 4 dimensional action space representing push actions
    """
    def __init__(self, env: PushPandaEnv,
                 prediction_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 encoder_fn: Callable[[np.ndarray], np.ndarray],
                 random_actions_p: float = 0.1):
        self.env: PushPandaEnv = env
        self.random_actions_p = random_actions_p
        self._prediction_fn = prediction_fn
        self._encoder_fn = encoder_fn

    def sample_actions(self, n_actions: int = 1):
        n_random_actions = int(np.ceil(n_actions * self.random_actions_p))
        random_actions = self.env.sample_actions(n_random_actions) if n_random_actions > 0 else np.empty((0, 4))
        n_push_actions = int(np.floor(n_actions * (1-self.random_actions_p)))
        push_actions = self.env.sample_push_actions(self.env.state, n_push_actions) if n_push_actions > 0 else np.empty((0, 4))
        return np.concatenate((push_actions, random_actions), axis=0)

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """predict next state"""
        return self._prediction_fn(states, actions)

    def score(self, state_1: np.ndarray, state_2: np.ndarray) -> np.ndarray:
        return self.negtive_l2_distance(state_1, state_2)

    @staticmethod
    def negtive_l2_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """expect x1 and x2 to be n_samples X n_features"""
        return -1 * np.linalg.norm(x1 - x2, axis=-1)

    def run(self,
            goal_observation: np.ndarray,
            goal_position: np.ndarray,
            start_position: np.ndarray = None,
            save_path: str = None,
            n_steps: int = 15,
            n_trials: int = 100,
            stop_when_not_improving: bool = False,
            tolerance_to_goal_in_cm: float = 3.,
            verbose: bool = False,
            ) -> [[float, float, int, np.ndarray, float, int], np.ndarray]:
        """
        param: goal_observation: np.ndarray,
        param: goal_position: np.ndarray,
        param: start_position: np.ndarray = None,
        param: save_path: str = None,
        param: n_steps: int = 15,
        param: stop_when_not_improving: bool = False,
        param: tolerance_to_goal_in_cm: float = 3.,
        param: verbose: bool = False,
        return: (init_dist, final_dist, n_trails, gains, min_dist, n_trails_to_min_dist), images
        """

        # encode goal observation
        goal_state = self._encoder_fn(goal_observation)
        if verbose:
            plt.imshow(goal_observation)

        # encode start observation
        self.env.reset()
        if start_position is not None:
            self.env.state = start_position
        current_observation = self.env.render()
        current_state = self._encoder_fn(current_observation)
        current_position = self.env.state

        # init stats
        dist = 100 * np.linalg.norm(goal_position - current_position)
        distances = [dist]
        images = []
        if verbose:
            print(f"goal position: {goal_position}, current state: {current_position}, dist: {dist:.2f} [cm]")
            plt.imshow(goal_observation)
            plt.show()

        # main MPC loop
        for i in range(n_steps):

            # run planning
            selected_action, pred_next_state, is_better = self.plan(start_state=current_state,
                                                                    goal_state=goal_state,
                                                                    n_trials=n_trials,
                                                                    n_steps=1)

            # if early stop condition and the action did not improve, stop planning
            if stop_when_not_improving and not is_better:
                break

            # perform selected action and capture "next" observation
            next_observation, _, done, _ = self.env.step(selected_action)
            images.append(self.env.visualize_action(image=np.copy(current_observation), action=selected_action))
            if verbose:
                plt.imshow(images[-1])
                plt.show()

            # encode "next" observation
            current_observation = next_observation
            current_state = self._encoder_fn(current_observation)
            current_position = self.env.state

            # report stats
            dist = 100 * np.linalg.norm(goal_position - current_position)
            distances.append(dist)
            if verbose:
                print(f"goal position: {goal_position}, current position: {current_position}, dist: {dist:.2f} [cm], Pred error: {np.linalg.norm(pred_next_state - current_state):.4f}")

            # if in tolerance_to_goal_in_cm [cm] from goal, stop, defualte is 2.5 [cm]
            # only relevant for
            if not stop_when_not_improving and dist < tolerance_to_goal_in_cm:
                break

        init_dist = distances[0]
        final_dist = distances[-1]
        n_trails = i + 1
        min_dist = np.min(distances)
        n_trails_to_min_dist = np.argmin(distances) + 1
        gains = -np.diff(distances) if len(distances) > 1 else np.array([0.])

        title = f"init dist: {init_dist:.2f} [cm], final dist: {final_dist:.2f} [cm], trails: {n_trails}/{n_steps}, " \
                f"mean dist_gains: {np.mean(gains):.2f} +- {np.std(gains):.2f} [cm], " \
                f"min_dist: {min_dist:.2f} [cm], n_trails_to_min_dist: {n_trails_to_min_dist:.2f}"
        print(title)
        if save_path is not None:
            visualize_mpc_trajectory(images, goal_observation, save_path=save_path, title=title)
            plt.show()

        return (init_dist, final_dist, n_trails, gains, min_dist, n_trails_to_min_dist), images


if __name__ == '__main__':

    env: PushPandaEnv = PybulletPushCubePandaEnv(load_frame=True,
                                                 random_texture=True,
                                                 random_light=True,
                                                 random_size=False,
                                                 random_camera=False,
                                                 use_ui=True,
                                                 textures_path=r"D:\Representation_Learning\textures",
                                                 alternative_textures_path=r"D:\Representation_Learning\textures")

    def simple_cube_forward_model(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.stack([simple_cube_forward_model_(state, action) for state, action in zip(states, actions)])

    def simple_cube_forward_model_(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        cube_size = 0.2
        ee_start = action[:2]
        ee_goal = action[2:]
        d = np.cross(ee_goal - ee_start, state - ee_start) / np.linalg.norm(ee_goal - ee_start)
        # only if the cube is in the middle of the action
        if np.minimum(ee_goal[0], ee_start[0]) <= state[0] <= np.maximum(ee_goal[0], ee_start[0]) \
                and np.minimum(ee_goal[1], ee_start[1]) <= state[1] <= np.maximum(ee_goal[1], ee_start[1]) \
                and d < cube_size:
            next_state = ee_goal
            vec = ee_goal - ee_start
            next_state += vec / np.linalg.norm(vec) * cube_size * 0.5
        else:
            next_state = state
        return next_state

    def identity_encoder(state: np.ndarray) -> np.ndarray:
        return state

    def oracle_encoder(observation: np.ndarray) -> np.ndarray:
        return env.state

    mpc = PushPandaMPC(env=env,
                       prediction_fn=simple_cube_forward_model,
                       encoder_fn=oracle_encoder,
                       random_actions_p=0.)

    # randomly sample start and goal state
    goal_state = env.state
    goal_image = env.render(mode="alternative")

    results, images = mpc.run(goal_observation=goal_image,
                              goal_position=goal_state,
                              start_position=None,
                              verbose=True,
                              save_path=r"D:\Representation_Learning\mpc_test")

    env.close()