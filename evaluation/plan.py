import argparse
import os
import torch
import numpy as np
import yaml

from data.utils import image_transform
from franka_panda.environment import PushPandaEnv
from franka_panda.mpc import PushPandaMPC
from franka_panda.pybullet_simulation.environment import PybulletPushCubePandaEnv
from model.cpc import ControlCPC
from utils import run_single, load_from_checkpoint


def generate_start_goal_pairs(size: int = 1) -> [np.ndarray, np.ndarray]:
    """ generate random start positions and goal positions"""
    starts = np.array([np.random.uniform(-0.2, 0.2, size=size), np.random.uniform(-0.6, -0.4, size=size)]).transpose()
    goals = np.array([np.random.uniform(-0.25, 0.25, size=size), np.random.uniform(-0.65, -0.35, size=size)]).transpose()

    return starts, goals
# starts, goals = generate_start_goal_pairs(100)
# np.save("evaluation/cube_start_goal_dataset", [starts, goals])


def evaluate_model_planning(model: ControlCPC, device: str,
                            config_path: str,
                            use_oracle: bool = False,
                            n_experiments: int = -1,
                            verbose: bool = False,
                            save_dir: str = None,
                            random_actions_p: float = 0.1,
                            n_steps: int = 15,
                            n_process: int = 1,
                            tolerance_to_goal_in_cm: float = 3,
                            stop_when_not_improving: bool = False,
                            use_goal_from_different_domain: bool = False) -> [list, list, list, list, list, list]:
    """
    :param model: model wich holds encoder and forward model
    :param device: pytorch device
    :param use_oracle: use position state as image representations
    :param n_experiments: number of experiments to run
    :param verbose: whether to print msg and images to the screen while running
    :param random_actions_p: percentage of random actions which does not guarantee cube in the middle
    :return: score for model
    """
    assert 0. <= random_actions_p <= 1., "random_actions_p < 0 or > 1"
    model.to(device)
    # load dataset
    starts, goals = np.load("evaluation/cube_start_goal_dataset_bigger_10_cm.npy")
    n_experiments = min(n_experiments, len(starts))
    if n_experiments > 0:
        starts = starts[:n_experiments]
        goals = goals[:n_experiments]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    env: PushPandaEnv = PybulletPushCubePandaEnv(config_path=config_path,
                                                 load_frame=True,
                                                 random_texture=True,
                                                 random_light=True,
                                                 random_size=False,
                                                 random_camera=False,
                                                 use_ui=False)

    def pytorch_forward_model_predict(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            actions = torch.from_numpy(actions).float().to(device)
            states = torch.from_numpy(states).float().to(device)
            next_states = model.forward_model(states, actions)
        return next_states.cpu().numpy()

    def pytorch_encode_observation(observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            observation = image_transform(observation).float().to(device)
            state = run_single(model.encoder, observation)
        return state.cpu().numpy()

    # define MPC
    mpc = PushPandaMPC(env=env,
                       prediction_fn=pytorch_forward_model_predict,
                       encoder_fn=pytorch_encode_observation,
                       random_actions_p=random_actions_p)


    final_dists = []
    init_dists = []
    one_step_dist_gains = []
    trials = []
    min_dists = []
    trails_to_min_dist = []
    for n, (goal_state, start_state) in enumerate(zip(goals, starts)):

        # randomly sample start and goal state
        env.reset_environment_appearance()
        env.reset()
        env.state = goal_state
        if use_goal_from_different_domain:
            goal_image = env.render(mode="alternative")
        else:
            goal_image = env.render()

        results, images = mpc.run(goal_observation=goal_image,
                                  goal_position=goal_state,
                                  start_position=start_state,
                                  verbose=verbose,
                                  n_steps=n_steps,
                                  n_trials=1000,
                                  stop_when_not_improving=stop_when_not_improving,
                                  tolerance_to_goal_in_cm=tolerance_to_goal_in_cm)

        (init_dist, final_dist, n_trail, dist_gains, min_dist, n_trails_to_min_dist) = results
        one_step_dist_gains.extend(dist_gains)
        final_dists.append(final_dist)
        init_dists.append(init_dist)
        trials.append(n_trail)
        min_dists.append(min_dist)
        trails_to_min_dist.append(n_trails_to_min_dist)

    env.close()

    if save_dir is not None:
        np.save(os.path.join(save_dir, f"results_{n_experiments}_experiments"), [init_dists, final_dists, trials, min_dists, trails_to_min_dist])
    return final_dists, init_dists, trials, one_step_dist_gains, min_dists, trails_to_min_dist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to trained model')
    parser.add_argument('-o', '--output', type=str, default='../planning_results', help='path to output directory')
    parser.add_argument('-c', '--simulation_config', type=str, help='path to simulation config')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load trained model
    assert os.path.exists(args.model_path), f"can't find model file in: {args.model_path}"
    base_path, checkpoint = os.path.split(args.model_path)
    _, model_name = os.path.split(base_path)
    cfg_path = os.path.join(base_path, "config.yml")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model, _, _ = load_from_checkpoint(path=args.model_path, config=config, device=device)
    dist_func = config["dist_function"]
    print(f"Using distance function: {dist_func}")
    image_size = model.encoder.image_size if hasattr(model.encoder, "image_size") else 128
    output_dir = os.path.join(args.output, f"{model_name}_epoch_{checkpoint.split('_')[-1]}_{dist_func}_mpc")

    n_experiments = 2
    results = evaluate_model_planning(model=model,
                                      device=device,
                                      use_oracle=False,
                                      n_experiments=n_experiments,
                                      verbose=False,
                                      save_dir=output_dir,
                                      random_actions_p=0.,
                                      n_process=1,
                                      config_path=args.simulation_config,
                                      use_goal_from_different_domain=True)
    final_dists, init_dists, trials, gains, min_dists, trails_to_min_dist = results
    print(init_dists)
    print(final_dists)
    print(trials)
    print(f"init_dists: {np.mean(init_dists)} +- {np.std(init_dists)}")
    print(f"final_dists: {np.mean(final_dists)} +- {np.std(final_dists)}")
    print(f"trials: {np.mean(trials)} +- {np.std(trials)}")
    print(f"Dist gains: {np.mean(gains)} +- {np.std(gains)}")


