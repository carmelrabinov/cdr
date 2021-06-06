import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from data.utils import image_transform
from franka_panda.environment import PushPandaEnv
from franka_panda.mpc import PushPandaMPC, visualize_mpc_trajectory
from franka_panda.pybullet_simulation.environment import PybulletPushCubePandaEnv
from franka_panda.real_robot.environment import RealPushCubePandaEnv

from utils import run_single, load_from_checkpoint
import os


def print_total_restuls(results):
    for key in results[-1].keys():
        key_stats = [r[key] for r in results if key in r.keys()]
        if key != 'texture' and key != 'real':
            print(f"{key}: {np.mean(key_stats)} +- {np.std(key_stats)}")


if __name__ == '__main__':

    # init
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/home/robot-lab/Documents/carmel/control_models/cube_sim2real_250221_contrastive_mse_Z4_RESNETenc_OriginalCFMfm32_b256_t1.0_contrastiveAug_250221/checkpoint_epoch_85"
    config_path = "/home/robot-lab/Documents/carmel/default_push.yml"
    texture = "plain_obstacles"  #"colored_map" #"random_colors_obstacles"   # "plain_obstacles"

    # results output directory
    model_name = model_path.split("/")[-2]
    output_dir = os.path.join("/home/robot-lab/Documents/carmel", "mpc_results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = []

    # load config
    assert os.path.exists(model_path), f"can't find model file in: {model_path}"
    base_path, checkpoint = os.path.split(model_path)
    cfg_path = os.path.join(base_path, "config.yml")
    assert os.path.exists(cfg_path), f"can't find config file in: {cfg_path}"
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load model
    model, _, _ = load_from_checkpoint(path=model_path, config=config, device=device)
    encoder = model.encoder
    forward_model = model.forward_model
    # encoder = Resnet18Encoder(input_dim=[128, 128, 3], output_dim=16)
    # forward_model = CFMForwardModel(z_dim=16, action_dim=4, z_hidden_dim=16)

    # init environment
    env: PushPandaEnv = RealPushCubePandaEnv()
    sim_env = PybulletPushCubePandaEnv(config_path=config_path,
                                       load_frame=False,
                                       random_texture=False,
                                       random_light=True,
                                       random_size=False,
                                       random_camera=False,
                                       use_ui=False,
                                       texture_path=None)

    # define forward model function
    def pytorch_forward_model_predict(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            # forward model was trained in simulation, first convert to simulation action space
            actions = env.convert_real_2_sim_actions(actions)
            actions = torch.from_numpy(actions).float().to(device)
            states = torch.from_numpy(states).float().to(device)
            next_states = forward_model(states, actions)
        return next_states.cpu().numpy()

    # define encoder function
    def pytorch_encode_observation(observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            observation = image_transform(observation).float().to(device)
            state = run_single(encoder, observation)
        return state.cpu().numpy()

    # define MPC
    mpc = PushPandaMPC(env=env,
                       prediction_fn=pytorch_forward_model_predict,
                       encoder_fn=pytorch_encode_observation,
                       random_actions_p=0.)

    # randomly sample start and goal state
    # goal_state = env.state
    # goal_image = env.render()
    #

    real = False
    if real:
        goal_image = env.render()
        goal_image_full = env.render(resize=False)
        cube_position = env.state
        plt.figure(1)
        plt.imshow(goal_image_full)
        plt.waitforbuttonpress()
    else:
        # display goal image
        goal_image = sim_env.render()
        cube_position = sim_env.convert_sim_2_real_coordinate(sim_env.state)
        plt.figure(1)
        plt.imshow(goal_image)
        plt.waitforbuttonpress()
        # plt.pause(2)

    n_steps = 10
    results, images = mpc.run(goal_observation=goal_image,
                              goal_position=cube_position,
                              start_position=None,
                              verbose=True,
                              n_steps=n_steps,
                              n_trials=1000,
                              stop_when_not_improving=False,
                              tolerance_to_goal_in_cm=3.)

    (init_dist, final_dist, n_trails, gains, min_dist, n_trails_to_min_dist) = results

    # save results
    print("Press any key to save results")
    # plt.waitforbuttonpress()
    results.append({"init_dist": init_dist, "final_dist": final_dist, "n_trail": n_trails,
                    "mean_gain": np.mean(gains), "std_gain": np.std(gains),
                    "min_dist": min_dist, "n_trails_to_min_dist": n_trails_to_min_dist, "texture": texture})
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # get last saved trajectory and visualize
    indexes = [int(f.split("_")[-1][:-4]) for f in os.listdir(output_dir) if f.endswith("png")]
    last_results_index = max(indexes) if len(indexes) > 0 else 0
    save_path = os.path.join(output_dir, f"trajectory_{last_results_index+1}")
    title = f"init dist: {init_dist:.2f} [cm], final dist: {final_dist:.2f} [cm], trails: {n_trails}/{n_steps}, " \
            f"mean dist_gains: {np.mean(gains):.2f} +- {np.std(gains):.2f} [cm], " \
            f"min_dist: {min_dist:.2f} [cm], n_trails_to_min_dist: {n_trails_to_min_dist:.2f}"
    visualize_mpc_trajectory(images, goal_image, save_path=save_path, title=title)
    print_total_restuls(results)
    plt.show()

