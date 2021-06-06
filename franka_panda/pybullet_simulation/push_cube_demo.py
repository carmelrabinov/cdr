import os
import numpy as np

from franka_panda.pybullet_simulation.environment import PybulletPushCubePandaEnv
from franka_panda.pybullet_simulation.utils import save_video_as_mp4, save_video_as_npy


def cube_demo(output_dir: str,
              id: int,
              use_ui=False,
              override_saved_data=False,
              generate_alternative=True,
              random_size=False,
              random_light=True,
              random_camera=False,
              random_texture=True,
              n_frames=15,
              config_path="configs/default_push.yml",
              record_robot_view: bool = False):

    # for the case this is a single
    if not isinstance(id, list):
        id = [id]


    # init random seed
    np.random.seed()

    # init simulation
    os.makedirs(output_dir, exist_ok=True)
    env = PybulletPushCubePandaEnv(config_path=config_path,
                                   load_frame=True,
                                   random_texture=random_texture,
                                   random_light=random_light,
                                   random_size=random_size,
                                   random_camera=random_camera,
                                   use_ui=use_ui,
                                   textures_path="../textures",
                                   alternative_textures_path="../textures")

    if record_robot_view:
        env._simulation.recoder_robot = True
        env._simulation.enable_rendering = True

    # generate videos
    for simulation_id in id:

        # reset simulation
        env.reset_environment_appearance()
        env.reset()

        # define save path
        save_path = os.path.join(output_dir, f"video_{simulation_id}")
        if os.path.exists(save_path + ".mp4") and not override_saved_data:
            print(f"skipping video_{simulation_id}")
            return

        # init results lists
        label = {"action": [], "state": [], "next_state": []}
        segmnetation_video = []
        video = []
        video_2 = []
        for i in range(n_frames):

            # save 2D state (x,z)
            state = env.state
            label["state"].append(state)

            # save_observations:
            video.append(env.render())
            if generate_alternative:
                video_2.append(env.render(mode="alternative"))
            segmnetation_video.append(env.render(mode="segmentation"))

            # sample random action with some probability
            if np.random.rand() < 0.1:
                action = env.sample_actions(n_actions=1)[0]
            else:
                action = env.sample_push_actions(pushed_coordinate=env.state, n_actions=1)[0]

            observation, reward, done, info = env.step(action)

            # save 2D next state (x,z)
            next_state = env.state
            label["next_state"].append(next_state)

            # save 4D action (x1,z1,x2,z2)
            label["action"].append(info["actual_action"])

            # stop if cube is out of state space.
            if done:
                break

        # save
        if len(label["state"]) > min(6, n_frames):
            if generate_alternative:
                video_2 = np.stack(video_2)
                save_video_as_mp4(video_2, save_path=save_path+"_2")
                save_video_as_npy(video_2, save_path=save_path+"_2")
            video = np.stack(video)
            segmnetation_video = np.stack(segmnetation_video)
            if record_robot_view:
                save_video_as_mp4(env._simulation.robot_video, save_path=save_path + "_robot_view", fps=30)
            save_video_as_mp4(video, save_path=save_path)
            save_video_as_npy(label, save_path=save_path+"_state_action_label")
            save_video_as_npy(segmnetation_video, save_path=save_path+"_seg_mask")
            save_video_as_npy(video, save_path=save_path)
            print(f"Video {simulation_id} with trajectory length = {i+1} was saved to {save_path}")

        else:
            print(f"Warning, trajectory is too short: {i+1}, skipping simulation {simulation_id}")

    env.close()


if __name__ == '__main__':
    n_demos = 1
    for i in range(n_demos):
        cube_demo(output_dir="../datasets/simple_cube",
                  id=i,
                  use_ui=True,
                  random_size=True,
                  random_light=True,
                  random_camera=False,
                  random_texture=True,
                  override_saved_data=True,
                  generate_alternative=False,
                  record_robot_view=False)
