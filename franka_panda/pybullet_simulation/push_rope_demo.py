import os
import numpy as np

from franka_panda.pybullet_simulation.environment import PybulletPushRopePandaEnv
from franka_panda.pybullet_simulation.utils import save_video_as_mp4, save_video_as_npy

#
# def rope_demo(output_dir: str,
#               id: int,
#               use_ui=False,
#               override_saved_data=False,
#               two_texture=True,
#               random_positions=True,
#               random_size=False,
#               random_light=True,
#               n_frames=40,
#               image_width=128,
#               image_hight=128,
#               sim2real_camera_view=False):
#
#     np.random.seed()
#     os.makedirs(output_dir, exist_ok=True)
#     save_path = os.path.join(output_dir, f"video_{id}")
#     if os.path.exists(save_path + ".mp4") and not override_saved_data:
#         print(f"skipping video_{id}")
#         return
#
#     if sim2real_camera_view:
#         simulation = Sim2RealPushPandaSimulation(use_ui=use_ui)
#     else:
#         simulation = PushPandaSimulation(use_ui=use_ui)
#     simulation.enable_rendering = False
#     simulation.real_time = False
#     simulation.print_msg = False
#     simulation.recoder_robot = False
#     # simulation.load_texture(object_id=simulation.plane_id, path=simulation.tray_textures_path, random=True)
#     # simulation.load_frame()
#
#     simulation.load_rope(random_positions=random_positions, random_rope=random_size)
#     simulation.let_objects_settle()
#     if random_positions:
#         start_record_ind = np.random.randint(0, 3)
#
#     # load textures for simulation
#     if two_texture:
#         simulation.init_random_textures(texture_dir=os.path.join(os.getcwd(), "../textures/dtd_train"),
#                                         objects=[simulation.rope_id])
#         simulation.apply_textures(texture_plain=False)
#
#     # load random light for simulation
#     if random_light:
#         simulation.init_random_light_directions()
#         simulation.apply_random_light()
#
#     orientation = list(p.getQuaternionFromEuler([math.pi / 2., 0., 0.]))
#     for i in range(start_record_ind + n_frames):
#
#         # render
#         simulation.panda_robot.reset()
#         if i >= start_record_ind:
#             simulation.render_object_2_configs() if two_texture else simulation.render_object()
#
#         # sample action
#         source_state = simulation.sample_not_object_position()
#         if source_state is None:
#             print("No source state")
#             break
#         dist = 11 * image_width / 128
#         target_state = simulation.sample_object_position(curr_pos=source_state, max_dist=dist)
#         if target_state is None:
#             print("No target state")
#             break
#         new_target_state = increase_action_vector(source=source_state, target=target_state, factor=[1., 2.5])
#         if simulation.valid_target(new_target_state, conservative=True):
#             target_state = new_target_state
#         source_state += orientation
#         target_state += orientation
#
#         # act - source position
#         simulation.step_to_state(source_state, eps=0.012)
#         if i >= start_record_ind:
#             simulation.recored_state_action(target_state, render=False)
#         simulation.step_to_state(target_state, eps=0.008)
#
#     simulation.objects.pop()
#     if i>=0:
#         if two_texture:
#             # simulation.save_as_mp4(simulation.video_2, save_path=save_path+"_2")
#             # simulation.save_as_mp4(np.concatenate(tuple([simulation.video_2 for _ in range(10)]), axis=0),
#             #                        save_path=save_path+"_2")
#             simulation.save_as_npy(simulation.video_2, save_path=save_path+"_2")
#         if simulation.recoder_robot:
#             simulation.save_as_mp4(simulation.robot_video, save_path=save_path + "_robot_view", fps=30)
#         simulation.save_as_npy(simulation.states_and_actions, save_path=save_path+"_state_action_label")
#         simulation.save_as_npy(simulation.segmentation_mask, save_path=save_path+"_seg_mask")
#         simulation.save_as_mp4(simulation.video, save_path=save_path)
#         # simulation.save_as_mp4(np.concatenate(tuple([simulation.video for _ in range(10)]), axis=0), save_path=save_path)
#         simulation.save_as_npy(simulation.video, save_path=save_path)
#
#         # save_object = {"video": simulation.video,
#         #                "video_2": simulation.video_2,
#         #                "label": simulation.states_and_actions,
#         #                "segmentation_mask": simulation.segmentation_mask}
#         # simulation.save_as_npy(save_object, save_path=save_path)
#         print(f"video saved to {save_path}")


def rope_demo(output_dir: str,
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

    os.makedirs(output_dir, exist_ok=True)

    # generate videos
    for simulation_id in id:

        # init simulation
        env = PybulletPushRopePandaEnv(config_path=config_path,
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
                rope_point = env.sample_point_on_rope()
                action = env.sample_push_actions(pushed_coordinate=rope_point, n_actions=1)[0]

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
        rope_demo(output_dir="../datasets/simple_rope",
                  id=i,
                  use_ui=True,
                  random_size=True,
                  random_light=True,
                  random_camera=False,
                  random_texture=True,
                  override_saved_data=True,
                  generate_alternative=True,
                  record_robot_view=False)