import argparse
import multiprocessing
import numpy as np
from shutil import copyfile
import os

from franka_panda.pybullet_simulation.push_cube_demo import cube_demo
from franka_panda.pybullet_simulation.push_rope_demo import rope_demo


class SimulationWorker(multiprocessing.Process):
    def __init__(self, id_range, run_function, output_dir, texture, n_frames, config_path: str):
        multiprocessing.Process.__init__(self)
        self.id_range = id_range
        self.run_function = run_function
        self.output_dir = output_dir
        self.texture = texture
        self.n_frames = n_frames
        self.config_path=config_path

    def run(self):
        self.run_function(output_dir=self.output_dir,
                          id=list(range(self.id_range[0], self.id_range[1])),
                          use_ui=False,
                          generate_alternative=self.texture,
                          random_size=True,
                          random_light=True,
                          random_camera=False,
                          n_frames=self.n_frames,
                          config_path=self.config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output_dir', required=True, type=str, help='path to store generated data.')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--n_start', default=1, type=int, help='init video.')
    parser.add_argument('--n_end', default=11, type=int, help='end video.')
    parser.add_argument('-t', '--texture', action='store_true', required=False, help='2 textures')
    parser.add_argument('-l', '--video_length', type=int, default=10, required=False, help='video length')
    parser.add_argument('-e', '--env', type=str, default="cube", required=False, help='either cube or rope')
    parser.add_argument('-c', '--config_path', type=str, default="default_push.yml",
                        required=False, help='path to config file')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = args.config_path
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "pybullet_simulation", "configs", config_path)

    copyfile(config_path, os.path.join(args.output_dir, "simulation_config.yml"))
    n_per_worker = int((args.n_end - args.n_start) / args.n_workers)
    id_range = [[i, np.minimum(i + n_per_worker, args.n_end)] for i in range(args.n_start, args.n_end, n_per_worker)]

    env_func = rope_demo if args.env == "rope" else cube_demo
    workers = [SimulationWorker(output_dir=args.output_dir,
                                id_range=id_range[i],
                                run_function=env_func,
                                texture=args.texture,
                                n_frames=args.video_length,
                                config_path=config_path)
               for i in range(args.n_workers)]

    if args.n_workers == 1:
        workers[0].run()
    else:
        for w in workers:
            w.start()
