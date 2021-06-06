import argparse

import os
import pickle

import torch
import numpy as np
import yaml

from evaluation.metrics import PlanningMetric
from utils import load_from_checkpoint

if __name__ == '__main__':

    starts, goals = np.load("evaluation/cube_start_goal_dataset.npy")

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, default=None,
                        required=False, action='append', help='path to trained model')
    parser.add_argument('-o', '--output', type=str, default='../control_results', help='path to output directory')
    parser.add_argument('-c', '--simulation_config_path', type=str, help='path to simulation config')
    parser.add_argument('-n', '--n_experiments', type=int, default=2, help='number of planning experiments')

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # output directory
    os.makedirs(args.output, exist_ok=True)
    results_file = os.path.join(args.output, "results.pkl")
    pre_calculated_model = []

    # load pre existing results and print them
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
            for model_results in results:
                if isinstance(model_results, dict):
                    for model_name, model_stats in model_results.items():
                        pre_calculated_model.append(model_name)
                        print(f"model {model_name} :")
                        model_stats.print_mean_std()

    else:
        results = []
        results.append({f"n_experiments: {args.n_experiments}"})
    for model_path in args.models:

        # init stats
        metric = PlanningMetric(n_experiments=args.n_experiments,
                                device=device,
                                stop_when_not_improving=False,
                                config_path=args.simulation_config_path,
                                tolerance_to_goal_in_cm=3.,
                                random_actions_p=0.,
                                use_goal_from_different_domain=True)

        # reload config and model
        assert os.path.exists(model_path), f"can't find model file in: {model_path}"
        base_path, checkpoint = os.path.split(model_path)
        _, model_name = os.path.split(base_path)
        full_model_name = f"{model_name}_epoch_{checkpoint.split('_')[-1]}"
        model_save_name = os.path.join(args.output, full_model_name)
        os.makedirs(model_save_name, exist_ok=True)

        # skip pre calculated results
        if full_model_name in pre_calculated_model:
            print(f"Skipping model {full_model_name}")
            continue

        # load config
        cfg_path = os.path.join(base_path, "config.yml")
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # init model and optimizer
        model, _, _ = load_from_checkpoint(path=model_path, config=config, device=device)

        model_stats = metric.calculate(model)
        results.append({full_model_name: model_stats})
        print(full_model_name + ":")
        model_stats.print_mean_std()
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

        with open(os.path.join(model_save_name, "results.pkl"), "wb") as f:
            pickle.dump([{model_name: model_stats}], f)
