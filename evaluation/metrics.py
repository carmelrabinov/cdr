from typing import List, Any, Callable

import torch
from torch import Tensor
from sklearn.linear_model import LinearRegression
import numpy as np

from evaluation.IOU import calc_IOU
from evaluation.mpc import evaluate_model_planning
from evaluation.nearest_neighbours import calc_nearest_neighbours


class Statistics:
    """ object to save statistics """
    def __init__(self):
        self.statistics = dict()

    def add(self, key: str, value: Any):
        if key not in self.statistics:
            self.statistics[key] = []
        self.statistics[key].append(value)

    def keys(self):
        return list(self.statistics.keys())

    def __getitem__(self, key: str):
        return self.statistics[key]

    def items(self):
        return self.statistics.items()

    def print(self):
        for k, v in self.statistics.items():
            print(f"{k}: {np.mean(v)}")

    def print_mean_std(self):
        for k, v in self.statistics.items():
            if k.endswith("mean") and k[:-4]+"std" in self.statistics.keys():
                print(f"{k[:-4]}: {v[0]:.2f} +- {self.statistics[k[:-4]+'std'][0]:.2f}")

    def update(self, key: str, values: List[Any]):
        """ add values to the Statistics objects """
        if key not in self.statistics:
            self.statistics[key] = values
        self.statistics[key].extend(values)

    def update_from_stats(self, stats):
        """ add values from Statistics objects """
        for key, values in stats.items():
            self.update(key, values)


class Metric():
    def __init__(self, prefix: str = ""):
        self._prefix = prefix

    def calculate(self, input: Any) -> Statistics:
        raise NotImplementedError


class GeomLinearRegressionMetric(Metric):
    def calculate(self, features_mat, states_mat, test_features=None, test_states=None) -> Statistics:
        """
        train a linear regression model Wz = state, calculate the euclidean error in cm
        and the 95% and 99% percentiles
        if test_features and test_states are not provided, error will be reported on features_mat and states_mat.
        :param features_mat: n_samples X n_features matrix
        :param states: n_samples X 2 euclidean position matrix
        :param test_features: n_batch_samples X n_features matrix
        :param test_states: n_batch_samples X 2 euclidean position matrix
        :return:
        """

        # fit the data to the linear model
        model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
        x = features_mat.cpu().numpy() if isinstance(features_mat, torch.Tensor) else features_mat
        y = states_mat.cpu().numpy() if isinstance(states_mat, torch.Tensor) else states_mat
        model.fit(x, y)
        w = model.coef_
        bias = model.intercept_

        # if test_features and test_states are not provided, error will be reported on features_mat and states_mat.
        if test_features is not None and test_states is not None:
            x_test = test_features.cpu().numpy() if isinstance(test_features, torch.Tensor) else test_features
            y_test = test_states.cpu().numpy() if isinstance(test_states, torch.Tensor) else test_states
            geom_error = np.linalg.norm((w.dot(x_test.T) + np.expand_dims(bias, axis=1)).T - y_test, axis=1)
        else:
            geom_error = np.linalg.norm((w.dot(x.T) + np.expand_dims(bias, axis=1)).T - y, axis=1)

        errors = Statistics()
        errors.add(f"{self._prefix}predict_geom_state_from_features_error", geom_error.mean())
        errors.add(f"{self._prefix}predict_geom_state_from_features_99_percentile_error", np.percentile(geom_error, 99))
        errors.add(f"{self._prefix}predict_geom_state_from_features_95_percentile_error", np.percentile(geom_error, 95))
        return errors


class ForwardModelMSEPredictionMetric(Metric):
    def calculate(self, z_next: Tensor, z_next_hat: Tensor) -> Statistics:
        """
        compute the MSE of error of the forward model ||z_next - z_next_hat||
        :param z_next: the true next state representations, n_batch_samples X n_features matrix
        :param z_next_hat: the predicted next state representations, n_batch_samples X n_features matrix
        :return:
        """
        errors = Statistics()
        fm_mse_error = torch.norm(z_next - z_next_hat, dim=-1).detach().cpu().numpy()
        errors.add(f"{self._prefix}fm_MSE_prediction_error", fm_mse_error.mean())
        errors.add(f"{self._prefix}fm_MSE_prediction_99_percentile_error", np.percentile(fm_mse_error, 99))
        errors.add(f"{self._prefix}fm_MSE_prediction_95_percentile_error", np.percentile(fm_mse_error, 95))
        return errors


class NearestNeighboursGeometricMetric(Metric):
    def __init__(self, similarity_func: Callable[[Tensor, Tensor], Tensor], prefix: str = ""):   #fm_prediction
        self._similarity_func = similarity_func
        self._prefix = prefix

    def calculate(self, features_mat: Tensor, states_mat: Tensor, true_next_state: Tensor, z: Tensor) -> Statistics:
        """
        given the state representations, find their nearest neighbour and compute euclidean error in cm
        and the 95% and 99% percentiles
        :param features_mat: n_samples X n_features matrix
        :param states_mat: n_samples X 2 euclidean position matrix
        :param true_next_state: the positive next state
        :param z: the state representations, n_batch_samples X n_features matrix
        :return:
        """
        assert true_next_state is not None, "Error, NearestNeighboursGeometricMetric, must provide state"
        stats = Statistics()

        # calculate the next Z prediction nearest neighbour and compare it to the real next state
        top_scores, top_scores_ind = calc_nearest_neighbours(z, features_mat, k_neighbours=3, similarity_func=self._similarity_func)
        next_state_nn = states_mat[top_scores_ind[:, 0]]
        geom_error = torch.norm(true_next_state - next_state_nn, dim=-1).detach().cpu().numpy()
        stats.add(f"{self._prefix}nn_geom_error", geom_error.mean())
        stats.add(f"{self._prefix}nn_geom_99_percentile_error", np.percentile(geom_error, 99))
        stats.add(f"{self._prefix}nn_geom_95_percentile_error", np.percentile(geom_error, 95))
        return stats


class NearestNeighboursIOUMetric(Metric):
    def __init__(self, similarity_func: Callable[[Tensor, Tensor], Tensor], prefix: str = ""):
        self._similarity_func = similarity_func
        self._prefix = prefix

    def calculate(self, features_mat: Tensor, seg_masks: Tensor, true_next_seg_mask: Tensor, z: Tensor) -> Statistics:
        """
        given the state representations, find their nearest neighbour and compute IOU
        :param features_mat: n_samples X n_features matrix
        :param seg_masks: n_samples X image_shape segmentation masks
        :param true_next_seg_mask: the positive next state segmentation mask
        :param z: the state representations, n_batch_samples X n_features matrix
        :return:
        """
        stats = Statistics()
        top_scores, top_scores_ind = calc_nearest_neighbours(z, features_mat, k_neighbours=3, similarity_func=self._similarity_func)
        stats.add(f"{self._prefix}nn_IOU", calc_IOU(true_next_seg_mask, seg_masks[top_scores_ind[:, 0]]))
        return stats


class NearestNeighboursAccuracyMetric(Metric):
    def __init__(self, similarity_func: Callable[[Tensor, Tensor], Tensor], prefix: str = "",
                 top_k: List[int] = [1, 3], ignore_first: bool = False):
        self._similarity_func = similarity_func
        self._prefix = prefix
        self._top_k = [k+1 for k in top_k] if ignore_first else top_k
        self._max_k = int(np.max(top_k)) + 1 if ignore_first else int(np.max(top_k))
        self._init_index = 1 if ignore_first else 0

    def calculate(self, features_mat: Tensor, info, batch_path, batch_index, z: Tensor) -> Statistics:
        """
        given the state representations, find their nearest neighbour and compute recover accuracy
        :param features_mat: n_samples X n_features matrix
        :param info: n_samples X batch information (file path and names)
        :param batch_path: n_batch_samples X batch path information (file path)
        :param batch_index: n_batch_samples X batch index information (index within file)
        :param z: the state representations, n_batch_samples X n_features matrix
        :return: Statistics
        """
        stats = Statistics()
        (video_path, frame_ind, _, _) = info

        top_scores, top_scores_ind = calc_nearest_neighbours(z, features_mat, k_neighbours=self._max_k,
                                                             similarity_func=self._similarity_func)

        # test if the nearest neighbours of the z_next_hat is equal to z_next.
        same_video = np.array(video_path)[top_scores_ind] == np.tile(batch_path, (self._max_k, 1)).transpose()
        same_index = np.array(frame_ind)[top_scores_ind] == np.tile(batch_index, (self._max_k, 1)).transpose()
        for k in self._top_k:
            stats.add(f"{self._prefix}nn_top_{k-self._init_index}_accuracy", (same_video[:, self._init_index:k] & same_index[:, self._init_index:k]).mean() * (k-self._init_index))
        return stats


class PlanningMetric(Metric):
    def __init__(self, device: str,
                 config_path: str,
                 prefix: str = "Planning/",
                 n_experiments: int = 20,
                 stop_when_not_improving: bool = False,
                 tolerance_to_goal_in_cm: float = 3.,
                 random_actions_p: float = 0.,
                 use_goal_from_different_domain: bool = False):
        self._prefix = prefix
        self._device = device
        self._n_experiments = n_experiments
        self.config_path = config_path
        self._tolerance_to_goal_in_cm = tolerance_to_goal_in_cm
        self._stop_when_not_improving = stop_when_not_improving
        self._random_actions_p = random_actions_p
        self._use_goal_from_different_domain=use_goal_from_different_domain

    def calculate(self, model) -> Statistics:
        """
        plan and report stats
        :param model: model
        :return:
        """
        stats = Statistics()

        final_dists, init_dists, trials, gains, min_dists, trails_to_min_dist = evaluate_model_planning(model=model,
                                                                         device=self._device,
                                                                         use_oracle=False,
                                                                         n_experiments=self._n_experiments,
                                                                         verbose=False,
                                                                         save_dir=None,
                                                                         random_actions_p=self._random_actions_p,
                                                                         n_process=1,
                                                                         n_steps=10,
                                                                         tolerance_to_goal_in_cm=self._tolerance_to_goal_in_cm,
                                                                         stop_when_not_improving=self._stop_when_not_improving,
                                                                         config_path=self.config_path,
                                                                         use_goal_from_different_domain=self._use_goal_from_different_domain)
        stats.add(f"{self._prefix}final_dist_mean", np.mean(final_dists))
        stats.add(f"{self._prefix}final_dist_std", np.std(final_dists))
        stats.add(f"{self._prefix}init_dist_mean", np.mean(init_dists))
        stats.add(f"{self._prefix}init_dist_std", np.std(init_dists))
        stats.add(f"{self._prefix}trails_mean", np.mean(trials))
        stats.add(f"{self._prefix}trails_std", np.std(trials))
        stats.add(f"{self._prefix}dist_gains_mean", np.mean(gains))
        stats.add(f"{self._prefix}dist_gains_std", np.std(gains))
        stats.add(f"{self._prefix}min_dist_mean", np.mean(min_dists))
        stats.add(f"{self._prefix}trails_to_min_mean", np.mean(trails_to_min_dist))
        stats.add(f"{self._prefix}min_dist_std", np.std(min_dists))
        stats.add(f"{self._prefix}trails_to_min_std", np.std(trails_to_min_dist))
        return stats
