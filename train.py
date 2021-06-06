import argparse
import os

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

from data.dataset import get_dataloader
from evaluation.nearest_neighbours import generate_dataset_features_matrix
from model.cpc import ControlCPC
from utils import load_from_checkpoint, save_checkpoint, freeze_weights, load_config, dump_config
from evaluation.metrics import Statistics, GeomLinearRegressionMetric, NearestNeighboursAccuracyMetric, \
    NearestNeighboursIOUMetric, NearestNeighboursGeometricMetric, ForwardModelMSEPredictionMetric, PlanningMetric


def train_epoch(model: ControlCPC, optimizer: Optimizer, train_loader: data.DataLoader, config, epoch: int, device: str,
                freeze_encoder: bool = False):
    """
    performs training for a single epoch
    :param model:
    :param optimizer:
    :param train_loader:
    :param config: config file
    :param epoch: number of epoch
    :param device: cuda device to run on
    :param freeze_encoder: if True, encoder parameters are frozen, used to fine-tune forward model parameters
    :return:
    """
    epoch_stats = Statistics()
    model.train()
    fm_mse_prediction = ForwardModelMSEPredictionMetric(prefix="Train_Epochs/")

    if freeze_encoder:
        freeze_weights(model.encoder)

    for i, (batch, _, _) in enumerate(tqdm(train_loader, desc="Train")):

        # load to device
        observations, next_observations, actions = [b.float().to(device) for b in batch]

        # forward pass
        z, z_next, z_next_hat = model(observations, next_observations, actions)
        assert not (z != z).any(), "Nan values z"
        assert not (z_next != z_next).any(), "Nan values z_next"
        assert not (z_next_hat != z_next_hat).any(), "Nan values z_next_hat"

        # norm
        z_norm = z_next.norm(p=2, dim=-1).mean().item()
        z_hat_norm = z_next_hat.norm(p=2, dim=-1).mean().item()
        epoch_stats.add("Train_Epochs/Z_norm", z_norm)
        epoch_stats.add("Train_Epochs/Z_hat_norm", z_hat_norm)
        epoch_stats.update_from_stats(fm_mse_prediction.calculate(z_next, z_next_hat))

        # loss and accuracy
        loss, accuracy = model.compute_loss(z, z_next, z_next_hat, actions)
        if "add_mse_loss" in config and config["add_mse_loss"]:
            loss += config["mse_weight"]*torch.nn.MSELoss()(z_next, z_next_hat)
        epoch_stats.add("Train_Epochs/Loss", loss.item())
        for k, v in accuracy.items():
            epoch_stats.add(f"Train_Epochs/top_{k}_accuracy", v.item())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 20)
        nn.utils.clip_grad_value_(model.parameters(), 0.3)
        optimizer.step()

    avg_loss = np.mean(epoch_stats['Train_Epochs/Loss'][-50:])
    avg_acc = np.mean(epoch_stats['Train_Epochs/top_1_accuracy'][-50:])
    print(f'Epoch {epoch}, Train Loss {avg_loss:.4f}, Train Accuracy {avg_acc:.4f}')

    return epoch_stats


def validation(model: ControlCPC, val_loader: data.DataLoader, epoch: int, device: str, config, test_loader: data.DataLoader = None):
    epoch_stats = Statistics()
    model.eval()
    total_loss = 0.
    n_samples = 0.

    # valitaion mertics
    nn_acc = NearestNeighboursAccuracyMetric(similarity_func=model.similarity_func_name, top_k=[1, 3, 10], ignore_first=True, prefix="Val_NN_Metrics/")
    nn_fm_acc = NearestNeighboursAccuracyMetric(similarity_func=model.similarity_func_name, prefix="Val_NN_Metrics/fm_prediction_", top_k=[1, 3, 10])
    nn_geom = NearestNeighboursGeometricMetric(similarity_func=model.similarity_func_name, prefix="Val_NN_Metrics/")
    nn_fm_geom = NearestNeighboursGeometricMetric(similarity_func=model.similarity_func_name, prefix="Val_NN_Metrics/fm_prediction_")
    nn_iou = NearestNeighboursIOUMetric(similarity_func=model.similarity_func_name, prefix="Val_NN_Metrics/")
    nn_fm_iou = NearestNeighboursIOUMetric(similarity_func=model.similarity_func_name, prefix="Val_NN_Metrics/fm_prediction_")

    geom_linear_regression = GeomLinearRegressionMetric(prefix="Val_Metrics/")
    fm_mse_prediction = ForwardModelMSEPredictionMetric(prefix="Val_Metrics/")

    # test mertics
    test_nn_geom = NearestNeighboursGeometricMetric(similarity_func=model.similarity_func_name, prefix="Test_NN_Metrics/")
    test_nn_fm_geom = NearestNeighboursGeometricMetric(similarity_func=model.similarity_func_name, prefix="Test_NN_Metrics/fm_prediction_")
    test_nn_iou = NearestNeighboursIOUMetric(similarity_func=model.similarity_func_name, prefix="Test_NN_Metrics/")
    test_nn_fm_iou = NearestNeighboursIOUMetric(similarity_func=model.similarity_func_name, prefix="Test_NN_Metrics/fm_prediction_")
    test_nn_fm_acc = NearestNeighboursAccuracyMetric(similarity_func=model.similarity_func_name, prefix="Test_NN_Metrics/fm_prediction_", top_k=[1, 3, 10])

    test_geom_linear_regression = GeomLinearRegressionMetric(prefix="Test_Metrics/")
    test_fm_mse_prediction = ForwardModelMSEPredictionMetric(prefix="Test_Metrics/")

    with torch.no_grad():

        features_mat, info = generate_dataset_features_matrix(model.encoder, val_loader, device=device, project_features=True)
        (video_path, frame_ind, seg_masks, state) = info

        ####################################
        # run validation on test data (real)
        ####################################
        if test_loader is not None:

            test_batch, test_segmentation, test_batch_info = next(iter(test_loader))
            # load to device
            test_observations, test_next_observations, test_actions = [b.float().to(device) for b in test_batch]

            # forward pass
            test_z, test_z_next, test_z_next_hat = model(test_observations, test_next_observations, test_actions)
            test_curr_seg_masks, test_next_seg_mask = test_segmentation

            test_features, test_info = generate_dataset_features_matrix(model.encoder, test_loader, device=device, project_features=True)
            (_, _, test_seg_masks, test_state) = test_info

            # add classification metrics:
            if test_state is not None:
                epoch_stats.update_from_stats(test_geom_linear_regression.calculate(features_mat, state, test_features, test_state))

            # fm mse_similarity prediction error
            epoch_stats.update_from_stats(test_fm_mse_prediction.calculate(test_z_next, test_z_next_hat))

            # Information retrieval (nearest neighbours)
            # calculate the next Z prediction nearest neighbour and compare it to the real next state
            epoch_stats.update_from_stats(test_nn_fm_acc.calculate(features_mat, info, test_batch_info.first_path, test_batch_info.next_observation_index,  test_z_next_hat))
            if test_state is not None:
                epoch_stats.update_from_stats(test_nn_geom.calculate(features_mat, state, test_batch_info.next_state, test_z))
                epoch_stats.update_from_stats(test_nn_fm_geom.calculate(features_mat, state, test_batch_info.next_state, test_z_next_hat))
            epoch_stats.update_from_stats(test_nn_iou.calculate(features_mat, seg_masks, test_next_seg_mask, test_z))
            epoch_stats.update_from_stats(test_nn_fm_iou.calculate(features_mat, seg_masks, test_next_seg_mask, test_z_next_hat))

        ####################################
        # run validation on validation data (sim)
        ####################################
        # add classification metrics:
        if state is not None:
            epoch_stats.update_from_stats(geom_linear_regression.calculate(features_mat, state))

        for i, (batch, segmentation, batch_info) in enumerate(tqdm(val_loader, desc="Validation")):

            # load to device
            observations, next_observations, actions = [b.float().to(device) for b in batch]

            # forward pass
            z, z_next, z_next_hat = model(observations, next_observations, actions)
            curr_seg_masks, next_seg_mask = segmentation

            # formard model accuracy
            if model.n_neg_actions > 0:
                fm_accuracy = model.compute_fm_accuracy(z, z_next, z_next_hat, actions)
                for k in model.top_k:
                    epoch_stats.add(f"Val_Epochs/negative_actions_top_{k}_accuracy", fm_accuracy[k].item())

            # fm mse_similarity prediction error
            epoch_stats.update_from_stats(fm_mse_prediction.calculate(z_next, z_next_hat))

            # Information retrieval (nearest neighbours)
            # calculate the next Z prediction nearest neighbour and compare it to the real next state
            epoch_stats.update_from_stats(nn_acc.calculate(features_mat, info, batch_info.first_path, batch_info.observation_index, z))
            epoch_stats.update_from_stats(nn_fm_acc.calculate(features_mat, info, batch_info.first_path, batch_info.next_observation_index, z_next_hat))
            if state is not None:
                epoch_stats.update_from_stats(nn_geom.calculate(features_mat, state, batch_info.state, z))
                epoch_stats.update_from_stats(nn_fm_geom.calculate(features_mat, state, batch_info.next_state, z_next_hat))
            epoch_stats.update_from_stats(nn_iou.calculate(features_mat, seg_masks, curr_seg_masks, z))
            epoch_stats.update_from_stats(nn_fm_iou.calculate(features_mat, seg_masks, next_seg_mask, z_next_hat))

            # # calculate the next Z prediction nearest neighbour and compare it to the real next state
            # _, top_scores_ind = calc_nearest_neighbours(z_next_hat, features_mat, num_neighbours=3, dist_func=model.dist_func_type)
            # if batch_info.next_state is not None:
            #     next_state = batch_info.next_state
            #     next_state_nn = state[top_scores_ind[:, 0]]
            #     epoch_stats.add(f"Val_GEOM_next_prediction",torch.norm(next_state - next_state_nn, dim=-1).mean())
            # epoch_stats.add(f"Val_IOU_next_prediction", calc_IOU(next_seg_mask, seg_masks[top_scores_ind[:, 0]]))
            # # test if the nearest neighbours of the z_next_hat is equal to z_next.
            # same_video = np.array(video_path)[top_scores_ind] == np.tile(batch_info.first_path, (3, 1)).transpose()
            # same_index = np.array(frame_ind)[top_scores_ind] == np.tile(batch_info.next_observation_index, (3, 1)).transpose()
            # epoch_stats.add(f"Val_forward_model_accuracy", (same_video[:, 0] & same_index[:, 0]).mean())
            # epoch_stats.add(f"Val_forward_model_top_3_accuracy", (same_video & same_index).mean()*3.)

            # norm
            z_norm = z_next.norm(p=2, dim=-1).mean().item()
            z_hat_norm = z_next_hat.norm(p=2, dim=-1).mean().item()
            epoch_stats.add("Val_Epochs/Z_norm", z_norm)
            epoch_stats.add("Val_Epochs/Z_hat_norm", z_hat_norm)

            # loss and accuracy
            loss, accuracy = model.compute_loss(z, z_next, z_next_hat, actions)
            if "add_mse_loss" in config and config["add_mse_loss"]:
                loss += config["mse_weight"] * torch.nn.MSELoss()(z_next, z_next_hat)

            total_loss += loss * z.size()[0]
            n_samples += z.size()[0]
            epoch_stats.add("Val_Epochs/Loss", loss.item())
            for k, v in accuracy.items():
                epoch_stats.add(f"Val_Epochs/top_{k}_accuracy", v.item())

    total_loss /= n_samples
    total_acc = np.mean(epoch_stats['Val_Epochs/top_1_accuracy'])
    print(f'Epoch {epoch}, Val Loss {total_loss:.4f}, Val Accuracy {total_acc:.4f}')

    return epoch_stats, total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('-i', '--input', type=str, help='path to dataset directory')
    parser.add_argument('-v', '--val_input', type=str, help='path to dataset directory')
    parser.add_argument('-t', '--test_input', type=str, default=None, help='path to dataset directory')
    parser.add_argument('-o', '--output', type=str, default='/mnt/data/carmel_data/control_models', help='path to output directory')
    parser.add_argument('-s', '--save_name', type=str, default='model', help='model results save name')
    parser.add_argument('-c', '--cfg', type=str, default='config/cdr.yml', help='path to config file')
    parser.add_argument('-w', '--workers', type=int, default=6, help='num workers for DataLoader')
    parser.add_argument('--log_dir', type=str, default="../runs/logs/",
                        required=False, help='tensorboard log directory')

    # Learning Parameters
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs to run; default: 100')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='number of epochs to run until saving model parameters; default: 5')
    parser.add_argument('-p', '--eval_planning_interval', type=int, default=-1,
                        help='default: -1, meaning no planning evaluation will be applied')

    # Other
    parser.add_argument('-r', '--reload_model', action='store_true',
                        required=False, help='check if model already exists and reload from last saved checkpoint')
    parser.add_argument('-f', '--freeze_encoder', action='store_true',
                        required=False, help='re-train model on forward model only')
    parser.add_argument('-m', '--model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.model_path is not None and args.reload_model:
        assert False, "Can't use \"--model_path\" and \"--reload_model\" simultaneity"

    # init seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reload config and model
    output_dir_path = os.path.join(args.output, args.save_name)
    if args.reload_model:
        # load config
        assert os.path.exists(output_dir_path + "/config.yml"), f"no config files in {output_dir_path}"
        config = load_config(path=output_dir_path+"/config.yml")
        print(f"Loaded config file from: {output_dir_path}")

        # find and load checkpoint from last saved epoch
        epochs_saved = [int(f.split("_")[-1]) for f in os.listdir(output_dir_path) if f.startswith("checkpoint")]
        if len(epochs_saved) > 0:
            max_checkpoint = max(epochs_saved)
            checkpoint_path = os.path.join(output_dir_path, f"checkpoint_epoch_{max_checkpoint}")

            # init model and optimizer
            model, optimizer, start_epoch = load_from_checkpoint(path=checkpoint_path, config=config, device=device)
            print(f"\nContinue training from epoch {start_epoch}")
        else:
            # init model and optimizer
            model = ControlCPC(config)
            model.to(device)
            parameters = list(model.parameters())
            optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
            start_epoch = 0

    elif args.model_path is not None:
        # load config
        if args.cfg is None:
            base_path, checkpoint = os.path.split(args.model_path)
            args.cfg = os.path.join(base_path, "config.yml")
            assert os.path.exists(base_path + "/config.yml"), f"no config files in {base_path}"
        config = load_config(path=args.cfg)
        dump_config(output_directory=output_dir_path, config=config)

        # reload model
        assert os.path.exists(args.model_path), f"can't find model file in: {args.model_path}"
        model, optimizer, start_epoch = load_from_checkpoint(path=args.model_path, config=config, device=device)
        print(f"Continue training from epoch {start_epoch}")

    # dump config file and start training
    else:
        # load config
        config = load_config(path=args.cfg)
        dump_config(output_directory=output_dir_path, config=config)

        # init model and optimizer
        model = ControlCPC(config).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        start_epoch = 0

    # Load simulation config only if evaluating planning during training as metric
    if args.eval_planning_interval > 0:
        if "simulation_config_path" in config and config.simulation_config_path != -1:
            simulation_config = config.simulation_config_path
        elif os.path.exists(os.path.join(args.input, "simulation_config.yml")):
            simulation_config = os.path.join(args.input, "simulation_config.yml")
        else:
            assert False, "Error, need to provide simulation_config.yml file path"

    # train data loaders
    train_loader = get_dataloader(args.input, config["train_n_videos"], config["train_range"], config,
                                  workers=args.workers, load_seg_masks=False)

    # validation data loader
    val_workers = min(args.workers, 4)
    val_config = copy.deepcopy(config)
    val_config.use_data_aug = False
    val_config.contrastive_dr = False
    val_loader = get_dataloader(args.val_input, config.val_n_videos, config.val_range, val_config,
                                workers=val_workers, load_seg_masks=True)
    val_loader.dataset.contrastive_and_mse = False

    # test data loaders
    if args.test_input is not None:
        test_config = copy.deepcopy(config)
        test_config["use_both_textures"] = False
        test_config["contrastive_dr"] = False
        test_loader = get_dataloader(args.test_input,
                                     n_videos=-1,
                                     sample_range=-1,
                                     config=test_config,
                                     workers=val_workers,
                                     load_seg_masks=True,
                                     use_full_batch_size=True)
    else:
        test_loader = None

    # tensorboard logs
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.save_name))

    if args.freeze_encoder:
        print(f"Fine tune training with frozen encoder")

    n_updates = 0
    best_val_loss = float('inf')    # can be used for early stop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Train epoch and save statistics
        epoch_stats = train_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            epoch=epoch,
            device=device,
            freeze_encoder=args.freeze_encoder)

        # run validation and test and save statistics
        val_stats, val_loss = validation(model=model,
                                         val_loader=val_loader,
                                         test_loader=test_loader,
                                         epoch=epoch, device=device, config=config)

        # log to Tensorboard epoch summery
        for k, values in epoch_stats.items():
            writer.add_scalar(k, np.mean(values), epoch)
        for k, values in val_stats.items():
            writer.add_scalar(k, np.mean(values), epoch)

        # log to Tensorboard updates summery for each batch
        for k, values in epoch_stats.items():
            for i, v in enumerate(values):
                writer.add_scalar(k.replace("Train_Epochs", "Train_Updates"), v, n_updates + i)
        n_updates += i+1

        # dump model parameters
        if epoch % args.log_interval == 0:
            save_checkpoint(output_dir_path, model, optimizer, epoch, val_loss)
        # this can be used to save only when validation loss is improving
        # if val_loss <= best_val_loss:
        #     best_val_loss = val_loss
        #     save_checkpoint(output_dir_path, model, optimizer, epoch, val_loss)

        # report how good this model performs planning
        if args.eval_planning_interval > 0 and epoch % args.eval_planning_interval == 0:
            planning_stats = PlanningMetric(n_experiments=20,
                                            device=device,
                                            stop_when_not_improving=False,
                                            use_goal_from_different_domain=False,
                                            random_actions_p=0.,
                                            prefix="Planning/same_domains_",
                                            config_path=simulation_config).calculate(model)
            for k, values in planning_stats.items():
                writer.add_scalar(k, np.mean(values), epoch)
            planning_stats = PlanningMetric(n_experiments=20,
                                            device=device,
                                            stop_when_not_improving=False,
                                            use_goal_from_different_domain=True,
                                            random_actions_p=0.,
                                            prefix="Planning/different_domains_",
                                            config_path=simulation_config).calculate(model)
            for k, values in planning_stats.items():
                writer.add_scalar(k, np.mean(values), epoch)




