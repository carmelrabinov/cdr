import argparse
import os
from typing import List

import numpy as np
import torch
import yaml
from torch.utils import data
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from data.dataset import TwoStateActionPairIndex, get_dataloader
from data.utils import load_single_image, load_frame_from_video, load_seg_masks_from_video, seg_mask_transform, normalize
from evaluation.IOU import calc_mean_iou
from model.encoder import Encoder
from utils import load_from_checkpoint


def calc_nearest_neighbours(query_features: torch.FloatTensor, external_dataset_mat: torch.FloatTensor, k_neighbours: int = 5,
                            similarity_func: str = "dot_product_similarity") -> [np.ndarray, np.ndarray]:
    """
    for each features vector, calculate k nearest neighbours (NN) in an external dataset
    :param query_features: a matrix where each row is a latent representation to calculate NN for
    :param external_dataset_mat: the external dataset to look NN in as latent representations
    :param k_neighbours: number of NN to look
    :param similarity_func: similarity function between neighbours features_vec
    :return: top_scores - array of k similarity measures for each vector in query_features
             top_scores_ind - the indexes in the external dataset of the k NN
    """
    if similarity_func == "mse_similarity":
        vec = query_features.repeat(external_dataset_mat.size()[0], 1, 1).transpose(0, 1).squeeze()
        if len(vec.shape) == 3:
            external_dataset_mat = external_dataset_mat.repeat(vec.shape[0], 1, 1)
        similarity = -1 * torch.nn.MSELoss(reduction="none")(vec, external_dataset_mat).mean(-1).squeeze()
    elif similarity_func == "dot_product_similarity":
        similarity = torch.mm(query_features, external_dataset_mat.T).squeeze()
    elif similarity_func == "cosine_similarity":
        similarity = torch.mm(query_features.div(query_features.norm(p=2, dim=-1, keepdim=True)),
                              external_dataset_mat.div(external_dataset_mat.norm(p=2, dim=-1, keepdim=True)).T).squeeze()

    else:
        assert False, f"Similarity function is not recognized: {similarity_func}"
    top_scores, top_scores_ind = similarity.topk(k=k_neighbours, largest=True, sorted=True)
    return top_scores.cpu().numpy(), top_scores_ind.cpu().numpy()


def generate_dataset_features_matrix(encoder: Encoder, loader: data.DataLoader, device: str,
                                     project_features: bool = True) -> [torch.Tensor, tuple]:
    """
    run all dataset through the encoder and generates latent representations for each image
    :param encoder: Encoder module
    :param loader: DataLoader
    :param device: device
    :param project_features: bool, if True use projected features, else use full features
    :return: features_mat - a matrix of size n_samples X n_features
             info - a tuple of (video_path, frame_ind, seg_masks, states) corresponding to each image in the features_mat
    """

    # init
    encoder.eval()
    encoder.to(device)
    features_mat: torch.FloatTensor = None
    video_path = []
    frame_ind = []
    seg_masks: torch.boolTensor = None
    states = None

    label: TwoStateActionPairIndex
    for (batch, seg, label) in tqdm(loader, desc="Generating features"):
        with torch.no_grad():
            # load to device
            seg_mask, _ = seg
            observations, _, _ = [b.float().to(device) for b in batch]
            first_path = label.first_path
            observation_index = label.observation_index
            state = label.state

            # forward pass
            if project_features:
                z = encoder(observations)
            else:
                z = encoder.extract_features(observations)

            features_mat = z if features_mat is None else torch.cat((features_mat, z), dim=0)
            video_path.extend(first_path)
            frame_ind.extend(observation_index)
            if len(seg_mask):
                seg_masks = seg_mask if seg_masks is None else torch.cat((seg_masks, seg_mask), dim=0)
            if state is not None:
                states = state if states is None else torch.cat((states, state), dim=0)

    return features_mat, (video_path, frame_ind, seg_masks, states)


def visualize_nearest_neighbours(results: List[tuple], save_path: str = None, title: str = None) -> None:
    """
    plots nearest neighbours images
    results should be a list of tuples (ref_image, top_scores, top_imgs, mean_iou, iou, top_ind)
    :param results: list of tuples the holds the results
    :param save_path: path to save NN results as image
    :param title: plot title
    :return: None
    """

    n_rows = len(results)
    num_nn = len(results[0][1])
    n_cols = num_nn + 1
    fig = plt.figure(figsize=(20, int(20*n_rows/5.)))
    if title is not None:
        fig.suptitle(title, fontsize=60)
    for row, info in enumerate(results):
        (ref_image, top_scores, top_imgs, mean_iou, iou, top_ind) = info
        fig.add_subplot(n_rows, n_cols, row * n_cols + 1).set_title(f"Ref: IOU: {mean_iou:.4f}", fontsize=20)
        plt.imshow(ref_image)
        plt.axis('off')
        for i in range(num_nn):
            fig.add_subplot(n_rows, n_cols, row * n_cols + i + 2).set_title(f"{i+1}: {top_scores[i]:.4f}, {iou[i]:.4f}", fontsize=20)
            plt.imshow(top_imgs[i])
            plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path + ".png")
    # plt.show()


def find_nearest_neighbours(images_path_list: List[str], encoder: Encoder, loader: data.DataLoader, device: str,
                            save_path: str, project_features: bool = True, similarity_func: str = "dot_product_similarity",
                            compare_segmentation_mask: bool = True, dump_raw_results_path: str = None) -> None:
    """
    find nearest neighbours (NN) for each image in images_path_list in an external dataset defined by a DataLoader
    :param images_path_list: list of images to find NN for them
    :param encoder: Encoder
    :param loader: DataLoader
    :param device: device
    :param save_path: path to save NN results as image
    :param project_features: bool, if True use projected features, else use full features
    :param similarity_func: similarity measurement
    :param compare_segmentation_mask: whether to calculate IOU for NN
    :param dump_raw_results_path: path to dump the external data as latent features
    :return: None
    """
    with torch.no_grad():
        # generate external_dataset_mat from data
        if os.path.exists(os.path.join(dump_raw_results_path, "external_dataset_mat.pkl")):
            with open(os.path.join(dump_raw_results_path, "external_dataset_mat.pkl"), "rb") as f:
                (external_dataset_mat, info) = pickle.load(f)
        else:
            external_dataset_mat, info = generate_dataset_features_matrix(encoder, loader, device=device,
                                                                          project_features=project_features)

        if dump_raw_results_path is not None:
            with open(os.path.join(dump_raw_results_path, "external_dataset_mat.pkl"), "wb") as f:
                pickle.dump((external_dataset_mat, info), f)

        (video_path, frame_ind, seg_masks, geom_location) = info

        # for each observation find nearest neighbours
        results = []
        iou_scores = []
        ref_features_vec = None
        for _, img_path in enumerate(images_path_list):
            ref_image = load_single_image(img_path)
            obs = normalize(loader.dataset.image_to_tensor(ref_image))
            if compare_segmentation_mask:
                ref_seg_mask = load_seg_masks_from_video(img_path[:-4] + config["seg_masks_files_sfx"])
                ref_seg_mask = seg_mask_transform(ref_seg_mask)
            features_vec = encoder(obs.float().to(device).unsqueeze(0))
            ref_features_vec = features_vec if ref_features_vec is None else torch.cat((ref_features_vec, features_vec), dim=0)
            top_scores, top_scores_ind = calc_nearest_neighbours(features_vec, external_dataset_mat, k_neighbours=5, similarity_func=similarity_func)
            top_img = []
            for top_ind in top_scores_ind:
                top_img.append(load_frame_from_video(video_path[top_ind], frame_ind[top_ind]))
            if compare_segmentation_mask:
                mean_iou, iou = calc_mean_iou(ref_seg_mask, [seg_masks[top_ind] for top_ind in top_scores_ind])
                iou_scores.append(mean_iou)
            else:
                mean_iou = 0.
                iou = np.zeros(len(top_scores))
            results.append((ref_image, top_scores, top_img, mean_iou, iou, top_scores_ind))

    if dump_raw_results_path is not None:
        os.remove(os.path.join(dump_raw_results_path, "external_dataset_mat.pkl"))
        with open(os.path.join(dump_raw_results_path, "nn_results.pkl"), "wb") as f:
            pickle.dump((results, external_dataset_mat, ref_features_vec, geom_location), f)

    if compare_segmentation_mask:
        print(f"Mean IOU score: {np.mean(iou_scores)}")
    visualize_nearest_neighbours(results, save_path=save_path)


def find_nearest_neighbours_from_list(images_path_list, encoder: Encoder, loader: data.DataLoader, device,
                                      save_path, project_features=True, dist_func="dot_product_similarity",
                                      compare_segmentation_mask=True, dump_raw_results_path: str = None):
    with torch.no_grad():

        # generate_feature_mat
        seg_masks = None
        external_dataset_mat = None
        images = []
        ref_features_vec = None
        for _, img_path in enumerate(images_path_list):
            ref_image = load_single_image(img_path)
            obs = normalize(loader.dataset.image_to_tensor(ref_image)).float().to(device).unsqueeze(0)
            if compare_segmentation_mask:
                seg_mask = load_seg_masks_from_video(img_path[:-4] + config["seg_masks_files_sfx"])
                seg_mask = seg_mask_transform(seg_mask)
                seg_masks = seg_mask if seg_masks is None else torch.cat((seg_masks, seg_mask), dim=0)
            features_vec = encoder(obs) if project_features else encoder.extract_features(obs)
            ref_features_vec = features_vec if ref_features_vec is None else torch.cat((ref_features_vec, features_vec), dim=0)
            external_dataset_mat = features_vec if external_dataset_mat is None else torch.cat((external_dataset_mat, features_vec), dim=0)
            images.append(ref_image)

        # for each observation find nearest neighbours
        results = []
        iou_scores = []
        for i, img_path in enumerate(images_path_list):
            features_vec = external_dataset_mat[i]
            ref_image = images[i]
            top_scores, top_scores_ind = calc_nearest_neighbours(features_vec, external_dataset_mat, k_neighbours=5, similarity_func=dist_func)
            top_img = []
            for top_ind in top_scores_ind:
                top_img.append(images[top_ind])
            if compare_segmentation_mask:
                mean_iou, iou = calc_mean_iou(seg_masks[i], [seg_masks[top_ind] for top_ind in top_scores_ind])
                iou_scores.append(mean_iou)
            else:
                mean_iou = 0.
                iou = np.zeros(len(top_scores))
            results.append((ref_image, top_scores, top_img, mean_iou, iou, top_scores_ind))

    if dump_raw_results_path is not None:
        with open(os.path.join(dump_raw_results_path, "nn_results.pkl"), "wb") as f:
            pickle.dump((results, external_dataset_mat, ref_features_vec, None), f)

    if compare_segmentation_mask:
        print(f"Mean IOU score: {np.mean(iou_scores)}")
    visualize_nearest_neighbours(results, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to trained model')
    parser.add_argument('-i', '--input', type=str, required=True, help='path to reference image directory')
    parser.add_argument('-s', '--save_name', type=str, default=None, help='model results save name, if None saves as model name')
    parser.add_argument('-c', '--cfg', type=str, default=None,
                        help='path to config file, if None will automatically take it from the model directory')
    parser.add_argument('--dataset', type=str, default='../datasets/textured_rope_val', help='path to dataset directory')
    parser.add_argument('-o', '--output', type=str, default='../control_results', help='path to output directory')
    parser.add_argument('-w', '--workers', type=int, default=6, help='num workers for DataLoader')

    parser.add_argument('--dist_func', type=str, default="mse_similarity", help='distance function type')
    parser.add_argument('--calc_iou', default=False, action="store_true", help='if True will calculate IOU')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reload config and model
    assert os.path.exists(args.model_path), f"can't find model file in: {args.model_path}"
    base_path, checkpoint = os.path.split(args.model_path)
    if args.cfg is None:
        args.cfg = os.path.join(base_path, "config.yml")
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # init model and optimizer
    config["batch_size"] = 1024
    model, _, _ = load_from_checkpoint(path=args.model_path, config=config, device=device)
    loader = get_dataloader(args.dataset, -1, -1, config, args.workers, load_seg_masks=args.calc_iou)

    images_path_list = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('png') or f.endswith("jpeg") or f.endswith("jpg")]
    assert len(images_path_list), "Error, no images where loaded"
    os.makedirs(args.output, exist_ok=True)
    if args.save_name is None:
        _, model_name = os.path.split(base_path)
        save_path = os.path.join(args.output, f"{model_name}_epoch_{checkpoint.split('_')[-1]}_{args.dist_func}")
    else:
        save_path = os.path.join(args.output, args.save_name + f"_{args.dist_func}")
    os.makedirs(save_path + "_results", exist_ok=True)
    find_nearest_neighbours_from_list(images_path_list,
                                      encoder=model.encoder,
                                      loader=loader,
                                      device=device,
                                      project_features=True,
                                      dist_func=args.dist_func,
                                      save_path=save_path + "_real",
                                      dump_raw_results_path=None,
                                      compare_segmentation_mask=args.calc_iou)
    find_nearest_neighbours(images_path_list,
                            encoder=model.encoder,
                            loader=loader,
                            device=device,
                            project_features=True,
                            similarity_func=args.dist_func,
                            save_path=save_path,
                            dump_raw_results_path=save_path + "_results",
                            compare_segmentation_mask=args.calc_iou)