import argparse
import os

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from data.utils import process_action, load_label, load_frame_from_video, add_arrow_to_image, load_video
from evaluation.nearest_neighbours import generate_dataset_features_matrix, calc_nearest_neighbours
from model.encoder import Encoder
from train import load_from_checkpoint, get_dataloader
from torch.utils import data


def visualize_trajectory(results, save_path=None, title=None):
    """plots nearest neighbours images"""

    n_cols = len(results)
    n_rows = 4
    fig = plt.figure(figsize=(int(12*n_cols/4.), 12))
    if title is not None:
        fig.suptitle(title, fontsize=60)
    for col, info in enumerate(results):
        (ref_image, ref_image_nn, ref_next_image, ref_next_image_nn) = info

        # state
        fig.add_subplot(n_rows, n_cols, col+1).set_title(f"Ref", fontsize=20)
        plt.imshow(ref_image)
        plt.axis('off')

        # state nn
        fig.add_subplot(n_rows, n_cols, n_cols + col+1).set_title(f"Ref NN", fontsize=20)
        plt.imshow(ref_image_nn)
        plt.axis('off')

        # next_state
        fig.add_subplot(n_rows, n_cols, 2*n_cols+col+1).set_title(f"Next Ref", fontsize=20)
        plt.imshow(ref_next_image)
        plt.axis('off')

        # next_state nn
        fig.add_subplot(n_rows, n_cols, 3*n_cols + col+1).set_title(f"Next Ref NN", fontsize=20)
        plt.imshow(ref_next_image_nn)
        plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path + ".jpg")
    # plt.show()


def predict_trajectory(input_path: str,
                       encoder: Encoder,
                       fm: torch.nn.Module,
                       loader: data.DataLoader,
                       device,
                       save_path,
                       dist_func="dot_product_similarity",
                       predict_full_trajectory=False):

    # load reference trajectory video
    ref_video = load_video(input_path)
    ref_seg_mask = load_video(input_path[:-4] + config["seg_masks_files_sfx"])
    actions = process_action(load_label(input_path[:-4] + config["action_files_sfx"]))
    results = []

    with torch.no_grad():
        # generate features_mat from data
        features_mat, info = generate_dataset_features_matrix(encoder, loader, device=device, )
        (video_path, frame_ind, seg_masks) = info

        for i in range(len(ref_video)-1):
            ref_image = ref_video[i]
            ref_next_image = ref_video[i+1]
            frame = loader.dataset.image_transform(ref_image)
            next_frame = loader.dataset.image_transform(ref_next_image)
            if i > 0 and predict_full_trajectory:
                # use forward model to predict all the trajectory
                z = torch.clone(z_next_hat)
            else:
                z = encoder(frame.float().to(device).unsqueeze(0))
            z_next = encoder(next_frame.float().to(device).unsqueeze(0))
            action = actions[i].unsqueeze(0).to(device).float()
            z_next_hat = fm(z, action)

            # fin nearest neighbour
            score, ind = calc_nearest_neighbours(z, features_mat, k_neighbours=1, similarity_func=dist_func)
            ref_image_nn = load_frame_from_video(video_path[ind[0]], frame_ind[ind[0]])

            next_score, next_ind = calc_nearest_neighbours(z_next_hat, features_mat, k_neighbours=1, similarity_func=dist_func)
            ref_next_image_nn = load_frame_from_video(video_path[next_ind[0]], frame_ind[next_ind[0]])

            action_ = actions[i].cpu().numpy()
            ref_image = add_arrow_to_image(np.copy(ref_image), action_)
            ref_image_nn = add_arrow_to_image(np.copy(ref_image_nn), action_)
            ref_next_image = add_arrow_to_image(np.copy(ref_next_image), action_)
            ref_next_image_nn = add_arrow_to_image(np.copy(ref_next_image_nn), action_)
            results.append((ref_image, ref_image_nn, ref_next_image, ref_next_image_nn))

    visualize_trajectory(results, save_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to trained model')
    parser.add_argument('-i', '--input', type=str, required=True, help='path to reference video')
    parser.add_argument('-s', '--save_name', type=str, default=None, help='model results save name, if None saves as model name')
    parser.add_argument('-c', '--cfg', type=str, default=None,
                        help='path to config file, if None will automatically take it from the model directory')
    parser.add_argument('--dataset', type=str, default='../datasets/textured_rope_val', help='path to dataset directory')
    parser.add_argument('-o', '--output', type=str, default='../control_results', help='path to output directory')
    parser.add_argument('-w', '--workers', type=int, default=5, help='num workers for DataLoader')

    parser.add_argument('--dist_func', type=str, default="dot_product_similarity", help='distance function type')
    parser.add_argument('-p', '--predict_full_trajectory', action='store_true',
                        required=False, help='use forward model to predict full trajectory')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert os.path.exists(args.input), f"input file does not exists: {args.input}"

    # reload config and model
    assert os.path.exists(args.model_path), f"can't find model file in: {args.model_path}"
    base_path, checkpoint = os.path.split(args.model_path)
    if args.cfg is None:
        args.cfg = os.path.join(base_path, "config.yml")
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # init model and optimizer
    model, _, _ = load_from_checkpoint(path=args.model_path, config=config, device=device)
    loader = get_dataloader(args.dataset, -1, -1, config, args.workers, load_seg_masks=True)

    fm_sfx = "_fm" if args.predict_full_trajectory else ""
    if args.save_name is None:
        _, model_name = os.path.split(base_path)
        save_path = os.path.join(args.output, f"{model_name}_epoch_{checkpoint.split('_')[-1]}_{args.dist_func}_trajectory{fm_sfx}")
    else:
        save_path = os.path.join(args.output, args.save_name + f"_{args.dist_func}_trajectory{fm_sfx}")

    predict_trajectory(args.input,
                       encoder=model.encoder,
                       fm=model.forward_model,
                       loader=loader,
                       device=device,
                       dist_func=args.dist_func,
                       predict_full_trajectory=args.predict_full_trajectory,
                       save_path=save_path)