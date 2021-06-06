import argparse
import os

import torch
import yaml
import numpy as np

from data.utils import load_single_image, load_seg_masks_from_video, load_frame_from_video, image_transform, \
    add_arrow_to_image, process_action, load_label, load_video, seg_mask_transform
from evaluation.IOU import calc_mean_iou
from evaluation.nearest_neighbours import generate_dataset_features_matrix, calc_nearest_neighbours, \
    visualize_nearest_neighbours
from model.encoder import Encoder
from train import load_from_checkpoint, get_dataloader
from evaluation.predict_trajectory import visualize_trajectory


def predict_trajectory(features_mat,
                       info: tuple,
                       input_path: str,
                       encoder: Encoder,
                       fm: torch.nn.Module,
                       device: str,
                       save_path: str,
                       similarity_func="dot_product_similarity",
                       predict_full_trajectory=False):

    # load reference trajectory video
    ref_video = load_video(input_path)
    ref_seg_mask = load_video(input_path[:-4] + config["seg_masks_files_sfx"])
    actions = process_action(load_label(input_path[:-4] + config["action_files_sfx"]))
    results = []

    with torch.no_grad():
        # generate features_mat from data
        (video_path, frame_ind, seg_masks) = info

        for i in range(len(ref_video)-1):
            ref_image = ref_video[i]
            ref_next_image = ref_video[i+1]
            frame = image_transform(ref_image, resize_to=encoder.image_size)
            next_frame = image_transform(ref_next_image, resize_to=encoder.image_size)
            if i > 0 and predict_full_trajectory:
                # use forward model to predict all the trajectory
                z = torch.clone(z_next_hat)
            else:
                z = encoder(frame.float().to(device).unsqueeze(0))
            z_next = encoder(next_frame.float().to(device).unsqueeze(0))
            action = actions[i].unsqueeze(0).to(device).float()
            z_next_hat = fm(z, action)

            # fin nearest neighbour
            score, ind = calc_nearest_neighbours(z, features_mat, k_neighbours=1, similarity_func=similarity_func)
            ref_image_nn = load_frame_from_video(video_path[ind[0]], frame_ind[ind[0]])

            next_score, next_ind = calc_nearest_neighbours(z_next_hat, features_mat, k_neighbours=1, similarity_func=similarity_func)
            ref_next_image_nn = load_frame_from_video(video_path[next_ind[0]], frame_ind[next_ind[0]])

            action_ = actions[i].cpu().numpy()
            ref_image = add_arrow_to_image(np.copy(ref_image), action_)
            ref_image_nn = add_arrow_to_image(np.copy(ref_image_nn), action_)
            ref_next_image = add_arrow_to_image(np.copy(ref_next_image), action_)
            ref_next_image_nn = add_arrow_to_image(np.copy(ref_next_image_nn), action_)
            results.append((ref_image, ref_image_nn, ref_next_image, ref_next_image_nn))

    print(f"Done computing trajectory with {similarity_func}, Full forward model: {predict_full_trajectory}")
    visualize_trajectory(results, save_path)


def find_nearest_neighbours(features_mat, info, images_path_list, encoder: Encoder, device,
                            save_path, project_features=True, dist_func="dot_product_similarity"):
    with torch.no_grad():
        # generate features_mat from data
        (video_path, frame_ind, seg_masks) = info

        # for each observation find nearest neighbours
        results = []
        iou_scores = []
        for _, img_path in enumerate(images_path_list):
            ref_image = load_single_image(img_path)
            obs = image_transform(ref_image, resize_to=encoder.image_size)
            ref_seg_mask = load_seg_masks_from_video(img_path[:-4] + config["seg_masks_files_sfx"])
            ref_seg_mask = seg_mask_transform(ref_seg_mask)
            features_vec = encoder(obs.float().to(device).unsqueeze(0))
            top_scores, top_scores_ind = calc_nearest_neighbours(features_vec,
                                                                 features_mat,
                                                                 k_neighbours=5,
                                                                 similarity_func=dist_func)
            top_img = []
            for top_ind in top_scores_ind:
                top_img.append(load_frame_from_video(video_path[top_ind], frame_ind[top_ind]))
            mean_iou, iou = calc_mean_iou(ref_seg_mask, [seg_masks[top_ind] for top_ind in top_scores_ind])
            results.append((ref_image, top_scores, top_img, mean_iou, iou))
            iou_scores.append(mean_iou)

    print(f"Mean IOU score {dist_func}: {np.mean(iou_scores)}")
    visualize_nearest_neighbours(results, save_path=save_path, title=f"IOU: {np.mean(iou_scores)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, default=None,
                        required=False, action='append', help='path to trained model')
    parser.add_argument('--input_images', type=str, required=True, help='path to reference image directory')
    parser.add_argument('--input_video', type=str, required=True, help='path to reference video file')
    parser.add_argument('-s', '--save_name', type=str, default=None, help='model results save name, if None saves as model name')
    parser.add_argument('--dataset', type=str, default='../datasets/textured_rope_val', help='path to dataset directory')
    parser.add_argument('-o', '--output', type=str, default='../control_results', help='path to output directory')
    parser.add_argument('-w', '--workers', type=int, default=5, help='num workers for DataLoader')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in args.models:
        args.model_path = model_name

        # reload config and model
        assert os.path.exists(args.model_path), f"can't find model file in: {args.model_path}"
        base_path, checkpoint = os.path.split(args.model_path)
        args.cfg = os.path.join(base_path, "config.yml")
        with open(args.cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # init model and optimizer
        model, _, _ = load_from_checkpoint(path=args.model_path, config=config, device=device)
        loader = get_dataloader(args.dataset, -1, -1, config, args.workers, load_seg_masks=True)

        images_path_list = [os.path.join(args.input_images, f) for f in os.listdir(args.input_images) if f.endswith('png')]
        if args.save_name is None:
            _, model_name = os.path.split(base_path)
            save_path = os.path.join(args.output, f"{model_name}_epoch_{checkpoint.split('_')[-1]}")
        else:
            save_path = os.path.join(args.output, args.save_name)
        os.makedirs(save_path, exist_ok=True)

        # generate features_mat from data
        with torch.no_grad():
            features_mat, info = generate_dataset_features_matrix(model.encoder, loader, device=device, project_features=True)
            (video_path, frame_ind, seg_masks) = info

        for similarity_func in ["dot_product_similarity", "mse_similarity", "cosine_similarity"]:
            find_nearest_neighbours(features_mat, info,
                                    images_path_list=images_path_list,
                                    encoder=model.encoder,
                                    device=device,
                                    save_path=os.path.join(save_path, f"nn_{similarity_func}"),
                                    project_features=True,
                                    dist_func=similarity_func)

            predict_trajectory(features_mat, info,
                               input_path=args.input_video,
                               encoder=model.encoder,
                               fm=model.forward_model,
                               device=device,
                               similarity_func=similarity_func,
                               predict_full_trajectory=False,
                               save_path=os.path.join(save_path, f"trajectory_{similarity_func}"))

            predict_trajectory(features_mat, info,
                               input_path=args.input_video,
                               encoder=model.encoder,
                               fm=model.forward_model,
                               device=device,
                               similarity_func=similarity_func,
                               predict_full_trajectory=True,
                               save_path=os.path.join(save_path, f"full_trajectory_{similarity_func}"))