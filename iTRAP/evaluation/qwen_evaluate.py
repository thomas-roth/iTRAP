from datetime import datetime
import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
from tqdm import tqdm
from dtw import *

sys.path.append(str(Path(__file__).absolute().parents[2]))
from iTRAP.models.Qwen3_VL.utils import extract_gripper_points_and_actions, draw_trajectory_onto_image


def parse_vlm_outputs(file_path: str) -> list:
    vlm_outputs = []
    with open(file_path, 'r') as file:
        for task in file:
            task = json.loads(task)
            vlm_outputs.append(task)
    
    return vlm_outputs


def get_alignment_of_gripper_points(pred, label):
    gripper_points_pred, _ = extract_gripper_points_and_actions(pred)
    gripper_points_label, _ = extract_gripper_points_and_actions(label)

    # filter edge cases with no gripper points in pred and/or label (can happen for gripper cam if traj completely out of bounds)
    if len(gripper_points_pred) == 0 and len(gripper_points_label) == 0:
        # no gripper points in pred or label => no error
        return 100, gripper_points_pred, gripper_points_label
    elif len(gripper_points_label) == 0:
        # no gripper points only in pred => max error
        return 0, gripper_points_pred, gripper_points_label
    elif len(gripper_points_pred) == 0:
        # no gripper points only in label => max error
        return 0, gripper_points_pred, gripper_points_label

    dtw_alignment = dtw(np.array(gripper_points_pred), np.array(gripper_points_label), keep_internals=True, dist_method=lambda p, l: np.linalg.norm(p - l))
    gripper_points_dist = dtw_alignment.normalizedDistance  # cumulative 3D world coord distance normalized by traj lengths

    gripper_points_score = transform_dist_to_similarity_score(gripper_points_dist)

    return gripper_points_score, gripper_points_pred, gripper_points_label


def get_alignment_of_gripper_actions(pred, label):
    _, gripper_actions_pred, _ = extract_gripper_points_and_actions(pred)
    _, gripper_actions_label, _ = extract_gripper_points_and_actions(label)

    # filter edge cases with no gripper actions in pred and/or label (can happen for gripper cam if traj completely out of bounds)
    if len(gripper_actions_pred) == 0 and len(gripper_actions_label) == 0:
        # no gripper actions in pred or label => no error
        return 100, 100, gripper_actions_pred, gripper_actions_label
    elif len(gripper_actions_label) == 0:
        # no gripper actions only in pred => max error
        return 0, 0, gripper_actions_pred, gripper_actions_label
    elif len(gripper_actions_pred) == 0:
        # no gripper actions only in label => max error
        return 0, 0, gripper_actions_pred, gripper_actions_label

    gripper_actions_cum_dist = 0
    gripper_actions_same_action = []
    for gripper_action_pred, gripper_action_label in zip(gripper_actions_pred, gripper_actions_label):
        # if lengths of gripper actions pred & label differ => zip automatically only takes elements until length of shorter list
        gripper_action_dist = np.linalg.norm(np.array(gripper_action_pred[0]) - np.array(gripper_action_label[0]))
        gripper_actions_cum_dist += gripper_action_dist

        gripper_actions_same_action.append(gripper_action_pred[1] == gripper_action_label[1])

    gripper_actions_cum_dist = gripper_actions_cum_dist / (len(gripper_actions_pred) + len(gripper_actions_label)) # normalize the same as for normalizedDistance of DTW
    gripper_actions_pos_score = transform_dist_to_similarity_score(gripper_actions_cum_dist)

    gripper_actions_type_score = np.mean(gripper_actions_same_action) * 100

    return gripper_actions_pos_score, gripper_actions_type_score, gripper_actions_pred, gripper_actions_label


def transform_dist_to_similarity_score(dist, max_dist=1.0):
    # TODO: figure out good max_dist value by testing workspace width in calvin env
    similarity_score = max(0, 100 * (1 - (dist / max_dist))) # clamped to [0, 100]
    return similarity_score


def build_and_save_trajectory_images(output_dir, img_arr, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label, task, total_score, cam, output_nr):
    env = None # TODO: figure out (store view & projection matrices of both cams in dataset?)
    traj_img_pred = draw_trajectory_onto_image(img_arr, gripper_points_pred, gripper_actions_pred, env, traj_color="green")
    traj_img_pred_label = draw_trajectory_onto_image(traj_img_pred, gripper_points_label, gripper_actions_label, env, traj_color="red")

    traj_img_pred_label_pil = Image.fromarray(traj_img_pred_label).convert("RGBA")
    
    # overlay simple legend
    img_size = traj_img_pred_label_pil.size[0]
    assert img_size == traj_img_pred_label_pil.size[1], "Image not square"
    alpha = 180
    overlay = Image.new('RGBA', traj_img_pred_label_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([3, img_size - 33, 88, img_size - 3], fill=(255, 255, 255, alpha), outline=(0, 0, 0, alpha))
    draw.ellipse([6, img_size - 30, 16, img_size - 20], fill=(0, 255, 0, alpha))
    draw.text((20, img_size - 31), "Prediction", fill=(0, 0, 0, alpha))
    draw.ellipse([6, img_size - 16, 16, img_size - 6], fill=(255, 0, 0, alpha))
    draw.text((20, img_size - 17), "Ground Truth", fill=(0, 0, 0, alpha))
    traj_img_pred_label_pil = Image.alpha_composite(traj_img_pred_label_pil, overlay).convert("RGB")

    os.makedirs(f"{output_dir}/traj_imgs/{cam}", exist_ok=True)
    traj_img_pred_label_pil.save(f"{output_dir}/traj_imgs/{cam}/total-score-{round(total_score, 2)}_index-{output_nr:04d}_{task.replace('_', '-')}.png")


def print_results(output_dir, points_pos_scores, actions_scores_pos, actions_scores_type, total_scores):
    avg_points_pos_score = np.mean(points_pos_scores)
    avg_actions_score_pos = np.mean(actions_scores_pos)
    avg_actions_score_type = np.mean(actions_scores_type)
    avg_total_score = np.mean(total_scores)

    results = f"Average gripper points position score: {round(avg_points_pos_score, 2)} (average 0-100 score for 3D world coord between pred & label gripper points normalized to traj lengths)\n" \
        + f"Average gripper actions position score: {round(avg_actions_score_pos, 2)} (average 0-100 score for 3D world coord between pred & label gripper actions normalized to traj lengths)\n" \
        + f"Average gripper actions type score: {round(avg_actions_score_type, 2)} (average percentage of matching action types between pred & label gripper actions)\n" \
        + f"Average total score: {round(avg_total_score, 2)} (unweighted average of the three sub-scores above)\n"

    print(results)

    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(results)


def main(gen_preds_path: str, val_imgs_dir: str, draw_trajectories=False):
    output_dir = Path(__file__).parents[2] / "outputs" / "qwen" / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    os.makedirs(output_dir, exist_ok=False)

    vlm_outputs = parse_vlm_outputs(gen_preds_path)

    if "image" in vlm_outputs[0]:
        first_static_img = Image.open(vlm_outputs[0]['image'][0])
        first_gripper_img = Image.open(vlm_outputs[0]['image'][1])
    else:
        dataset_val = {}
        dataset_val["static"] = sorted([
            os.path.join(val_imgs_dir, f)
            for f in os.listdir(val_imgs_dir)
            if f.endswith(('.png', '.jpg', '.jpeg')) and "static" in f
        ])
        dataset_val["gripper"] = sorted([
            os.path.join(val_imgs_dir, f)
            for f in os.listdir(val_imgs_dir)
            if f.endswith(('.png', '.jpg', '.jpeg')) and "gripper" in f
        ])
        first_static_img = Image.open(dataset_val["static"][0])
        first_gripper_img = Image.open(dataset_val["gripper"][0])
    
    assert first_static_img.size[0] == first_static_img.size[1]
    assert first_gripper_img.size[0] == first_gripper_img.size[1]
    img_sizes = {}
    img_sizes["static"] = first_static_img.size[0]
    img_sizes["gripper"] = first_gripper_img.size[0]
    
    traj_points_pos_scores = []
    traj_actions_pos_scores = []
    traj_actions_type_scores = []
    traj_total_scores = []
    img_counter = 0
    for vlm_output in tqdm(vlm_outputs, total=len(vlm_outputs), desc="Evaluating VLM outputs"):        
        traj_points_pos_score, traj_points_pred, traj_points_label = get_alignment_of_gripper_points(vlm_output["predict"], vlm_output["label"])
        traj_actions_pos_score, traj_actions_type_score, traj_actions_pred, traj_actions_label = get_alignment_of_gripper_actions(vlm_output["predict"], vlm_output["label"])
        traj_total_score = (traj_points_pos_score + traj_actions_pos_score + traj_actions_type_score) / 3

        if draw_trajectories:
            if "image" in vlm_output:
                for cam_id in range(0, 2):
                    img_arr = cv2.cvtColor(cv2.imread(vlm_output['image'][cam_id]), cv2.COLOR_BGR2RGB)
            else:
                for cam in ["static", "gripper"]:
                    img_arr = cv2.cvtColor(cv2.imread(dataset_val[cam][img_counter]), cv2.COLOR_BGR2RGB)
                    task = vlm_output["prompt"].split("<prompt>")[1].split("</prompt>")[0].replace(" ", "_")
                    build_and_save_trajectory_images(output_dir, img_arr, traj_points_pred, traj_actions_pred, traj_points_label, traj_actions_label,
                                                    task, traj_total_score, cam, output_nr=img_counter)

        traj_points_pos_scores.append(traj_points_pos_score)
        traj_actions_pos_scores.append(traj_actions_pos_score)
        traj_actions_type_scores.append(traj_actions_type_score)
        traj_total_scores.append(traj_total_score)
        img_counter += 1
    
    print_results(output_dir, traj_points_pos_scores, traj_actions_pos_scores, traj_actions_type_scores, traj_total_scores)


if __name__ == '__main__':
    main(gen_preds_path="/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/2026_02_19-unfrozen_vision_tower-longer_training/generated_predictions.jsonl",
         val_imgs_dir="/DATA/troth/iTRAP/data/calvin_vlm_dataset/2025-11-17_21-08-09_qwen3_both-cams_single-query_static-traj-only/validation", # old but only for imgs & last one where imgs created
         draw_trajectories=False) # TODO: draw_trajectories=True currently fails bc of missing view & projection matrices for drawing trajectories onto images (see other TODO)
