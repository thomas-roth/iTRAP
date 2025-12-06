from datetime import datetime
import os
from pathlib import Path
import sys
import cv2
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).absolute().parents[2]))
from iTRAP.models.Qwen3_VL.utils import setup_vlm_client, query_vlm
from iTRAP.evaluation.qwen_evaluate import get_alignment_of_gripper_points, get_alignment_of_gripper_actions, build_and_save_trajectory_images, print_results



def main(eval_dataset_path: str, val_imgs_dir: str, draw_trajectories=False):
    output_dir = Path(__file__).parents[2] / "outputs" / "qwen" / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    os.makedirs(output_dir, exist_ok=False)

    vlm_client = setup_vlm_client()

    with open(eval_dataset_path, "r") as file:
        eval_dataset = []
        lines = file.readlines()

        assert len(lines) % 2 == 0, "Expected even number of lines in eval dataset (pairs of static and gripper cam data)"
        for i in range(0, len(lines), 2):
            pred = {"prompt": json.loads(lines[i])["prompt"], "label": {"static": json.loads(lines[i])["label"], "gripper": json.loads(lines[i+1])["label"]}}
            eval_dataset.append(pred)

    dataset_val_static_imgs = sorted([
        os.path.join(val_imgs_dir, f)
        for f in os.listdir(val_imgs_dir)
        if f.endswith(('.png', '.jpg', '.jpeg')) and "static" in f
    ])

    dataset_val_gripper_imgs = sorted([
        os.path.join(val_imgs_dir, f)
        for f in os.listdir(val_imgs_dir)
        if f.endswith(('.png', '.jpg', '.jpeg')) and "gripper" in f
    ])
    
    static_traj_points_pos_scores = []
    static_traj_actions_pos_scores = []
    static_traj_actions_type_scores = []
    static_traj_total_scores = []
    gripper_traj_points_pos_scores = []
    gripper_traj_actions_pos_scores = []
    gripper_traj_actions_type_scores = []
    gripper_traj_total_scores = []
    for i, (eval_ds_entry, static_img_path, gripper_img_path) in tqdm(enumerate(zip(eval_dataset, dataset_val_static_imgs, dataset_val_gripper_imgs)), total=len(dataset_val_static_imgs), desc="Evaluating VLM outputs"):
        static_img_arr = cv2.cvtColor(cv2.imread(static_img_path), cv2.COLOR_BGR2RGB)
        assert static_img_arr.shape[0] == static_img_arr.shape[1]
        static_img_size = static_img_arr.shape[0]

        gripper_img_arr = cv2.cvtColor(cv2.imread(gripper_img_path), cv2.COLOR_BGR2RGB)
        assert gripper_img_arr.shape[0] == gripper_img_arr.shape[1]
        gripper_img_size = gripper_img_arr.shape[0]

        task = static_img_path.split("_static.png")[0].split("validation/")[1][5:]
        task_text = eval_ds_entry["prompt"].split("<prompt>")[1].split("</prompt>")[0]
        
        vlm_outputs_predict = query_vlm(static_img_arr, gripper_img_arr, vlm_client, task_text)

        vlm_outputs_label = eval_ds_entry["label"]

        static_traj_points_pos_score, static_traj_points_pred, static_traj_points_label = get_alignment_of_gripper_points(vlm_outputs_predict["static"], vlm_outputs_label["static"], static_img_size)
        static_traj_points_pos_scores.append(static_traj_points_pos_score)

        static_traj_actions_pos_score, static_traj_actions_type_score, static_traj_actions_pred, static_traj_actions_label = get_alignment_of_gripper_actions(vlm_outputs_predict["static"], vlm_outputs_label["static"], static_img_size)
        static_traj_actions_pos_scores.append(static_traj_actions_pos_score)
        static_traj_actions_type_scores.append(static_traj_actions_type_score)

        static_traj_total_score = (static_traj_points_pos_score + static_traj_actions_pos_score + static_traj_actions_type_score) / 3
        static_traj_total_scores.append(static_traj_total_score)

        gripper_traj_points_pos_score, gripper_traj_points_pred, gripper_traj_points_label = get_alignment_of_gripper_points(vlm_outputs_predict["gripper"], vlm_outputs_label["gripper"], gripper_img_size)
        gripper_traj_points_pos_scores.append(gripper_traj_points_pos_score)

        gripper_traj_actions_pos_score, gripper_traj_actions_type_score, gripper_traj_actions_pred, gripper_traj_actions_label = get_alignment_of_gripper_actions(vlm_outputs_predict["gripper"], vlm_outputs_label["gripper"], gripper_img_size)
        gripper_traj_actions_pos_scores.append(gripper_traj_actions_pos_score)
        gripper_traj_actions_type_scores.append(gripper_traj_actions_type_score)

        gripper_traj_total_score = (gripper_traj_points_pos_score + gripper_traj_actions_pos_score + gripper_traj_actions_type_score) / 3
        gripper_traj_total_scores.append(gripper_traj_total_score)

        if draw_trajectories:
            build_and_save_trajectory_images(output_dir, static_img_arr, static_traj_points_pred, static_traj_actions_pred, static_traj_points_label, static_traj_actions_label,
                                             task, static_traj_total_score, cam="static", output_nr=i)
            build_and_save_trajectory_images(output_dir, gripper_img_arr, gripper_traj_points_pred, gripper_traj_actions_pred, gripper_traj_points_label, gripper_traj_actions_label,
                                             task, gripper_traj_total_score, cam="gripper", output_nr=i)
    
    print_results(output_dir, static_traj_points_pos_scores, static_traj_actions_pos_scores, static_traj_actions_type_scores, static_traj_total_scores,
                  gripper_traj_points_pos_scores, gripper_traj_actions_pos_scores, gripper_traj_actions_type_scores, gripper_traj_total_scores)


if __name__ == '__main__':
    main(eval_dataset_path="/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/2025_12_06-both_cams-both_trajs/generated_predictions.jsonl",
         val_imgs_dir="/home/troth/data/iTRAP-flower/calvin_vlm_dataset/2025-11-17_21-08-09_qwen3_both-cams/validation",
         draw_trajectories=True)
