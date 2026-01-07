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

    dataset_val = {}
    for cam in ["static", "gripper"]:
        dataset_val[cam] = sorted([
            os.path.join(val_imgs_dir, f)
            for f in os.listdir(val_imgs_dir)
            if f.endswith(('.png', '.jpg', '.jpeg')) and cam in f
        ])
    
    img_arrs = {}
    img_sizes = {}
    traj_points_preds = {}
    traj_points_labels = {}
    traj_actions_preds = {}
    traj_actions_labels = {}
    traj_points_pos_scores = {"static": [], "gripper": []}
    traj_actions_pos_scores = {"static": [], "gripper": []}
    traj_actions_type_scores = {"static": [], "gripper": []}
    traj_total_scores = {"static": [], "gripper": []}
    for i, (eval_ds_entry, static_img_path, gripper_img_path) in tqdm(enumerate(zip(eval_dataset, dataset_val["static"], dataset_val["gripper"])), total=len(dataset_val["static"]), desc="Evaluating VLM outputs"):
        img_arrs["static"] = cv2.cvtColor(cv2.imread(static_img_path), cv2.COLOR_BGR2RGB)
        assert img_arrs["static"].shape[0] == img_arrs["static"].shape[1]
        img_sizes["static"] = img_arrs["static"].shape[0]

        img_arrs["gripper"] = cv2.cvtColor(cv2.imread(gripper_img_path), cv2.COLOR_BGR2RGB)
        assert img_arrs["gripper"].shape[0] == img_arrs["gripper"].shape[1]
        img_sizes["gripper"] = img_arrs["gripper"].shape[0]

        task = static_img_path.split("_static.png")[0].split("validation/")[1][5:]
        task_text = eval_ds_entry["prompt"].split("<prompt>")[1].split("</prompt>")[0]
        
        vlm_outputs_predict = query_vlm(img_arrs["static"], img_arrs["gripper"], vlm_client, task_text)

        vlm_outputs_label = eval_ds_entry["label"]
        
        for cam in ["static", "gripper"]:
            points_pos_score, points_pred, points_label = get_alignment_of_gripper_points(vlm_outputs_predict[cam], vlm_outputs_label[cam], img_sizes[cam])
            actions_pos_score, actions_type_score, actions_pred, actions_label = get_alignment_of_gripper_actions(vlm_outputs_predict[cam], vlm_outputs_label[cam], img_sizes[cam])
            total_score = (points_pos_score + actions_pos_score + actions_type_score) / 3
            
            traj_points_preds[cam] = points_pred
            traj_points_labels[cam] = points_label
            traj_actions_preds[cam] = actions_pred
            traj_actions_labels[cam] = actions_label
            
            traj_points_pos_scores[cam].append(points_pos_score)
            traj_actions_pos_scores[cam].append(actions_pos_score)
            traj_actions_type_scores[cam].append(actions_type_score)
            traj_total_scores[cam].append(total_score)
            
            if draw_trajectories:
                build_and_save_trajectory_images(output_dir, img_arrs[cam], points_pred, actions_pred, points_label, actions_label, task, total_score, cam, output_nr=i)
    
    print_results(output_dir, traj_points_pos_scores["static"], traj_actions_pos_scores["static"], traj_actions_type_scores["static"], traj_total_scores["static"],
                  traj_points_pos_scores["gripper"], traj_actions_pos_scores["gripper"], traj_actions_type_scores["gripper"], traj_total_scores["gripper"])


if __name__ == '__main__':
    main(eval_dataset_path="/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/2025_12_06-both_cams-both_trajs/generated_predictions.jsonl",
         val_imgs_dir="/home/troth/data/iTRAP-flower/calvin_vlm_dataset/2025-11-17_21-08-09_qwen3_both-cams/validation",
         draw_trajectories=True)
