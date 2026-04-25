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
        
        if "two views" in lines[0]:
            # single query for both cams => split label into two entries
            for line in lines:
                line = json.loads(line)
                
                assert "\n" in line["label"], f"Expected newline in eval dataset label splitting the two views : {repr(line['label'])}"
                
                pred = {"prompt": line["prompt"], "label": {"static": line["label"].split("\n")[0], "gripper": line["label"].split("\n")[1]}}
                eval_dataset.append(pred)
        else:
            # separate queries for both cams => pair every two lines
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
    
    traj_points_pos_scores = []
    traj_actions_pos_scores = []
    traj_actions_type_scores = []
    traj_total_scores = []
    for i, (eval_ds_entry, static_img_path, gripper_img_path) in tqdm(enumerate(zip(eval_dataset, dataset_val["static"], dataset_val["gripper"])), total=len(dataset_val["static"]), desc="Evaluating VLM outputs"):
        static_img = cv2.cvtColor(cv2.imread(static_img_path), cv2.COLOR_BGR2RGB)
        assert static_img.shape[0] == static_img.shape[1]

        gripper_img = cv2.cvtColor(cv2.imread(gripper_img_path), cv2.COLOR_BGR2RGB)
        assert gripper_img.shape[0] == gripper_img.shape[1]

        task = static_img_path.split("_static.png")[0].split("validation/")[1][5:]
        task_text = eval_ds_entry["prompt"].split("<prompt>")[1].split("</prompt>")[0]
        
        vlm_output_predict = query_vlm(static_img, gripper_img, vlm_client, task_text)
        vlm_output_label = eval_ds_entry["label"]
        
        points_pos_score, points_pred, points_label = get_alignment_of_gripper_points(vlm_output_predict, vlm_output_label)
        actions_pos_score, actions_type_score, actions_pred, actions_label = get_alignment_of_gripper_actions(vlm_output_predict, vlm_output_label)
        total_score = (points_pos_score + actions_pos_score + actions_type_score) / 3
        
        traj_points_pos_scores.append(points_pos_score)
        traj_actions_pos_scores.append(actions_pos_score)
        traj_actions_type_scores.append(actions_type_score)
        traj_total_scores.append(total_score)
        
        if draw_trajectories:
            build_and_save_trajectory_images(output_dir, static_img, points_pred, actions_pred, points_label, actions_label, task, total_score, "static", output_nr=i)
            build_and_save_trajectory_images(output_dir, gripper_img, points_pred, actions_pred, points_label, actions_label, task, total_score, "gripper", output_nr=i)
    
    print_results(output_dir, traj_points_pos_scores["static"], traj_actions_pos_scores["static"], traj_actions_type_scores["static"], traj_total_scores["static"],
                  traj_points_pos_scores["gripper"], traj_actions_pos_scores["gripper"], traj_actions_type_scores["gripper"], traj_total_scores["gripper"])


if __name__ == '__main__':
    main(eval_dataset_path="/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/2026_02_19-unfrozen_vision_tower-longer_training/generated_predictions.jsonl",
         val_imgs_dir="/DATA/troth/iTRAP/data/calvin_vlm_dataset/2025-11-17_21-08-09_qwen3_both-cams_single-query_static-traj-only/validation", # old but only for imgs & last one where imgs created
         draw_trajectories=False) # TODO: draw_trajectories=True currently fails bc of missing view & projection matrices for drawing trajectories onto images (see other TODO)
