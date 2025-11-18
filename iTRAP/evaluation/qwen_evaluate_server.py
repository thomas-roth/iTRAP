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
        eval_dataset = [json.loads(line) for line in file]

    dataset_val_static_imgs = sorted([
        os.path.join(val_imgs_dir, f)
        for f in os.listdir(val_imgs_dir)
        if f.endswith(('.png', '.jpg', '.jpeg')) and "static" in f
    ])
    
    gripper_points_pos_scores = []
    gripper_actions_pos_scores = []
    gripper_actions_type_scores = []
    total_scores = []
    for i, (eval_ds_entry, static_img_path) in tqdm(enumerate(zip(eval_dataset, dataset_val_static_imgs)), total=len(dataset_val_static_imgs), desc="Evaluating VLM outputs"):
        static_img_arr = cv2.cvtColor(cv2.imread(static_img_path), cv2.COLOR_BGR2RGB)
        assert static_img_arr.shape[0] == static_img_arr.shape[1]
        static_img_size = static_img_arr.shape[0]

        gripper_img_arr = cv2.cvtColor(cv2.imread(static_img_path.replace("static", "gripper")), cv2.COLOR_BGR2RGB)
        assert gripper_img_arr.shape[0] == gripper_img_arr.shape[1]

        task = static_img_path.split("_static.png")[0].split("validation/")[1][5:]
        task_text = eval_ds_entry["prompt"].split("<prompt>")[1].split("</prompt>")[0]
        
        vlm_output_predict = query_vlm(static_img_arr, gripper_img_arr, vlm_client, task_text)

        vlm_output_label = eval_ds_entry["label"]

        gripper_points_pos_score, gripper_points_pred, gripper_points_label = get_alignment_of_gripper_points(vlm_output_predict, vlm_output_label, static_img_size)
        gripper_points_pos_scores.append(gripper_points_pos_score)

        gripper_actions_pos_score, gripper_actions_type_score, gripper_actions_pred, gripper_actions_label = get_alignment_of_gripper_actions(vlm_output_predict, vlm_output_label, static_img_size)
        gripper_actions_pos_scores.append(gripper_actions_pos_score)
        gripper_actions_type_scores.append(gripper_actions_type_score)

        total_score = (gripper_points_pos_score + gripper_actions_pos_score + gripper_actions_type_score) / 3
        total_scores.append(total_score)

        if draw_trajectories:
            build_and_save_trajectory_images(output_dir, static_img_arr, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label,
                                             task, total_score, output_nr=i)
    
    print_results(output_dir, gripper_points_pos_scores, gripper_actions_pos_scores, gripper_actions_type_scores, total_scores)


if __name__ == '__main__':
    main(eval_dataset_path="/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/qwen3_vl_8b-calvin_abc-2025_11_17-both_cams/merged_best/generated_predictions.jsonl",
         val_imgs_dir="/home/troth/data/iTRAP-flower/calvin_vlm_dataset/2025-11-17_21-08-09_qwen3_both-cams/validation",
         draw_trajectories=True)
