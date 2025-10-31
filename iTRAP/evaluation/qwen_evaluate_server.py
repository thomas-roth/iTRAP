from datetime import datetime
import os
from pathlib import Path
import sys
import cv2
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).absolute().parents[2]))
from iTRAP.evaluation.qwen_evaluate import get_alignment_of_gripper_points, get_alignment_of_gripper_actions, build_and_save_trajectory_images, print_results
from iTRAP.evaluation.utils import setup_vlm_client, query_vlm



def main(eval_dataset_path: str, val_imgs_dir: str, draw_trajectories=False):
    output_dir = Path(__file__).parents[2] / "outputs" / "qwen" / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    os.makedirs(output_dir, exist_ok=False)

    vlm_client = setup_vlm_client()

    dataset_val_imgs = sorted([
        os.path.join(val_imgs_dir, f)
        for f in os.listdir(val_imgs_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    with open(eval_dataset_path, "r") as file:
        eval_dataset = [json.loads(line) for line in file]
    
    gripper_points_pos_scores = []
    gripper_actions_pos_scores = []
    gripper_actions_type_scores = []
    total_scores = []
    for i, (eval_ds_entry, img_path) in tqdm(enumerate(zip(eval_dataset, dataset_val_imgs)), total=len(dataset_val_imgs), desc="Evaluating VLM outputs"):
        img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        assert img_arr.shape[0] == img_arr.shape[1]
        img_size = img_arr.shape[0]

        task = img_path.split("_static.png")[0].split("validation/")[1][5:]

        vlm_output_predict = query_vlm(img_arr, vlm_client, task)

        vlm_output_label = eval_ds_entry["label"]

        gripper_points_pos_score, gripper_points_pred, gripper_points_label = get_alignment_of_gripper_points(vlm_output_predict, vlm_output_label, img_size)
        gripper_points_pos_scores.append(gripper_points_pos_score)

        gripper_actions_pos_score, gripper_actions_type_score, gripper_actions_pred, gripper_actions_label = get_alignment_of_gripper_actions(vlm_output_predict, vlm_output_label, img_size)
        gripper_actions_pos_scores.append(gripper_actions_pos_score)
        gripper_actions_type_scores.append(gripper_actions_type_score)

        total_score = (gripper_points_pos_score + gripper_actions_pos_score + gripper_actions_type_score) / 3
        total_scores.append(total_score)

        if draw_trajectories:
            build_and_save_trajectory_images(output_dir, img_arr, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label,
                                             task, total_score, output_nr=i)
    
    print_results(output_dir, gripper_points_pos_scores, gripper_actions_pos_scores, gripper_actions_type_scores, total_scores)


if __name__ == '__main__':
    main(eval_dataset_path="/home/troth/data/iTRAP-flower/vlm_val_predictions/qwen3_vl/generated_predictions.jsonl",
         val_imgs_dir="/home/troth/data/iTRAP-flower/calvin_vlm_dataset/2025-10-20_16-24-18_qwen3_abc/validation",
         draw_trajectories=True)
