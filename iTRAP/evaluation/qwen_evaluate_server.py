from datetime import datetime
import os
from pathlib import Path
from PIL import Image, ImageDraw
from dtw import *
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[2]))
from iTRAP.evaluation.utils import draw_trajectory_onto_image, extract_gripper_points_and_actions, query_vlm, setup_vlm_client


def get_alignment_of_gripper_points(pred, label, img_size):
    gripper_points_pred, _ = extract_gripper_points_and_actions(pred, orig_img_height=img_size, orig_img_width=img_size)
    gripper_points_label, _ = extract_gripper_points_and_actions(label, orig_img_height=img_size, orig_img_width=img_size)

    dtw_alignment = dtw(np.array(gripper_points_pred), np.array(gripper_points_label), keep_internals=True, dist_method=lambda p, l: np.linalg.norm(p - l))
    gripper_points_dist = dtw_alignment.normalizedDistance  # cumulative pixel distance normalized by traj lengths

    gripper_points_score = transform_dist_to_similarity_score(gripper_points_dist, img_size)

    return gripper_points_score, gripper_points_pred, gripper_points_label


def get_alignment_of_gripper_actions(pred, label, img_size):
    _, gripper_actions_pred = extract_gripper_points_and_actions(pred, orig_img_height=img_size, orig_img_width=img_size)
    _, gripper_actions_label = extract_gripper_points_and_actions(label, orig_img_height=img_size, orig_img_width=img_size)

    if len(gripper_actions_pred) == 0 or len(gripper_actions_label) == 0:
        # no gripper actions in traj => no errors # FIXME: not ideal behavior
        return 100, 100, gripper_actions_pred, gripper_actions_label

    gripper_actions_cum_dist = 0
    gripper_actions_same_action = []
    for gripper_action_pred, gripper_action_label in zip(gripper_actions_pred, gripper_actions_label):
        # if lengths of gripper actions pred & label differ => zip automatically only takes elements until length of shorter list
        gripper_action_dist = np.linalg.norm(np.array(gripper_action_pred[0]) - np.array(gripper_action_label[0]))
        gripper_actions_cum_dist += gripper_action_dist

        gripper_actions_same_action.append(gripper_action_pred[1] == gripper_action_label[1])

    gripper_actions_cum_dist = gripper_actions_cum_dist / (len(gripper_actions_pred) + len(gripper_actions_label)) # normalize the same as for normalizedDistance of DTW
    gripper_actions_pos_score = transform_dist_to_similarity_score(gripper_actions_cum_dist, img_size)

    gripper_actions_type_score = np.mean(gripper_actions_same_action) * 100

    return gripper_actions_pos_score, gripper_actions_type_score, gripper_actions_pred, gripper_actions_label


def transform_dist_to_similarity_score(dist, img_size):
    max_dist = img_size / 2  # score 0 if distance half the image size
    similarity_score = max(0, 100 * (1 - (dist / max_dist))) # clamped to [0, 100]
    return similarity_score


def build_and_save_trajectory_images(output_dir, img, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label, task, total_score, output_nr):
    traj_img_pred = draw_trajectory_onto_image(np.array(img), gripper_points_pred, gripper_actions_pred, traj_color="green")
    traj_img_pred_label = draw_trajectory_onto_image(traj_img_pred, gripper_points_label, gripper_actions_label, traj_color="red")

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

    os.makedirs(f"{output_dir}/traj_imgs", exist_ok=True)
    traj_img_pred_label_pil.save(f"{output_dir}/traj_imgs/total-score-{round(total_score, 2)}_index-{output_nr:04d}_{task.replace('_', '-')}.png")


def print_results(output_dir, gripper_points_pos_scores, gripper_actions_scores_pos, gripper_actions_scores_type, total_scores):
    avg_gripper_points_pos_score = np.mean(gripper_points_pos_scores)
    avg_gripper_actions_score_pos = np.mean(gripper_actions_scores_pos)
    avg_gripper_actions_score_type = np.mean(gripper_actions_scores_type)
    avg_total_score = np.mean(total_scores)

    results = f"Average gripper points position score: {round(avg_gripper_points_pos_score, 2)} (average 0-100 score for pixel distance between pred & label gripper points normalized to traj lengths)\n" \
        + f"Average gripper actions position score: {round(avg_gripper_actions_score_pos, 2)} (average 0-100 score for pixel distance between pred & label gripper actions normalized to traj lengths)\n" \
        + f"Average gripper actions type score: {round(avg_gripper_actions_score_type, 2)} (average percentage of matching action types between pred & label gripper actions)\n" \
        + f"Average total score: {round(avg_total_score, 2)} (unweighted average of the three sub-scores above)"

    print(results)

    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(results)


def main(eval_dataset_path: str, draw_trajectories=False):
    output_dir = Path(__file__).parents[2] / "outputs" / "qwen" / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    os.makedirs(output_dir, exist_ok=False)

    vlm_client = setup_vlm_client()

    dataset_val_imgs_dir = "/home/troth/data/iTRAP-flower/calvin_vlm_dataset/2025-10-20_16-24-18_qwen3_abc/validation"
    dataset_val_imgs = sorted([
        os.path.join(dataset_val_imgs_dir, f)
        for f in os.listdir(dataset_val_imgs_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    eval_dataset = open(eval_dataset_path, "r").readlines()
    
    gripper_points_pos_scores = []
    gripper_actions_pos_scores = []
    gripper_actions_type_scores = []
    total_scores = []
    for i, (eval_ds_entry, img_path) in tqdm(enumerate(zip(eval_dataset, dataset_val_imgs)), total=len(dataset_val_imgs), desc="Evaluating VLM outputs"):
        img_path = eval_ds_entry.images[0]
        img = Image.open(img_path)
        assert img.size[0] == img.size[1]
        img_size = img.size[0]

        task = img_path.split("_static.png")[0][5:]
        task_ds = eval_ds_entry.split("<prompt>")[1].split("</prompt>")[0].replace(" ", "_")
        assert task == task_ds, f"Task from image path ({task}) does not match task from dataset entry ({task_ds})"

        vlm_output_predict = query_vlm(img, vlm_client, task)

        vlm_output_label = eval_ds_entry.messages[1].content

        gripper_points_pos_score, gripper_points_pred, gripper_points_label = get_alignment_of_gripper_points(vlm_output_predict, vlm_output_label, img_size)
        gripper_points_pos_scores.append(gripper_points_pos_score)

        gripper_actions_pos_score, gripper_actions_type_score, gripper_actions_pred, gripper_actions_label = get_alignment_of_gripper_actions(vlm_output_predict, vlm_output_label, img_size)
        gripper_actions_pos_scores.append(gripper_actions_pos_score)
        gripper_actions_type_scores.append(gripper_actions_type_score)

        total_score = (gripper_points_pos_score + gripper_actions_pos_score + gripper_actions_type_score) / 3
        total_scores.append(total_score)

        if draw_trajectories:
            build_and_save_trajectory_images(output_dir, img, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label,
                                             task, total_score, output_nr=i)
    
    print_results(output_dir, gripper_points_pos_scores, gripper_actions_pos_scores, gripper_actions_type_scores, total_scores)


if __name__ == '__main__':
    main(eval_dataset_path="/home/troth/data/iTRAP-flower/vlm_val_predictions/qwen3_vl/generated_predictions.jsonl", draw_trajectories=True)
