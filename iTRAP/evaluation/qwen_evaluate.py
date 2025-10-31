from datetime import datetime
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dtw import *

sys.path.append(str(Path(__file__).absolute().parents[2]))
from iTRAP.evaluation.itrap_evaluate import draw_trajectory_onto_image, extract_gripper_points_and_actions


def parse_vlm_outputs(file_path: str) -> list:
    vlm_outputs = []
    with open(file_path, 'r') as file:
        for task in file:
            task = json.loads(task)
            vlm_outputs.append(task)
    return vlm_outputs


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


def plot_trajs_basic(index, gripper_points_pred, gripper_points_label, base_path):
    pred_x = [point[0] for point in gripper_points_pred]
    pred_y = [point[1] for point in gripper_points_pred]
    label_x = [point[0] for point in gripper_points_label]
    label_y = [point[1] for point in gripper_points_label]

    plt.figure()
    plt.plot(pred_x, pred_y, label="Prediction", color="green")
    plt.plot(label_x, label_y, label="Label", color="red")
    plt.legend()

    ax = plt.gca()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.invert_yaxis()

    plt.savefig(f"{base_path}/traj_imgs/trajs-{index:04d}_basic.png")


def plot_trajs_dtw(index, dtw_alignment, base_path):
    # TODO: dtw plot expects arrays of shape (n,) but points & actions are both of shape (n, 2)
    dtw_alignment.plot(xlab="Prediction", ylab="Label", type="twoway")
    plt.title(f"DTW distance of {index}: {dtw_alignment.distance}")
    plt.savefig(f"{base_path}/traj_imgs/index-{index:04d}_dtw.png")


def build_and_save_trajectory_images(output_dir, img_arr, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label, task, total_score, output_nr):
    traj_img_pred = draw_trajectory_onto_image(img_arr, gripper_points_pred, gripper_actions_pred, traj_color="green")
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


def main(gen_preds_path: str, val_imgs_dir: str, draw_trajectories=False):
    output_dir = Path(__file__).parents[2] / "outputs" / "qwen" / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    os.makedirs(output_dir, exist_ok=False)

    vlm_outputs = parse_vlm_outputs(gen_preds_path)

    if "image" in vlm_outputs[0]:
        first_img = Image.open(vlm_outputs[0]['image'][0])
    else:
        dataset_val_imgs = sorted([
            os.path.join(val_imgs_dir, f)
            for f in os.listdir(val_imgs_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        first_img = Image.open(dataset_val_imgs[0])
    assert first_img.size[0] == first_img.size[1]
    img_size = first_img.size[0]
    
    gripper_points_pos_scores = []
    gripper_actions_pos_scores = []
    gripper_actions_type_scores = []
    total_scores = []
    for i, vlm_output in tqdm(enumerate(vlm_outputs), total=len(vlm_outputs), desc="Evaluating VLM outputs"):
        gripper_points_pos_score, gripper_points_pred, gripper_points_label = get_alignment_of_gripper_points(vlm_output["predict"], vlm_output["label"], img_size)
        gripper_points_pos_scores.append(gripper_points_pos_score)

        gripper_actions_pos_score, gripper_actions_type_score, gripper_actions_pred, gripper_actions_label = get_alignment_of_gripper_actions(vlm_output["predict"], vlm_output["label"], img_size)
        gripper_actions_pos_scores.append(gripper_actions_pos_score)
        gripper_actions_type_scores.append(gripper_actions_type_score)

        total_score = (gripper_points_pos_score + gripper_actions_pos_score + gripper_actions_type_score) / 3
        total_scores.append(total_score)

        if draw_trajectories:
            if "image" in vlm_output:
                img_arr = cv2.cvtColor(cv2.imread(vlm_output['image'][0]), cv2.COLOR_BGR2RGB)
            else:
                img_arr = cv2.cvtColor(cv2.imread(dataset_val_imgs[i]), cv2.COLOR_BGR2RGB)
            task = vlm_output["prompt"].split("<prompt>")[1].split("</prompt>")[0].replace(" ", "_")
            build_and_save_trajectory_images(output_dir, img_arr, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label,
                                             task, total_score, output_nr=i)
    
    print_results(output_dir, gripper_points_pos_scores, gripper_actions_pos_scores, gripper_actions_type_scores, total_scores)


if __name__ == '__main__':
    main(gen_preds_path="/home/troth/data/iTRAP-flower/vlm_val_predictions/qwen3_vl/generated_predictions.jsonl",
         val_imgs_dir="/home/troth/data/iTRAP-flower/calvin_vlm_dataset/2025-10-20_16-24-18_qwen3_abc/validation",
         draw_trajectories=True)
