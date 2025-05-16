import json
import os
from pathlib import Path
from PIL import Image
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


def get_alignment_of_gripper_points(pred, label):
    gripper_points_pred, _ = extract_gripper_points_and_actions(pred)
    gripper_points_label, _ = extract_gripper_points_and_actions(label)

    dtw_alignment = dtw(np.array(gripper_points_pred), np.array(gripper_points_label), keep_internals=True)

    return dtw_alignment, gripper_points_pred, gripper_points_label


def get_alignment_of_gripper_actions(pred, label):
    _, gripper_actions_pred = extract_gripper_points_and_actions(pred)
    _, gripper_actions_label = extract_gripper_points_and_actions(label)

    gripper_action_dists = []
    same_gripper_actions = []
    for gripper_action_pred, gripper_action_label in zip(gripper_actions_pred, gripper_actions_label):
        # TODO: handle different lengths of gripper actions
        gripper_action_dist = np.linalg.norm(np.array(gripper_action_pred[0]) - np.array(gripper_action_label[0]))
        gripper_action_dists.append(gripper_action_dist)

        same_gripper_actions.append(gripper_action_pred[1] == gripper_action_label[1])

    if len(gripper_action_dists) == 0:
        # no gripper actions in trajectory => no errors
        return 0, 100, gripper_actions_pred, gripper_actions_label
    else:
        avg_gripper_action_dist = np.mean(gripper_action_dists)
        percent_same_gripper_action = np.mean(same_gripper_actions) * 100

        return avg_gripper_action_dist, percent_same_gripper_action, gripper_actions_pred, gripper_actions_label


def build_and_save_trajectory_images(base_path, img, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label, prompt, dtw_dist, output_nr):
    traj_img_pred = draw_trajectory_onto_image(np.array(img), gripper_points_pred, gripper_actions_pred, traj_color="green")
    traj_img_pred_label = draw_trajectory_onto_image(traj_img_pred, gripper_points_label, gripper_actions_label, traj_color="red")

    task = prompt.split("<prompt>")[1].split("</prompt>")[0].replace(" ", "_")
    os.makedirs(f"{base_path}/traj_imgs", exist_ok=True)
    Image.fromarray(traj_img_pred_label).save(f"{base_path}/traj_imgs/{output_nr:03d}_{task}_{round(dtw_dist * 100 / img.size[0], 2)}.png")


def plot_trajs_basic(gripper_points_pred, gripper_points_label):
    pred_x = [point[0] for point in gripper_points_pred]
    pred_y = [point[1] for point in gripper_points_pred]
    label_x = [point[0] for point in gripper_points_label]
    label_y = [point[1] for point in gripper_points_label]

    plt.plot(pred_x, pred_y, label="Prediction")
    plt.plot(label_x, label_y, label="Label")
    plt.legend()
    plt.show()


def print_results(gripper_points_dists, avg_gripper_action_dists, percent_same_gripper_actions, img_size):
    avg_gripper_points_dist = round(np.mean(gripper_points_dists) * 10000 / img_size, 2)
    avg_gripper_action_dist = round(np.mean(avg_gripper_action_dists) * 10000 / img_size, 2)
    avg_per_cent_same_gripper_action = round(np.mean(percent_same_gripper_actions), 2)

    print(f"Average distance between gripper points: {avg_gripper_points_dist} % of image size") # FIXME: seems wrong
    print(f"Average distance between gripper actions: {avg_gripper_action_dist} % of image size") # FIXME: seems wrong
    print(f"Average percentage of same gripper actions: {avg_per_cent_same_gripper_action} %") # FIXME?


def main(base_path: str, file_path: str, draw_trajectories=False):
    vlm_outputs = parse_vlm_outputs(base_path + '/' + file_path)

    first_img = Image.open(vlm_outputs[0]['image'][0])
    assert first_img.size[0] == first_img.size[1]
    img_size = first_img.size[0]
    
    gripper_points_dists = []
    avg_gripper_action_dists = []
    percent_same_gripper_actions = []
    for i, vlm_output in tqdm(enumerate(vlm_outputs), total=len(vlm_outputs), desc="Evaluating VLM outputs"):
        dtw_alignment, gripper_points_pred, gripper_points_label = get_alignment_of_gripper_points(vlm_output["predict"], vlm_output["label"])
        gripper_points_dists.append(dtw_alignment.distance)

        #plot_trajs_basic(gripper_points_predict, gripper_points_label)
        
        #dtw_alignment.plot(type="threeway")
        #plt.show()

        avg_gripper_action_dist, percent_same_gripper_action, gripper_actions_pred, gripper_actions_label = get_alignment_of_gripper_actions(vlm_output["predict"], vlm_output["label"])
        avg_gripper_action_dists.append(avg_gripper_action_dist)
        percent_same_gripper_actions.append(percent_same_gripper_action)
        
        if draw_trajectories:
            img = Image.open(vlm_output['image'][0])
            build_and_save_trajectory_images(base_path, img, gripper_points_pred, gripper_actions_pred, gripper_points_label, gripper_actions_label,
                                             vlm_output["prompt"], dtw_alignment.distance, output_nr=i)
    
    print_results(gripper_points_dists, avg_gripper_action_dists, percent_same_gripper_actions, img_size)


if __name__ == '__main__':
    main(base_path="/home/troth/bt/data/vlm_val_predictions", file_path="vlm_generated_predictions.jsonl", draw_trajectories=True)
