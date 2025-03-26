import base64
import io
import logging
import os
import re
from PIL import Image
import cv2
from openai import OpenAI
from termcolor import colored



def setup_vlm_client():
    client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return client


def query_vlm(static_img_start, vlm_client, task):
    # get base64 encoded image of first frame of static camera
    img_bytes = io.BytesIO()
    Image.fromarray(static_img_start).save(img_bytes, format="PNG")
    base64_img = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    # build prompt text
    prompt = f"<image.png>In the image, please execute the command described in <prompt>{task.replace('_', ' ')}</prompt>. " \
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(0.252, 0.328), (0.327, 0.174), " \
            "(0.139, 0.242), <action>Open Gripper</action>, (0.746, 0.218), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
            "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action. " \
            "The coordinates should be floats ranging between 0 and 1, indicating the relative location of the points in the image."
    
    # send request to vlm
    response = vlm_client.chat.completions.create(
        model="qwen2-vl",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }
                }
            ]
        }]
    )

    return response.choices[0].message.content


def extract_gripper_points(response):
    regex_ans = r"<ans>(.*?)</ans>"
    regex_gripper_points = r"\(([0-9.]+),\s*([0-9.]+)\)"
    regex_gripper_actions = r"<action>(.*?)</action>"

    response_content = re.search(regex_ans, response, re.DOTALL)
    if response_content is None:
        print(colored(f"Error: Invalid response: {response}. Skipping task", "red"))
        return [], []
    else:
        response_content = response_content.group(1)

    gripper_points = []
    for match in re.finditer(regex_gripper_points, response_content):
        if match is None:
            # should not happen, but did happen once ):
            print(colored(f"Error: Invalid gripper point in response: {response_content}. Skipping match", "red"))
            continue

        x = float(match.group(1))
        y = float(match.group(2))
        gripper_points.append((x, y))
    
    gripper_actions = []
    for match in re.finditer(regex_gripper_actions, response_content):
        if match is None:
            # should not happen, but did happen once ):
            print(colored(f"Error: Invalid gripper action in response: {response_content}. Skipping match", "red"))
            continue

        action = match.group(1)
        action_start_pos = match.start()
        gripper_actions.append((action_start_pos, action))
    
    gripper_actions.sort()

    points_before_gripper_actions = []
    for action_start_pos, action in gripper_actions:
        prev_point_index = -1
        for i, match in enumerate(re.finditer(regex_gripper_points, response_content)):
            if match.end() < action_start_pos:
                prev_point_index = i
            else:
                break
        
        if prev_point_index >= 0:
            points_before_gripper_actions.append((gripper_points[prev_point_index], action))

    return gripper_points, points_before_gripper_actions


def draw_trajectory(img, gripper_points, gripper_actions, thickness=2, traj_color="red"):
    if gripper_points == []:
        # gripper_actions is then empty as well, error msg already printed in extract_gripper_points
        return img

    img_copy = img.copy()
    
    assert img_copy.shape[0] == img_copy.shape[1]
    img_size = img_copy.shape[0]

    scaled_gripper_points = [(int(x * img_size), int(y * img_size)) for (x, y) in gripper_points]
    scaled_gripper_actions = [((int(x * img_size), int(y * img_size)), action) for ((x, y), action) in gripper_actions]

    for i in range(len(scaled_gripper_points) - 1):
        if traj_color == "red":
            color = (round((i+1) / len(scaled_gripper_points) * 255), 0, 0) # black to red over time
        elif traj_color == "green":
            color = (0, round((i+1) / len(scaled_gripper_points) * 255), 0) # black to green over time
        else:
            color = (0, 0, round((i+1) / len(scaled_gripper_points) * 255)) # black to blue over time
        
        cv2.line(img_copy, scaled_gripper_points[i], scaled_gripper_points[i+1], color, thickness)
    
    for point, action in scaled_gripper_actions:
        circle_outer_radius = 2 * thickness
        if action == "Close Gripper":
            # green circle
            cv2.circle(img_copy, point, radius=circle_outer_radius, color=(0, 255, 0), thickness=thickness)
        elif action == "Open Gripper":
            # blue circle
            cv2.circle(img_copy, point, radius=circle_outer_radius, color=(0, 0, 255), thickness=thickness)
    
    return img_copy


def build_trajectory_image(static_img_start, vlm_response, save_traj_imgs, task_nr=-1, task="", thickness=None, output_dir=None):
    gripper_points, gripper_actions = extract_gripper_points(vlm_response)
    static_traj_img = draw_trajectory(static_img_start, gripper_points, gripper_actions, thickness)

    if save_traj_imgs:
        if task_nr == -1 or task == "":
            print(colored("Warning: No task description provided for traj img saving. Make sure to provide task and task_nr", "yellow"))
        traj_imgs_dir = "traj_imgs" if output_dir is None else f"{output_dir}/traj_imgs"
        os.makedirs(traj_imgs_dir, exist_ok=True)
        Image.fromarray(static_traj_img).save(f"{traj_imgs_dir}/{task_nr:04d}_{task}.png")

    return static_traj_img
