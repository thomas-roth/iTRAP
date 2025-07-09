import base64
import io
import logging
import os
import re
from PIL import Image
import cv2
from openai import OpenAI
from termcolor import colored


QWEN25VL_RESIZED_SIZE = 196



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
             "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(25, 32), (33, 18), " \
             "(14, 24), <action>Open Gripper</action>, (20, 41), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
             "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action. " \
             f"The coordinates should be integers ranging between 0 and {QWEN25VL_RESIZED_SIZE}, " \
             "indicating the absolute location of the points in the image."
    
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


def extract_gripper_points_and_actions(response, error_logger=None, stretch_factor=1.1):
    regex_ans = r"<ans>(.*?)</ans>"
    regex_gripper_points = r"\(([0-9.]+),\s*([0-9.]+)\)"
    regex_gripper_actions = r"<action>(.*?)</action>"

    try:
        response_content = re.search(regex_ans, response, re.DOTALL) # re.DOTALL to match newlines to broaden accepted response syntax
        response_content = response_content.group(1)

        # TODO: remove after testing VLM responses
        tmp_stricter_response_content = re.search(r"^" + regex_ans + r"$", response, re.DOTALL)
        if tmp_stricter_response_content is None:
            if error_logger is not None:
                error_logger.error(f"Invalid VLM response if strict: {response}")
            else:
                print(colored(f"Error: Invalid VLM response if strict: {response}", "red"))

        gripper_points = []
        for i, match in enumerate(re.finditer(regex_gripper_points, response_content)):
            if match is None:
                # should not happen, but did happen once ):
                if error_logger is not None:
                    error_logger.error(f"Invalid gripper point in response: {response_content}. Skipping match")
                else:
                    print(colored(f"Error: Invalid gripper point in response: {response_content}. Skipping match", "red"))
                continue

            x = float(match.group(1))
            y = float(match.group(2))

            if i > 0 and stretch_factor != 1.0:
                x_start = gripper_points[0][0]
                y_start = gripper_points[0][1]

                # stretch coord points (start point stays the same, end point stretched by stretch factor, points in between stretched accordingly)
                progression = i / (len(list(re.finditer(regex_gripper_points, response_content))) - 1)
                x = x_start + (x - x_start) * (1.0 + (stretch_factor - 1.0) * progression)
                y = y_start + (y - y_start) * (1.0 + (stretch_factor - 1.0) * progression)

                # make sure strected coords aren't out of bounds
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))

            gripper_points.append((x, y))
        
        gripper_actions = []
        for match in re.finditer(regex_gripper_actions, response_content):
            if match is None:
                # should not happen, but did happen once ):
                if error_logger is not None:
                    error_logger.error(f"Invalid gripper action in response: {response_content}. Skipping match")
                else:
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
    except Exception as e:
        if error_logger is not None:
            error_logger.error(f"Invalid VLM response: {response}. Error msg: {e}. Skipping task")
        else:
            print(colored(f"Error: Invalid VLM response: {response}. Error msg: {e}. Skipping task", "red"))
        return [], []

    return gripper_points, points_before_gripper_actions


def draw_trajectory_onto_image(img, gripper_points, gripper_actions, traj_color="red", thickness=2):
    if gripper_points == []:
        # gripper_actions is then empty as well, error msg already printed in extract_gripper_points
        return img

    img_copy = img.copy()
    
    assert img_copy.shape[0] == img_copy.shape[1]
    img_size = img_copy.shape[0]

    scaled_gripper_points = [(round(float(x) / QWEN25VL_RESIZED_SIZE * img_size), round(float(y) / QWEN25VL_RESIZED_SIZE * img_size)) for (x, y) in gripper_points]
    scaled_gripper_actions = [((round(float(x) / QWEN25VL_RESIZED_SIZE * img_size), round(float(y) / QWEN25VL_RESIZED_SIZE * img_size)), action) for ((x, y), action) in gripper_actions]

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


def save_trajectory_image(traj_img, task, local_rank, seq_nr, subtask_nr, step_nr, root_output_dir=None):
    traj_imgs_dir = "traj_imgs" if root_output_dir is None else f"{root_output_dir}/traj_imgs"
    os.makedirs(traj_imgs_dir, exist_ok=True)

    Image.fromarray(traj_img).save(f"{traj_imgs_dir}/rank-{local_rank:01d}_seq-{seq_nr:03d}_task-{subtask_nr:01d}-{task}_step-{step_nr:03d}.png")
