import base64
import io
import logging
import os
from pathlib import Path
import re
import sys
from PIL import Image
import cv2
from openai import OpenAI
from termcolor import colored
import pylineclip

sys.path.append(str(Path(__file__).absolute().parents[3])) # Add repo root to path
from iTRAP.models.Qwen3_VL.resize_utils import resize_point_back_to_original_for_qwen3_vl



STATIC_IMG_SIZE = 200
GRIPPER_IMG_SIZE = 84


def setup_vlm_client():
    client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return client


def get_prompt(task: str, cam:str=None, single_query=False) -> str:
    if single_query:
        return f"<image><image>The images show two views of the same scene. In the images, please execute the command described in <prompt>{task.replace('_', ' ')}</prompt>. " \
                "Provide two sequences of points denoting the trajectory of a robot gripper in each of the views to achieve the goal. " \
                "Format your answer as two lists of tuples each enclosed by <ans> and </ans> tags. For example: <ans>[(x_1, y_1), (x_2, y_2), " \
                "(x_3, y_3), <action>Open Gripper</action>, (x_4, y_4), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
                "an x and y location of the end effector of the gripper in one of the images. The action tags indicate the gripper action."
    else:
        if cam != "static" and cam != "gripper":
            raise ValueError(f"Invalid cam type {cam} for prompt generation")
        
        return f"<image>The image shows the view from the {cam} camera of the scene. In the image, please execute the command described in <prompt>{task.replace('_', ' ')}</prompt>. " \
                "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
                "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(x_1, y_1), (x_2, y_2), " \
                "(x_3, y_3), <action>Open Gripper</action>, (x_4, y_4), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
                "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action."


def query_vlm(static_img_start, gripper_img_start, vlm_client, task, single_query=False):
    # get base64 encoded image of first frame of static camera
    img_buffer = io.BytesIO()
    Image.fromarray(static_img_start).save(img_buffer, format="PNG")
    base64_static_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
    # reset buffer
    img_buffer.seek(0)
    img_buffer.truncate(0)
    
    # get base64 encoded image of first frame of gripper camera
    Image.fromarray(gripper_img_start).save(img_buffer, format="PNG")
    base64_gripper_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    responses = {}
    if single_query:
        prompt = get_prompt(task, single_query=True)
        
        # send request to vlm
        response_single_query = vlm_client.chat.completions.create(
            model="qwen3_vl",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_static_img}"
                        }
                    },{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_gripper_img}"
                        }
                    }
                ]
            }],
            temperature=0.7, # matches SFT generation temp
        )
        
        assert "\n" in response_single_query.choices[0].message.content, "Expected two answers separated by newline for single VLM query with two views"
        responses["static"] = response_single_query.choices[0].message.content.split("\n")[0]
        responses["gripper"] = response_single_query.choices[0].message.content.split("\n")[1]
    else:

        prompt_static = get_prompt(task, cam="static")
        prompt_gripper = get_prompt(task, cam="gripper")
        
        # send requests to vlm
        response_static = vlm_client.chat.completions.create(
            model="qwen3_vl",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_static
                    },{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_static_img}"
                        }
                    }
                ]
            }],
            temperature=0.7, # matches SFT generation temp
        )
        responses["static"] = response_static.choices[0].message.content

        response_gripper = vlm_client.chat.completions.create(
            model="qwen3_vl",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_gripper
                    },{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_gripper_img}"
                        }
                    }
                ]
            }],
            temperature=0.7, # matches SFT generation temp
        )
        responses["gripper"] = response_gripper.choices[0].message.content

    return responses


def extract_gripper_points_and_actions(response, orig_img_height, orig_img_width, logger=None, stretch_factor=1.0):
    regex_ans = r"<ans>(.*?)</ans>"
    regex_ans_no_end = r"<ans>(.*)"
    regex_gripper_points = r"\(([0-9.]+),\s*([0-9.]+)\)"
    regex_gripper_actions = r"<action>(.*?)</action>"

    try:
        response_content = re.search(regex_ans, response, re.DOTALL) # re.DOTALL to match newlines to broaden accepted response syntax

        if response_content is None:
            _log_warning(logger, "No valid answer tags found in VLM response, trying to parse without closing </ans> tag")
            
            response_content = re.search(regex_ans_no_end, response, re.DOTALL)

            if response_content is None:
                _log_error(logger, f"Invalid VLM response: {response}. Skipping task")
                return [], []

        response_content = response_content.group(1)

        gripper_points = []
        for i, match in enumerate(re.finditer(regex_gripper_points, response_content)):
            if match is None:
                # should not happen, but did happen once ):
                _log_error(logger, f"Invalid gripper point in response: {response_content}. Skipping match")
                continue

            x = int(match.group(1))
            y = int(match.group(2))

            # resize coords from Qwen3-VL internal size to original size
            x, y = resize_point_back_to_original_for_qwen3_vl((x, y), orig_img_height, orig_img_width)

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
                _log_error(logger, f"Invalid gripper action in response: {response_content}. Skipping match")
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
        _log_error(logger, f"Invalid VLM response: {response}. Error msg: {e}. Skipping task")
        return [], []
    
    gripper_points, points_before_gripper_actions, dont_draw_line_between = _clip_gripper_traj_to_image_bounds(gripper_points, points_before_gripper_actions, orig_img_height, orig_img_width)
    
    return gripper_points, points_before_gripper_actions, dont_draw_line_between


def draw_trajectory_onto_image(img, gripper_points, gripper_actions, dont_draw_line_between=None, traj_color="red", thickness=2):
    if gripper_points == []:
        # gripper_actions is then empty as well, error msg already printed in extract_gripper_points
        return img
        
    assert (img.shape[0] == STATIC_IMG_SIZE and img.shape[1] == STATIC_IMG_SIZE or
            img.shape[0] == GRIPPER_IMG_SIZE and img.shape[1] == GRIPPER_IMG_SIZE), \
            f"Image size {img.shape[0]}x{img.shape[1]} not supported for drawing trajectory"
    
    img_copy = img.copy()

    for i in range(len(gripper_points) - 1):
        if traj_color == "red":
            color = (round((i+1) / len(gripper_points) * 255), 0, 0) # black to red over time
        elif traj_color == "green":
            color = (0, round((i+1) / len(gripper_points) * 255), 0) # black to green over time
        else:
            color = (0, 0, round((i+1) / len(gripper_points) * 255)) # black to blue over time
        
        if dont_draw_line_between is None or \
            (gripper_points[i], gripper_points[i+1]) not in dont_draw_line_between: 
            cv2.line(img_copy, gripper_points[i], gripper_points[i+1], color, thickness)

    for point, action in gripper_actions:
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

    if traj_img.shape[0] == STATIC_IMG_SIZE and traj_img.shape[1] == STATIC_IMG_SIZE:
        cam = "static"
    elif traj_img.shape[0] == GRIPPER_IMG_SIZE and traj_img.shape[1] == GRIPPER_IMG_SIZE:
        cam = "gripper"
    else:
        raise ValueError(f"Image size {traj_img.shape[0]}x{traj_img.shape[1]} not supported for saving trajectory image")

    Image.fromarray(traj_img).save(f"{traj_imgs_dir}/rank-{local_rank:01d}_seq-{seq_nr:03d}_task-{subtask_nr:01d}-{task}_step-{step_nr:03d}_{cam}.png")


def _log_error(logger, msg):
    if logger is None:
        print(colored(f"Error: {msg}", "red"))
    else:
        logger.error(msg)


def _log_warning(logger, msg):
    if logger is None:
        print(colored(f"Warning: {msg}", "yellow"))
    else:
        logger.warning(msg)


def _clip_gripper_traj_to_image_bounds(gripper_points, gripper_actions, orig_img_width, orig_img_height):
    # uses Cohen-Sutherland line clipping algorithm
    
    gripper_action_points = [point for point, action in gripper_actions]
    
    clipped_gripper_points = []
    clipped_gripper_actions = gripper_actions.copy()
    dont_draw_line_between = []
    for i in range(len(gripper_points) - 1): # results in empty list if only single point in original list, should be fine since then no line drawn anyway
        clipped_x1, clipped_y1, clipped_x2, clipped_y2 = pylineclip.cohensutherland(xmin=0, xmax=orig_img_width-1, ymin=0, ymax=orig_img_height-1, 
                                                                                    x1=gripper_points[i][0], y1=gripper_points[i][1],
                                                                                    x2=gripper_points[i+1][0], y2=gripper_points[i+1][1])
        
        line_fully_out_of_bounds = clipped_x1 is None and clipped_y1 is None and clipped_x2 is None and clipped_y2 is None
        if line_fully_out_of_bounds:
            # line segment completely out of image bounds => skip both points
            
            # remove gripper actions associated with the two out-of-bounds points if they exist
            if gripper_points[i] in gripper_action_points:
                gripper_action = [action for point, action in clipped_gripper_actions if point == gripper_points[i]][0]
                clipped_gripper_actions.remove((gripper_points[i], gripper_action))
                gripper_action_points.remove(gripper_points[i])
            if gripper_points[i+1] in gripper_action_points:
                gripper_action = [action for point, action in clipped_gripper_actions if point == gripper_points[i+1]][0]
                clipped_gripper_actions.remove((gripper_points[i+1], gripper_action))
                gripper_action_points.remove(gripper_points[i+1])
            
            continue
        
        curr_first_point_clipped_to_any_border = (clipped_x1, clipped_y1) != gripper_points[i]
        curr_second_point_clipped_to_any_border = (clipped_x2, clipped_y2) != gripper_points[i+1]
        if curr_first_point_clipped_to_any_border or curr_second_point_clipped_to_any_border:
            # line segment intersects with image borders => add clipped points
            
            # clipped points may be floats, but need to be ints for cv2 drawing functions
            clipped_x1 = round(clipped_x1)
            clipped_y1 = round(clipped_y1)
            clipped_x2 = round(clipped_x2)
            clipped_y2 = round(clipped_y2)
            
            # avoid drawing lines along the border for out-of-bounds lines
            last_added_point_clipped_to_any_border = (len(clipped_gripper_points) > 0) and \
                ((clipped_gripper_points[-1][0] == 0) or (clipped_gripper_points[-1][0] == orig_img_width - 1) or 
                 (clipped_gripper_points[-1][1] == 0) or (clipped_gripper_points[-1][1] == orig_img_height - 1))
            if last_added_point_clipped_to_any_border:
                if curr_first_point_clipped_to_any_border:
                    # don't draw line between last added point & current 1st point
                    dont_draw_line_between.append((clipped_gripper_points[-1], (clipped_x1, clipped_y1)))
                elif curr_second_point_clipped_to_any_border:
                    # don't draw line between last added point & current 2nd point
                    dont_draw_line_between.append((clipped_gripper_points[-1], (clipped_x2, clipped_y2)))

            # remove gripper actions associated with the out-of-bounds points if they exist
            if curr_first_point_clipped_to_any_border and gripper_points[i] in gripper_action_points:
                gripper_action = [action for point, action in clipped_gripper_actions if point == gripper_points[i]][0]
                clipped_gripper_actions.remove((gripper_points[i], gripper_action))
                gripper_action_points.remove(gripper_points[i])
            if curr_second_point_clipped_to_any_border and gripper_points[i+1] in gripper_action_points:
                gripper_action = [action for point, action in clipped_gripper_actions if point == gripper_points[i+1]][0]
                clipped_gripper_actions.remove((gripper_points[i+1], gripper_action))
                gripper_action_points.remove(gripper_points[i+1])
        else:
            # line segment completely in image bounds => add both points
            
            assert clipped_x1 == gripper_points[i][0] and clipped_y1 == gripper_points[i][1] and \
                clipped_x2 == gripper_points[i+1][0] and clipped_y2 == gripper_points[i+1][1], \
                "Cohen-Sutherland should not modify the points if the connecting line is completely within bounds"
        
        # add clipped points if not already added
        if (clipped_x1, clipped_y1) not in clipped_gripper_points:
            clipped_gripper_points.append((clipped_x1, clipped_y1))
        if (clipped_x2, clipped_y2) not in clipped_gripper_points:
            clipped_gripper_points.append((clipped_x2, clipped_y2))           
    
    return clipped_gripper_points, clipped_gripper_actions, dont_draw_line_between
