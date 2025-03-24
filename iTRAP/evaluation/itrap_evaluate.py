import base64
import io
import logging
import os
from pathlib import Path
import re
import sys
from collections import Counter
import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from openai import OpenAI
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).absolute().parents[2]))
sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "MoDE_Diffusion_Policy"))
from iTRAP.models.MoDE_Diffusion_Policy.mode.evaluation.utils import LangEmbeddings, get_default_mode_and_env, get_env_state_for_initial_condition
from iTRAP.models.MoDE_Diffusion_Policy.mode.evaluation.multistep_sequences import get_sequences
from iTRAP.models.MoDE_Diffusion_Policy.mode.rollout.rollout_video import RolloutVideo



MODEL_PATH = "path/to/model"



class ItrapEvaluator:
    def __init__(self):
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        with hydra.initialize(config_path="../models/MoDE_Diffusion_Policy/conf"):
            self.mode_eval_cfg = hydra.compose(config_name="mode_evaluate")
            complete_calvin_cfg = hydra.compose(config_name="config_calvin")
        
        self.device = self.mode_eval_cfg.device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        if self.device != "cpu":
            self.device = torch.device(f"cuda:{self.mode_eval_cfg.device}")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.record = True
        if self.record:
            self.rollout_video = RolloutVideo(
                logger=self.logger,
                empty_cache=False,
                log_to_file=True,
                save_dir=Path(self._get_log_dir()) / "rollout_videos",
                resolution_scale=1,
            )

        self.vlm_client = setup_vlm_client()

        self.setup_policy()
        
        val_transforms_cfg = complete_calvin_cfg.datamodule.transforms.val.rgb_static
        self.val_transforms = []
        for val_transform_cfg in val_transforms_cfg:
            self.val_transforms.append(hydra.utils.instantiate(val_transform_cfg))
        
        self.task_oracle = hydra.utils.instantiate(self.mode_eval_cfg.tasks)


    def _get_log_dir(self):
        log_dir = Path(self.mode_eval_cfg.log_dir)
        latest_run_day = sorted(log_dir.iterdir(), key=os.path.getmtime)[-1]
        latest_run_time = sorted(latest_run_day.iterdir(), key=os.path.getmtime)[-1]
        return latest_run_time


    def setup_policy(self):
        seed_everything(0, workers=True)

        sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "MoDE_Diffusion_Policy" / "calvin_env"))
        self.policy, self.env, _, self.lang_embeddings = get_default_mode_and_env(
            self.mode_eval_cfg.train_folder,
            self.mode_eval_cfg.dataset_path,
            self.mode_eval_cfg.checkpoint,
            eval_cfg_overwrite=self.mode_eval_cfg.eval_cfg_overwrite,
            device_id=self.mode_eval_cfg.device,
        )

        self.policy = self.policy.to(self.device)

        if self.mode_eval_cfg.num_sampling_steps is not None:
            self.policy.num_sampling_steps = self.mode_eval_cfg.num_sampling_steps
        if self.mode_eval_cfg.sampler_type is not None:
            self.policy.sampler_type = self.mode_eval_cfg.sampler_type
        if self.mode_eval_cfg.multistep is not None:
            self.policy.multistep = self.mode_eval_cfg.multistep
        if self.mode_eval_cfg.sigma_min is not None:
            self.policy.sigma_min = self.mode_eval_cfg.sigma_min
        if self.mode_eval_cfg.sigma_max is not None:
            self.policy.sigma_max = self.mode_eval_cfg.sigma_max
        if self.mode_eval_cfg.noise_scheduler is not None:
            self.policy.noise_scheduler = self.mode_eval_cfg.noise_scheduler

        self.policy.eval()

        self.policy_global_step = int(self.mode_eval_cfg.checkpoint.split('=')[-1].split('_')[0]) * 1000 # 1000 steps per epoch
    
    
    def evaluate_itrap(self):
        eval_sequences = get_sequences(num_sequences=self.mode_eval_cfg.num_sequences)

        results = []
        for seq_nr, (initial_state, eval_sequence) in tqdm(enumerate(eval_sequences), desc="Evaluating policy"):
            success_counter = self.evaluate_sequence(seq_nr, initial_state, eval_sequence)
            results.append(success_counter)

            if self.record:
                # FIXME: add goal image as video thumbnail
                #normalized_goal_img = goal["vis_image"] / 127.5 - 1 # normalize to [-1, 1]
                #rollout_video.add_goal_thumbnail(normalized_goal_img.permute(2, 0, 1)) # shape: C, H, W

                self.rollout_video.log(self.policy_global_step)
        
        self.log_success_rate(results)
    

    def evaluate_sequence(self, seq_nr, initial_state, eval_sequence):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        if self.record:
            self.rollout_video.new_video(tag=f"lh-eval_seq-nr-{seq_nr:04d}", caption=" | ".join(eval_sequence))

        success_counter = Counter()
        for subtask in eval_sequence:
            if self.record:
                self.rollout_video.new_subtask()
            
            success = self.rollout_subtask(subtask)

            if self.record:
                self.rollout_video.draw_outcome(success)

            if success:
                success_counter[subtask] += 1
            else:
                return success_counter
        
        return success_counter
    

    def rollout_subtask(self, subtask):
        obs = self.env.get_obs()

        goal = self.lang_embeddings.get_lang_goal(subtask)

        self.policy.reset()
        start_info = self.env.get_info()

        success = False
        for step in tqdm(range(self.mode_eval_cfg.ep_len), desc=f"Rolling out policy for task {subtask}", leave=False):
            if step % self.policy.multistep == 0:
                static_img_start = self.env.cameras[0].render()[0].squeeze()
                response = query_vlm(static_img_start, self.vlm_client, subtask)
                static_traj_img = build_trajectory_image(static_img_start, response, save_traj_imgs=False)

                transformed_static_traj_img = static_traj_img.permute(2, 0, 1).unsqueeze(0)
                for transform in self.val_transforms:
                    transformed_static_traj_img = transform(transformed_static_traj_img)

            action = self.policy.step(rgb_static=transformed_static_traj_img.to(self.device), rgb_gripper=obs["rgb_obs"]["rgb_gripper"].to(self.device), goal=goal)
            obs, _, _, current_info = self.env.step(action)

            if self.record:
                normalized_rgb_static = obs["rgb_obs"]["rgb_static"] / 127.5 - 1 # normalize to [-1, 1] # TODO: check if neccesary
                self.rollout_video.update(normalized_rgb_static) # shape: B, F, C, H, W

            # check if current steps solves task
            current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break
        
        if self.record:
            self.rollout_video.add_language_instruction(goal["lang_text"])
        
        return success
    

    def log_success_rate(self, results):
        results_counter = Counter(results)
        for subtask_nr in range(1, 6):
            num_success = sum(results_counter[j] for j in reversed(range(subtask_nr, 6)))
            success_rate = num_success / len(results)
            self.logger.log(f"{subtask_nr} / 5 subtasks: {num_success} / {len(results)} sequences, SR: {success_rate * 100:.1f}%")
        
        avg_seq_len = np.mean(results)
        self.logger.log(f"Average successful sequence length: {avg_seq_len:.1f}")


def log_success_rate(logger, total_task_counter, success_counter):
    max_task_len = max([len(task) for task in total_task_counter])

    logger.info("Evaluation results:")
    for task in total_task_counter:
        logger.info(f"{task:<{max_task_len}}: {success_counter[task]} / {total_task_counter[task]} | SR: {success_counter[task] / total_task_counter[task] * 100:.2f} %")
    logger.info(f"Total SR: {sum(success_counter.values()) / sum(total_task_counter.values()) * 100:.2f}%")



"""
def generate_seqs(num_sequences) -> list:
    sequences = get_sequences(num_sequences=num_sequences)

    # turn sequences into tasks => only keep first task
    tasks = []
    for initial_state, sequence in sequences:
        task = initial_state, sequence[0]
        tasks.append(task)

    return tasks
"""


def setup_vlm_client():
    client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return client


def build_prompt(task_desc: str) -> list:
    return f"<image.png>In the image, please execute the command described in <prompt>{task_desc.replace('_', ' ')}</prompt>. " \
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(0.252, 0.328), (0.327, 0.174), " \
            "(0.139, 0.242), <action>Open Gripper</action>, (0.746, 0.218), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
            "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action. " \
            "The coordinates should be floats ranging between 0 and 1, indicating the relative location of the points in the image."


def query_vlm(static_img_start, vlm_client, task):
    # get base64 encoded image of first frame of static camera
    img_bytes = io.BytesIO()
    Image.fromarray(static_img_start).save(img_bytes, format="PNG")
    base64_img = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    # query VLM for trajectory
    prompt = build_prompt(task)

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


def draw_trajectory(img, gripper_points, gripper_actions, traj_color="red"):
    if gripper_points == []:
        # gripper_actions is empty as well, error msg already printed in extract_gripper_points
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
        
        cv2.line(img_copy, scaled_gripper_points[i], scaled_gripper_points[i+1], color, thickness=2)
    
    for point, action in scaled_gripper_actions:
        if action == "Close Gripper":
            # green circle
            cv2.circle(img_copy, point, radius=5, color=(0, 255, 0), thickness=2)
        elif action == "Open Gripper":
            # blue circle
            cv2.circle(img_copy, point, radius=5, color=(0, 0, 255), thickness=2)
    
    return img_copy


def build_trajectory_image(static_img_start, vlm_response, save_traj_imgs, task_nr=-1, task=""):
    gripper_points, gripper_actions = extract_gripper_points(vlm_response)
    static_traj_img = draw_trajectory(static_img_start, gripper_points, gripper_actions)

    if save_traj_imgs:
        if task_nr == -1 or task == "":
            print(colored("Warning: No task description provided for traj img saving. Make sure to provide task and task_nr", "yellow"))
        os.makedirs("traj_imgs", exist_ok=True)
        Image.fromarray(static_traj_img).save(f"traj_imgs/{task_nr:04d}_{task}.png")

    return torch.tensor(static_traj_img)


if __name__ == "__main__":
    itrap_evaluator = ItrapEvaluator()
    itrap_evaluator.evaluate_itrap()
