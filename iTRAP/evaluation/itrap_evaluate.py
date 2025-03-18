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
from omegaconf import DictConfig
from openai import OpenAI
from pytorch_lightning import seed_everything
import torch
from tqdm import tqdm
from PIL import Image

sys.path.append(str(Path(__file__).absolute().parents[2]))
sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "MoDE_Diffusion_Policy"))
from iTRAP.models.MoDE_Diffusion_Policy.mode.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition
from iTRAP.models.MoDE_Diffusion_Policy.mode.evaluation.multistep_sequences import get_sequences
from iTRAP.models.MoDE_Diffusion_Policy.mode.rollout.rollout_video import RolloutVideo



MODEL_PATH = "path/to/model"



def _get_log_dir(cfg_log_dir):
    log_dir = Path(cfg_log_dir)
    latest_run_day = sorted(log_dir.iterdir(), key=os.path.getmtime)[-1]
    latest_run_time = sorted(latest_run_day.iterdir(), key=os.path.getmtime)[-1]
    return latest_run_time


def setup_vlm_client():
    client = OpenAI(api_key="0", base_url="http://localhost:8000/v1")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return client


def setup_policy(cfg):
    seed_everything(0, workers=True)

    model, env, _, _ = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
    )

    model = model.to(cfg.device)

    if cfg.num_sampling_steps is not None:
        model.num_sampling_steps = cfg.num_sampling_steps
    if cfg.sampler_type is not None:
        model.sampler_type = cfg.sampler_type
    if cfg.multistep is not None:
        model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler

    model.eval()

    global_step = int(cfg.checkpoint.split('=')[-1].split('_')[0]) * 1000 # 1000 steps per epoch

    return model, env, global_step


def generate_tasks(num_tasks) -> list:
    sequences = get_sequences(num_sequences=num_tasks)

    # turn sequences into tasks => only keep first task
    tasks = []
    for initial_state, sequence in sequences:
        task = initial_state, sequence[0]
        tasks.append(task)

    return tasks


def build_prompt(task_desc: str) -> list:
    return f"<image.png>In the image, please execute the command described in <prompt>{task_desc.replace('_', ' ')}</prompt>. " \
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(0.252, 0.328), (0.327, 0.174), " \
            "(0.139, 0.242), <action>Open Gripper</action>, (0.746, 0.218), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
            "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action. " \
            "The coordinates should be floats ranging between 0 and 1, indicating the relative location of the points in the image."


def query_vlm(env, vlm_client, task):
    # get base64 encoded image of first frame of static camera
    static_img_start, _ = env.cameras[0].render()
    img_bytes = io.BytesIO()
    Image.fromarray(static_img_start).save(img_bytes, format="PNG")
    base64_img = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    # build prompt text from task
    prompt = build_prompt(task)
    
    # query VLM for trajectory string
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
        }],
    )

    return response.choices[0].message.content


def extract_gripper_points(response):
    regex_ans = r"<ans>(.*?)</ans>"
    regex_gripper_points = r"\(([0-9.]+),\s*([0-9.]+)\)"
    regex_gripper_actions = r"<action>(.*?)</action>"

    response = re.search(regex_ans, response, re.DOTALL).group(1)

    gripper_points = []
    for match in re.finditer(regex_gripper_points, response):
        x = float(match.group(1))
        y = float(match.group(2))
        gripper_points.append((x, y))
    
    gripper_actions = []
    for match in re.finditer(regex_gripper_actions, response):
        action = match.group(1)
        action_start_pos = match.start()
        gripper_actions.append((action_start_pos, action))
    
    gripper_actions.sort()

    points_before_gripper_actions = []
    for action_start_pos, action in gripper_actions:
        prev_point_index = -1
        for i, match in enumerate(re.finditer(regex_gripper_points, response)):
            if match.end() < action_start_pos:
                prev_point_index = i
            else:
                break
        
        if prev_point_index >= 0:
            points_before_gripper_actions.append((gripper_points[prev_point_index], action))

    return gripper_points, points_before_gripper_actions


def draw_trajectory(img, gripper_points, gripper_actions, traj_color="red"):
    img_copy = img.copy()
    
    assert img_copy.shape[0] == img_copy.shape[1]
    img_size = img_copy.shape[0]

    scaled_gripper_points = [(int(x * img_size), int(y * img_size)) for x, y in gripper_points]
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


def build_trajectory_image(env, vlm_response, save_traj_imgs, task_nr, task):
    static_img_start, _ = env.cameras[0].render()

    gripper_points, gripper_actions = extract_gripper_points(vlm_response)
    static_traj_img = draw_trajectory(static_img_start, gripper_points, gripper_actions)

    if save_traj_imgs:
        os.makedirs("traj_imgs", exist_ok=True)
        Image.fromarray(static_traj_img).save(f"traj_imgs/{task_nr:04d}_{task}.png")

    return torch.tensor(static_traj_img)


def rollout_policy(policy_cfg, env, policy, task_nr, task, static_traj_img, task_oracle, rollout_video, policy_global_step):
    obs = env.get_obs()
    goal = {"vis_image": torch.tensor(static_traj_img), "lang_text": task}

    policy.reset()
    start_info = env.get_info()

    rollout_video.new_video(tag=f"{task_nr:04d}_{task}", caption=task)

    success = False
    for step in tqdm(range(policy_cfg.ep_len), desc=f"Rolling out policy for task {task}", leave=False):
        action = policy.step(obs, goal)
        obs, _, _, current_info = env.step(action)

        normalized_rgb_static = obs["rgb_obs"]["rgb_static"] / 127.5 - 1 # normalize to [-1, 1]
        rollout_video.update(normalized_rgb_static) # shape: B, F, C, H, W

        # check if current steps solves task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        if len(current_task_info) > 0:
            success = True
            break
    
    rollout_video.draw_outcome(success)
    rollout_video.add_language_instruction(task)

    # FIXME: add goal image as video thumbnail
    #normalized_goal_img = goal["vis_image"] / 127.5 - 1 # normalize to [-1, 1]
    #rollout_video.add_goal_thumbnail(normalized_goal_img.permute(2, 0, 1)) # shape: C, H, W

    rollout_video.log(policy_global_step)

    return success


def log_success_rate(logger, total_task_counter, success_counter):
    max_task_len = max([len(task) for task in total_task_counter])

    logger.info("Evaluation results:")
    for task in total_task_counter:
        logger.info(f"{task:<{max_task_len}}: {success_counter[task]} / {total_task_counter[task]} | SR: {success_counter[task] / total_task_counter[task] * 100:.2f} %")
    logger.info(f"Total SR: {sum(success_counter.values()) / sum(total_task_counter.values()) * 100:.2f}%")


@hydra.main(config_path="../models/MoDE_Diffusion_Policy/conf", config_name="mode_evaluate")
def main(policy_cfg: DictConfig) -> None:
    # make sure policy runs on GPUs 1, 2, 3 (GPU 0 used by VLM)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    policy, env, policy_global_step = setup_policy(policy_cfg)

    vlm_client = setup_vlm_client()

    logger = logging.getLogger(__name__)
    log_dir = _get_log_dir(policy_cfg.log_dir)

    rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=Path(log_dir) / "rollout_videos",
            resolution_scale=1,
        )

    tasks = generate_tasks(policy_cfg.num_sequences)
    task_oracle = hydra.utils.instantiate(policy_cfg.tasks)

    success_counter = Counter()
    failure_counter = Counter()
    for task_nr, (initial_state, task) in tqdm(enumerate(tasks), total=len(tasks), desc="Evaluating model"):
        # reset env
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        response = query_vlm(env, vlm_client, task)

        static_traj_img = build_trajectory_image(env, response, save_traj_imgs, task_nr, task)

        success = rollout_policy(policy_cfg, env, policy, task_nr, task, static_traj_img, task_oracle, rollout_video, policy_global_step)
        if success:
            success_counter[task] += 1
        else:
            failure_counter[task] += 1
    
    total_task_counter = success_counter + failure_counter
    log_success_rate(logger, total_task_counter, success_counter)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "MoDE_Diffusion_Policy" / "calvin_env"))

    save_traj_imgs = True
    main()
