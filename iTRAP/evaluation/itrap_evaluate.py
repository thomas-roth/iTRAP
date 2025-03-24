from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from collections import Counter
import hydra
import numpy as np
from pytorch_lightning import seed_everything
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[2]))
sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "MoDE_Diffusion_Policy"))
from iTRAP.evaluation.utils import build_trajectory_image, query_vlm, setup_vlm_client
from iTRAP.models.MoDE_Diffusion_Policy.mode.evaluation.utils import LangEmbeddings, get_default_mode_and_env, get_env_state_for_initial_condition
from iTRAP.models.MoDE_Diffusion_Policy.mode.evaluation.multistep_sequences import get_sequences
from iTRAP.models.MoDE_Diffusion_Policy.mode.rollout.rollout_video import RolloutVideo



MODEL_PATH = "path/to/model"



class ItrapEvaluator:
    def __init__(self):
        with hydra.initialize(config_path="../models/MoDE_Diffusion_Policy/conf"):
            self.mode_eval_cfg = hydra.compose(config_name="mode_evaluate")
            complete_calvin_cfg = hydra.compose(config_name="config_calvin")
        
        self.device = self.mode_eval_cfg.device
        if self.device != "cpu":
            self.device = torch.device(f"cuda:{self.mode_eval_cfg.device}")

        self.output_dir = Path(self.mode_eval_cfg.log_dir) / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        _stream_handler = logging.StreamHandler(stream=sys.stdout)
        _stream_handler.setFormatter(logging.Formatter("\033[94m[%(asctime)s][%(levelname)s]\033[0m %(message)s"))
        self.logger.addHandler(_stream_handler)
        _file_handler = logging.FileHandler(self.output_dir / "itrap_evaluate.log", mode="w")
        _file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        self.logger.addHandler(_file_handler)

        self.record = True
        if self.record:
            self.rollout_video = RolloutVideo(
                logger=self.logger,
                empty_cache=False,
                log_to_file=True,
                save_dir=self.output_dir / "rollout_videos",
                resolution_scale=1,
            )

        self.vlm_client = setup_vlm_client()

        self.setup_policy()
        
        val_transforms_cfg = complete_calvin_cfg.datamodule.transforms.val.rgb_static
        self.val_transforms = []
        for val_transform_cfg in val_transforms_cfg:
            self.val_transforms.append(hydra.utils.instantiate(val_transform_cfg))
        
        self.task_oracle = hydra.utils.instantiate(self.mode_eval_cfg.tasks)


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

        self.logger.info(f"Loaded policy from checkpoint {self.mode_eval_cfg.checkpoint}.")
    
    
    def evaluate_itrap(self):
        self.logger.info("Start generating evaluation sequences.")
        eval_sequences = get_sequences(num_sequences=self.mode_eval_cfg.num_sequences)
        self.logger.info("Done generating evaluation sequences.")

        results = []
        for seq_nr, (initial_state, eval_sequence) in tqdm(enumerate(eval_sequences), total=len(eval_sequences), desc="Evaluating policy"):
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
            tag = f"lh-eval_seq-nr-{seq_nr:04d}_global-step"
            caption = " | ".join(eval_sequence)
            self.rollout_video.new_video(tag, caption)

        success_counter = 0
        for subtask in eval_sequence:
            if self.record:
                self.rollout_video.new_subtask()
            
            success = self.rollout_subtask(subtask)

            if self.record:
                self.rollout_video.draw_outcome(success)

            if success:
                success_counter += 1
            else:
                break
        
        return success_counter
    

    def rollout_subtask(self, subtask):
        obs = self.env.get_obs()

        goal = self.lang_embeddings.get_lang_goal(subtask)

        self.policy.reset()
        start_info = self.env.get_info()

        success = False
        for step in tqdm(range(self.mode_eval_cfg.ep_len), desc=f"Rolling out policy for task {subtask}", leave=False):
            if step % self.policy.multistep == 0:
                static_img_start = self.env.cameras[0].render()[0]
                response = query_vlm(static_img_start, self.vlm_client, task_text=goal["lang_text"])
                static_traj_img = build_trajectory_image(static_img_start, response, save_traj_imgs=False, task_nr=step, task=subtask, output_dir=self.output_dir)

                transformed_static_traj_img = torch.tensor(static_traj_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                for transform in self.val_transforms:
                    transformed_static_traj_img = transform(transformed_static_traj_img)

            action = self.policy.step(rgb_static=transformed_static_traj_img, rgb_gripper=obs["rgb_obs"]["rgb_gripper"].to(self.device), goal=goal)
            obs, _, _, current_info = self.env.step(action)

            if self.record:
                normalized_rgb_static = self.env.cameras[0].render()[0] / 127.5 - 1 # normalize to [-1, 1]
                normalized_rgb_static = torch.tensor(normalized_rgb_static).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(self.device)
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
            self.logger.info(f"{subtask_nr} / 5 subtasks: {num_success} / {len(results)} sequences, SR: {success_rate * 100:.1f}%")
        
        avg_seq_len = np.mean(results)
        self.logger.info(f"Average successful sequence length: {avg_seq_len:.1f}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    itrap_evaluator = ItrapEvaluator()
    itrap_evaluator.evaluate_itrap()
