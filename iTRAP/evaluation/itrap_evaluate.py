from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from collections import Counter
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb

sys.path.append(str(Path(__file__).absolute().parents[2]))
sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "flower_vla_calvin"))
from iTRAP.evaluation.utils import setup_vlm_client, query_vlm, extract_gripper_points_and_actions, draw_trajectory_onto_image, save_trajectory_image
from iTRAP.models.flower_vla_calvin.flower.evaluation.utils import get_default_mode_and_env, get_env_state_for_initial_condition
from iTRAP.models.flower_vla_calvin.flower.evaluation.multistep_sequences import get_sequences
from iTRAP.models.flower_vla_calvin.flower.rollout.rollout_video import RolloutVideo



class ItrapEvaluator:
    def __init__(self):
        with hydra.initialize(config_path="../models/flower_vla_calvin/conf"):
            self.flower_eval_cfg = hydra.compose(config_name="eval_calvin")
            complete_calvin_cfg = hydra.compose(config_name="config_calvin")
        
        self.device = self.flower_eval_cfg.device
        if self.device != "cpu":
            self.device = torch.device(f"cuda:{self.flower_eval_cfg.device}")

        self.output_dir = Path(self.flower_eval_cfg.log_dir) / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
        os.makedirs(self.output_dir, exist_ok=False)
        if self.flower_eval_cfg.record:
            os.makedirs(self.output_dir / "rollout_videos", exist_ok=False)
        if self.flower_eval_cfg.save_traj_imgs:
            os.makedirs(self.output_dir / "traj_imgs", exist_ok=False)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        _stream_handler = logging.StreamHandler(stream=sys.stdout)
        _stream_handler.setFormatter(logging.Formatter("\033[94m[%(asctime)s][%(levelname)s]\033[0m %(message)s"))
        self.logger.addHandler(_stream_handler)
        _file_handler = logging.FileHandler(self.output_dir / "itrap_evaluate.log", mode="w")
        _file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        self.logger.addHandler(_file_handler)

        if self.flower_eval_cfg.wandb.log:
            wandb.init(
                project=self.flower_eval_cfg.wandb.project,
                group=self.flower_eval_cfg.wandb.group,
                name=self.flower_eval_cfg.wandb.name,
                entity=self.flower_eval_cfg.wandb.entity,
                config=OmegaConf.to_object(self.flower_eval_cfg),
                dir=self.output_dir
            )

        if self.flower_eval_cfg.record:
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
        
        self.task_oracle = hydra.utils.instantiate(self.flower_eval_cfg.tasks)


    def setup_policy(self):
        seed_everything(0, workers=True)

        sys.path.append(str(Path(__file__).absolute().parents[1] / "models" / "flower_vla_calvin" / "calvin_env"))
        self.policy, self.env, _, self.lang_embeddings = get_default_mode_and_env(
            self.flower_eval_cfg.train_folder,
            self.flower_eval_cfg.dataset_path,
            self.flower_eval_cfg.checkpoint,
            eval_cfg_overwrite=self.flower_eval_cfg.eval_cfg_overwrite,
            device_id=self.flower_eval_cfg.device,
        )

        self.policy = self.policy.to(self.device)

        if self.flower_eval_cfg.num_sampling_steps is not None:
            self.policy.num_sampling_steps = self.flower_eval_cfg.num_sampling_steps
        if self.flower_eval_cfg.multistep is not None:
            self.policy.multistep = self.flower_eval_cfg.multistep

        self.policy.eval()

        self.policy_global_step = int(self.flower_eval_cfg.checkpoint.split('=')[-1].split('_')[0]) * 1000 # 1000 steps per epoch

        self.logger.info(f"Loaded policy from checkpoint {self.flower_eval_cfg.checkpoint}.")
    
    
    def evaluate_itrap(self):
        self.logger.info("Start generating evaluation sequences.")
        eval_sequences = get_sequences(num_sequences=self.flower_eval_cfg.num_sequences)
        self.logger.info("Done generating evaluation sequences.")

        results = []
        for seq_nr, (initial_state, eval_sequence) in tqdm(enumerate(eval_sequences), total=len(eval_sequences), desc="Evaluating policy"):
            success_counter = self.evaluate_sequence(seq_nr, initial_state, eval_sequence)
            results.append(success_counter)

            if self.flower_eval_cfg.record:
                self.rollout_video.log(self.policy_global_step)
        
        self.log_success_rate(results)

        if self.flower_eval_cfg.wandb.log:
            wandb.finish()


    def evaluate_sequence(self, seq_nr, initial_state, eval_sequence):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        if self.flower_eval_cfg.record:
            tag = f"lh-eval_seq-nr-{seq_nr:03d}_global-step"
            caption = " | ".join(eval_sequence)
            self.rollout_video.new_video(tag, caption)

        success_counter = 0
        for subtask_nr, subtask in enumerate(eval_sequence):
            if self.flower_eval_cfg.record:
                self.rollout_video.new_subtask()
            
            success = self.rollout_subtask(subtask, seq_nr, subtask_nr)

            if self.flower_eval_cfg.record:
                self.rollout_video.draw_outcome(success)

            if success:
                success_counter += 1
            else:
                break
        
        return success_counter
    

    def rollout_subtask(self, subtask, seq_nr, subtask_nr):
        obs = self.env.get_obs()

        goal = self.lang_embeddings.get_lang_goal(subtask)

        local_rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0

        # get trajectory points & actions from initial state of scene & robot (static camera image untransformed as render() used instead of get_obs())
        static_img_start = self.env.cameras[0].render()[0].squeeze()
        vlm_response = query_vlm(static_img_start, self.vlm_client, subtask)
        traj_gripper_points, traj_gripper_actions = extract_gripper_points_and_actions(vlm_response, error_logger=self.logger)

        if self.flower_eval_cfg.save_traj_imgs:
            untransformed_static_traj_img = draw_trajectory_onto_image(static_img_start, traj_gripper_points, traj_gripper_actions)
            save_trajectory_image(untransformed_static_traj_img, subtask, local_rank, seq_nr, subtask_nr, step_nr=0, root_output_dir=self.output_dir)

        self.policy.reset()
        start_info = self.env.get_info()

        if self.flower_eval_cfg.record:
            # update video with initial state
            static_img = self.env.cameras[0].render()[0].squeeze()
            static_traj_img = draw_trajectory_onto_image(static_img, traj_gripper_points, traj_gripper_actions)
            normalized_static_traj_img = static_traj_img / 127.5 - 1 # normalize to [-1, 1]
            self.rollout_video.update(torch.tensor(normalized_static_traj_img).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(self.device)) # shape: B, F, C, H, W

        success = False
        for step in tqdm(range(self.flower_eval_cfg.ep_len), desc=f"Rolling out policy for task {subtask}", leave=False):
            if step == self.flower_eval_cfg.ep_len / 2:
                # query vlm again to help robot out of wrong state
                untransformed_static_img = self.env.cameras[0].render()[0].squeeze()
                vlm_response = query_vlm(untransformed_static_img, self.vlm_client, subtask)
                traj_gripper_points, traj_gripper_actions = extract_gripper_points_and_actions(vlm_response, error_logger=self.logger)

                if self.flower_eval_cfg.save_traj_imgs:
                    untransformed_static_traj_img = draw_trajectory_onto_image(untransformed_static_img, traj_gripper_points, traj_gripper_actions)
                    save_trajectory_image(untransformed_static_traj_img, subtask, local_rank, seq_nr, subtask_nr, step, self.output_dir)

            if step % self.policy.multistep == 0:
                # model predicts multistep actions per step => only draw trajectory once per multistep
                untransformed_static_img = self.env.cameras[0].render()[0].squeeze()
                untransformed_static_traj_img = draw_trajectory_onto_image(untransformed_static_img, traj_gripper_points, traj_gripper_actions)

                # apply transforms to trajectory image
                transformed_static_traj_img = torch.tensor(untransformed_static_traj_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                for transform in self.val_transforms:
                    transformed_static_traj_img = transform(transformed_static_traj_img)
                
                obs["vis_image"] = transformed_static_traj_img.unsqueeze(0)

            action = self.policy.step(obs, goal)
            obs, _, _, current_info = self.env.step(action)

            if self.flower_eval_cfg.record:
                static_img = self.env.cameras[0].render()[0].squeeze()
                static_traj_img = draw_trajectory_onto_image(static_img, traj_gripper_points, traj_gripper_actions)
                normalized_static_traj_img = static_traj_img / 127.5 - 1 # normalize to [-1, 1]
                self.rollout_video.update(torch.tensor(normalized_static_traj_img).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(self.device))

            # check if current steps solves task
            current_task_info = self.task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break

        if self.flower_eval_cfg.record:
            self.rollout_video.add_language_instruction(goal["lang_text"])
        
        return success
    

    def log_success_rate(self, results):
        results_counter = Counter(results)
        total_successes_counter = Counter()
        success_rates = []
        for subtask_nr in range(1, 6):
            num_success = sum(results_counter[j] for j in reversed(range(subtask_nr, 6)))
            total_successes_counter[subtask_nr] = num_success
            success_rate = num_success / len(results)
            success_rates.append(success_rate)
            self.logger.info(f"{subtask_nr} / 5 subtasks: {num_success} / {len(results)} sequences, SR: {success_rate * 100:.1f}%")
        
        avg_seq_len = np.mean(results)
        chain_sr = {i+1: sr for i, sr in enumerate(success_rates)}
        task_info = {}
        for subtask_nr in range(1, 6):
            task_info[f"subtask_{subtask_nr}"] = {"success": total_successes_counter[subtask_nr], "total": len(results)}

        self.logger.info(f"Average successful sequence length: {avg_seq_len:.1f}")
        wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    itrap_evaluator = ItrapEvaluator()
    itrap_evaluator.evaluate_itrap()
