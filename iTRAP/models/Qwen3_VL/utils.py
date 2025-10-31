from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_model_plus_checkpoint_to_model(base_model_path, checkpoint_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)

    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)


def get_prompt(task: str) -> str:
    return f"<image>In the image, please execute the command described in <prompt>{task.replace('_', ' ')}</prompt>. " \
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(x_1, y_1), (x_2, y_2), " \
            "(x_3, y_3), <action>Open Gripper</action>, (x_4, y_4), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
            "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action."


if __name__ == "__main__":
    base_model_path = "/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/qwen3_vl_8b-calvin_abc"
    checkpoint_path = "/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/qwen3_vl_8b-calvin_abc/checkpoint-1400"
    save_path = "/home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/qwen3_vl_8b-calvin_abc/merged-1400"

    convert_model_plus_checkpoint_to_model(base_model_path, checkpoint_path, save_path)
