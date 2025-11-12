from peft import PeftModel
from termcolor import colored
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def convert_model_plus_checkpoint_to_model(base_model_path, checkpoint_path, save_path):
    print(colored("Loading base model and checkpoint", "blue"))
    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    print(colored("Loading LoRA checkpoint", "blue"))
    lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print(colored("Merging LoRA weights into base model", "blue"))
    merged_model = lora_model.merge_and_unload()

    print(colored("Saving merged model and processor", "blue"))
    merged_model.save_pretrained(save_path, safe_serialization=True)
    processor.save_pretrained(save_path)


def get_prompt(task: str) -> str:
    return f"<image>In the image, please execute the command described in <prompt>{task.replace('_', ' ')}</prompt>. " \
            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(x_1, y_1), (x_2, y_2), " \
            "(x_3, y_3), <action>Open Gripper</action>, (x_4, y_4), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
            "an x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action."


if __name__ == "__main__":
    base_model_path = "Qwen/Qwen3-VL-8B-Instruct"  # Use HuggingFace model path
    checkpoint_path = "/DATA/troth/iTRAP/pretrained/qwen3_vl_8b-calvin_abc_img-1000/checkpoint-1400"
    save_path = "/DATA/troth/iTRAP/pretrained/qwen3_vl_8b-calvin_abc_img-1000/merged-1400"

    convert_model_plus_checkpoint_to_model(base_model_path, checkpoint_path, save_path)
