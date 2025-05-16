CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
 --model /home/troth/bt/iTRAP/iTRAP/models/Qwen2-VL/pretrained/qwen2_vl_lora_sft_calvin \
 --host 0.0.0.0 --port 8000 \
 --gpu-memory-utilization 0.5 --max-model-len 32768 \
 --served-model-name qwen2-vl
 