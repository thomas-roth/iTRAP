CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
 --model /home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen2_5_VL/pretrained/qwen2_5_vl_7b-calvin_abc \
 --host 0.0.0.0 --port 8000 \
 --gpu-memory-utilization 0.9 --max-model-len 32768 \
 --served-model-name qwen2_5_vl
