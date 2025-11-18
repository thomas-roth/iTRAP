CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model /home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/qwen3_vl_8b-calvin_abc-2025_11_17-both_cams/merged_best \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2800 \
    --enforce-eager \
    --max-num-seqs 1 \
    --served-model-name qwen3_vl
