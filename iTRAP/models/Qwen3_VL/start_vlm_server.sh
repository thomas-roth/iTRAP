CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /home/troth/code/hiwi/iTRAP/iTRAP/models/Qwen3_VL/pretrained/2026_01_22-both_cams-both_trajs-single_query \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2800 \
    --enforce-eager \
    --max-num-seqs 1 \
    --served-model-name qwen3_vl
