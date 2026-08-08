CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /DATA/troth/iTRAP/pretrained/qwen3_vl_8b/2026_05_01-re_itrap/merged_best \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2800 \
  --enforce-eager \
  --max-num-seqs 1 \
  --served-model-name qwen3_vl
