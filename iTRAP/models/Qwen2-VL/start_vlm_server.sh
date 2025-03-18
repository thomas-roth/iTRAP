CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /DATA/models/calvin_lora/qwen2_vl_lora_sft_calvin \
                                                                    --host 0.0.0.0 --port 8000 \
                                                                    --gpu-memory-utilization 0.95 --max-model-len 25000 \
                                                                    --served-model-name qwen2-vl
