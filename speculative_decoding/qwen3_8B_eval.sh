source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_USE_V1=0

# python evaluation.py \
#     --base-model-path /home/Qwen3-8B \
#     --model-id baseline-qwen3-8b-fp16 \
#     --question-end 10 \
#     --bench-name humaneval > vllm_sd.log 2>&1

# python evaluation.py \
#     --base-model-path /home/Qwen3-8B \
#     --model-id sd-qwen3-8b-fp16 \
#     --question-end 10 \
#     --use_sd \
#     --bench-name humaneval >> vllm_sd.log 2>&1

# python speed.py --bench-name humaneval \
#     --jsonl-file sd-qwen3-8b-fp16-temperature-0.0 \
#     --jsonl-file-base baseline-qwen3-8b-fp16-temperature-0.0

python evaluation_qwen.py \
    --base-model-path /home/Qwen3-8B-W8A8 \
    --model-id quan-qwen3-8b \
    --question-end 10 \
    --use_quan \
    --bench-name humaneval > vllm_quan.log 2>&1

python evaluation_qwen.py \
    --base-model-path /home/Qwen3-8B-W8A8 \
    --model-id quan_sd-qwen3-8b \
    --question-end 10 \
    --use_sd \
    --use_quan \
    --bench-name humaneval >> vllm_quan.log 2>&1

python speed.py --bench-name humaneval \
    --jsonl-file quan_sd-qwen3-8b-temperature-0.0 \
    --jsonl-file-base quan-qwen3-8b-temperature-0.0  >> vllm_quan.log 2>&1