export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
# export MODEL_PATH="/home/Qwen3-8B"
# export SAVE_PATH="/home/Qwen3-8B-W8A8"

cd /root/pangu_infer/quantization/msit/msmodelslim/example/Qwen

# msmodelslim quant --model_path /opt/pangu/openPangu-Embedded-7B-V1.1 --save_path /home/Pangu-7B-W8A8 --device npu --model_type Qwen3-8B --quant_type w8a8 --trust_remote_code True
# msmodelslim quant --model_path /home/Qwen3-8B --save_path /home/Qwen3-8B-W8A8 --device npu --model_type Qwen3-8B --quant_type w8a8 --trust_remote_code True --is_dynamic True

python quant_qwen.py \
          --model_path /opt/pangu/openPangu-Embedded-7B-V1.1 \
          --save_directory /home/Pangu-7B-W8A8 \
          --device_type npu \
          --calib_file /root/pangu_infer/quantization/calib_data/calib_prompt.jsonl \
          --w_bit 8 \
          --a_bit 8 \
          --trust_remote_code True \
          --anti_method m2 \
          --pdmix False \
          --is_dynamic True \
          --disable_names "lm_head" \