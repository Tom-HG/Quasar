"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
import torch


# from vllm import LLM, SamplingParams
# import json
# import vllm, importlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

# import sys
# import typing

# _original_infer_schema = torch.library.infer_schema

# def _patched_infer_schema(func, *args, **kwargs):
#     if hasattr(func, "__annotations__"):
#         for arg_name, arg_type in list(func.__annotations__.items()):
#             type_str = str(arg_type)
            
#             if type_str == "list[int]":
#                 print(f"   [Patching] Fixing '{arg_name}': list[int] -> List[int]")
#                 func.__annotations__[arg_name] = typing.List[int]
            
#             elif "list[int]" in type_str and ("Optional" in type_str or "Union" in type_str):
#                 print(f"   [Patching] Fixing '{arg_name}': {type_str} -> Optional[List[int]]")
#                 func.__annotations__[arg_name] = typing.Optional[typing.List[int]]

#     return _original_infer_schema(func, *args, **kwargs)

# # 挂载补丁
# torch.library.infer_schema = _patched_infer_schema


# # 1. 导入官方库 (此时它会注册 'ascend')
# import vllm_ascend.models.open_pangu
# import vllm_ascend.quantization.quant_config


# from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS


# if "ascend" in QUANTIZATION_METHODS:
#     QUANTIZATION_METHODS.remove("ascend")

# from my_open_pangu import OpenPanguMLP as myOpenPanguMLP
# from my_open_pangu import OpenPanguForCausalLM as myOpenPanguForCausalLM
# from my_quant_config import AscendQuantConfig as myAscendQuantConfig


# vllm_ascend.models.open_pangu.OpenPanguMLP = myOpenPanguMLP
# vllm_ascend.models.open_pangu.OpenPanguForCausalLM = myOpenPanguForCausalLM
# vllm_ascend.quantization.quant_config.AscendQuantConfig = myAscendQuantConfig

# 4. 启动 vLLM
from vllm import LLM, SamplingParams



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 
    print(f"Total questions: {len(questions)}, chunk size: {chunk_size}")
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    # model = EaModel.from_pretrained(
    #     base_model_path=base_model_path,
    #     ea_model_path=ea_model_path,
    #     total_token=args.total_token,
    #     depth=args.depth,
    #     top_k=args.top_k,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     # load_in_8bit=True,
    #     device_map="auto",
    #     use_eagle3=args.use_eagle3,
    # )

    # tokenizer = model.get_tokenizer()


    # if temperature > 1e-5:
    #     logits_processor = prepare_logits_processor(temperature=temperature)
    # else:
    #     logits_processor = None


    if args.use_quan and args.use_sd:
        print("Using both quantization and speculative decoding")
        model = LLM(
            model=base_model_path,
            trust_remote_code=True, 
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            quantization="ascend",
            speculative_config={
                "method":"ngram",
                "num_speculative_tokens":5,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 3,
            },
        )
    elif args.use_quan:
        print("Using quantization")
        model = LLM(
            model=base_model_path,
            trust_remote_code=True, 
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            quantization="ascend",
        )
    elif args.use_sd:
        print("Using speculative decoding")
        model = LLM(
            model=base_model_path,
            trust_remote_code=True, 
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            speculative_config={
                "method":"ngram",
                "num_speculative_tokens":5,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 3,
            },
        )
    else:
        print("Using base model")
        model = LLM(
            model=base_model_path,
            trust_remote_code=True, 
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_model_len=2048,
        )
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token)
    # tokenizer = LLM.get_tokenizer()

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     trust_remote_code=True,
    #     torch_dtype="auto",
    #     device_map="npu",
    #     local_files_only=True,
    #     # quantization_config=quantization_config
    # )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    # print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    npu_devices = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    print('NPU VISIBLE DEVICES:', npu_devices)
    device_config = model.llm_engine.vllm_config.device_config
    print(f"运行设备 (Device):  {device_config.device}")

    try:
        # 访问 Driver Worker 中的 Model Runner
        vllm_model = model.llm_engine.model_executor.driver_worker.model_runner.model

        print(f"\n 模型结构 (Model Structure): \n")
        print(vllm_model)
        
    except AttributeError:
        pass

    question = questions[0]

    conv = get_conversation_template("qwen3")

    # sys_prompt = "你必须严格遵守法律法规和社会道德规范。" \
    # "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。" \
    # "一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。例如，如果输入内容包含暴力威胁或色情描述，" \
    # "应返回错误信息：“您的输入包含不当内容，无法处理。”"

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        # messages = [
        #     {"role": "system",
        #      "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        # ]
        conv = get_conversation_template("qwen3")
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []

        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # try:
            torch.npu.synchronize()
            start_time = time.time()

            # output_ids, new_token, idx = model.eagenerate(
            #     torch.as_tensor(input_ids).cuda(),
            #     temperature=temperature,
            #     log=True,
            #     is_llama3=True,
            # )
            outputs = model.generate(prompt, sampling_params)
            output_ids = outputs[0].outputs[0].token_ids
            new_token = len(output_ids)
            idx = new_token 

            torch.npu.synchronize()
            total_time = time.time() - start_time

            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()



            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            # messages = [
            #     {"role": "system",
            #      "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            # ]
            # messages = [
            #     {"role": "system", "content": sys_prompt}, # define your system prompt here
            # ]
            conv = get_conversation_template("qwen3")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                # input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids
                prompt = conv.get_prompt()

                # try:
                torch.npu.synchronize()
                start_time = time.time()

                outputs = model.generate([prompt], sampling_params)
                output_ids = outputs[0].outputs[0].token_ids
                new_token = len(output_ids)
                idx = new_token 
                torch.npu.synchronize()
                total_time = time.time() - start_time
                # output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                # for special_token in tokenizer.special_tokens_map.values():
                #     if isinstance(special_token, list):
                #         for special_tok in special_token:
                #             output = output.replace(special_tok, "")
                #     else:
                #         output = output.replace(special_token, "")
                # output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
                # print("total_time:", total_time)
            # torch.npu.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle3/llama31chat/8B/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", 
                        type=str, 
                        default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                        # default="/home/Pangu-7B-W8A8",
                        # default="/home/Qwen3-8B",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="openpangu-7b-baseline")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="gsm8k",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=0,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        default=10,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="total-token = The total number of drafted tokens in the tree + 1",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="depth = The maximum number of draft length - 1",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of drafted tokens in each layer.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use_eagle3",
        action="store_true"
    )
    parser.add_argument(
        "--use_quan",
        action="store_true"
    )
    parser.add_argument(
        "--use_sd",
        action="store_true"
    )


    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"./data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"./data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.draft_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
    from vllm.spec_decode.spec_decode_worker import decode_step,accept_token
    print("Overall accept length:", accept_token/decode_step if decode_step>0 else 0)