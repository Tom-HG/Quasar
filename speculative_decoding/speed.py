import json
from transformers import AutoTokenizer
import numpy as np
import argparse



parser = argparse.ArgumentParser()

parser.add_argument(
    "--bench-name",
    type=str,
    default="gsm8k",
    help="The name of the benchmark question set.",
)
parser.add_argument(
    "--jsonl-file",
    type=str,
    default="sd-pangu-7b-fp16-temperature-0.0",
    help="The name of the benchmark question set.",
)

parser.add_argument(
    "--jsonl-file-base",
    type=str,
    default="baseline-pangu-7b-fp16-temperature-0.0",
    help="The name of the benchmark question set.",
)


args = parser.parse_args()


tokenizer=AutoTokenizer.from_pretrained("/opt/pangu/openPangu-Embedded-7B-V1.1",trust_remote_code=True)
jsonl_file = "./data/{}/model_answer/{}.jsonl".format(args.bench_name, args.jsonl_file)
jsonl_file_base = "./data/{}/model_answer/{}.jsonl".format(args.bench_name, args.jsonl_file_base)

data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



# print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())