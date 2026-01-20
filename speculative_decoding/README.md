# Model Inference - Speculative Inference

## Overview

The current software configuration is as follows:

| Component | Version | Path / Note |
| :--- | :--- | :--- |
| **CANN** | 8.1.rc1 | |
| **vLLM** | 0.9.2 | |
| **vLLM-Ascend** | 0.9.2rc1 | |
| **Torch** | 2.5.1 | |
| **Torch_NPU** | 2.5.1.post1 | |

| **Pangu-7B** | | `/opt/pangu/openPangu-Embedded-7B-V1.1` |
| **Qwen3-8B** | | `/home/Qwen3-8B` |


## Installation

```bash
pip install -r requirements.txt
pip install pydantic==2.11.9



## Inference

- Environment Setup

If you are performing inference in a Conda environment (non-Docker image), please activate the pangu environment first:

```bash
conda activate pangu
```

Then, set the required environment variables:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

- Run Tests
Execute the evaluation script:

```bash
bash pangu_7B_eval.sh
```

Note: The `--bench-name` parameter specifies the dataset to be tested.

- Evaluation Scripts
To run the Python evaluation scripts directly:
```bash
python evaluation.py
```
or
```bash
python evaluation_qwen.py
```

- Calculate Speedup
To generate the speedup results:

```bash
python speed.py
```

