# Inference - Quantization

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
| **Qwen-8B** | | `/home/Qwen3-8B` |

## MSIT Usage

Run the quantization script:

```bash
bash pangu_quant.sh
```
## References
MSIT Example: MSIT Qwen Example on Gitee \url{[https://gitee.com/ascend/msit/tree/master/msmodelslim/example/Qwen](https://gitee.com/ascend/msit/tree/master/msmodelslim/example/Qwen)}
