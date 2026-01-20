# 盘古模型推理 - 投机推理

## 概述

当前软件配置如下：

- CANN: 8.1.rc1
- vllm: 0.9.2
- vllm-ascend: 0.9.2rc1
- torch: 2.5.1
- torch_npu: 2.5.1.post1

并且在/opt/panguy目录下已经预下载盘古模型权重：

- 盘古1B：/opt/pangu/openPangu-Embedded-1B-V1.1
- 盘古7B：/opt/pangu/openPangu-Embedded-7B-V1.1



## 安装



```bash
pip install -r requriment.txt
pip install pydantic===2.11.9
```



## 7B模型推理

- 设置环量

如果您是基于conda环境进行推理（非docker镜像），请先切换到`pangu`环境:

```bash
conda activate pangu
```

然后设置相应环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

- 测试运行

```bash
bash pangu_7B_eval.sh
```

其中，`--bench-name`为测试的数据集。

- 得到加速比结果

```bash
python speed.py
```

