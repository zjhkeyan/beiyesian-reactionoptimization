# 代码与实验复现指南

## 1. 运行环境

在 `submission/code/` 目录下创建并激活锁定的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate reaction-optimization-submission
```

主要环境为 Python 3.11，依赖版本记录在 `environment.yml`；本地扩展包通过可编辑模式安装。

生成中文 PDF 前需安装文泉驿正黑字体。在 Ubuntu/Debian 环境可执行：

```bash
apt-get install -y fonts-wqy-zenhei
```

## 2. 验证数据契约

```bash
python scripts/validate_dataset_metadata.py
```

该脚本验证以下内容：

- `options.json` 仅包含可选择的优化变量；
- Suzuki 修正后的偶联产物作为固定 `ProductContext`，而不是候选变量；
- 在标准化产物结构未经审核前，不启用依赖产物结构的分子差分编码；
- Buchwald 元数据中包含固定催化剂 `palladium(2+) diacetate`。

## 3. 构建模型直接特征

```bash
python main/build_model_features.py \
  --dataset Suzuki \
  --input ../../Suzuki/suzuki_train.csv \
  --output /tmp/suzuki_model_features.csv
```

入口脚本直接读取 `metadata.json`，不维护第二套硬编码特征列表。对于 Suzuki，模型使用亲电试剂、亲核试剂、配体、碱、溶剂以及修正后的固定 `ProductContext`；有标签输入还会保留 `Yield`。固定产物上下文不会被视为可选择的实验条件。

对于 Buchwald，由于训练文件合并了五种产物，`Product` 保留为多任务上下文特征。固定反应物、溶剂和催化剂不作为可优化特征输出。

## 4. 复现最终优化算法

最终配置为高斯过程与低秩交互模型组成的在线加权集成。第一轮仅使用高斯过程后验均值，后续轮次使用稳定 LogEI。运行以下命令可复现两个数据集、20 个正式随机种子、每次 40 轮的全部 40 个优化过程：

```bash
python main/run_final_optimizer.py --dataset-root ../.. --jobs 6
```

运行器只允许离线 oracle 在候选被正式提交后返回对应标签。入口脚本按照实验来源分为两个阶段：随机种子 100、200、300 写入 `refinement_screening`，随机种子 400 至 2000 写入 `refinement_confirmation`。

方法名称、基础配置、策略覆盖项、随机种子列表、轮数预算和 t95 未达到阈值时的惩罚值均锁定在 `config/final_method.json`。

## 5. 生成正式轨迹和指标

从完整 JSON 运行记录重新生成 `.pt` 文件并重算全部指标：

```bash
python scripts/package_final_results.py
```

每个 `.pt` 文件均为只含标准 Python 容器和数值的字典，可使用 PyTorch 安全读取：

```python
import torch

run = torch.load("seed_100.pt", map_location="cpu", weights_only=True)
trajectory = run["trajectory"]
```

## 6. 校验提交目录

以下校验不读取测试目标值，仅使用 feature-only 候选文件：

```bash
python scripts/validate_submission.py \
  --submission-root .. \
  --dataset-root ../..
```

校验内容包括文件集合、随机种子、40 轮长度、候选索引范围、运行内索引唯一性、候选条件逐字段匹配以及 best-so-far 数值一致性。

## 7. 重新生成技术方案 PDF

```bash
python scripts/build_method_pdf.py
```

## 8. 防止数据泄漏的边界

优化代码在作出推荐之前只能读取训练表、无标签候选池、选项文件、元数据文件和已经查询点的顺序反馈。带标签测试表和完整搜索空间不得用于训练、特征构造、超参数选择、提示信息构造或候选排序。公开离线查询实验用于评估既定方法，不作为隐藏赛道泛化性能的证明。
