# A Systematic Evaluation of GNN Explainers for Vulnerability Detection: Faithfulness, Semantic Alignment, and Robustness

欢迎来到我们的代码仓库 🌟。

这里我们提供了我们论文 📚 "A Systematic Evaluation of GNN Explainers for Vulnerability Detection: Faithfulness, Semantic Alignment, and Robustness" 的PyTorch实现。

我们很高兴与社区分享我们的工作，并鼓励协作探索和讨论。

如果您有任何问题或在使用我们的代码时遇到问题，请随时通过 `Issues` 部分提交。

我们的实现也在上展示。



**仓库概览：**
- [环境配置](#环境配置) - 设置运行我们代码所需环境的指南
- [数据准备](#数据准备) - 如何为我们的模型准备数据的说明
- [训练基于GNN的漏洞检测器](#训练基于gnn的漏洞检测器) - 训练图神经网络模型进行漏洞检测的步骤
- [解释基于GNN的漏洞检测器](#解释基于gnn的漏洞检测器) - 生成GNN模型决策解释的步骤
- [评估指标（三维度）](#评估指标三维度) - 忠实性（PN/PS）、定位（F1/TLC/FLC）与鲁棒性（NI-SI-PC）的评估命令

# 环境配置

## CUDA依赖

我们的工作依赖于特定版本的CUDA工具包和cuDNN。请确保您安装了以下版本：
- **CUDA工具包**: 版本 11.7.0
- **cuDNN**: 版本 8.8.1 (与CUDA 11.x兼容)

设置步骤如下：
1. 下载所需版本：
    - [CUDA工具包归档](https://developer.nvidia.com/cuda-toolkit-archive)
    - [cuDNN归档](https://developer.nvidia.com/rdp/cudnn-archive)
2. 安装CUDA：
    ```shell
    sudo sh cuda_11.7.0_515.43.04_linux.run
    ```
3. 设置cuDNN：
    ```shell
    tar -zxvf cudnn-linux-x86_64-8.8.1.3_cuda11-archive.tar.xz
    sudo cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/include/cudnn.h  /usr/local/cuda-11.7/include
    sudo cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/lib/libcudnn*  /usr/local/cuda-11.7/lib64
    sudo chmod a+r /usr/local/cuda-11.7/include/cudnn.h  /usr/local/cuda-11.7/lib64/libcudnn*
    ```
4. 配置环境变量：
    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
    export PATH=$PATH:/usr/local/cuda-11.7/bin
    export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.7
    ```

## Python库依赖

首先创建一个Conda环境：
```shell
conda create -n cfvd python=3.9
conda activate cfvd
```

我们的实现基于特定版本的PyTorch和Pytorch Geometric：
- **PyTorch**: 版本 2.0.0
- **Pytorch Geometric**: 版本 2.3.1

使用以下命令安装它们：
```shell
pip install https://download.pytorch.org/whl/cu117/torch-2.0.0%2Bcu117-cp39-cp39-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu117/torchvision-0.15.1%2Bcu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/pyg_lib-0.2.0%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_cluster-1.6.1%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.1%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_sparse-0.6.17%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_spline_conv-1.2.2%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_geometric
```

其他必需的Python包如下：
```shell
pip install numpy==1.24.3
pip install pandas==2.0.1
pip install scikit-learn==1.2.2
pip install tensorboard==2.13.0
pip install transformers==4.29.1
pip install tqdm==4.65.0
pip install scipy==1.10.1
pip install graphviz==0.20.1
pip install unidiff==0.7.5
pip install dive-into-graphs==1.1.0
pip install captum==0.2.0
pip install matplotlib==3.7.1
pip install rdkit
```

## Joern

对于这个项目，我们利用Joern为易受攻击和不易受攻击的代码段生成图。需要注意的是，Joern是一个积极开发的工具，经常更新可能会引入功能变化。如果您希望无缝复制我们的图生成过程，我们建议使用Joern版本1.1.260。以下是设置方法：
```shell
wget https://github.com/joernio/joern/releases/download/v1.1.260/joern-install.sh
chmod +x ./joern-install.sh
printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n'  | sudo ./joern-install.sh --interactive
```

对于那些尝试使用较新Joern版本的用户，或者如果您对Joern的功能有具体疑问，我们建议访问Joern的官方仓库：[Joern GitHub仓库](https://github.com/joernio/joern)。它提供了关于代码图生成等的全面文档和见解。

# 数据准备

我们的数据准备过程与[LineVd](https://github.com/davidhin/linevd)项目密切相关。以下是设置和处理数据集的逐步指南。

## 下载

首先，通过从[此链接](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view)获取`MSR_data_cleaned.csv`文件来下载Big-Vul数据集的清理版本。有关Big-Vul的详细信息，请访问其[官方仓库](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)。

下载后，使用以下命令解压并将数据集移动到适当位置：
```shell
unzip MSR_data_cleaned.zip
rm MSR_data_cleaned.zip
mv MSR_data_cleaned.csv cfexplainer/storage/external
```

运行以下命令配置数据存储路径：
```shell
cd cfexplainer
export SINGSTORAGE=$(pwd)
```

## 数据预处理

要预处理Big-Vul数据集，请运行：

```shell
python data_pre.py
```

成功执行后，将创建一个`storage/cache`目录。这用作存储脚本运行缓存数据的位置。在此目录中，您将找到两个子目录：`minimal_datasets`和`bigvul`。这些设计用于快速访问预处理的Big-Vul数据集。


## 代码图生成

要使用Joern工具生成代码图，请按顺序执行以下命令：
```shell
python code_graph_gen.py 1
python code_graph_gen.py 2
python code_graph_gen.py 3
python code_graph_gen.py 4
python code_graph_gen.py 5
```

这些命令将创建一个目录`storage/processed/bigvul`，其中包含两个目录`before`和`after`。`before`目录存放易受攻击代码片段的代码图，而`after`目录存放其相应修复版本的图。例如，文件`before/177736.c`、`before/177736.nodes.json`和`before/177736.edges.json`存储Big-Vul数据集中ID为`177736`的样本的原始源代码、节点属性和控制/数据流边。

然后，您可以为`train/valid/test`分区构建代码图数据集，执行这些命令：
```shell
python graph_dataset.py train
python graph_dataset.py val
python graph_dataset.py test
```
它将生成一个目录`storage/cache/vul_graph_feat`来缓存代码的图特征。同时，将形成三个新目录`storage/processed/vul_graph_dataset/train_processed`、`storage/processed/vul_graph_dataset/val_processed`和`storage/processed/vul_graph_dataset/test_processed`，存放分区的代码图数据集。在这些图数据集中，每个单独的代码图都有一个节点特征矩阵$\mathbf{X} \in \mathcal{R}^{n \times d}$和一个邻接矩阵$\mathbf{A} \in \mathcal{R}^{n \times n}$，其中$n$表示节点数，$d$是特征维度。为了节省内存空间，邻接矩阵采用`edge index`数据结构。`edge index`是一个$2 \times E$矩阵，其中$E$是边的数量。两行表示：
- 第一行包含边的源节点。
- 第二行包含边的目标节点。

例如，如果`edge index`包含一列$[3, 5]$，这表示从节点3到节点5的边。

## 提取代码版本差异

要提取从修复前版本中删除的行和添加到修复后版本中的行，特别是针对易受攻击的代码，请执行以下命令：
```shell
python line_extract.py
```
此命令将在`storage/processed/bigvul/eval/statement_labels.pkl`生成一个文件，包含提取的"删除/添加"行。


# 训练基于GNN的漏洞检测器

这项工作研究了四种基于图神经网络的漏洞检测器：DeepWukong、Devign、IVDetect 和 Reveal。

要训练这些检测器，请执行以下命令：
```shell
python main.py --do_train --do_test --gnn_model DeepWukong --cuda_id 0
python main.py --do_train --do_test --gnn_model Devign --cuda_id 0
python main.py --do_train --do_test --gnn_model IVDetect --cuda_id 0
python main.py --do_train --do_test --gnn_model Reveal --cuda_id 0
```

成功执行后，训练好的模型检查点将保存到目录：`storage/cache/saved_models`。这些检查点代表在验证集上获得最佳性能的基于GNN的检测器。

# 解释基于GNN的漏洞检测器

一旦基于GNN的漏洞检测器训练完成，您可以使用不同的解释器来解释基于GNN的检测器的预测结果。为了评估不同解释器的效果，我们的研究以六个事后解释器作为基线：`gnnexplainer`、`pgexplainer`、`subgraphx`、`gnn_lrp`、`deeplift`、`gradcam`和`cfexplainer`。

运行以下命令在不同基于GNN的检测器上训练这些解释器：
```shell
python main.py --do_test --do_explain --gnn_model DeepWukong --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model Devign --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model IVDetect --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model Reveal --ipt_method specific_explainer --KM 8 --cuda_id 0
```


# 评估指标（三维度）

我们的评估体系从三个维度全面衡量解释器：

1. **维度一：忠实性指标（Faithfulness）** — PN / PS
2. **维度二：定位指标（Localization）** — 传统定位指标（Accuracy / Precision / Recall / F1）与因果指标（TLC、FLC）
3. **维度三：鲁棒性指标（Robustness）** — NI-SI-PC 三维框架

> **说明：** 维度一与维度二的指标由同一次 `--eval_only` 评估输出；维度三需先 `python generate_variants.py` 生成变体数据，再以 `--do_robust` 评估。各维度"生成解释缓存"的步骤与[解释基于GNN的漏洞检测器](#解释基于gnn的漏洞检测器)一节相同，已执行过可直接复用缓存，无需重复运行。

## 维度一：忠实性指标（Faithfulness）—— PN / PS

**文字说明：** PN（Probability of Necessity）为移除 top-$K_M$ 重要边后模型预测发生翻转的比例；PS（Probability of Sufficiency）为仅保留 top-$K_M$ 重要边时模型预测保持不变的比例。两者由维度二的 `--eval_only` 评估一次性输出，例如：

```shell
python main.py --do_test --eval_only --gnn_model DeepWukong --ipt_method cfexplainer --KM 8 --cuda_id 0
```

## 维度二：定位指标（Localization）

**文字说明：** 传统定位指标（Accuracy / Precision / Recall / F1）基于解释出的代码行与"被删除的漏洞行"的重叠衡量定位精度；因果指标 TLC（Triggering Location Coverage）与 FLC（Fixing Location Coverage）分别衡量解释对漏洞触发行 VTS 与修复行映射 VFS 的覆盖率。指标计算见 `main.py` 的 `eval_exp`，结果写入 `storage/cache/results/{gnn_model}/{ipt_method}.res`。

**步骤1：生成解释缓存**（与[解释基于GNN的漏洞检测器](#解释基于gnn的漏洞检测器)一节相同；已执行过可跳过）

完整命令参考 `explain.sh`（4 模型 × 6 解释器，每个组合生成一次解释缓存）：

```shell
# ========== 阶段1：生成解释缓存 ==========

# DeepWukong
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model DeepWukong --ipt_method $method --cuda_id 0
done

# Devign
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model Devign --ipt_method $method --cuda_id 0
done

# IVDetect
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model IVDetect --ipt_method $method --cuda_id 0
done

# Reveal
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model Reveal --ipt_method $method --cuda_id 0
done
```

**步骤2：按 $K_M$ 评估**（输出维度一的 PN/PS 与维度二的全部定位指标，$K_M \in \{2, 4, \dots, 20\}$）

```shell
# ========== 阶段2：从缓存评估所有 KM ==========

# DeepWukong
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model DeepWukong --ipt_method $method --KM $KM --cuda_id 0
    done
done

# Devign
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model Devign --ipt_method $method --KM $KM --cuda_id 0
    done
done

# IVDetect
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model IVDetect --ipt_method $method --KM $KM --cuda_id 0
    done
done

# Reveal
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model Reveal --ipt_method $method --KM $KM --cuda_id 0
    done
done
```

## 维度三：鲁棒性指标（Robustness）—— NI-SI-PC 三维框架

**步骤1：生成语义等价变体数据（前置，全数据集仅需一次）**

```shell
python generate_variants.py
```

**步骤2：生成解释缓存**

```shell
# ========== 阶段1：生成解释缓存 ==========

# DeepWukong
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model DeepWukong --ipt_method $method --cuda_id 0
done

# Devign
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model Devign --ipt_method $method --cuda_id 0
done

# IVDetect
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model IVDetect --ipt_method $method --cuda_id 0
done

# Reveal
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model Reveal --ipt_method $method --cuda_id 0
done
```

**步骤3：鲁棒性评估**（完整命令参考 `robustness_eval.sh`，4 模型 × 6 解释器 × $K_M \in \{2, 4, \dots, 20\}$）：

```shell
# ========== 阶段1（如维度二已跑过可跳过）：生成解释缓存 ==========
for MODEL in DeepWukong Devign IVDetect Reveal
do
    for EXPLAINER in subgraphx gradcam deeplift gnnexplainer cfexplainer pgexplainer
    do
        python main.py --do_test --do_explain --gnn_model $MODEL --ipt_method $EXPLAINER --cuda_id 0
    done
done

# ========== 阶段2：鲁棒性评估 ==========
for MODEL in DeepWukong Devign IVDetect Reveal
do
    for EXPLAINER in subgraphx gradcam deeplift gnnexplainer cfexplainer pgexplainer
    do
        for KM in {2..20..2}
        do
            python main.py --do_test --eval_only --do_robust --gnn_model $MODEL --ipt_method $EXPLAINER --KM $KM --cuda_id 0
        done
    done
done
```

或一键运行仓库脚本：

```shell
bash robustness_eval.sh phase1        # 生成解释缓存（每模型×解释器一次）
bash robustness_eval.sh phase2 0      # 按 K_M 循环评估 NI/SI/PC
```

> 注：`robustness_eval.sh` 中 `MODELS` 与 `EXPLAINERS` 两个数组可按需修改。

# 项目文件架构

以下是文件结构的概述，帮助您理解仓库的组织结构：

## 仓库结构

```
counterfactual-vulnerability-detection
├─ README.md                       # 项目说明（本文档）
├─ Framework.jpg                   # 框架图（README 引用）
├─ .gitignore                      # 上传过滤规则（忽略环境/数据/结果）
├─ cfexplainer/                    # 核心代码目录（全部上传）
│  ├─ cfvd                         # conda 环境导出文件（依赖清单）
│  ├─ main.py                      # 统一入口：训练 / 测试 / 解释 / 评估（--do_robust）
│  ├─ data_pre.py                  # 数据预处理
│  ├─ code_graph_gen.py            # Joern 代码图生成
│  ├─ graph_dataset.py             # 图数据集构建（train/val/test）
│  ├─ line_extract.py              # 修复前后代码行差异提取
│  ├─ generate_variants.py         # 语义等价变体生成（鲁棒性评估前置）
│  ├─ explain.sh                   # 解释 + 评估一键脚本
│  ├─ robustness_eval.sh           # 鲁棒性（NI-SI-PC）评估一键脚本
│  ├─ models/                      # 检测器与解释器实现
│  │  ├─ vul_detector.py           #   Detector 实现（按 --gnn_model 选择架构）
│  │  ├─ cfexplainer.py            #   本文提出的反事实解释器
│  │  ├─ gnnexplainer.py / pgexplainer.py / subgraphx.py
│  │  ├─ deeplift.py / gradcam.py / gnn_lrp.py / shapley.py / pcf_explainer.py
│  │  └─ graphcodebert-base/       #   ⚠ 预训练权重，不入库
│  └─ helpers/                     # 公共工具（全部上传）
│     ├─ utils.py                  #   缓存/路径工具
│     ├─ joern.py                  #   Joern 调用封装
│     └─ git.py                    #   git 相关工具
```

## 数据与结果存放结构

`storage/` 下所有目录均由上文「数据准备」「训练」「解释」等命令生成，位于 `cfexplainer/storage/`：

```
storage/
├─ external/                       # 原始数据与工具链（需自行下载/安装）
│  ├─ MSR_data_cleaned.csv/.zip    #   Big-Vul 数据集（下载后放入）
│  ├─ joern-cli/                   #   Joern 安装目录
│  └─ get_func_graph.scala         #   Joern 图生成脚本
├─ cache/
│  ├─ minimal_datasets/            #   预处理缓存（data_pre.py）
│  ├─ bigvul/                      #   版本差异等中间缓存
│  ├─ vul_graph_feat/              #   图特征缓存（graph_dataset.py）
│  ├─ saved_models/{模型}/         #   检测器 checkpoint（main.py --do_train）
│  │  └─ checkpoint-best-acc/model.bin
│  ├─ explainer_cache/{模型}/{方法}.pt   # 解释缓存（main.py --do_explain）
│  ├─ variant_data.pt              #   语义变体数据（generate_variants.py）
│  └─ results/{模型}/              #   评估结果
│     ├─ {方法}.res                #   忠实性/定位指标：PN、PS、F1、TLC、FLC 等
│     └─ {方法}_robustness.res     #   鲁棒性指标：NI / SI / PC
├─ processed/
│  ├─ bigvul/                      #   Joern 代码图（code_graph_gen.py）
│  │  ├─ before/                   #     修复前（漏洞）代码图 .c/.nodes.json/.edges.json
│  │  ├─ after/                    #     修复后代码图
│  │  └─ eval/statement_labels.pkl #     删除/添加行标签（line_extract.py）
│  └─ vul_graph_dataset/           #   分区图数据集（graph_dataset.py）
│     ├─ train_processed/ val_processed/ test_processed/
```

