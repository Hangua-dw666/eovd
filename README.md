# A Systematic Evaluation of GNN Explainers for Vulnerability Detection: Faithfulness, Semantic Alignment, and Robustness

Welcome to our code repository 🌟.

Here we provide the PyTorch implementation of our paper 📚 "A Systematic Evaluation of GNN Explainers for Vulnerability Detection: Faithfulness, Semantic Alignment, and Robustness".

We are excited to share our work with the community and encourage collaborative exploration and discussion.

If you have any questions or encounter issues while using our code, please feel free to submit them through the `Issues` section.

Our implementation is also showcased at https://github.com/Hangua-dw666/eovd.



**Repository Overview:**
- [Environment Setup](#environment-setup) - Guide to setting up the environment required to run our code
- [Data Preparation](#data-preparation) - Instructions on how to prepare data for our model
- [Training GNN-based Vulnerability Detectors](#training-gnn-based-vulnerability-detectors) - Steps to train Graph Neural Network models for vulnerability detection
- [Explaining GNN-based Vulnerability Detectors](#explaining-gnn-based-vulnerability-detectors) - Steps to generate explanations for GNN model decisions
- [Evaluation Metrics (Three Dimensions)](#evaluation-metrics-three-dimensions) - Evaluation commands for faithfulness (PN/PS), localization (F1/TLC/FLC), and robustness (NI-SI-PC)

# Environment Setup

## CUDA Dependencies

Our work relies on specific versions of the CUDA Toolkit and cuDNN. Please ensure you have the following versions installed:
- **CUDA Toolkit**: version 11.7.0
- **cuDNN**: version 8.8.1 (compatible with CUDA 11.x)

Setup steps are as follows:
1. Download the required versions:
    - [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
    - [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
2. Install CUDA:
    ```shell
    sudo sh cuda_11.7.0_515.43.04_linux.run
    ```
3. Set up cuDNN:
    ```shell
    tar -zxvf cudnn-linux-x86_64-8.8.1.3_cuda11-archive.tar.xz
    sudo cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/include/cudnn.h  /usr/local/cuda-11.7/include
    sudo cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/lib/libcudnn*  /usr/local/cuda-11.7/lib64
    sudo chmod a+r /usr/local/cuda-11.7/include/cudnn.h  /usr/local/cuda-11.7/lib64/libcudnn*
    ```
4. Configure environment variables:
    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
    export PATH=$PATH:/usr/local/cuda-11.7/bin
    export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.7
    ```

## Python Library Dependencies

First, create a Conda environment:
```shell
conda create -n cfvd python=3.9
conda activate cfvd
```

Our implementation is based on specific versions of PyTorch and PyTorch Geometric:
- **PyTorch**: version 2.0.0
- **PyTorch Geometric**: version 2.3.1

Install them using the following commands:
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

Other required Python packages are as follows:
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

For this project, we utilize Joern to generate graphs for vulnerable and non-vulnerable code snippets. Note that Joern is an actively developed tool, and frequent updates may introduce functional changes. If you wish to seamlessly replicate our graph generation process, we recommend using Joern version 1.1.260. Here is how to set it up:
```shell
wget https://github.com/joernio/joern/releases/download/v1.1.260/joern-install.sh
chmod +x ./joern-install.sh
printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n'  | sudo ./joern-install.sh --interactive
```

For those attempting to use a newer version of Joern, or if you have specific questions about Joern's functionality, we recommend visiting Joern's official repository: [Joern GitHub Repository](https://github.com/joernio/joern). It provides comprehensive documentation and insights on code graph generation and more.

# Data Preparation

Our data preparation process is closely related to the [LineVd](https://github.com/davidhin/linevd) project. Below is a step-by-step guide to setting up and processing the dataset.

## Download

First, download the cleaned version of the Big-Vul dataset by obtaining the `MSR_data_cleaned.csv` file from [this link](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view). For details about Big-Vul, please visit its [official repository](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset).

After downloading, use the following commands to unzip and move the dataset to the appropriate location:
```shell
unzip MSR_data_cleaned.zip
rm MSR_data_cleaned.zip
mv MSR_data_cleaned.csv cfexplainer/storage/external
```

Run the following commands to configure the data storage path:
```shell
cd cfexplainer
export SINGSTORAGE=$(pwd)
```

## Data Preprocessing

To preprocess the Big-Vul dataset, run:

```shell
python data_pre.py
```

Upon successful execution, a `storage/cache` directory will be created. This serves as the location for storing cached data from script runs. Within this directory, you will find two subdirectories: `minimal_datasets` and `bigvul`. These are designed for fast access to the preprocessed Big-Vul dataset.


## Code Graph Generation

To generate code graphs using the Joern tool, execute the following commands in order:
```shell
python code_graph_gen.py 1
python code_graph_gen.py 2
python code_graph_gen.py 3
python code_graph_gen.py 4
python code_graph_gen.py 5
```

These commands will create a directory `storage/processed/bigvul` containing two directories: `before` and `after`. The `before` directory stores the code graphs of vulnerable code snippets, while the `after` directory stores the graphs of their corresponding fixed versions. For example, the files `before/177736.c`, `before/177736.nodes.json`, and `before/177736.edges.json` store the original source code, node attributes, and control/data flow edges for the sample with ID `177736` in the Big-Vul dataset.

Then, you can build the code graph dataset for the `train/valid/test` partitions by executing these commands:
```shell
python graph_dataset.py train
python graph_dataset.py val
python graph_dataset.py test
```
It will generate a directory `storage/cache/vul_graph_feat` to cache the graph features of the code. Meanwhile, three new directories `storage/processed/vul_graph_dataset/train_processed`, `storage/processed/vul_graph_dataset/val_processed`, and `storage/processed/vul_graph_dataset/test_processed` will be formed to store the partitioned code graph datasets. In these graph datasets, each individual code graph has a node feature matrix $\mathbf{X} \in \mathcal{R}^{n \times d}$ and an adjacency matrix $\mathbf{A} \in \mathcal{R}^{n \times n}$, where $n$ denotes the number of nodes and $d$ is the feature dimension. To save memory space, the adjacency matrix adopts the `edge index` data structure. The `edge index` is a $2 \times E$ matrix, where $E$ is the number of edges. The two rows represent:
- The first row contains the source nodes of the edges.
- The second row contains the target nodes of the edges.

For example, if the `edge index` contains a column $[3, 5]$, this indicates an edge from node 3 to node 5.

## Extracting Code Version Diffs

To extract the lines removed from the pre-fix version and the lines added to the post-fix version, specifically for vulnerable code, execute the following command:
```shell
python line_extract.py
```
This command will generate a file at `storage/processed/bigvul/eval/statement_labels.pkl` containing the extracted "deleted/added" lines.


# Training GNN-based Vulnerability Detectors

This work studies four Graph Neural Network-based vulnerability detectors: DeepWukong, Devign, IVDetect, and Reveal.

To train these detectors, execute the following commands:
```shell
python main.py --do_train --do_test --gnn_model DeepWukong --cuda_id 0
python main.py --do_train --do_test --gnn_model Devign --cuda_id 0
python main.py --do_train --do_test --gnn_model IVDetect --cuda_id 0
python main.py --do_train --do_test --gnn_model Reveal --cuda_id 0
```

Upon successful execution, the trained model checkpoints will be saved to the directory: `storage/cache/saved_models`. These checkpoints represent the GNN-based detectors that achieved the best performance on the validation set.

# Explaining GNN-based Vulnerability Detectors

Once the GNN-based vulnerability detectors are trained, you can use different explainers to explain the predictions of the GNN-based detectors. To evaluate the effectiveness of different explainers, our study uses six post-hoc explainers as baselines: `gnnexplainer`, `pgexplainer`, `subgraphx`, `gnn_lrp`, `deeplift`, `gradcam`, and `cfexplainer`.

Run the following commands to train these explainers on different GNN-based detectors:
```shell
python main.py --do_test --do_explain --gnn_model DeepWukong --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model Devign --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model IVDetect --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model Reveal --ipt_method specific_explainer --KM 8 --cuda_id 0
```


# Evaluation Metrics (Three Dimensions)

Our evaluation system comprehensively measures explainers from three dimensions:

1. **Dimension 1: Faithfulness Metrics** — PN / PS
2. **Dimension 2: Localization Metrics** — Traditional localization metrics (Accuracy / Precision / Recall / F1) and causal metrics (TLC, FLC)
3. **Dimension 3: Robustness Metrics** — NI-SI-PC three-dimensional framework

> **Note:** The metrics for Dimension 1 and Dimension 2 are output by a single `--eval_only` evaluation run; Dimension 3 requires first running `python generate_variants.py` to generate variant data, then evaluating with `--do_robust`. The "generate explanation cache" step for each dimension is identical to the one in [Explaining GNN-based Vulnerability Detectors](#explaining-gnn-based-vulnerability-detectors); if already executed, the cache can be directly reused without rerunning.

## Dimension 1: Faithfulness Metrics — PN / PS

**Description:** PN (Probability of Necessity) is the proportion of cases where the model's prediction flips after removing the top-$K_M$ important edges; PS (Probability of Sufficiency) is the proportion of cases where the model's prediction remains unchanged when only the top-$K_M$ important edges are retained. Both are output in a single run by the Dimension 2 `--eval_only` evaluation, for example:

```shell
python main.py --do_test --eval_only --gnn_model DeepWukong --ipt_method cfexplainer --KM 8 --cuda_id 0
```

## Dimension 2: Localization Metrics

**Description:** Traditional localization metrics (Accuracy / Precision / Recall / F1) measure localization accuracy based on the overlap between the explained code lines and the "deleted vulnerability lines"; the causal metrics TLC (Triggering Location Coverage) and FLC (Fixing Location Coverage) measure the coverage of the explanation over the vulnerability triggering lines VTS and the fix line mapping VFS, respectively. The metric computation is in the `eval_exp` function of `main.py`, and the results are written to `storage/cache/results/{gnn_model}/{ipt_method}.res`.

**Step 1: Generate explanation cache** (same as in [Explaining GNN-based Vulnerability Detectors](#explaining-gnn-based-vulnerability-detectors); skip if already executed)

The full commands are referenced in `explain.sh` (4 models × 6 explainers, generating the explanation cache once for each combination):

```shell
# ========== Phase 1: Generate explanation cache ==========

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

**Step 2: Evaluate by $K_M$** (outputs the PN/PS of Dimension 1 and all localization metrics of Dimension 2, $K_M \in \{2, 4, \dots, 20\}$)

```shell
# ========== Phase 2: Evaluate all KM from cache ==========

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

## Dimension 3: Robustness Metrics — NI-SI-PC Three-dimensional Framework

**Step 1: Generate semantically equivalent variant data (prerequisite, only once for the entire dataset)**

```shell
python generate_variants.py
```

**Step 2: Generate explanation cache**

```shell
# ========== Phase 1: Generate explanation cache ==========

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

**Step 3: Robustness evaluation** (full commands referenced in `robustness_eval.sh`, 4 models × 6 explainers × $K_M \in \{2, 4, \dots, 20\}$):

```shell
# ========== Phase 1 (skip if already run in Dimension 2): Generate explanation cache ==========
for MODEL in DeepWukong Devign IVDetect Reveal
do
    for EXPLAINER in subgraphx gradcam deeplift gnnexplainer cfexplainer pgexplainer
    do
        python main.py --do_test --do_explain --gnn_model $MODEL --ipt_method $EXPLAINER --cuda_id 0
    done
done

# ========== Phase 2: Robustness evaluation ==========
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

Or run the repository script with one command:

```shell
bash robustness_eval.sh phase1        # Generate explanation cache (once per model × explainer)
bash robustness_eval.sh phase2 0      # Evaluate NI/SI/PC by looping over K_M
```

> Note: The `MODELS` and `EXPLAINERS` arrays in `robustness_eval.sh` can be modified as needed.

# Project File Architecture

Below is an overview of the file structure to help you understand the organization of the repository:

## Repository Structure

```
counterfactual-vulnerability-detection
├─ README.md                       # Project description (this document)
├─ Framework.jpg                   # Framework figure (referenced by README)
├─ .gitignore                      # Upload filter rules (ignores environment/data/results)
├─ cfexplainer/                    # Core code directory (all uploaded)
│  ├─ cfvd                         # Conda environment export file (dependency list)
│  ├─ main.py                      # Unified entry point: train / test / explain / evaluate (--do_robust)
│  ├─ data_pre.py                  # Data preprocessing
│  ├─ code_graph_gen.py            # Joern code graph generation
│  ├─ graph_dataset.py             # Graph dataset construction (train/val/test)
│  ├─ line_extract.py              # Pre/post-fix code line diff extraction
│  ├─ generate_variants.py         # Semantically equivalent variant generation (robustness evaluation prerequisite)
│  ├─ explain.sh                   # Explanation + evaluation one-click script
│  ├─ robustness_eval.sh           # Robustness (NI-SI-PC) evaluation one-click script
│  ├─ models/                      # Detector and explainer implementations
│  │  ├─ vul_detector.py           #   Detector implementation (architecture selected via --gnn_model)
│  │  ├─ cfexplainer.py            #   The counterfactual explainer proposed in this work
│  │  ├─ gnnexplainer.py / pgexplainer.py / subgraphx.py
│  │  ├─ deeplift.py / gradcam.py / gnn_lrp.py / shapley.py / pcf_explainer.py
│  │  └─ graphcodebert-base/       #   ⚠ Pretrained weights, not included in the repository
│  └─ helpers/                     # Common utilities (all uploaded)
│     ├─ utils.py                  #   Cache/path utilities
│     ├─ joern.py                  #   Joern invocation wrapper
│     └─ git.py                    #   Git-related utilities
```

## Data and Results Storage Structure

All directories under `storage/` are generated by the commands in the "Data Preparation", "Training", "Explaining", etc. sections above, located at `cfexplainer/storage/`:

```
storage/
├─ external/                       # Raw data and toolchain (download/install yourself)
│  ├─ MSR_data_cleaned.csv/.zip    #   Big-Vul dataset (place after download)
│  ├─ joern-cli/                   #   Joern installation directory
│  └─ get_func_graph.scala         #   Joern graph generation script
├─ cache/
│  ├─ minimal_datasets/            #   Preprocessing cache (data_pre.py)
│  ├─ bigvul/                      #   Intermediate cache such as version diffs
│  ├─ vul_graph_feat/              #   Graph feature cache (graph_dataset.py)
│  ├─ saved_models/{model}/        #   Detector checkpoints (main.py --do_train)
│  │  └─ checkpoint-best-acc/model.bin
│  ├─ explainer_cache/{model}/{method}.pt   # Explanation cache (main.py --do_explain)
│  ├─ variant_data.pt              #   Semantic variant data (generate_variants.py)
│  └─ results/{model}/             #   Evaluation results
│     ├─ {method}.res              #   Faithfulness/localization metrics: PN, PS, F1, TLC, FLC, etc.
│     └─ {method}_robustness.res   #   Robustness metrics: NI / SI / PC
├─ processed/
│  ├─ bigvul/                      #   Joern code graphs (code_graph_gen.py)
│  │  ├─ before/                   #     Pre-fix (vulnerable) code graphs .c/.nodes.json/.edges.json
│  │  ├─ after/                    #     Post-fix code graphs
│  │  └─ eval/statement_labels.pkl #     Deleted/added line labels (line_extract.py)
│  └─ vul_graph_dataset/           #   Partitioned graph datasets (graph_dataset.py)
│     ├─ train_processed/ val_processed/ test_processed/
```
