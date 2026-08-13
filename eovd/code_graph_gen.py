"""
代码图生成模块 - 使用Joern工具生成程序依赖图

本模块负责使用Joern工具为BigVul数据集中的代码片段生成程序依赖图(PDG)，
包括修复前(before)和修复后(after)的代码图。

主要功能：
- preprocess: 并行处理单个样本的代码图生成
- 支持多进程并行处理以提高效率
- 生成before和after两个版本的代码图

处理流程：
1. 读取BigVul数据集
2. 将数据分割为多个作业
3. 并行处理每个作业中的样本
4. 为每个样本生成修复前后的代码图
"""

import os
import argparse
import sys
import numpy as np
from glob import glob
from pathlib import Path
import pandas as pd
from helpers import utils
from helpers import joern
from data_pre import bigvul


_CFG_DATASET_OVERRIDE = None
_CFG_ONLY_BEFORE = False
_CFG_USE_EXISTING_C = False
_CFG_PARTITION = None


def _resolved_processed_subdir(dataset_root: str, partition: str, sub: str) -> Path:
    base = utils.processed_dir() / dataset_root
    if partition:
        return utils.get_dir(base / partition / sub)
    return utils.get_dir(base / sub)


def preprocess(row):
    """
    并行处理单个样本的代码图生成
    
    为给定的数据行生成修复前(before)和修复后(after)的代码图：
    1. 将代码写入C文件
    2. 使用Joern工具生成AST和程序依赖图
    3. 保存节点和边信息到JSON文件
    
    Args:
        row: DataFrame行，包含以下字段：
            - id: 样本ID
            - dataset: 数据集名称
            - before: 修复前代码
            - after: 修复后代码
            - diff: 代码差异
            
    Output Files:
        - {id}.c: 原始C代码文件
        - {id}.nodes.json: AST节点信息
        - {id}.edges.json: 程序依赖边信息
        
    Example:
        df = bigvul()
        row = df.iloc[180189]  # 论文示例
        row = df.iloc[177860]  # 边界情况1
        preprocess(row)
    """
    dataset_root = _CFG_DATASET_OVERRIDE if _CFG_DATASET_OVERRIDE else row["dataset"]

    part = None
    if _CFG_PARTITION and str(_CFG_PARTITION) != "all":
        part = str(_CFG_PARTITION)
    else:
        try:
            part = str(row.get("label", "")) if isinstance(row, dict) else str(getattr(row, "label", ""))
        except Exception:
            part = ""
        if part not in ("train", "val", "test"):
            part = ""

    savedir_before = _resolved_processed_subdir(dataset_root, part, "before")
    savedir_after = _resolved_processed_subdir(dataset_root, part, "after")

    # === 写入C代码文件 ===
    # 写入修复前的代码
    fpath1_default = savedir_before / f"{row['id']}.c"
    fpath1 = fpath1_default
    if _CFG_USE_EXISTING_C:
        candidates = [
            fpath1_default,
            utils.processed_dir() / dataset_root / "before" / f"{row['id']}.c",
        ]
        for cand in candidates:
            try:
                if os.path.exists(str(cand)):
                    fpath1 = cand
                    break
            except Exception:
                continue
        if not os.path.exists(str(fpath1)):
            return
    else:
        with open(fpath1, "w") as f:
            f.write(row["before"])
    
    # 写入修复后的代码（如果有差异）
    fpath2_default = savedir_after / f"{row['id']}.c"
    fpath2 = fpath2_default
    if not _CFG_ONLY_BEFORE:
        if _CFG_USE_EXISTING_C:
            candidates2 = [
                fpath2_default,
                utils.processed_dir() / dataset_root / "after" / f"{row['id']}.c",
            ]
            for cand in candidates2:
                try:
                    if os.path.exists(str(cand)):
                        fpath2 = cand
                        break
                except Exception:
                    continue
        else:
            if len(row["diff"]) > 0:
                with open(fpath2, "w") as f:
                    f.write(row["after"])

    # === 使用Joern生成代码图 ===
    # 为修复前代码生成图（如果不存在）
    if not os.path.exists(f"{fpath1}.edges.json"):
        joern.full_run_joern(fpath1, verbose=3)

    # 为修复后代码生成图（如果不存在且有差异）
    if not _CFG_ONLY_BEFORE:
        if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
            joern.full_run_joern(fpath2, verbose=3)


if __name__ == "__main__":
    """
    主程序入口：并行生成代码图
    
    支持多作业并行处理，通过命令行参数指定作业编号：
    - python code_graph_gen.py 1  # 处理第1个作业
    - python code_graph_gen.py 2  # 处理第2个作业
    - ...
    - python code_graph_gen.py 5  # 处理第5个作业
    
    处理流程：
    1. 读取BigVul数据集
    2. 将数据分割为5个作业
    3. 根据命令行参数选择要处理的作业
    4. 使用8个进程并行处理该作业中的样本
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "job",
        type=int,
        nargs="?",
        default=1,
        help=("Job index in [1..5], compatible with the original usage: python code_graph_gen.py 1"),
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Override output dataset root under processed_dir. Example: --dataset bigvul_sp/t01_xxx. "
            "Default: use the dataset field from bigvul()."
        ),
    )
    ap.add_argument(
        "--only_before",
        action="store_true",
        help="Only generate before graphs (skip after).",
    )
    ap.add_argument(
        "--partition",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="If set, read/write processed files under <dataset>/<partition>/. If 'all', use each row's label if available.",
    )
    ap.add_argument(
        "--use_existing_c",
        action="store_true",
        help="Do not overwrite .c files; only run Joern if *.edges.json is missing.",
    )
    ap.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional comma-separated list of sample ids to process (debugging).",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of samples to process after filtering (debugging).",
    )
    args = ap.parse_args()

    # === 作业配置 ===
    NUM_JOBS = 5  # 总作业数

    # 获取当前作业编号（从1开始，转换为0开始的索引）
    JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(args.job) - 1
    
    # === 数据准备 ===
    # 读取BigVul数据集
    df = bigvul()
    df = df.iloc[::-1]  # 反转数据顺序（从后往前处理）

    if str(getattr(args, "partition", "all")) != "all":
        try:
            df = df[df.label == str(args.partition)].copy()
        except Exception:
            pass

    if args.dataset and bool(getattr(args, 'use_existing_c', False)):
        try:
            part = str(getattr(args, 'partition', 'all'))
            parts = [part] if part in ("train", "val", "test") else ["train", "val", "test"]
            rows = []
            total_c = 0
            for p in parts:
                cand_dirs = [
                    utils.processed_dir() / str(args.dataset) / p / 'before',
                    utils.processed_dir() / p / str(args.dataset) / 'before',
                    utils.processed_dir() / p / 'before',
                ]
                fps = []
                for bd in cand_dirs:
                    fps = glob(str(Path(bd) / '*.c'))
                    if fps:
                        before_dir = bd
                        break
                else:
                    before_dir = cand_dirs[0]
                total_c += len(fps)
                for fp in fps:
                    try:
                        stem = os.path.splitext(os.path.basename(fp))[0]
                        rows.append({'id': int(stem), 'label': p, 'dataset': str(args.dataset), 'diff': ''})
                    except Exception:
                        continue

            if not rows:
                before_dir = utils.processed_dir() / str(args.dataset) / 'before'
                fps = glob(str(before_dir / '*.c'))
                total_c = len(fps)
                for fp in fps:
                    try:
                        stem = os.path.splitext(os.path.basename(fp))[0]
                        rows.append({'id': int(stem), 'label': part if part in ("train", "val", "test") else '', 'dataset': str(args.dataset), 'diff': ''})
                    except Exception:
                        continue

            existing_ids = sorted(set(int(r['id']) for r in rows))
            print(f"[*] dataset_scan: dataset={str(args.dataset)} | num_c_files={int(total_c)} | num_ids={len(existing_ids)}")

            if existing_ids:
                df2 = df[df.id.isin(existing_ids)].copy()
                if int(len(df2)) == 0:
                    df = pd.DataFrame(rows)
                else:
                    df = df2
        except Exception:
            pass

    if args.ids:
        try:
            ids = [int(x) for x in str(args.ids).split(',') if str(x).strip()]
            if ids:
                df = df[df.id.isin(ids)].copy()
        except Exception:
            pass

    if args.max_samples and int(args.max_samples) > 0:
        try:
            df = df.head(int(args.max_samples)).copy()
        except Exception:
            pass

    try:
        print(f"[*] will_process: n={int(len(df))} | first_ids={df.id.head(10).tolist()}")
    except Exception:
        pass

    _CFG_DATASET_OVERRIDE = args.dataset
    _CFG_ONLY_BEFORE = bool(getattr(args, 'only_before', False))
    _CFG_USE_EXISTING_C = bool(getattr(args, 'use_existing_c', False))
    _CFG_PARTITION = str(getattr(args, 'partition', 'all'))
    
    # 将数据分割为多个作业
    splits = np.array_split(df, NUM_JOBS)
    
    # === 并行处理 ===
    # 使用8个进程并行处理当前作业中的样本
    utils.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)