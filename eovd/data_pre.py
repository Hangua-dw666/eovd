"""
数据预处理模块 - BigVul数据集预处理

本模块负责处理BigVul漏洞检测数据集，包括：
1. 数据集划分（训练/验证/测试）
2. 代码注释清理
3. 代码差异提取
4. 数据质量过滤
5. 缓存管理

主要功能：
- train_val_test_split_df: 数据集划分
- remove_comments: 代码注释清理
- bigvul: BigVul数据集主处理函数
"""

import os
import re
import random
import numpy as np
import pandas as pd
from helpers import utils
from helpers import git
from sklearn.model_selection import train_test_split


def train_val_test_split_df(df, idcol, labelcol):
    """
    将数据集划分为训练集、验证集和测试集
    
    使用分层抽样确保各集合中漏洞样本和非漏洞样本的比例保持一致
    
    Args:
        df: 输入数据框
        idcol: ID列名
        labelcol: 标签列名（用于分层抽样）
        
    Returns:
        DataFrame: 添加了'label'列的数据框，包含'train'、'val'、'test'标签
    """
    X = df[idcol]  # 获取ID列
    y = df[labelcol]  # 获取标签列
    
    # 定义数据集划分比例
    train_rat = 0.8  # 训练集占80%
    val_rat = 0.1    # 验证集占10%
    test_rat = 0.1   # 测试集占10%

    # 第一次划分：训练集 vs (验证集+测试集)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    
    # 第二次划分：验证集 vs 测试集
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    
    # 转换为集合以便快速查找
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        """
        根据ID确定样本属于哪个数据集
        
        Args:
            path: 样本ID
            
        Returns:
            str: 'train', 'val', 或 'test'
        """
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    # 为每个样本分配数据集标签
    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """
    从C/C++代码中移除注释
    
    支持移除以下类型的注释：
    - 单行注释：// 注释内容
    - 多行注释：/* 注释内容 */
    - 保留字符串字面量中的内容（单引号和双引号内）
    
    Args:
        text: 包含注释的代码文本
        
    Returns:
        str: 移除注释后的代码文本
    """

    def replacer(match):
        """
        正则表达式匹配替换函数
        
        Args:
            match: 正则表达式匹配对象
            
        Returns:
            str: 替换后的内容（注释用空格替换）
        """
        s = match.group(0)
        if s.startswith("/"):  # 匹配到注释
            return " "  # 用空格替换注释（保持代码结构）
        else:
            return s  # 保留其他内容（如字符串）

    # 正则表达式模式：
    # //.*?$ : 单行注释（//到行尾）
    # /\*.*?\*/ : 多行注释（/*...*/）
    # \'(?:\\.|[^\\\'])*\' : 单引号字符串
    # "(?:\\.|[^\\"])*" : 双引号字符串
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,  # 支持多行匹配
    )
    return re.sub(pattern, replacer, text)


def bigvul(minimal=True, sample=False, return_raw=False, splits="default"):
    """
    BigVul数据集主处理函数
    
    负责加载、清理、过滤和预处理BigVul漏洞检测数据集
    
    主要处理步骤：
    1. 加载原始数据集或缓存数据
    2. 移除代码注释
    3. 提取代码差异信息
    4. 数据质量过滤
    5. 数据集划分
    6. 保存处理结果
    
    Args:
        minimal (bool): 是否使用最小化数据集（从缓存加载）
        sample (bool): 是否使用采样数据集（仅用于测试）
        return_raw (bool): 是否返回原始数据（不进行后处理）
        splits (str): 数据集划分方式，支持：
            - "default": 默认随机划分
            - "crossproject-(linux|Chrome|Android|qemu)": 跨项目划分
            
    Returns:
        DataFrame: 处理后的数据集
        
    Note:
        特殊处理：ID为177860的样本不应包含before/after中的注释
    """
    # 设置缓存目录
    savedir = utils.get_dir(utils.cache_dir() / "minimal_datasets")
    
    # 尝试从缓存加载最小化数据集
    if minimal:
        try:
            # 加载预处理的parquet文件
            df = pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()

            # 加载元数据
            md = pd.read_csv(utils.cache_dir() / "bigvul/bigvul_metadata.csv")
            md.groupby("project").count().sort_values("id")

            # 加载预定义的数据集划分
            default_splits = utils.external_dir() / "bigvul_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            return df
        except Exception as E:
            print(E)
            pass
    
    # 加载原始数据集
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(utils.external_dir() / filename)
    df = df.rename(columns={"Unnamed: 0": "id"})  # 重命名ID列
    df["dataset"] = "bigvul"  # 标记数据集来源

    # === 代码清理阶段 ===
    # 移除函数代码中的注释
    df["func_before"] = utils.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = utils.dfmp(df, remove_comments, "func_after", cs=500)

    # 如果只需要原始数据，直接返回
    if return_raw:
        return df

    # === 代码差异提取阶段 ===
    # 提取代码差异信息（added/removed lines）
    cols = ["func_before", "func_after", "id", "dataset"]
    utils.dfmp(df, git._c2dhelper, columns=cols, ordr=False, cs=300)

    # 提取函数信息（如函数名、参数等）
    df["info"] = utils.dfmp(df, git.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # === 数据质量过滤阶段 ===
    # 只处理漏洞样本
    dfv = df[df.vul == 1]
    
    # 过滤1：移除没有添加或删除行的漏洞样本（可能是误标）
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    
    # 过滤2：移除函数结尾异常的样本（不以}或;结尾）
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    
    # 过滤3：移除以");"结尾的函数（可能是宏定义）
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # 过滤4：移除修改比例过高的样本（>70%）
    # 计算修改比例：修改行数 / 总差异行数
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    
    # 过滤5：移除过短的函数（<5行）
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    
    # 应用过滤条件到整个数据集
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # === 数据集划分阶段 ===
    df = train_val_test_split_df(df, "id", "vul")

    # === 保存处理结果 ===
    # 选择要保存的列
    keepcols = [
        "dataset",    # 数据集名称
        "id",         # 样本ID
        "label",      # 数据集标签（train/val/test）
        "removed",    # 被删除的行号
        "added",      # 被添加的行号
        "diff",       # 代码差异
        "before",      # 修复前代码
        "after",      # 修复后代码
        "vul",        # 漏洞标签
    ]
    
    # 保存最小化数据集
    df_savedir = savedir / f"minimal_bigvul_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",  # 处理复杂数据类型
        index=0,
        compression="gzip",      # 压缩存储
        engine="fastparquet",
    )
    
    # 保存元数据
    metadata_cols = df.columns[:17].tolist() + ["project"]
    df[metadata_cols].to_csv(utils.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    
    return df


if __name__ == "__main__":
    """
    主程序入口：运行BigVul数据集预处理脚本
    
    执行BigVul数据集的完整预处理流程，包括：
    - 数据加载和清理
    - 代码注释移除
    - 代码差异提取
    - 数据质量过滤
    - 数据集划分
    - 结果保存
    """
    bigvul()
