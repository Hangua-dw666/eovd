"""
漏洞图数据集处理模块

该模块实现了VulGraphDataset类，用于处理代码漏洞检测的图数据。
主要功能包括：
1. 从BigVul数据集中加载代码样本
2. 使用Joern工具提取代码的AST和CFG信息
3. 构建程序依赖图(PDG)
4. 将代码转换为图神经网络可处理的格式
"""

import sys, json, os
import os.path as osp
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import pickle as pkl
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, Batch
from tqdm.std import trange
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from helpers import utils
from helpers import joern
from data_pre import bigvul


class VulGraphDataset(Dataset):
    """
    漏洞图数据集类
    
    继承自PyTorch Geometric的Dataset类，用于处理代码漏洞检测的图数据。
    将代码转换为程序依赖图(PDG)，其中节点表示代码行，边表示数据依赖和控制依赖关系。
    """
    
    def __init__(self, root: Optional[str] = "storage/processed/vul_graph_dataset", 
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True, 
                 encoder = None, tokenizer = None, partition = None,
                 vulonly = False, sample = -1, splits = "default",
                 ):
        """
        初始化漏洞图数据集
        
        Args:
            root: 数据集存储根目录
            transform: 数据变换函数
            pre_transform: 预处理变换函数
            pre_filter: 预处理过滤函数
            log: 是否启用日志
            encoder: 预训练的语言模型编码器(如RoBERTa)
            tokenizer: 对应的分词器
            partition: 数据集分区(train/val/test)
            vulonly: 是否只包含漏洞样本
            sample: 采样数量(-1表示使用全部数据)
            splits: 数据集分割方式
        """
        # 创建存储目录
        os.makedirs(root, exist_ok=True)
        
        # 保存模型和分词器
        self.encoder = encoder
        # 提取词嵌入权重用于特征提取
        self.word_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy() if self.encoder is not None else None
        
        self.tokenizer = tokenizer
        self.partition = partition
        
        # 数据集配置参数
        self.vulonly = vulonly
        self.sample = sample
        self.splits = splits
        
        # 调用父类初始化
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
        # 加载已处理的数据
        self.data_list = torch.load(self.processed_paths[0])
        
    @property
    def processed_dir(self) -> str:
        """返回处理后的数据存储目录路径"""
        return osp.join(self.root, f'{self.partition}_processed')
    
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """返回处理后的文件名"""
        return 'data.pt'
    
    def process(self):
        """
        处理数据集的主要方法
        
        1. 加载BigVul数据集
        2. 过滤和平衡数据
        3. 验证样本有效性
        4. 提取特征并构建图
        5. 保存处理后的数据
        """
        # 获取已完成Joern处理的样本ID列表
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(utils.processed_dir() / "bigvul/before/*nodes*"))
        ]
        
        # 加载BigVul数据集
        self.df = bigvul(splits=self.splits)
        # 根据分区过滤数据(train/val/test)
        self.df = self.df[self.df.label == self.partition]
        # 只保留已完成Joern处理的样本
        self.df = self.df[self.df.id.isin(self.finished)]

        # 平衡数据集：确保漏洞样本和非漏洞样本数量相等
        vul = self.df[self.df.vul == 1]  # 漏洞样本
        nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)  # 随机采样等量的非漏洞样本
        self.df = pd.concat([vul, nonvul])

        # 调试模式：使用小样本
        if self.sample > 0:
            self.df = self.df.sample(self.sample, random_state=0)

        # 如果只处理漏洞样本
        if self.vulonly:
            self.df = self.df[self.df.vul == 1]

        # 验证样本有效性：检查Joern输出是否包含有效的行号信息
        self.df["valid"] = utils.dfmp(
            self.df, VulGraphDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # 创建索引到样本ID的映射
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        # 处理每个样本，构建图数据
        data_list = []
        for idx in trange(self.df.shape[0]):
            _id = self.idx2id[idx]
            # 提取特征并构建图
            result = self.feature_extraction(VulGraphDataset.itempath(_id))
            if result is None:
                print(f"Warning: feature_extraction returned None for id {_id}, skipping...")
                continue
            n, e = result  # n: 节点信息, e: 边信息
            
            # 构建PyTorch Geometric的Data对象
            x = np.array(list(n.subseq_feat.values))  # 节点特征矩阵
            edge_index = np.array(e)  # 边索引矩阵
            code_graph = Data(x=torch.FloatTensor(x), edge_index=torch.LongTensor(edge_index))
            
            # 添加漏洞标签信息
            n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)
            code_graph.__setitem__("_VULN", torch.Tensor(n["vuln"].astype(int).to_numpy()))  # 漏洞标签
            code_graph.__setitem__("_LINE", torch.Tensor(n["id"].astype(int).to_numpy()))     # 行号信息
            code_graph.__setitem__("_SAMPLE", torch.Tensor([_id] * len(n)))                  # 样本ID
            data_list.append(code_graph)

        # 保存处理后的数据
        print('Saving...')
        torch.save(data_list, self.processed_paths[0])
        
    def len(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """根据索引获取图数据"""
        return self.data_list[idx]
    
    @staticmethod
    def itempath(_id):
        """
        根据样本ID获取对应的C文件路径
        
        Args:
            _id: 样本ID
            
        Returns:
            对应的C文件路径
        """
        return utils.processed_dir() / f"bigvul/before/{_id}.c"
    
    @staticmethod
    def check_validity(_id):
        """
        检查样本是否有效（包含必要的节点和边信息）
        
        验证条件：
        1. 节点文件存在且包含多个不同行号的节点
        2. 边文件存在且包含REACHING_DEF或CDG类型的边
        
        Args:
            _id: 样本ID
            
        Returns:
            bool: 样本是否有效
        """
        valid = 0
        try:
            # 检查节点文件
            with open(str(VulGraphDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:  # 需要多个不同行号的节点
                            valid = 1
                            break
                if valid == 0:
                    return False
                    
            # 检查边文件
            with open(str(VulGraphDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])  # 提取边类型
                # 必须包含数据依赖(REACHING_DEF)或控制依赖(CDG)边
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(VulGraphDataset.itempath(_id)))
            return False
        
    def get_vuln_indices(self, _id):
        """
        获取样本中的漏洞行索引
        
        Args:
            _id: 样本ID
            
        Returns:
            dict: 漏洞行号到标签的映射字典
        """
        df = self.df[self.df.id == _id]
        removed = df.removed.item()  # 获取被修复的行号列表
        return dict([(i, 1) for i in removed])  # 将漏洞行标记为1
    
    def feature_extraction(self, filepath):
        """
        从C文件提取特征并构建程序依赖图(PDG)
        
        主要步骤：
        1. 检查缓存，如果存在则直接加载
        2. 使用Joern提取AST节点和边信息
        3. 处理代码序列，生成词嵌入特征
        4. 构建程序依赖图(PDG)
        5. 缓存结果
        
        Args:
            filepath: C文件路径
            
        Returns:
            tuple: (节点DataFrame, 边元组) 或 None（如果处理失败）
        """
        # 生成缓存文件名
        cache_name = "_".join(str(filepath).split("/")[-3:])
        cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(cache_name).stem
        
        # 尝试从缓存加载
        try:
            with open(cachefp, "rb") as f:
                return pkl.load(f)
        except:
            pass

        # 使用Joern提取节点和边信息
        try:
            nodes, edges = joern.get_node_edges(filepath)
        except Exception as e: 
            print(f"Error in get_node_edges for {filepath}: {e}")
            return None
            
        # === 处理代码序列特征 ===
        # 按代码长度排序，每组行号只保留最长的代码
        subseq = (
            nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
        )
        subseq = subseq[["lineNumber", "code", "local_type"]].copy()
        
        # 将类型信息添加到代码中
        subseq.code = subseq.local_type + " " + subseq.code
        subseq = subseq.drop(columns="local_type")
        
        # 清理空值和无效代码
        subseq = subseq[~subseq.eq("").any(axis='columns')]
        subseq = subseq[subseq.code != " "]
        subseq = subseq[subseq.code.notnull()]
        subseq.lineNumber = subseq.lineNumber.astype(int)
        subseq = subseq.sort_values("lineNumber")
        
        # 标准化代码格式
        subseq.code = subseq.code.apply(lambda s: ' '.join(s.split()))
        
        # 分词处理
        subseq.code = subseq.code.apply(lambda s: [self.tokenizer.cls_token] + self.tokenizer.tokenize(s) + [self.tokenizer.sep_token])
        subseq["code_feat"] = subseq.code.apply(lambda tokens: self.tokenizer.convert_tokens_to_ids(tokens))
        subseq.code = subseq.code.apply(lambda tokens: ' '.join(tokens))
        
        # 生成词嵌入特征（使用预训练模型的词嵌入）
        subseq.code_feat = subseq.code_feat.apply(lambda token_ids: np.mean(self.word_embeddings[token_ids], axis=0))
        
        # 分离代码文本和特征
        subseq_feat = subseq.drop(columns="code")
        subseq = subseq.drop(columns="code_feat")
        subseq = subseq.set_index("lineNumber").to_dict()["code"]
        subseq_feat = subseq_feat.set_index("lineNumber").to_dict()["code_feat"]

        # === 构建程序依赖图(PDG) ===
        # 处理节点信息
        nodesline = nodes[nodes.lineNumber != ""].copy()
        nodesline.lineNumber = nodesline.lineNumber.astype(int)
        nodesline = (
            nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
        )
        
        # 处理边信息
        edgesline = edges.copy()
        edgesline.innode = edgesline.line_in
        edgesline.outnode = edgesline.line_out
        nodesline.id = nodesline.lineNumber
        
        # 构建程序依赖图
        edgesline = joern.rdg(edgesline, "pdg")
        nodesline = joern.drop_lone_nodes(nodesline, edgesline)
        
        # 去重边
        edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
        
        # 将REACHING_DEF边重命名为DDG（数据依赖图）
        edgesline["etype"] = edgesline.apply(
            lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
        )
        
        # 过滤有效的边（确保节点ID是数值类型）
        edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
        edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
        
        # 构建无向边（添加反向边）
        edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
        edgesline_reverse.columns = ["outnode", "innode", "etype"]
        uedge = pd.concat([edgesline, edgesline_reverse])
        uedge = uedge[uedge.innode != uedge.outnode]  # 移除自环
        
        # 按节点和边类型分组，聚合出边节点
        uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
        uedge = uedge.reset_index()
        
        if len(uedge) > 0:
            # 透视表：将边类型作为列
            uedge = uedge.pivot(index="innode", columns="etype", values="outnode")
            
            # 确保DDG和CDG列存在
            if "DDG" not in uedge.columns:
                uedge["DDG"] = None
            if "CDG" not in uedge.columns:
                uedge["CDG"] = None
                
            uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
            uedge.columns = ["lineNumber", "control", "data"]
            
            # 转换集合为列表
            uedge.control = uedge.control.apply(
                lambda x: list(x) if isinstance(x, set) else []
            )
            uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
            
            # 转换为字典格式
            data = uedge.set_index("lineNumber").to_dict()["data"]
            control = uedge.set_index("lineNumber").to_dict()["control"]
        else:
            data = {}
            control = {}

        # === 生成最终的PDG ===
        pdg_nodes = nodesline.copy()
        pdg_nodes = pdg_nodes[["id"]].sort_values("id")
        
        # 添加代码序列和特征
        pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
        pdg_nodes["subseq_feat"] = pdg_nodes.id.map(subseq_feat).fillna("")
        
        # 添加依赖关系
        pdg_nodes["data"] = pdg_nodes.id.map(data)
        pdg_nodes["control"] = pdg_nodes.id.map(control)
        
        # 处理边信息
        pdg_edges = edgesline.copy()
        pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
        
        # 创建节点ID到索引的映射
        pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
        pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
        pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
        pdg_edges = pdg_edges.dropna()
        
        # 转换为PyTorch Geometric格式的边索引
        pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

        # 缓存结果
        with open(cachefp, "wb") as f:
            pkl.dump([pdg_nodes, pdg_edges], f)
        return pdg_nodes, pdg_edges


def collate(data_list):
    """
    将多个图数据合并为一个批次
    
    Args:
        data_list: 图数据列表
        
    Returns:
        Batch: 合并后的批次数据
    """
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == '__main__':
    """
    主程序入口：用于测试和演示VulGraphDataset的使用
    
    支持的语言模型类型：
    - gpt2: GPT-2模型
    - openai-gpt: OpenAI GPT模型  
    - bert: BERT模型
    - roberta: RoBERTa模型
    - distilbert: DistilBERT模型
    - t5: T5模型
    """
    # 定义支持的语言模型类型
    MODEL_CLASSES = {
        'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
        't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
    }
    
    # 配置模型参数
    model_type = "roberta"
    model_name_or_path = "models/graphcodebert-base"  # 使用本地模型路径
    tokenizer_name = "models/graphcodebert-base"      # 使用本地tokenizer路径
    
    # 从命令行参数获取数据集分区
    partition = sys.argv[1]
    
    # 加载模型和分词器
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    language_model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)
    
    # 创建数据集实例
    dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), encoder=language_model, tokenizer=tokenizer, partition=partition)
    
    # 打印数据集信息
    print(dataset)
    print(dataset.data_list[0])
    print(dataset.data_list[0].x)
    print(dataset.data_list[0].edge_index)
    print(dataset.data_list[0]._SAMPLE)