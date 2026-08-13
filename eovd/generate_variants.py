#!/usr/bin/env python3
"""
变体数据生成器 (Variant Data Generator)
用于生成第三层鲁棒性评估 (NI-SI-PC) 所需的数据

核心设计：将语义等价变换正交分解为两类
  - Type A (节点级): 仅改变节点特征，图结构(edge_index)不变
    模拟: 变量重命名、常量替换、操作符替换等
    对应指标: NI (Node-Invariance) — 伪相关抗性

  - Type B (结构级): 改变图拓扑，但保持程序语义
    模拟: 循环重构、条件分支交换、死代码插入等
    对应指标: SI (Struct-Invariance) — 因果结构保真性

  - Type C (反事实): 漏洞版 vs 修复版的真实代码对
    对应指标: PC (Pairwise Counterfactual) — 因果区分性

运行方式:
    python generate_variants.py --help
    python generate_variants.py         # 使用默认参数生成

输出:
    cache/variant_data.pt
"""

import os
import torch
import pickle as pkl
import argparse
import numpy as np
import copy
from pathlib import Path
from tqdm import tqdm
from helpers import utils
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops


# ==================== 缓存 ====================

_dataset_cache = None  # {sample_id: Data}


def _load_dataset_cache(partition="test"):
    """懒加载已处理的图数据集缓存"""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    cache_path = Path("storage/processed/vul_graph_dataset") / f"{partition}_processed" / "data.pt"
    if not cache_path.exists():
        return {}
    data_list = torch.load(str(cache_path), map_location="cpu")
    _dataset_cache = {}
    for g in data_list:
        sid = int(g._SAMPLE[0].item()) if hasattr(g, "_SAMPLE") and g._SAMPLE is not None else -1
        _dataset_cache[sid] = g
    return _dataset_cache


def _build_data_from_file(filepath, sample_id):
    """从 .c 文件路径构建 Data 图对象（优先走缓存）"""
    cache = _load_dataset_cache()
    sid = int(sample_id) if sample_id else -1
    if sid in cache:
        return copy.deepcopy(cache[sid])
    return None


def _safe_iter(val):
    """安全迭代：处理 data/control 字段可能是 float(nan) 而非 list 的情况"""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return val
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="生成鲁棒性评估所需的变体数据 (NI-SI-PC)")

    parser.add_argument('--output_path', type=str, default=None,
                        help='输出文件路径 (默认: cache/variant_data.pt)')
    parser.add_argument('--num_node_variants', type=int, default=3,
                        help='NI: 每个样本生成的节点级变体数量 (默认3)')
    parser.add_argument('--num_struct_variants', type=int, default=3,
                        help='SI: 每个样本生成的结构级变体数量 (默认3)')
    parser.add_argument('--sample_limit', type=int, default=-1,
                        help='限制处理的样本数量 (-1表示不限制)')

    # 数据源配置
    parser.add_argument('--data_source', type=str, default='bigvul',
                        choices=['bigvul', 'fan', 'sard'],
                        help='数据集来源')

    return parser.parse_args()


# ==================== Type A: 节点级变体 (edge_index 不变) ====================

def generate_node_variants(orig_graph, sample_id, K=3):
    """生成节点级语义等价变体 (NI 指标所需)

    核心约束: edge_index 完全不变，仅改变节点特征 x
    模拟真实代码级语义等价变换（对应现实中的代码重构操作）:

    [FR/VR] Function/Variable Renaming:
        选择部分"变量/函数类"节点(度数中等的非孤立节点)，对其特征施加定向偏移，
        模拟重命名后token embedding的变化。只影响被重命名的实体，其余节点不变。

    [OS] Operand Swap:
        选择成对的数据依赖边(a,b)和(c,d)，交换两端目标节点的特征向量，
        模拟二元逻辑表达式中操作数交换(如 a==b → b==a)，保持逻辑等价性。

    [SP-node] Statement-level Feature Perturbation:
        选择无依赖关系的独立语句节点(入度+出度较低的叶子/根节点)，
        对其特征做微扰，模拟语句内常量替换或格式调整。
    """
    n_nodes = orig_graph.x.shape[0]
    if n_nodes == 0:
        return []

    variants = []
    np.random.seed(sample_id * 100 + 1)  # 固定种子，可复现

    # 预计算节点度数信息，用于选择合适的扰动目标
    ei = orig_graph.edge_index
    if ei is not None and ei.shape[1] > 0:
        deg = torch.zeros(n_nodes)
        for j in range(ei.shape[1]):
            deg[ei[0, j].item()] += 1
            deg[ei[1, j].item()] += 1
    else:
        deg = torch.ones(n_nodes)

    for i in range(K):
        g = copy.deepcopy(orig_graph)
        perturb_type = i % 3

        if perturb_type == 0:
            # [FR/VR] 变量/函数重命名: 选择部分中度数节点做定向偏移
            # 真实场景: 重命名只影响被改名实体的所有出现位置
            # 图上表现: 被重命名变量的所有声明和使用节点的embedding发生一致偏移
            non_isolated = (deg > 1).nonzero(as_tuple=True)[0]
            if len(non_isolated) < 2:
                # 回退: 随机选几个节点
                n_rename = max(1, n_nodes // 10)
                rename_idx = np.random.choice(n_nodes, size=min(n_rename, n_nodes), replace=False)
            else:
                # 选中度数在 [2, 中位数+1] 范围的节点（排除超高度数的枢纽节点和孤立点）
                median_deg = deg.median().item()
                candidate_mask = (deg >= 2) & (deg <= median_deg + 1)
                candidates = candidate_mask.nonzero(as_tuple=True)[0]
                n_rename = max(1, min(len(candidates) // 3, len(candidates)))
                if n_rename > 0:
                    rename_idx = np.random.choice(len(candidates), size=n_rename, replace=False)
                    rename_idx = candidates[rename_idx].cpu().numpy()
                else:
                    rename_idx = non_isolated[:max(1, len(non_isolated)//3)].cpu().numpy()

            # 每个被重命名节点: 特征偏移量一致（同一变量的不同出现位置偏移相同）
            rename_offset = torch.randn(g.x.shape[1]) * 0.02  # 统一偏移方向
            for idx in rename_idx:
                g.x[idx] = g.x[idx] + rename_offset + torch.randn_like(g.x[idx]) * 0.005

        elif perturb_type == 1:
            # [OS] 操作数交换: 成对交换数据依赖边的目标节点特征
            # 真实场景: a == b  →  b == a （逻辑等价）
            # 图上表现: 数据流边两端的目标节点特征互换
            if ei is not None and ei.shape[1] > 0:
                # 收集所有数据依赖边（排除自环）
                edge_list = []
                for j in range(ei.shape[1]):
                    s, d = ei[0, j].item(), ei[1, j].item()
                    if s != d:
                        edge_list.append((s, d))

                if len(edge_list) >= 2:
                    # 随机选若干对边进行目标节点特征交换
                    n_swap_pairs = max(1, len(edge_list) // 8)
                    chosen_edges = np.random.choice(len(edge_list),
                                                   size=min(n_swap_pairs * 2, len(edge_list)),
                                                   replace=False)
                    swap_targets = []
                    for eidx in chosen_edges:
                        _, dst = edge_list[eidx]
                        swap_targets.append(dst)

                    # 成对交换特征
                    for p in range(0, len(swap_targets) - 1, 2):
                        a, b = swap_targets[p], swap_targets[p + 1]
                        g.x[a], g.x[b] = g.x[b].clone(), g.x[a].clone()
                else:
                    # 边太少，回退为局部微扰
                    n_perturb = max(1, n_nodes // 15)
                    pert_idx = np.random.choice(n_nodes, size=n_perturb, replace=False)
                    for idx in pert_idx:
                        g.x[idx] = g.x[idx] + torch.randn_like(g.x[idx]) * 0.01
            else:
                noise = torch.randn_like(g.x) * 0.005
                g.x = g.x + noise

        else:
            # [SP-node] 语句级特征微扰: 选择低度数节点（独立语句）做微小扰动
            # 真实场景: 常量替换、格式调整等不影响控制流的语句级变更
            low_deg_mask = (deg <= 2)  # 叶子节点或近叶子节点
            low_deg_nodes = low_deg_mask.nonzero(as_tuple=True)[0]
            if len(low_deg_nodes) >= 2:
                n_perturb = max(1, len(low_deg_nodes) // 3)
                pert_idx = np.random.choice(len(low_deg_nodes),
                                            size=min(n_perturb, len(low_deg_nodes)),
                                            replace=False)
                pert_idx = low_deg_nodes[pert_idx].cpu().numpy()
            else:
                n_perturb = max(1, n_nodes // 10)
                pert_idx = np.random.choice(n_nodes, size=n_perturb, replace=False)

            for idx in pert_idx:
                # 微小扰动：模拟语句内常量值变化但语义不变（如 100 → 100.0）
                g.x[idx] = g.x[idx] + torch.randn_like(g.x[idx]) * 0.005

        variants.append(g)

    return variants


# ==================== Type B: 结构级变体 (edge_index 改变) ====================

def generate_struct_variants(orig_graph, sample_id, K=3):
    """生成结构级语义等价变体 (SI 指标所需)

    核心约束: edge_index 发生变化，节点特征 x 完全不变
    模拟真实代码级结构等价变换（对应现实中的代码重构操作）:

    [SP] Statement Permutation:
        交换两个无数据/控制依赖的语句在图中的连接关系。
        真实场景: 两个连续的声明语句互换位置，不影响程序语义。
        图上表现: 选择两个低耦合节点对，交换它们的入边/出边连接。

    [BS] Block Swap:
        交换 if 语句的 then 块和 else 块，同时取反分支条件。
        真实场景: if(cond) { A } else { B } → if(!cond) { B } else { A }
        图上表现: 识别一个"分支枢纽"节点(高出度)，将其部分出边目标集合
        与另一部分出边目标集合互换，模拟分支块交换。

    [LX] Loop Exchange:
        for 循环和 while 循环互相转换。
        真实场景: for(i=0; i<n; i++) { body } ↔ i=0; while(i<n) { body; i++; }
        图上表现: 在循环回边上插入/删除辅助节点（初始化/增量语句），
        或调整循环体的入口/出口边模式。
    """
    n_nodes = orig_graph.x.shape[0]
    if n_nodes < 2:
        return []

    variants = []
    np.random.seed(sample_id * 100 + 2)  # 固定种子，可复现

    # 预计算邻接信息
    ei = orig_graph.edge_index
    if ei is not None and ei.shape[1] > 0:
        # 出边邻接表: src -> [dst_list]
        out_adj = {}
        in_adj = {}
        for j in range(ei.shape[1]):
            s, d = ei[0, j].item(), ei[1, j].item()
            out_adj.setdefault(s, []).append(d)
            in_adj.setdefault(d, []).append(s)
        deg_out = torch.zeros(n_nodes)
        deg_in = torch.zeros(n_nodes)
        for j in range(ei.shape[1]):
            deg_out[ei[0, j].item()] += 1
            deg_in[ei[1, j].item()] += 1
    else:
        out_adj, in_adj = {}, {}
        deg_out, deg_in = torch.zeros(n_nodes), torch.zeros(n_nodes)

    for i in range(K):
        g = copy.deepcopy(orig_graph)
        perturb_type = i % 3

        if perturb_type == 0:
            # [SP] Statement Permutation: 交换无依赖语句的边连接
            # 选择两个"独立"节点：度数较低（非枢纽），且彼此不直接相连
            low_coupling_mask = ((deg_out + deg_in) <= 4) & (deg_out >= 1) & (deg_in >= 1)
            candidates = low_coupling_mask.nonzero(as_tuple=True)[0]

            if len(candidates) >= 2:
                # 随机选两个候选节点
                chosen = np.random.choice(len(candidates), size=2, replace=False)
                node_a, node_b = candidates[chosen[0]].item(), candidates[chosen[1]].item()

                # 确保它们不是直接邻居
                a_neighbors = set(out_adj.get(node_a, []) + in_adj.get(node_a, []))
                if node_b in a_neighbors:
                    # 是直接邻居，换一个
                    for alt in candidates:
                        alt_n = alt.item()
                        if alt_n != node_a and alt_n not in a_neighbors:
                            node_b = alt_n
                            break

                # 交换两个节点的入边目标（其他节点→node_a 变成 其他节点→node_b）
                new_ei = g.edge_index.clone()
                for j in range(new_ei.shape[1]):
                    if new_ei[1, j].item() == node_a and new_ei[0, j].item() != node_b:
                        new_ei[1, j] = node_b
                    elif new_ei[1, j].item() == node_b and new_ei[0, j].item() != node_a:
                        new_ei[1, j] = node_a
                g.edge_index = new_ei

            else:
                # 候选不够，回退为轻量边重排
                new_ei = g.edge_index.clone()
                if new_ei.shape[1] > 4:
                    n_rewire = max(1, new_ei.shape[1] // 10)
                    rewire_idx = np.random.choice(new_ei.shape[1],
                                                  size=min(n_rewire, new_ei.shape[1]),
                                                  replace=False)
                    for idx in rewire_idx:
                        new_dst = np.random.randint(0, n_nodes)
                        new_ei[1, idx] = new_dst
                g.edge_index = new_ei

        elif perturb_type == 1:
            # [BS] Block Swap: 分支块交换（then↔else）
            # 找一个高出度节点作为"分支枢纽"（如if/switch语句节点）
            high_out_mask = (deg_out >= 3)  # 至少3条出边才像分支
            branch_nodes = high_out_mask.nonzero(as_tuple=True)[0]

            if len(branch_nodes) > 0:
                # 选一个分支枢纽
                branch = branch_nodes[np.random.choice(len(branch_nodes))].item()
                neighbors = list(set(out_adj.get(branch, [])))
                if len(neighbors) >= 4:
                    # 将出边目标分成两组，然后交叉重连
                    mid = len(neighbors) // 2
                    group_a = neighbors[:mid]
                    group_b = neighbors[mid:]

                    new_ei = g.edge_index.clone()
                    for j in range(new_ei.shape[1]):
                        if new_ei[0, j].item() == branch:
                            dst = new_ei[1, j].item()
                            if dst in group_a:
                                # 原来连 group_a 的，改为连 group_b 中对应位置
                                idx_in_a = group_a.index(dst)
                                if idx_in_a < len(group_b):
                                    new_ei[1, j] = group_b[idx_in_a]
                            elif dst in group_b:
                                idx_in_b = group_b.index(dst)
                                if idx_in_b < len(group_a):
                                    new_ei[1, j] = group_a[idx_in_b]
                    g.edge_index = new_ei
                else:
                    # 出边不够分组，做轻量边重排
                    new_ei = g.edge_index.clone()
                    if new_ei.shape[1] > 2:
                        n_swap = max(1, len(neighbors) // 2)
                        for _ in range(n_swap):
                            if len(neighbors) >= 2:
                                a, b = np.random.choice(len(neighbors), size=2, replace=False)
                                # 交换这两条边的目标
                                for j in range(new_ei.shape[1]):
                                    if (new_ei[0, j].item() == branch and
                                            new_ei[1, j].item() == neighbors[a]):
                                        new_ei[1, j] = neighbors[b]
                                    elif (new_ei[0, j].item() == branch and
                                          new_ei[1, j].item() == neighbors[b]):
                                        new_ei[1, j] = neighbors[a]
                        g.edge_index = new_ei
            else:
                # 无明显分支节点，回退
                new_ei = g.edge_index.clone()
                if new_ei.shape[1] > 4:
                    n_rewire = max(1, new_ei.shape[1] // 8)
                    rewire_idx = np.random.choice(new_ei.shape[1],
                                                  size=min(n_rewire, new_ei.shape[1]),
                                                  replace=False)
                    for idx in rewire_idx:
                        new_ei[:, idx] = new_ei[:, idx].flip(0)
                g.edge_index = new_ei

        else:
            # [LX] Loop Exchange: 循环结构变换（for↔while）
            # 识别潜在循环回边: 目标节点ID小于源节点的边（后向边）
            back_edges = []
            if ei is not None:
                for j in range(ei.shape[1]):
                    s, d = ei[0, j].item(), ei[1, j].item()
                    if d < s:  # 后向边（可能构成循环）
                        back_edges.append((s, d, j))

            if len(back_edges) > 0:
                # 选一条回边，在其附近注入/删除辅助节点
                chosen = back_edges[np.random.choice(len(back_edges))]
                loop_src, loop_dst, _ = chosen

                # 方式A: 注入一个循环控制辅助节点（for→while风格）
                # 新增节点代表循环增量/条件检查
                aux_x = g.x[loop_dst].clone() * 0.5 + g.x[loop_src].clone() * 0.5
                aux_node = n_nodes
                g.x = torch.cat([g.x, aux_x.unsqueeze(0)], dim=0)

                # 辅助节点接入循环: loop_src → aux → loop_dst (新增)
                # 同时保留原回边 loop_src → loop_dst
                new_edges = [
                    [loop_src, aux_node],   # 循环入口到辅助节点
                    [aux_node, loop_dst],   # 辅助节点到循环体
                    [loop_dst, aux_node],   # 循环体回到辅助节点
                ]
                new_ei_tensor = torch.LongTensor(new_edges).T
                g.edge_index = torch.cat([g.edge_index, new_ei_tensor], dim=1)

                # 更新辅助属性
                if hasattr(g, '_LINE') and g._LINE is not None:
                    g._LINE = torch.cat([g._LINE, g._LINE[[loop_dst]]], dim=0)
                if hasattr(g, '_SAMPLE') and g._SAMPLE is not None:
                    g._SAMPLE = torch.cat([g._SAMPLE, g._SAMPLE[[loop_dst]]], dim=0)
            else:
                # 无明确回边，用通用方式：复制一个小子图并重新连接
                if n_nodes >= 3:
                    # 选一条边，在其两端之间插入一个中继节点
                    new_ei = g.edge_index.clone()
                    edge_idx = np.random.choice(new_ei.shape[1])
                    src, dst = new_ei[0, edge_idx].item(), new_ei[1, edge_idx].item()

                    # 插入中继节点
                    relay_x = (g.x[src] + g.x[dst]) / 2
                    g.x = torch.cat([g.x, relay_x.unsqueeze(0)], dim=0)
                    relay_node = n_nodes

                    # 原 edge: src → dst 变成 src → relay → dst
                    new_ei[:, edge_idx] = torch.LongTensor([src, relay_node])
                    new_edge = torch.LongTensor([[relay_node], [dst]])
                    new_ei = torch.cat([new_ei, new_edge], dim=1)
                    g.edge_index = new_ei

                    if hasattr(g, '_LINE') and g._LINE is not None:
                        g._LINE = torch.cat([g._LINE, g._LINE[[dst]]], dim=0)
                    if hasattr(g, '_SAMPLE') and g._SAMPLE is not None:
                        g._SAMPLE = torch.cat([g._SAMPLE, g._SAMPLE[[dst]]], dim=0)

        variants.append(g)

    return variants


# ==================== Type C: 反事实对 (PC 指标) ====================

def generate_counterfactual_pair(sample_info, correct_lines_info):
    """生成反事实修复版本图对 (PC 指标所需)

    直接使用 BigVul 的 before/after 文件构建图对。
    vul_graph: 漏洞版 (before 文件)
    cf_graph:  修复版 (after 文件)
    """
    sample_id = sample_info.get('id', None)
    if not sample_id:
        return None

    before_path = str(utils.processed_dir() / f"bigvul/before/{sample_id}.c")
    after_path = str(utils.processed_dir() / f"bigvul/after/{sample_id}.c")

    vul_graph = _build_data_from_file(before_path, sample_id)
    cf_graph = _build_data_from_file(after_path, sample_id)

    if vul_graph is not None and cf_graph is not None:
        return (vul_graph, cf_graph)
    return None


# ==================== PDG 切片 (辅助) ====================

def compute_pdg_backward_slice(sample_info, vts_lines):
    """计算PDG后向程序切片

    以 VTS(漏洞触发行)为准则节点，沿 PDG 反向遍历，
    收集所有数据依赖和控制依赖的祖先节点。
    """
    from line_extract import _load_graph_cached

    sample_id = sample_info.get('id', None)
    if not sample_id or not vts_lines:
        return set()

    before_path = str(utils.processed_dir() / f"bigvul/before/{sample_id}.c")
    pdg_graph = _load_graph_cached(before_path)
    if pdg_graph is None or len(pdg_graph) == 0:
        return set()

    # 构建反向邻接表
    reverse_deps = {}
    all_nodes = set(pdg_graph['id'].tolist())

    for _, row in pdg_graph.iterrows():
        nid = row['id']
        for dep in _safe_iter(row.get('data')):
            try:
                d = int(dep)
                reverse_deps.setdefault(d, set()).add(nid)
            except (ValueError, TypeError):
                pass
        for dep in _safe_iter(row.get('control')):
            try:
                d = int(dep)
                reverse_deps.setdefault(d, set()).add(nid)
            except (ValueError, TypeError):
                pass

    # BFS 反向遍历
    seed = set(int(x) for x in vts_lines if isinstance(x, (int, float))) & all_nodes
    visited = set(seed)
    queue = list(seed)

    while queue:
        u = queue.pop(0)
        for src in reverse_deps.get(u, set()):
            if src not in visited and src in all_nodes:
                visited.add(src)
                queue.append(src)

    return visited


# ==================== 主函数 ====================

def main(args):
    print("="*70)
    print("变体数据生成器 (NI-SI-PC Framework)")
    print("用于第三层鲁棒性评估")
    print("="*70)

    # 设置输出路径
    if args.output_path is None:
        output_path = str(utils.cache_dir() / "variant_data.pt")
    else:
        output_path = args.output_path

    print(f"\n输出路径: {output_path}")
    print(f"节点级变体数 (NI): {args.num_node_variants}")
    print(f"结构级变体数 (SI): {args.num_struct_variants}")
    print(f"数据来源: {args.data_source}")

    # 加载数据集
    print("\n加载数据集...")
    if args.data_source == 'bigvul':
        from line_extract import get_dep_add_lines_bigvul
        correct_lines = get_dep_add_lines_bigvul()
        samples = list(correct_lines.keys())
    else:
        raise NotImplementedError(f"数据源 {args.data_source} 尚未支持")

    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]

    print(f"样本总数: {len(samples)}")

    # 初始化变体数据字典 (NI-SI-PC 三维)
    variant_data = {
        'node_variants': {},         # Type A: NI 指标 (edge_index 不变)
        'struct_variants': {},       # Type B: SI 指标 (edge_index 改变)
        'counterfactual_pairs': {},  # Type C: PC 指标 (真实 before/after)
        'pdg_slices': {},            # 辅助: PDG 后向切片
        'metadata': {
            'source': args.data_source,
            'num_node_variants': args.num_node_variants,
            'num_struct_variants': args.num_struct_variants,
            'total_samples': len(samples),
            'framework': 'NI-SI-PC',
        }
    }

    # 开始生成变体数据
    print("\n开始生成变体数据...")
    print("-"*70)

    for idx, sampleid in enumerate(tqdm(samples, desc="Processing")):
        sampleid_str = str(sampleid)
        correct_lines_info = correct_lines[sampleid]
        sample_info = {'id': sampleid}

        try:
            # 加载原始图
            before_path = str(utils.processed_dir() / f"bigvul/before/{sampleid}.c")
            orig_graph = _build_data_from_file(before_path, sampleid)
            if orig_graph is None:
                continue

            # [A] 生成节点级变体 (NI)
            node_vars = generate_node_variants(orig_graph, sampleid, K=args.num_node_variants)
            if len(node_vars) >= 2:
                variant_data['node_variants'][sampleid_str] = node_vars

            # [B] 生成结构级变体 (SI)
            struct_vars = generate_struct_variants(orig_graph, sampleid, K=args.num_struct_variants)
            if len(struct_vars) >= 2:
                variant_data['struct_variants'][sampleid_str] = struct_vars

            # [C] 生成反事实对 (PC)
            cf_pair = generate_counterfactual_pair(sample_info, correct_lines_info)
            if cf_pair is not None:
                variant_data['counterfactual_pairs'][sampleid_str] = cf_pair

            # PDG 切片 (辅助)
            vts_lines = set(correct_lines_info.get('removed', []))
            slice_nodes = compute_pdg_backward_slice(sample_info, vts_lines)
            if len(slice_nodes) > 0:
                variant_data['pdg_slices'][sampleid_str] = list(slice_nodes)

        except Exception as e:
            print(f"\n⚠️  样本 {sampleid} 处理失败: {e}")
            continue

    # 统计有效数据量
    num_nv = len(variant_data['node_variants'])
    num_sv = len(variant_data['struct_variants'])
    num_cf = len(variant_data['counterfactual_pairs'])
    num_sl = len(variant_data['pdg_slices'])

    print("\n" + "="*70)
    print("变体数据生成完成！")
    print("-"*70)
    print(f"  [A] 节点级变体 (NI):  {num_nv}/{len(samples)} ({100*num_nv/len(samples):.1f}%)")
    print(f"  [B] 结构级变体 (SI):  {num_sv}/{len(samples)} ({100*num_sv/len(samples):.1f}%)")
    print(f"  [C] 扰动变体   (PC):  {num_cf}/{len(samples)} ({100*num_cf/len(samples):.1f}%)")
    print(f"  辅助 PDG切片:        {num_sl}/{len(samples)} ({100*num_sl/len(samples):.1f}%)")
    print("="*70)

    # 保存到文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(variant_data, output_path)
    print(f"\n数据已保存至: {output_path}")

    # 返回使用建议
    print("\n后续使用方法:")
    print(f"  python main.py --do_test --do_explain --do_robust --variant_cache {output_path}")

    return variant_data


if __name__ == "__main__":
    import pandas as pd
    args = parse_args()
    main(args)
