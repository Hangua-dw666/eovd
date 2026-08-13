import os
import gc
import json
import random
import argparse
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
import torch_scatter
from transformers import AdamW, get_linear_schedule_with_warmup

from models.vul_detector import Detector
from helpers import utils
from line_extract import get_dep_add_lines_bigvul
from graph_dataset import VulGraphDataset, collate
from models.gnnexplainer import XGNNExplainer
from models.cfexplainer import CFExplainer
from models.pcf_explainer import PCFExplainerA, PCFExplainerB
from models.pgexplainer import XPGExplainer, PGExplainer_edges
from models.subgraphx import SubgraphX
from models.gnn_lrp import GNN_LRP
from models.deeplift import DeepLIFT
from models.gradcam import GradCAM

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary'), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary'), 4),
        'binary_f1': round(f1_score(y_true, y_pred, average='binary'), 4),
    }
    return results


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 注意：不使用 torch.use_deterministic_algorithms(True)，
    # 因为 PyG 的 scatter_reduce_cuda 在 CUDA 上无确定性实现
    # 已通过 cudnn.deterministic=True 保证大部分操作可复现


def train(args, train_dataloader, valid_dataloader, test_dataloader, model):

    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    checkpoint_last = os.path.join(args.model_checkpoint_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location=args.device))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location=args.device))

    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataloader)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Total optimization steps = {}".format(args.max_steps))

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0

    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch_data in enumerate(bar):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1
            model.train()
            probs = model(x, edge_index, batch)
            labels = F.one_hot(1 - labels, 2)
            loss = F.binary_cross_entropy(probs, labels.float())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    results = evaluate(args, valid_dataloader, model)
                    print(f"  Valid acc:{results['eval_acc']}")

                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        print("  " + "*" * 20)
                        print("  Best acc:{}".format(round(best_acc, 4)))
                        print("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        print("Saving model checkpoint to {}".format(output_dir))

                        test_result = evaluate(args, test_dataloader, model)
                        for key, value in test_result.items():
                            print("  {} = {}".format(key, round(value, 4)))
        bar.close()


def evaluate(args, eval_dataloader, model):

    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataloader)))

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_dataloader):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1
            probs = model(x, edge_index, batch)
            probs = F.one_hot(torch.argmax(probs, dim=-1), 2)[:, 0]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, 0)
    all_labels = np.concatenate(all_labels, 0)
    eval_acc = np.mean(all_labels == all_probs)

    result = {"eval_acc": round(eval_acc, 4)}
    eval_results = calculate_metrics(all_labels, all_probs)
    result.update(eval_results)

    return result


def gen_exp_lines(edge_index, edge_weight, index, num_nodes, lines):
    # 安全对齐：缓存可能来自旧环境，edge_index 与 edge_weight 长度不一致
    min_edges = min(edge_index.shape[1], edge_weight.shape[0])
    if min_edges < edge_index.shape[1] or min_edges < edge_weight.shape[0]:
        edge_index = edge_index[:, :min_edges]
        edge_weight = edge_weight[:min_edges]
        index = index[index < min_edges]

    temp = torch.zeros_like(edge_weight).to(edge_index.device)
    temp[index] = edge_weight[index]

    adj_mask = torch.sparse_coo_tensor(edge_index, temp, [num_nodes, num_nodes])
    adj_mask_binary = to_dense_adj(edge_index[:, temp != 0], max_num_nodes=num_nodes).squeeze(0)

    out_degree = torch.sum(adj_mask_binary, dim=1)
    out_degree[out_degree == 0] = 1e-8
    in_degree = torch.sum(adj_mask_binary, dim=0)
    in_degree[in_degree == 0] = 1e-8

    line_importance_init = torch.ones(num_nodes).unsqueeze(-1).to(edge_index.device)
    line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
    line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
    line_importance = line_importance_out + line_importance_in

    ret = sorted(
        list(zip(line_importance.squeeze(-1).cpu().numpy(), lines)),
        reverse=True,
    )

    filtered_ret = []
    for i in ret:
        if i[0] > 0:
            filtered_ret.append(int(i[1]))

    return filtered_ret


def eval_exp(exp_saved_path, model, correct_lines, args):
    """
    Evaluate explanations with PN, PS, Sparsity, and FLC/VFS metrics.

    Traditional Metrics (based on 'removed' lines):
    - Accuracy/Precision/Recall/F1: How well explainer locates deleted vulnerable lines

    Beyond-Fidelity Metrics (from "Beyond Fidelity: Explaining Vulnerability Localization of Learning-based Detectors"):
    - TLC (Triggering Location Coverage): |S_e ∩ VTS| / |VTS|
      VTS = removed lines ∪ their data/control dependencies (漏洞触发行及其依赖扩展)
    - FLC (Fixing Location Coverage): |S_e ∩ VFS| / |VFS|
      VFS = added lines mapped to before-version dependencies (修复行映射到before版本)

    PN(K_M): After removing top-K important edges, does the prediction change?
    PS(K_M): Using only top-K important edges, does the prediction stay the same?
    Sparsity(K_M): |V_E| / |V_G|
    """
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("Number of explanations:", len(graph_exp_list))

    accuracy = 0
    precisions = []
    recalls = []
    F1s = []
    pn_list = []   # Probability of Necessity
    ps_list = []   # Probability of Sufficiency
    sparsity_list = []   # Sparsity: |V_E| / |V_G|
    flc_list = []   # Fix Location Coverage (VFS)
    tlc_list = []   # Triggering Location Coverage (VTS)

    for graph in graph_exp_list:
        graph.to(args.device)
        x, edge_index, edge_weight, pred, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch

        # 安全对齐：缓存可能来自旧环境，edge_index 与 edge_weight 长度不一致
        if edge_index.shape[1] != edge_weight.shape[0]:
            min_edges = min(edge_index.shape[1], edge_weight.shape[0])
            edge_index = edge_index[:, :min_edges]
            edge_weight = edge_weight[:min_edges]

        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        # 安全检查：跳过不在真值数据中的样本（缓存可能来自不同版本）
        if int(sampleid) not in correct_lines:
            continue
        exp_label_data = correct_lines[int(sampleid)]
        exp_label_lines_removed = list(exp_label_data["removed"])
        exp_label_lines_depadd_removed = list(exp_label_data.get("depadd_removed", []))
        exp_label_lines_depadd_added = list(exp_label_data.get("depadd_added", []))

        # TLC: VTS = removed ∪ depadd_removed (漏洞触发行及其依赖扩展)
        vts_lines = set(exp_label_lines_removed) | set(exp_label_lines_depadd_removed)
        # FLC: VFS = depadd_added (修复行映射到before版本的依赖行)
        vfs_lines = set(exp_label_lines_depadd_added) if len(exp_label_lines_depadd_added) > 0 else set(exp_label_lines_removed)

        if len(edge_weight) > args.KM:
            value, index = torch.topk(edge_weight, k=args.KM)
        else:
            index = torch.arange(edge_weight.shape[0])

        temp = torch.ones_like(edge_weight)
        temp[index] = 0
        cf_index = temp != 0

        lines = graph._LINE.cpu().numpy()
        exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)
        exp_lines_set = set(exp_lines)

        for i, l in enumerate(exp_lines):
            if l in exp_label_lines_removed:
                accuracy += 1
                break

        hit = 0
        for i, l in enumerate(exp_lines):
            if l in exp_label_lines_removed:
                hit += 1
        if len(exp_lines) > 0 and len(exp_label_lines_removed) > 0 and hit != 0:
            precision = hit / len(exp_lines)
            recall = hit / len(exp_label_lines_removed)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(f1)

        flc_hit = len(exp_lines_set & vfs_lines)
        flc = flc_hit / len(vfs_lines) if len(vfs_lines) > 0 else 0
        flc_list.append(round(flc, 4))

        # TLC: 解释器对漏洞触发行(VTS)的覆盖率
        tlc_hit = len(exp_lines_set & vts_lines)
        tlc = tlc_hit / len(vts_lines) if len(vts_lines) > 0 else 0
        tlc_list.append(round(tlc, 4))

        topk_edge_index = edge_index[:, index]
        unique_nodes = torch.unique(topk_edge_index)
        num_explanation_nodes = unique_nodes.shape[0]
        num_total_nodes = x.shape[0]
        sparsity = round(num_explanation_nodes / num_total_nodes, 4)
        sparsity_list.append(sparsity)

        fac_edge_index = edge_index[:, index]
        fac_edge_index, _ = add_self_loops(fac_edge_index, num_nodes=x.shape[0])
        fac_logits = model(x, fac_edge_index, batch)
        fac_pred = F.one_hot(torch.argmax(fac_logits, dim=-1), 2)[0][0]

        cf_edge_index = edge_index[:, cf_index]
        cf_edge_index, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])
        cf_logits = model(x, cf_edge_index, batch)
        cf_pred = F.one_hot(torch.argmax(cf_logits, dim=-1), 2)[0][0]

        pn_list.append(int(cf_pred != pred))
        ps_list.append(int(fac_pred == pred))

        # Case Study: 保存指定样本的解释图
        if args.case_sample_ids and str(sampleid) in args.case_sample_ids:
            case_saving_dir = str(utils.cache_dir() / f"cases")
            os.makedirs(case_saving_dir, exist_ok=True)
            case_path = os.path.join(case_saving_dir, f"{args.gnn_model}_{args.ipt_method}_{sampleid}.pt")
            torch.save(graph, case_path)
            print(f"  [Case Study] 保存样本 {sampleid} → {case_path}")

    accuracy = round(accuracy / len(graph_exp_list), 4)
    print("Accuracy:", accuracy)
    precision = round(np.mean(precisions), 4)
    print("Precision:", precision)
    recall = round(np.mean(recalls), 4)
    print("Recall:", recall)
    f1 = round(np.mean(F1s), 4)
    print("F1:", f1)

    PN = round(sum(pn_list) / len(pn_list), 4)
    print("Probability of Necessity (PN):", PN)

    PS = round(sum(ps_list) / len(ps_list), 4)
    print("Probability of Sufficiency (PS):", PS)

    Sparsity = round(np.mean(sparsity_list), 4)
    print("Sparsity (|V_E|/|V_G|):", Sparsity)

    FLC = round(np.mean(flc_list), 4)
    print("Fix Location Coverage (FLC):", FLC)

    TLC = round(np.mean(tlc_list), 4)
    print("Triggering Location Coverage (TLC):", TLC)

    KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}

    # 结果保存：统一到 cache/results/{gnn_model}/ 下
    results_base = str(utils.cache_dir() / "results" / args.gnn_model)
    os.makedirs(results_base, exist_ok=True)

    if getattr(args, 'hyper_para', False):
        if args.ipt_method == "cfexplainer":
            suffix = f"_L1_{args.cfexp_alpha}" if args.cfexp_L1 else f"_{args.cfexp_alpha}"
            results_saving_path = os.path.join(results_base, f"{args.ipt_method}{suffix}.res")
        else:
            results_saving_path = os.path.join(results_base, f"{args.ipt_method}.res")
    else:
        results_saving_path = os.path.join(results_base, f"{args.ipt_method}.res")

    if os.path.isfile(results_saving_path):
        result = json.load(open(results_saving_path, "r"))
    else:
        result = {}

    if args.gnn_model not in result:
        result[args.gnn_model] = {}
        for metric in [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN", r"PS", r"Sparsity", r"FLC", r"TLC"]:
            result[args.gnn_model][metric] = [0.] * 10

    result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
    result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
    result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
    result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
    result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
    result[args.gnn_model][r"PS"][KM_index_map[args.KM]] = PS
    result[args.gnn_model][r"Sparsity"][KM_index_map[args.KM]] = Sparsity
    result[args.gnn_model][r"FLC"][KM_index_map[args.KM]] = FLC
    result[args.gnn_model][r"TLC"][KM_index_map[args.KM]] = TLC

    json.dump(result, open(results_saving_path, "w"))
    print(f"\nResults saved to: {results_saving_path}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "PN": PN, "PS": PS, "Sparsity": Sparsity, "FLC": FLC, "TLC": TLC}


def eval_robustness(exp_saved_path, model, variant_data, correct_lines, args):
    """
    第三层：鲁棒性评估 (NI-SI-PC 三维框架)

    ┌─────────────┬──────────────────────────┬────────┬──────────────────────────────────────┐
    │ 指标         │ 全称                     │ 趋势   │ 诊断含义                             │
    ├─────────────┼──────────────────────────┼────────┼──────────────────────────────────────┤
    │ NI           │ Node-Invariance          │ ↑越高越好│ 伪相关抗性：解释是否依赖节点表层特征  │
    │              │ 节点不变性               │        │ (命名风格/代码格式/嵌入偏差)          │
    ├─────────────┼──────────────────────────┼────────┼──────────────────────────────────────┤
    │ SI           │ Struct-Invariance        │ ↑越高越好│ 因果结构保真性：解释是否抓住抽象的    │
    │              │ 结构不变性               │        │ 因果依赖模式而非特定图拓扑            │
    ├─────────────┼──────────────────────────┼────────┼──────────────────────────────────────┤
    │ PC           │ Perturbation-based       │ ↑越高越好│ 因果区分性：解释子图是否包含真正的    │
    │              │ Counterfactual           │        │ 因果特征(移除后预测翻转)              │
    └─────────────┴──────────────────────────┴────────┴──────────────────────────────────────┘
    """
    print("\n" + "="*60)
    print("第三层：鲁棒性评估 (NI-SI-PC Framework)")
    print("="*60)

    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    results = {}
    KM = args.KM  # 使用 KM 作为 Top-K 边数，与前两层一致

    # ========== 变体解释缓存: 避免对同一变体重复跑完整解释器 ==========
    # 缓存路径: explainer_cache/{model}/variant_expl_{method}.pt
    # Key: (sample_id, variant_type, variant_index) → edge_weights (Tensor)
    # 变体生成是确定性的(固定seed)，同一(sid, vtype, vidx)的图完全相同
    variant_expl_cache_path = os.path.join(
        str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}"),
        f"variant_expl_{args.ipt_method}.pt"
    )
    variant_expl_cache = {}
    if os.path.exists(variant_expl_cache_path):
        print(f"  加载变体解释缓存: {variant_expl_cache_path}")
        variant_expl_cache = torch.load(variant_expl_cache_path, map_location='cpu')
    else:
        print(f"  变体解释缓存不存在，将新建: {variant_expl_cache_path}")

    def _get_variant_weights_cached(vg, sid_str, vtype, vidx):
        """获取变体图的边权重（带磁盘缓存）"""
        cache_key = (sid_str, vtype, vidx)
        if cache_key in variant_expl_cache:
            return variant_expl_cache[cache_key]
        # 缓存未命中，运行完整解释器 (fast_mode: 缩减迭代次数加速)
        nonlocal variant_cache_dirty
        variant_cache_dirty = True
        w = _get_edge_weights(model, vg, args, fast_mode=True)
        variant_expl_cache[cache_key] = w.cpu()  # 存到CPU避免GPU内存膨胀
        return w

    variant_cache_dirty = False  # 标记是否有新计算需要保存

    # [1/3] NI: 节点不变性 — 伪相关抗性
    if 'node_variants' in variant_data and variant_data['node_variants']:
        print("\n[1/3] NI (Node-Invariance / 节点不变性)...")
        print("      诊断: 解释是否依赖命名风格等伪相关特征?")
        ni_scores = []
        pred_consistent = 0
        pred_total = 0

        for g in tqdm(graph_exp_list, desc="NI评估", leave=False):
            sid = str(g._SAMPLE.max().int().item())
            if sid not in variant_data['node_variants']:
                continue
            variants = variant_data['node_variants'][sid]
            if len(variants) < 2:
                continue

            # 原始解释: 使用已缓存的解释器结果 (g.edge_weight)
            orig_w = g.edge_weight if hasattr(g, 'edge_weight') and g.edge_weight is not None else None
            if orig_w is None:
                orig_w = _get_edge_weights(model, g, args)
            orig_ei = g.edge_index.long()
            # 安全对齐: 缓存的 edge_weight 可能与 edge_index 长度不一致(旧版PGExplainer bug)
            if len(orig_ei.shape) == 2 and len(orig_w) > orig_ei.shape[1]:
                orig_w = orig_w[:orig_ei.shape[1]]
            orig_k = min(KM, len(orig_w))
            if len(orig_w) > orig_k:
                orig_topk_idx = torch.topk(orig_w, orig_k)[1]
            else:
                orig_topk_idx = torch.arange(len(orig_w))
            # NI: 节点变体 edge_index 不变，用边索引位置比较
            orig_topk = set(orig_topk_idx.cpu().numpy().tolist())

            for vidx, vg in enumerate(variants):
                vg.to(args.device)
                pred_total += 1

                # 预测一致性检查
                with torch.no_grad():
                    orig_pred = model(g.x, add_self_loops(g.edge_index.long(), num_nodes=g.x.shape[0])[0], g.batch).argmax(-1)[0].item()
                    var_pred = model(vg.x, add_self_loops(vg.edge_index.long(), num_nodes=vg.x.shape[0])[0], vg.batch).argmax(-1)[0].item()
                if orig_pred != var_pred:
                    continue
                pred_consistent += 1

                # 变体解释: 使用解释器重新解释 (带缓存)
                w = _get_variant_weights_cached(vg, sid, 'node', vidx)
                k = min(KM, len(w))
                var_topk = set(torch.topk(w, k)[1].cpu().numpy().tolist()) if len(w) > k else set(range(len(w)))

                # Jaccard 相似度（NI: edge_index不变，边索引位置可比）
                intersection = len(orig_topk & var_topk)
                union = len(orig_topk | var_topk)
                jaccard = intersection / union if union > 0 else 1.0
                ni_scores.append(jaccard)

        results['NI'] = round(np.mean(ni_scores), 4) if ni_scores else None
        consistency_rate = f"{pred_consistent}/{pred_total}" if pred_total > 0 else "N/A"
        print(f"  → NI = {results['NI']} (↑越高越好, KM={KM}, 预测一致: {consistency_rate}, 样本数={len(ni_scores)})")
        if results['NI'] is not None and results['NI'] < 0.5:
            print(f"  ⚠️  NI 偏低: 解释器可能依赖节点表层特征(伪相关)")
    else:
        results['NI'] = None
        print("\n[1/3] ⏭️  NI 跳过 (无节点级变体数据 node_variants)")

    # [2/3] SI: 结构不变性 — 因果结构保真性
    if 'struct_variants' in variant_data and variant_data['struct_variants']:
        print("\n[2/3] SI (Struct-Invariance / 结构不变性)...")
        print("      诊断: 解释是否抓住了抽象因果模式而非特定图拓扑?")
        si_scores = []
        pred_consistent = 0
        pred_total = 0

        for g in tqdm(graph_exp_list, desc="SI评估", leave=False):
            sid = str(g._SAMPLE.max().int().item())
            if sid not in variant_data['struct_variants']:
                continue
            variants = variant_data['struct_variants'][sid]
            if len(variants) < 2:
                continue

            # 原始解释: 使用已缓存的解释器结果 (g.edge_weight) + 转为(src,dst)节点对集合
            orig_w = g.edge_weight if hasattr(g, 'edge_weight') and g.edge_weight is not None else None
            if orig_w is None:
                orig_w = _get_edge_weights(model, g, args)
            orig_ei = g.edge_index.long()
            # 安全对齐: 缓存的 edge_weight 可能与 edge_index 长度不一致(旧版PGExplainer bug)
            if len(orig_ei.shape) == 2 and len(orig_w) > orig_ei.shape[1]:
                orig_w = orig_w[:orig_ei.shape[1]]
            orig_k = min(KM, len(orig_w))
            if len(orig_w) > orig_k:
                topk_idx = torch.topk(orig_w, orig_k)[1]
            else:
                topk_idx = torch.arange(len(orig_w))
            # 用 (src, dst) 元组作为边的唯一标识（跨图可比）
            orig_topk_pairs = set()
            for idx in topk_idx.cpu().tolist():
                s, d = orig_ei[0, idx].item(), orig_ei[1, idx].item()
                orig_topk_pairs.add((s, d))

            for vidx, vg in enumerate(variants):
                vg.to(args.device)
                pred_total += 1

                # 预测一致性检查
                with torch.no_grad():
                    orig_pred = model(g.x, add_self_loops(g.edge_index.long(), num_nodes=g.x.shape[0])[0], g.batch).argmax(-1)[0].item()
                    var_pred = model(vg.x, add_self_loops(vg.edge_index.long(), num_nodes=vg.x.shape[0])[0], vg.batch).argmax(-1)[0].item()
                if orig_pred != var_pred:
                    continue
                pred_consistent += 1

                # 变体解释: 使用解释器重新解释 (带缓存) + 转为(src,dst)节点对集合
                w = _get_variant_weights_cached(vg, sid, 'struct', vidx)
                var_ei = vg.edge_index.long()
                k = min(KM, len(w))
                if len(w) > k:
                    var_topk_idx = torch.topk(w, k)[1]
                else:
                    var_topk_idx = torch.arange(len(w))
                var_topk_pairs = set()
                for idx in var_topk_idx.cpu().tolist():
                    s, d = var_ei[0, idx].item(), var_ei[1, idx].item()
                    var_topk_pairs.add((s, d))

                # Jaccard 相似度（基于(src,dst)对，跨图可比）
                intersection = len(orig_topk_pairs & var_topk_pairs)
                union = len(orig_topk_pairs | var_topk_pairs)
                jaccard = intersection / union if union > 0 else 1.0
                si_scores.append(jaccard)

        results['SI'] = round(np.mean(si_scores), 4) if si_scores else None
        consistency_rate = f"{pred_consistent}/{pred_total}" if pred_total > 0 else "N/A"
        print(f"  → SI = {results['SI']} (↑越高越好, KM={KM}, 预测一致: {consistency_rate}, 样本数={len(si_scores)})")
        if results['SI'] is not None and results['SI'] < 0.5:
            print(f"  ⚠️  SI 偏低: 解释器可能未捕获抽象因果依赖模式")
    else:
        results['SI'] = None
        print("\n[2/3] ⏭️  SI 跳过 (无结构级变体数据 struct_variants)")

    # [3/3] PC: 扰动下因果对准一致性 (Perturbed Causal Alignment)
    # ─────────────────────────────────────────────────────────────
    # NI/SI 测的是"解释变没变"(稳定性)
    # PC 测的是"解释盯住的地方对不对"(因果对准性)
    #
    # 核心思路: 扰动非因果部分后，解释对VTS(漏洞触发行)的关注是否保持？
    #
    # 与 NI/SI 的正交性:
    #   NI/SI: 整个解释是否稳定？(包括因果和非因果部分)
    #   PC:    解释的因果部分(VTS重叠)是否稳定？(只关注真正重要的部分)
    #
    # 诊断矩阵:
    #   NI/SI高 + PC高 → 解释稳定且关注因果特征 ✓
    #   NI/SI高 + PC低 → 解释稳定但关注的是伪相关特征(稳定地错) ✗
    #   NI/SI低 + PC高 → 解释不稳定但因果部分始终被识别 → 部分可接受
    #   NI/SI低 + PC低 → 解释既不稳定又不关注因果特征 ✗✗
    print("\n[3/3] PC (Perturbed Causal Alignment / 扰动下因果对准一致性)...")
    print("      诊断: 解释对因果特征(VTS)的关注在扰动下是否保持?")
    print("      (与NI/SI区别: NI/SI=解释变没变, PC=解释盯的地方对不对)")

    # 合并所有变体类型
    all_variants = {}
    for sid_key, vlist in variant_data.get('node_variants', {}).items():
        all_variants.setdefault(sid_key, []).extend([(v, 'node') for v in vlist])
    for sid_key, vlist in variant_data.get('struct_variants', {}).items():
        all_variants.setdefault(sid_key, []).extend([(v, 'struct') for v in vlist])

    pc_scores = []
    pc_node_scores = []
    pc_struct_scores = []

    for g in tqdm(graph_exp_list, desc="PC评估", leave=False):
        g.to(args.device)
        sid = int(g._SAMPLE.max().int().item())
        sid_str = str(sid)

        # 需要 correct_lines 来获取 VTS 行号
        if sid not in correct_lines:
            continue
        if sid_str not in all_variants:
            continue

        exp_label_data = correct_lines[sid]
        vts_lines = set(exp_label_data["removed"]) | set(exp_label_data.get("depadd_removed", []))
        if len(vts_lines) == 0:
            continue

        x, ei, ew = g.x, g.edge_index.long(), g.edge_weight

        # 安全对齐
        if ei.shape[1] != ew.shape[0]:
            min_edges = min(ei.shape[1], ew.shape[0])
            ei = ei[:, :min_edges]
            ew = ew[:min_edges]

        # 原始解释与 VTS 的重叠率
        k = min(KM, len(ew))
        if len(ew) > k:
            topk_idx = torch.topk(ew, k)[1]
        else:
            topk_idx = torch.arange(len(ew))

        orig_exp_nodes = torch.unique(ei[:, topk_idx]).cpu().numpy().tolist()
        orig_lines = g._LINE[orig_exp_nodes].int().tolist() if hasattr(g, '_LINE') and g._LINE is not None else []
        orig_vts_overlap = len(set(orig_lines) & vts_lines) / len(vts_lines) if len(vts_lines) > 0 else 0

        # 在每个扰动变体上检查因果对准是否保持
        for vidx, (vg, vtype) in enumerate(all_variants[sid_str]):
            vg.to(args.device)

            # 预测一致性检查
            with torch.no_grad():
                orig_pred = model(x, add_self_loops(ei, num_nodes=x.shape[0])[0], g.batch).argmax(-1)[0].item()
                var_pred = model(vg.x, add_self_loops(vg.edge_index.long(), num_nodes=vg.x.shape[0])[0], vg.batch).argmax(-1)[0].item()
            if orig_pred != var_pred:
                continue

            # 变体解释: 使用解释器重新解释 (带缓存)
            w = _get_variant_weights_cached(vg, sid_str, vtype, vidx)
            vk = min(KM, len(w))
            if len(w) > vk:
                var_topk_idx = torch.topk(w, vk)[1]
            else:
                var_topk_idx = torch.arange(len(w))

            var_ei = vg.edge_index.long()
            # 安全对齐
            if var_ei.shape[1] != len(w):
                min_e = min(var_ei.shape[1], len(w))
                var_ei = var_ei[:, :min_e]
                var_topk_idx = var_topk_idx[var_topk_idx < min_e]

            var_exp_nodes = torch.unique(var_ei[:, var_topk_idx]).cpu().numpy().tolist()
            var_lines = vg._LINE[var_exp_nodes].int().tolist() if hasattr(vg, '_LINE') and vg._LINE is not None else []
            var_vts_overlap = len(set(var_lines) & vts_lines) / len(vts_lines) if len(vts_lines) > 0 else 0

            # 因果对准一致性: 扰动后的VTS重叠率不应显著下降
            # 用 min(pert/orig, 1.0) 衡量，orig=0 时直接用 var_vts_overlap
            if orig_vts_overlap > 0:
                causal_consistency = min(var_vts_overlap / orig_vts_overlap, 1.0)
            else:
                causal_consistency = var_vts_overlap  # 原始就没对准，扰动后对准了也算好

            pc_scores.append(causal_consistency)
            if vtype == 'node':
                pc_node_scores.append(causal_consistency)
            else:
                pc_struct_scores.append(causal_consistency)

    results['PC'] = round(np.mean(pc_scores), 4) if pc_scores else None
    results['PC_node'] = round(np.mean(pc_node_scores), 4) if pc_node_scores else None
    results['PC_struct'] = round(np.mean(pc_struct_scores), 4) if pc_struct_scores else None

    print(f"  → PC       = {results['PC']} (↑越高越好, KM={KM}, 样本数={len(pc_scores)})")
    if results['PC_node'] is not None:
        print(f"  → PC_node  = {results['PC_node']} (节点级扰动下)")
    if results['PC_struct'] is not None:
        print(f"  → PC_struct= {results['PC_struct']} (结构级扰动下)")
    if results['PC'] is not None and results['PC'] < 0.5:
        print(f"  ⚠️  PC 偏低: 解释对因果特征(VTS)的关注在扰动下丢失，可能依赖伪相关")

    # NI-SI-PC 三维诊断
    ni_val = results.get('NI')
    si_val = results.get('SI')
    pc_val = results.get('PC')
    if ni_val is not None and si_val is not None and pc_val is not None:
        print("\n" + "-"*60)
        print("NI-SI-PC 三维诊断:")

        # 稳定性维度 (NI vs SI)
        diff = ni_val - si_val
        if diff > 0.1:
            print(f"  [稳定性] NI({ni_val}) >> SI({si_val}): 对结构变化更敏感")
        elif diff < -0.1:
            print(f"  [稳定性] SI({si_val}) >> NI({ni_val}): 对节点特征更敏感")
        else:
            print(f"  [稳定性] NI({ni_val}) ≈ SI({si_val}): 两类扰动影响相当")

        # 因果对准维度 (PC)
        avg_stability = (ni_val + si_val) / 2
        if avg_stability > 0.5 and pc_val > 0.5:
            print(f"  [因果对准] 稳定({avg_stability:.2f}) + 因果对准({pc_val}): 解释可靠 ✓")
        elif avg_stability > 0.5 and pc_val <= 0.5:
            print(f"  [因果对准] 稳定({avg_stability:.2f}) + 因果偏离({pc_val}): 稳定地错(伪相关) ✗")
        elif avg_stability <= 0.5 and pc_val > 0.5:
            print(f"  [因果对准] 不稳定({avg_stability:.2f}) + 因果对准({pc_val}): 乱但碰对了 → 部分可接受")
        else:
            print(f"  [因果对准] 不稳定({avg_stability:.2f}) + 因果偏离({pc_val}): 既不稳定又不对准 ✗✗")
        print("-"*60)
    elif ni_val is not None and si_val is not None:
        print("\n" + "-"*60)
        print("NI-SI 诊断:")
        diff = ni_val - si_val
        if diff > 0.1:
            print(f"  NI({ni_val}) >> SI({si_val}): 解释器对结构变化更敏感")
        elif diff < -0.1:
            print(f"  SI({si_val}) >> NI({ni_val}): 解释器对节点特征更敏感")
        else:
            print(f"  NI({ni_val}) ≈ SI({si_val}): 两类扰动影响程度相当")
        print("-"*60)

    # 保存变体解释缓存 (跨KM复用)
    if variant_cache_dirty:
        os.makedirs(os.path.dirname(variant_expl_cache_path), exist_ok=True)
        torch.save(variant_expl_cache, variant_expl_cache_path)
        print(f"  变体解释缓存已保存: {variant_expl_cache_path} (共{len(variant_expl_cache)}条)")

    # 保存结果
    save_dir = str(utils.cache_dir() / "results" / args.gnn_model)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.ipt_method}_robustness.res")
    existing = json.load(open(save_path)) if os.path.exists(save_path) else {}
    key = f"{args.gnn_model}_KM{args.KM}"
    existing[key] = {k: v for k, v in results.items()}
    json.dump(existing, open(save_path, "w"), indent=2)
    print(f"\n结果已保存: {save_path}")

    return results


# ==================== 鲁棒性评估辅助函数 ====================

def _get_edge_weights_fast(model, graph, args):
    """快速获取边重要性权重 (基于梯度近似，无需重新训练解释器)

    核心思路: 对每条边计算其对预测损失的梯度，梯度越大→边越重要
    速度比完整解释器快 ~100x，适用于鲁棒性评估中的大量变体
    """
    x, ei = graph.x, graph.edge_index.long()
    ei_sl, _ = add_remaining_self_loops(ei, num_nodes=x.shape[0])

    x_req = x.clone().detach().requires_grad_(True)
    prob = model(x_req, ei_sl, graph.batch)
    label = prob.argmax(-1)

    # 计算对预测类别的梯度
    loss = prob[0, label[0]]
    loss.backward()

    # 节点梯度范数作为边重要性代理
    node_grad = x_req.grad.norm(dim=1)  # [num_nodes]

    # 每条边的重要性 = 源节点梯度 + 目标节点梯度
    edge_weights = []
    for j in range(ei.shape[1]):
        src, dst = ei[0, j].item(), ei[1, j].item()
        w = node_grad[src].item() + node_grad[dst].item()
        edge_weights.append(w)

    w = torch.FloatTensor(edge_weights)
    if w.max() > 0:
        w = w / w.max()  # 归一化到 [0, 1]
    return w


def _get_edge_weights(model, graph, args, fast_mode=False):
    """在图上运行解释器并返回边权重向量 (完整版，支持所有解释器)

    fast_mode=True: 用于鲁棒性评估的变体图解释，大幅缩减迭代次数以加速
                    只需粗略的边权重排序即可计算 Jaccard 相似度
    """
    x, ei = graph.x, graph.edge_index.long()
    ei_nosl, _ = remove_self_loops(ei)
    ei_sl, _ = add_remaining_self_loops(ei_nosl, num_nodes=x.shape[0])
    prob = model(x, ei_sl, graph.batch)
    label = prob.argmax(-1)

    if args.ipt_method == "gnnexplainer":
        _epochs = 150 if fast_mode else 800
        expl = XGNNExplainer(model, explain_graph=True, epochs=_epochs, lr=0.05,
                              coff_edge_size=0.001, coff_edge_ent=0.001)
        masks, _, _, slei = expl(x, ei_sl, False, None, num_classes=args.num_classes)
        w = masks[label]
        ei_w, w = remove_self_loops(slei.detach().cpu(), w.detach().cpu())

    elif args.ipt_method == "cfexplainer":
        _epochs = 150 if fast_mode else 800
        expl = CFExplainer(model, explain_graph=True, epochs=_epochs, lr=0.05,
                            alpha=getattr(args, 'cfexp_alpha', 1.0),
                            L1_dist=getattr(args, 'cfexp_L1', False))
        masks, _, _, slei = expl(x, ei_sl, False, None, num_classes=args.num_classes)
        w = 1 - masks[label]
        ei_w, w = remove_self_loops(slei.detach().cpu(), w.detach().cpu())

    elif args.ipt_method in ("pcf_a", "pcf_b"):
        _epochs = 150 if fast_mode else 800
        _prior_path = getattr(args, 'prior_path', None)
        if _prior_path is None:
            _prior_path = str(utils.cache_dir() / "prior" / args.gnn_model / f"KM{args.KM}" / "prior_clf.pkl")
        _prior_lambda = getattr(args, 'prior_lambda', 0.5)
        _alpha = getattr(args, 'cfexp_alpha', 0.9)
        _L1 = getattr(args, 'cfexp_L1', False)
        _cls = PCFExplainerA if args.ipt_method == "pcf_a" else PCFExplainerB
        expl = _cls(model, explain_graph=True, epochs=_epochs, lr=0.05,
                    alpha=_alpha, L1_dist=_L1,
                    prior_lambda=_prior_lambda, prior_path=_prior_path)
        masks, _, _, slei = expl(x, ei_sl, False, None, num_classes=args.num_classes)
        w = 1 - masks[label]
        ei_w, w = remove_self_loops(slei.detach().cpu(), w.detach().cpu())

    elif args.ipt_method == "pgexplainer":
        input_dim = args.gnn_hidden_size * 2
        pgexplainer = XPGExplainer(model=model, in_channels=input_dim, device=args.device,
                                    explain_graph=True, epochs=100, lr=0.005,
                                    coff_size=0.01, coff_ent=5e-4, sample_bias=0.0, t0=5.0, t1=1.0)
        pgexplainer_saving_path = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/pgexplainer.bin")
        if os.path.isfile(pgexplainer_saving_path):
            pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
        else:
            # PGExplainer 需要训练数据，这里无法训练，回退到梯度近似
            return _get_edge_weights_fast(model, graph, args)
        pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=model)
        _result = pgexplainer_edges(x, ei_nosl, **{"batch": graph.batch, "tmp_batch": None, "num_classes": args.num_classes})
        # PGExplainer内部会add_self_loop，返回的_result[3]是带self-loop的edge_index
        # edge_masks[0]长度与_result[3]一致(都带self-loop)，需用返回的edge_index做remove_self_loops
        pg_ei_with_sl = _result[3]
        w = _result[0][label]
        ei_w, w = remove_self_loops(pg_ei_with_sl.detach().cpu(), w.detach().cpu())

    elif args.ipt_method == "subgraphx":
        _rollout = 2 if fast_mode else 5
        explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/subgraphx")
        if not os.path.exists(explanation_saving_dir):
            os.makedirs(explanation_saving_dir)
        subgraphx = SubgraphX(model, args.num_classes, args.device, explain_graph=True,
                              verbose=False, c_puct=10.0, rollout=_rollout, high2low=False, min_atoms=5, expand_atoms=14,
                              reward_method='gnn_score', subgraph_building_method='zero_filling',
                              save_dir=explanation_saving_dir)
        prediction = label.item()
        explain_result, _ = subgraphx.explain(x, ei_sl, label=prediction, node_idx=0, saved_MCTSInfo_list=None)
        node_weight = torch.zeros(x.shape[0])
        for item in explain_result:
            # coalition 可能是节点列表或单个节点索引
            c = item['coalition']
            if isinstance(c, (list, tuple, torch.Tensor)):
                p_per_node = item['P'] / len(c)  # 概率均分到 coalition 中每个节点
                for n in c:
                    if isinstance(n, torch.Tensor):
                        n = n.item()
                    if isinstance(n, int):
                        node_weight[n] += p_per_node
            elif isinstance(c, int):
                node_weight[c] += item['P']
        node_weight = node_weight / max(len(explain_result), 1)
        ei_nosl_cpu, _ = remove_self_loops(ei_sl.detach().cpu())
        w = node_weight[ei_nosl_cpu[0]] + node_weight[ei_nosl_cpu[1]]
        ei_w, w = ei_nosl_cpu, w

    elif args.ipt_method == "deeplift":
        deep_lift = DeepLIFT(model, explain_graph=True)
        edge_masks, _, _, slei = deep_lift(x, ei_sl, sparsity=0.5, num_classes=args.num_classes)
        w = edge_masks[label].sigmoid()
        ei_w, w = remove_self_loops(slei.detach().cpu(), w.detach().cpu())

    elif args.ipt_method == "gradcam":
        gc_explainer = GradCAM(model, explain_graph=True)
        edge_masks, _, _, slei = gc_explainer(x, ei_sl, sparsity=0.5, num_classes=args.num_classes)
        w = edge_masks[label]
        ei_w, w = remove_self_loops(slei.detach().cpu(), w.detach().cpu())

    else:
        raise NotImplementedError(f"{args.ipt_method}")

    return w


def _topk_edges_set(graph, ratio):
    """从已计算权重的图中取 top-ratio 边索引集合"""
    w = graph.edge_weight
    k = max(1, int(len(w) * ratio))
    if len(w) > k:
        return set(torch.topk(w, k)[1].cpu().numpy().tolist())
    return set(range(len(w)))


def _topk_edges_set_by_k(graph, k):
    """从已计算权重的图中取 top-k 边索引集合 (k为绝对边数)"""
    w = graph.edge_weight
    k = min(k, len(w))
    if len(w) > k:
        return set(torch.topk(w, k)[1].cpu().numpy().tolist())
    return set(range(len(w)))


def _topk_edges_from_graph(model, graph, ratio, args):
    """在图上重新运行解释器并返回 top-ratio 边集合"""
    w = _get_edge_weights(model, graph, args)
    k = max(1, int(len(w) * ratio))
    if len(w) > k:
        return set(torch.topk(w, k)[1].cpu().numpy().tolist())
    return set(range(len(w)))


def _topk_nodes_set(graph, ratio):
    """从已计算权重的图中取 top-ratio 节点集合"""
    w, ei = graph.edge_weight, graph.edge_index
    k = max(1, int(len(w) * ratio))
    if len(w) > k:
        idx = torch.topk(w, k)[1]
        return set(torch.unique(ei[:, idx]).cpu().numpy().tolist())
    return set(torch.unique(ei).cpu().numpy().tolist())


# ==================== Explainer Functions ====================

def gnnexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = XGNNExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05,
        coff_edge_size=0.001, coff_edge_ent=0.001
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None, num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def cfexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = CFExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05, alpha=args.cfexp_alpha, L1_dist=args.cfexp_L1
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None, num_classes=args.num_classes)
        edge_weight = 1 - edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def pcf_explainer_run(args, model, test_dataset, correct_lines):
    """PCFExplainer (Prior-aware Counterfactual Explainer) 的解释流程.

    支持两个版本 (由 args.ipt_method 决定):
      - pcf_a: PCFExplainerA (no_grad 调制), mask = σ(z_cf - λ·z_align)
      - pcf_b: PCFExplainerB (loss 正则),   loss = CF_loss + λ·MSE(m, 1-z_align)

    与 cfexplainer_run 流程一致, 仅替换解释器为先验感知反事实解释器.
    先验由 train_prior.py 训练的逻辑回归提供, 加载路径由 --prior_path 指定.
    edge_weight 取 1 - masks[label] (与 CFExplainer 一致: 权重越大=越该被移除).
    模型无关: 先验只用输入节点特征 x 和边结构, 适配 ReVeal/Devign/DeepWukong 等.
    """
    graph_exp_list = []
    visited_sampleids = set()
    _cls = PCFExplainerA if args.ipt_method == "pcf_a" else PCFExplainerB
    # 默认 prior_path 查找: storage/cache/prior/<model>/KM<km>/prior_clf.pkl
    _prior_path = getattr(args, 'prior_path', None)
    if _prior_path is None:
        _prior_path = str(utils.cache_dir() / "prior" / args.gnn_model / f"KM{args.KM}" / "prior_clf.pkl")
    explainer = _cls(
        model=model, explain_graph=True, epochs=800, lr=0.05,
        alpha=getattr(args, 'cfexp_alpha', 0.9),
        L1_dist=getattr(args, 'cfexp_L1', False),
        prior_lambda=getattr(args, 'prior_lambda', 0.5),
        prior_path=_prior_path
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None, num_classes=args.num_classes)
        edge_weight = 1 - edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    input_dim = args.gnn_hidden_size * 2

    pgexplainer = XPGExplainer(model=model, in_channels=input_dim, device=args.device, explain_graph=True, epochs=100, lr=0.005,
                            coff_size=0.01, coff_ent=5e-4, sample_bias=0.0, t0=5.0, t1=1.0)
    pgexplainer_saving_path = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/pgexplainer.bin")
    if os.path.isfile(pgexplainer_saving_path) and not args.ipt_update:
        print("Load saved PGExplainer model...")
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
    else:
        pgexplainer.train_explanation_network(train_dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))

    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=eval_model)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        _result = pgexplainer_edges(x, edge_index, **{"batch": batch, "tmp_batch": None, "num_classes": args.num_classes})
        # PGExplainer内部会add_self_loop，返回的_result[3]是带self-loop的edge_index
        pg_ei_with_sl = _result[3]
        edge_weight = _result[0][torch.argmax(exp_prob_label, dim=-1)[0].item()]
        edge_index, edge_weight = remove_self_loops(pg_ei_with_sl.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def subgraphx_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/subgraphx")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    subgraphx = SubgraphX(model, args.num_classes, args.device, explain_graph=True,
                        verbose=False, c_puct=10.0, rollout=5, high2low=False, min_atoms=5, expand_atoms=14,
                        reward_method='gnn_score', subgraph_building_method='zero_filling',
                        save_dir=explanation_saving_dir)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        saved_MCTSInfo_list = None
        prediction = prob.argmax(-1).item()
        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        explain_result = subgraphx.explain(x, edge_index, label=prediction, node_idx=0, saved_MCTSInfo_list=saved_MCTSInfo_list)
        torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))
        node_weight = torch.zeros(x.shape[0])
        for item in explain_result:
            # coalition 可能是节点列表或单个节点索引
            c = item['coalition']
            if isinstance(c, (list, tuple, torch.Tensor)):
                p_per_node = item['P'] / len(c)  # 概率均分到 coalition 中每个节点
                for n in c:
                    if isinstance(n, torch.Tensor):
                        n = n.item()
                    if isinstance(n, int):
                        node_weight[n] += p_per_node
            elif isinstance(c, int):
                node_weight[c] += item['P']
        node_weight = node_weight / max(len(explain_result), 1)
        edge_index, _ = remove_self_loops(edge_index.detach().cpu())
        edge_weight = node_weight[edge_index[0]] + node_weight[edge_index[1]]
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gnn_lrp_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/gnn_lrp")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    gnnlrp_explainer = GNN_LRP(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            edge_masks, self_loop_edge_index = torch.load(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        else:
            walks, edge_masks, related_preds, self_loop_edge_index = gnnlrp_explainer(x, edge_index, sparsity=0.5, num_classes=args.num_classes)
            torch.save((edge_masks, self_loop_edge_index), os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph.detach().clone().cpu())
        visited_sampleids.add(sampleid)

        del graph
        gc.collect()

    return graph_exp_list


def deeplift_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    deep_lift = DeepLIFT(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = deep_lift(x, edge_index, sparsity=0.5, num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gradcam_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    gc_explainer = GradCAM(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = gc_explainer(x, edge_index, sparsity=0.5, num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument('--cuda_id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--seed', type=int, default=1, help="random seed")

    # GNN Model
    parser.add_argument("--model_checkpoint_dir", default="saved_models", type=str)
    parser.add_argument("--gnn_model", default="GCNConv", type=str, help="GNN core: GCNConv, GatedGraphConv, GINConv, GraphConv, Reveal, Devign, DeepWukong, IVDetect")
    parser.add_argument("--gnn_hidden_size", default=256, type=int)
    parser.add_argument("--gnn_feature_dim_size", default=768, type=int)
    parser.add_argument("--residual", action='store_true')
    parser.add_argument("--graph_pooling", default="mean", type=str)
    parser.add_argument("--num_gnn_layers", default=2, type=int)
    parser.add_argument("--num_ggnn_steps", default=3, type=int)
    parser.add_argument("--ggnn_aggr", default="add", type=str)
    parser.add_argument("--gin_eps", default=0., type=float)
    parser.add_argument("--gin_train_eps", action='store_true')
    parser.add_argument("--gconv_aggr", default="mean", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--num_classes", default=2, type=int)

    # Training
    parser.add_argument("--num_train_epochs", default=50, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_explain", action='store_true')
    parser.add_argument("--eval_only", action='store_true', help="跳过解释器，直接从缓存.pt评估（需先运行过 --do_explain）")
    parser.add_argument("--do_robust", action='store_true', help="运行第三层鲁棒性评估 (NI-SI-PC)")

    # Explainer
    parser.add_argument("--ipt_method", default="gnnexplainer", type=str)
    parser.add_argument("--ipt_update", action='store_true')
    parser.add_argument("--KM", default=8, type=int, help="The size of explanation subgraph (K_M for PN/PS curve)")
    parser.add_argument("--cfexp_L1", action='store_true')
    parser.add_argument("--cfexp_alpha", default=0.9, type=float)
    # PCFExplainer (pcf_a / pcf_b) 参数 —— 先验感知反事实解释器
    # 先验由 train_prior.py 训练, 复用 CFExplainer 的 alpha/L1_dist 参数
    parser.add_argument("--prior_lambda", default=0.5, type=float,
                        help="PCF 先验调制/正则强度 (λ=0 退化为 CFExplainer)")
    parser.add_argument("--prior_path", default=None, type=str,
                        help="先验逻辑回归模型路径 (pkl). 默认自动查找 storage/cache/prior/<model>/KM<km>/prior_clf.pkl")
    parser.add_argument("--hyper_para", action='store_true', help="超参数调优模式：结果存到 parameter_analysis/, 解释按alpha/L1分路径")
    parser.add_argument("--case_sample_ids", nargs='+', help="指定样本ID进行case study，保存解释图到 cases/")

    # Robustness Evaluation
    parser.add_argument("--variant_cache", default=None, type=str, help="变体数据缓存路径 (用于鲁棒性评估)")
    parser.add_argument("--robust_alpha", type=float, default=0.2, help="NI/SI/PC的top比例α")
    parser.add_argument("--robust_theta", type=float, default=0.3, help="PC的重叠率阈值θ")

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_dir}" / args.gnn_model)
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    model = Detector(args)
    model.to(args.device)

    train_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    print(train_dataset)

    valid_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='val')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
    print(valid_dataset)

    test_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(test_dataset)

    if args.do_train:
        train(args, train_dataloader, valid_dataloader, test_dataloader, model)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        model_checkpoint_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
        model.to(args.device)
        test_result = evaluate(args, test_dataloader, model)

        print("***** Test results *****")
        for key in sorted(test_result.keys()):
            print("  {} = {}".format(key, str(round(test_result[key], 4))))

        if args.do_explain or args.eval_only:
            correct_lines = get_dep_add_lines_bigvul()
            ipt_save_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}")
            if not os.path.exists(ipt_save_dir):
                os.makedirs(ipt_save_dir)
            # hyper_para 模式：按 alpha/L1 组合分路径
            if args.hyper_para and args.ipt_method == "cfexplainer":
                suffix = f"_L1_{args.cfexp_alpha}" if args.cfexp_L1 else f"_{args.cfexp_alpha}"
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}{suffix}.pt")
            else:
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}.pt")

            # eval_only 模式：检查缓存是否存在
            if args.eval_only:
                if not os.path.exists(ipt_save):
                    print(f"\n❌ 缓存文件不存在: {ipt_save}")
                    print("请先运行 --do_explain 生成解释缓存，或去掉 --eval_only")
                    return
                print(f"[eval_only] 直接加载缓存: {ipt_save}")
            else:
                print("Size of test dataset:", len(test_dataset))

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if not args.eval_only and (not os.path.exists(ipt_save) or args.ipt_update):
                graph_exp_list = []
                if args.ipt_method == "pgexplainer":
                    eval_model = Detector(args)
                    eval_model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
                    eval_model.to(args.device)
                    graph_exp_list = pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines)
                elif args.ipt_method == "subgraphx":
                    graph_exp_list = subgraphx_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "deeplift":
                    graph_exp_list = deeplift_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gradcam":
                    graph_exp_list = gradcam_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnnexplainer":
                    graph_exp_list = gnnexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnn_lrp":
                    graph_exp_list = gnn_lrp_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "cfexplainer":
                    graph_exp_list = cfexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method in ("pcf_a", "pcf_b"):
                    graph_exp_list = pcf_explainer_run(args, model, test_dataset, correct_lines)

                torch.save(graph_exp_list, ipt_save)
                print(f"\nExplanations saved to: {ipt_save}")

            print(f"\n{'='*60}")
            print(f"Evaluating with K_M = {args.KM}")
            print(f"{'='*60}")
            eval_exp(ipt_save, model, correct_lines, args)

            if args.do_robust:
                print(f"\n{'='*60}")
                print(f"第三层：鲁棒性评估 (NI-SI-PC Framework)")
                print(f"{'='*60}")

                variant_data = None
                if args.variant_cache and os.path.exists(args.variant_cache):
                    print(f"\n加载变体数据: {args.variant_cache}")
                    variant_data = torch.load(args.variant_cache, map_location=args.device)
                else:
                    default_variant_path = str(utils.cache_dir() / "variant_data.pt")
                    if os.path.exists(default_variant_path):
                        print(f"\n加载默认变体数据: {default_variant_path}")
                        variant_data = torch.load(default_variant_path, map_location=args.device)
                    else:
                        print("\n⚠️  未找到变体数据文件！")
                        print("请先运行 generate_variants.py 生成变体数据，或通过 --variant_cache 指定路径")
                        print("跳过鲁棒性评估...")

                if variant_data:
                    robustness_results = eval_robustness(ipt_save, model, variant_data, correct_lines, args)
                    print("\n✅ 鲁棒性评估完成！")
                else:
                    print("\n❌ 鲁棒性评估已跳过（缺少变体数据）")


if __name__ == "__main__":
    main()
