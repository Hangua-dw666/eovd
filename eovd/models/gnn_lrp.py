import math
import gc
import torch
from torch import Tensor
import torch.nn as nn
import copy
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.xgraph.models.utils import subgraph
from dig.xgraph.models.models import GraphSequential
from dig.xgraph.method.base_explainer import WalkBase
from typing import Tuple, List, Dict, Optional
from torch_geometric.nn import MessagePassing
from .vul_detector import GNNPool
EPS = 1e-15


def _safe_graph_sequential(modules: List[nn.Module], h: Tensor, edge_index: Tensor) -> Tensor:
    """Apply a list of modules sequentially.

    Some modules (e.g., MessagePassing layers) expect (x, edge_index), while
    others (ReLU/Dropout/Linear, etc.) expect only x.

    This helper tries (h, edge_index) first and falls back to (h) when the
    module does not accept the extra argument.
    """
    out = h
    for m in modules:
        try:
            out = m(out, edge_index)
        except TypeError:
            if isinstance(m, nn.Linear) and out.dim() >= 2 and int(out.shape[-1]) != int(m.in_features):
                continue
            out = m(out)
    return out


def _safe_graph_sequential_with_batch(modules: List[nn.Module], h: Tensor, edge_index: Tensor, batch: Optional[Tensor]) -> Tensor:
    """Apply modules sequentially while supporting pooling layers needing `batch`.

    Important: For PyG `MessagePassing` layers (e.g., `GCNConv`), the 3rd positional
    argument is typically `edge_weight`, NOT `batch`. Passing `batch` there will
    corrupt edge normalization and crash.
    """
    out = h
    for m in modules:
        if isinstance(m, GNNPool):
            if batch is None:
                batch = torch.zeros(out.shape[0], dtype=torch.long, device=out.device)
            out = m(out, batch)
            continue
        if isinstance(m, MessagePassing):
            out = m(out, edge_index)
            continue
        if isinstance(m, nn.Linear) and out.dim() >= 2 and int(out.shape[-1]) != int(m.in_features):
            continue
        out = m(out)
    return out


class GNN_LRP(WalkBase):
    r"""
    An implementation of GNN-LRP in
    `Higher-Order Explanations of Graph Neural Networks via Relevant Walks <https://arxiv.org/abs/2006.03589>`_.
    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)
    .. note::
            For node classification model, the :attr:`explain_graph` flag is False.
            GNN-LRP is very model dependent. Please be sure you know how to modify it for different models.
            For an example, see `benchmarks/xgraph
            <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.
    """

    def __init__(self, model: nn.Module, explain_graph=False):
        super().__init__(model=model, explain_graph=explain_graph)
        
    def extract_step(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None, detach: bool = True, split_fc: bool = False):

        layer_extractor = []
        hooks = []

        def register_hook(module: nn.Module):
            if not list(module.children()) or isinstance(module, MessagePassing):
                hooks.append(module.register_forward_hook(forward_hook))

        def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
            # input contains x and edge_index
            if detach:
                layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
            else:
                layer_extractor.append((module, input[0], output))

        # --- register hooks ---
        self.model.apply(register_hook)

        try:
            pred = self.model(x, edge_index, batch)
        except TypeError:
            pred = self.model(x, edge_index)

        for hook in hooks:
            hook.remove()
        
        transform_steps = []
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor[:3]:
            if isinstance(layer[0], nn.Linear):
                step['input'] = layer[1]
            step['module'].append(layer[0])
            step['output'] = layer[2]
            if isinstance(layer[0], nn.Dropout):
                transform_steps.append(step)

        walk_steps = []
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor[3:]:
            if isinstance(layer[0], GNNPool):
                break
            if isinstance(layer[0], MessagePassing):
                step = {'input': layer[1], 'module': [], 'output': None}
            step['module'].append(layer[0])
            step['output'] = layer[2]
            if isinstance(layer[0], nn.Dropout):
                walk_steps.append(step)
                step = {'input': None, 'module': [], 'output': None}
        
        fc_steps = []
        pool_flag = False
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor[3:]:
            if isinstance(layer[0], GNNPool):
                pool_flag = True
                step = {'input': layer[1], 'module': [layer[0]], 'output': layer[2]}
                fc_steps.append(step)
                step = {'input': None, 'module': [], 'output': None}
            if pool_flag:
                if isinstance(layer[0], nn.Linear):
                    step = {'input': layer[1], 'module': [], 'output': None}
                step['module'].append(layer[0])
                step['output'] = layer[2]
                if isinstance(layer[0], nn.Dropout) or isinstance(layer[0], nn.Softmax):
                    fc_steps.append(step)

        return transform_steps, walk_steps, fc_steps

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        r"""
        Run the explainer for a specific graph instance.
        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.
        :rtype:
            (walks, edge_masks, related_predictions),
            walks is a dictionary including walks' edge indices and corresponding explained scores;
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.
        """
        super().forward(x, edge_index, **kwargs)
        num_classes = int(kwargs.get('num_classes'))
        target_class = kwargs.get('target_class', None)
        if target_class is None:
            labels = tuple(i for i in range(num_classes))
        else:
            labels = (int(target_class),)
        self.model.eval()

        batch = kwargs.get('batch', None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        transform_steps, walk_steps, fc_steps = self.extract_step(x, edge_index, batch=batch, detach=False, split_fc=True)

        edge_index_with_loop, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)

        max_walks = kwargs.get('max_walks', None)
        try:
            max_walks = int(max_walks) if max_walks is not None else 0
        except Exception:
            max_walks = 0
        if max_walks > 0 and int(walk_indices_list.shape[0]) > int(max_walks):
            try:
                seed = kwargs.get('seed', 0)
                g = torch.Generator(device='cpu')
                g.manual_seed(int(seed) if seed is not None else 0)
                perm = torch.randperm(int(walk_indices_list.shape[0]), generator=g, device='cpu')[: int(max_walks)]
                walk_indices_list = walk_indices_list[perm.to(walk_indices_list.device)]
            except Exception:
                walk_indices_list = walk_indices_list[: int(max_walks)]
        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            node_idx = node_idx.reshape([1]).to(self.device)
            assert node_idx is not None
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]

        if kwargs.get('walks'):
            walks = kwargs.pop('walks')

        else:
            def compute_walk_score():

                # hyper-parameter gamma
                epsilon = 1e-30   # prevent from zero division
                gamma = [2, 1, 1]
                
                # --- record original weights of transform layer ---
                ori_transform_weights = []
                transform_gamma_modules = []
                for i, transform_step in enumerate(transform_steps):
                    modules = transform_step['module']
                    gamma_module = copy.deepcopy(modules[0])
                    if hasattr(modules[0], 'weight'):
                        ori_transform_weights.append(modules[0].weight.data)
                        gamma_ = 1
                        gamma_module.weight.data = ori_transform_weights[i] + gamma_ * ori_transform_weights[i].relu()
                    else:
                        ori_transform_weights.append(None)
                    transform_gamma_modules.append(gamma_module)

                # --- record original weights of GNN ---
                ori_gnn_weights = []
                gnn_gamma_modules = []
                # clear_probe = x
                for i, walk_step in enumerate(walk_steps):
                    modules = walk_step['module']
                    gamma_ = gamma[i] if i <= 1 else 1
                    base_mp = None
                    for mm in modules:
                        if isinstance(mm, MessagePassing):
                            base_mp = mm
                            break
                    if base_mp is None:
                        ori_gnn_weights.append(None)
                        gnn_gamma_modules.append(None)
                        continue

                    gamma_module = copy.deepcopy(base_mp)
                    if hasattr(base_mp, 'lin'):
                        ori_gnn_weights.append(base_mp.lin.weight.data)
                        gamma_module.lin.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(base_mp, 'nn'):
                        ori_gnn_weights.append(base_mp.nn.weight.data)
                        gamma_module.nn.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(base_mp, 'lin_r'):
                        ori_gnn_weights.append(base_mp.lin_l.weight.data)
                        gamma_module.lin_l.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(base_mp, 'lin_rel'):
                        ori_gnn_weights.append(base_mp.lin_rel.weight.data)
                        gamma_module.lin_rel.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(base_mp, 'weight'):
                        ori_gnn_weights.append(base_mp.weight.data)
                        gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    else:
                        ori_gnn_weights.append(None)
                        gnn_gamma_modules.append(None)
                        continue
                    gnn_gamma_modules.append(gamma_module)

                # --- record original weights of fc layer ---
                ori_fc_weights = []
                fc_gamma_modules = []
                for i, fc_step in enumerate(fc_steps):
                    modules = fc_step['module']
                    gamma_module = copy.deepcopy(modules[0])
                    if hasattr(modules[0], 'weight'):
                        ori_fc_weights.append(modules[0].weight.data)
                        gamma_ = 1
                        gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                    else:
                        ori_fc_weights.append(None)
                    fc_gamma_modules.append(gamma_module)

                # --- GNN_LRP implementation ---
                for walk_indices in walk_indices_list:
                    walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                    for walk_idx in walk_indices:
                        walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                    h = x.requires_grad_(True)
                    
                    # --- transform LRP_gamma ---
                    for i, transform_step in enumerate(transform_steps):
                        modules = transform_step['module']
                        std_h = nn.Sequential(*modules)(h)

                        # --- gamma ---
                        s = transform_gamma_modules[i](h)
                        ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                        h = ht
                    
                    max_steps = int(min(len(walk_steps), max(0, len(walk_node_indices) - 1)))
                    for i in range(max_steps):
                        walk_step = walk_steps[i]
                        modules = walk_step['module']
                        std_h = _safe_graph_sequential_with_batch(modules, h, edge_index, batch)

                        # --- LRP-gamma ---
                        gamma_m = gnn_gamma_modules[i] if i < len(gnn_gamma_modules) else None
                        if gamma_m is None:
                            p = std_h
                        elif isinstance(gamma_m, MessagePassing):
                            p = gamma_m(h, edge_index)
                        else:
                            p = gamma_m(h)
                        q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                        # --- pick a path ---
                        mk = torch.zeros((h.shape[0], 1), device=self.device)
                        k = walk_node_indices[i + 1]
                        mk[k] = 1
                        ht = q * mk + q.detach() * (1 - mk)
                        h = ht

                    # --- FC LRP_gamma ---
                    # debug that torch.zeros(h.shape[0], dtype=torch.long, device=self.device)
                    # should be an edge_index with [num_edge, 2]
                    for _, fc_step in enumerate(fc_steps):
                        modules = fc_step['module']
                        # FC steps may include pooling layers requiring `batch`. Use safe replay and
                        # skip gamma to keep compatibility across different detector heads.
                        h = _safe_graph_sequential_with_batch(modules, h, edge_index, batch)

                    if not self.explain_graph:
                        f = h[node_idx, label]
                    else:
                        f = h[0, label]
                    x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                    I = walk_node_indices[0]
                    r = x_grads[I, :] @ x[I].T
                    walk_scores.append(r.detach().clone())
                    del r, x_grads, f, h
                del ori_transform_weights, transform_gamma_modules, \
                    ori_gnn_weights, gnn_gamma_modules, \
                    ori_fc_weights, fc_gamma_modules
                gc.collect()
                torch.cuda.empty_cache()
            
            computed_scores: Dict[int, Tensor] = {}
            for label in labels:
                walk_scores = []
                compute_walk_score()
                computed_scores[int(label)] = torch.stack(walk_scores, dim=0).view(-1, 1)

            # Always build a full [num_walks, num_classes] score tensor so downstream
            # can safely index by class id.
            if computed_scores:
                any_t = next(iter(computed_scores.values()))
                num_walks = int(any_t.shape[0])
                device = any_t.device
            else:
                num_walks = int(walk_indices_list.shape[0])
                device = self.device

            walk_scores_tensor_list: List[Tensor] = []
            for c in range(int(num_classes)):
                t = computed_scores.get(int(c), None)
                if t is None:
                    t = torch.zeros((int(num_walks), 1), device=device)
                walk_scores_tensor_list.append(t)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_scores_tensor_list, dim=1)}

        del transform_steps, walk_steps, fc_steps
        gc.collect()
        torch.cuda.empty_cache()

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                edge_masks_full: List[Tensor] = [torch.zeros((int(edge_index_with_loop.shape[1]),), device=self.device) for _ in range(int(num_classes))]
                hard_edge_masks_full: List[Tensor] = [torch.zeros((int(edge_index_with_loop.shape[1]),), device=self.device) for _ in range(int(num_classes))]

                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    edge_mask = edge_attr.detach()
                    valid_mask = (edge_mask != -math.inf)
                    edge_mask[edge_mask == - math.inf] = edge_mask[valid_mask].min() - 1  # replace the negative inf

                    c = int(ex_label.view(-1)[0].item())
                    if 0 <= c < int(num_classes):
                        edge_masks_full[c] = edge_mask
                        hard_edge_masks_full[c] = self.control_sparsity(edge_attr, kwargs.get('sparsity')).sigmoid()

                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks_full, **kwargs)
                edge_masks = edge_masks_full
                hard_edge_masks = hard_edge_masks_full

        return walks, edge_masks, related_preds, edge_index_with_loop
    
    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module.explain = True
                module._edge_mask = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.explain = False
