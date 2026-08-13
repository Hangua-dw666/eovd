import os
import pickle as pkl
import math
import functools
import argparse
from pathlib import Path
from helpers import utils
from helpers import joern
from data_pre import bigvul
from typing import List
import pandas as pd

def get_dep_add_lines(filepath_before, filepath_after, added_lines):
    """Get lines that are dependent on added lines.

    Example:
    df = bigvul()
    filepath_before = "storage/processed/bigvul/before/177775.c"
    filepath_after = "storage/processed/bigvul/after/177775.c"
    added_lines = df[df.id==177775].added.item()

    """

    before_graph = _load_graph_cached(filepath_before)
    after_graph = _load_graph_cached(filepath_after)
    if before_graph is None or after_graph is None:
        return []

    # Get nodes in graph corresponding to added lines
    added_after_lines = after_graph[after_graph.id.isin(added_lines)]

    # Get lines dependent on added lines in added graph
    dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
    dep_add_lines = set([i for j in dep_add_lines for i in j])

    # Filter by lines in before graph
    before_lines = set(before_graph.id.tolist())
    dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

    return dep_add_lines


def _load_graph_cached(filepath: str):
    cache_name = "_".join(str(filepath).split("/")[-3:])
    cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(cache_name).stem
    try:
        if os.path.exists(str(cachefp)):
            return pkl.load(open(cachefp, "rb"))[0]
    except Exception:
        return None
    try:
        out = _build_graph_from_joern_outputs(str(filepath))
        if out is None:
            return None
        nodes_df, edges = out
        try:
            with open(cachefp, "wb") as f:
                pkl.dump([nodes_df, edges], f)
        except Exception:
            pass
        return nodes_df
    except Exception:
        return None


def _as_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _build_graph_from_joern_outputs(filepath: str):
    # Requires existing <filepath>.nodes.json and <filepath>.edges.json.
    nodes, edges = joern.get_node_edges(str(filepath))
    if nodes is None or edges is None:
        return None

    nodesline = nodes[nodes.lineNumber != ""].copy()
    if int(len(nodesline)) == 0:
        return None
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )

    edgesline = edges.copy()
    if 'line_in' not in edgesline.columns or 'line_out' not in edgesline.columns:
        return None
    edgesline['innode'] = edgesline['line_in']
    edgesline['outnode'] = edgesline['line_out']
    nodesline['id'] = nodesline['lineNumber']

    edgesline = joern.rdg(edgesline, "pdg")
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    edgesline["etype"] = edgesline.apply(
        lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )

    # Keep only valid numeric endpoints.
    def _is_num(v):
        return _as_int(v) is not None
    edgesline = edgesline[edgesline['innode'].apply(_is_num)]
    edgesline = edgesline[edgesline['outnode'].apply(_is_num)]
    edgesline = edgesline.copy()
    edgesline['innode'] = edgesline['innode'].apply(_as_int)
    edgesline['outnode'] = edgesline['outnode'].apply(_as_int)
    edgesline = edgesline[edgesline['innode'].notnull() & edgesline['outnode'].notnull()]

    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline[["innode", "outnode", "etype"]], edgesline_reverse], ignore_index=True)
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set}).reset_index()

    if int(len(uedge)) > 0:
        uedge = uedge.pivot(index="innode", columns="etype", values="outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(lambda x: list(x) if isinstance(x, set) else [])
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    pdg_nodes["data"] = pdg_nodes.id.map(data).fillna(pd.Series([[]] * int(len(pdg_nodes)))).tolist() if int(len(pdg_nodes)) > 0 else []
    pdg_nodes["control"] = pdg_nodes.id.map(control).fillna(pd.Series([[]] * int(len(pdg_nodes)))).tolist() if int(len(pdg_nodes)) > 0 else []

    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    try:
        e_in = edgesline['innode'].map(pdg_dict)
        e_out = edgesline['outnode'].map(pdg_dict)
        e_in = e_in.dropna().astype(int).tolist()
        e_out = e_out.dropna().astype(int).tolist()
        pdg_edges = (e_out, e_in)
    except Exception:
        pdg_edges = ([], [])

    # Restore id as lineNumber column (kept as id).
    pdg_nodes = pdg_nodes.drop(columns=['index'], errors='ignore')
    return pdg_nodes, pdg_edges


def get_dep_add_lines_bounded(
    filepath_before,
    filepath_after,
    added_lines,
    budget_abs: int = 50,
    budget_ratio: float = 0.1,
    filter_hubs: bool = False,
    hub_top_pct: float = 0.01,
):
    before_graph = _load_graph_cached(filepath_before)
    after_graph = _load_graph_cached(filepath_after)
    if before_graph is None or after_graph is None:
        return []

    try:
        before_lines = set(before_graph.id.tolist())
    except Exception:
        before_lines = set()

    try:
        added_after_lines = after_graph[after_graph.id.isin(added_lines)]
        dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
        dep_add_lines = set([i for j in dep_add_lines for i in j])
    except Exception:
        dep_add_lines = set()

    out_dep: List[int] = []
    for i in dep_add_lines:
        ii = _as_int(i)
        if ii is None:
            continue
        if int(ii) in before_lines:
            out_dep.append(int(ii))
    dep_add_lines = sorted(set(out_dep))

    hub_lines = set()
    if bool(filter_hubs) and len(before_lines) > 0:
        try:
            data_map = before_graph.set_index('id')['data'].to_dict()
        except Exception:
            data_map = {}
        try:
            control_map = before_graph.set_index('id')['control'].to_dict()
        except Exception:
            control_map = {}
        try:
            cnt = {}
            for u in before_lines:
                for v in (data_map.get(u, []) or []):
                    try:
                        vv = int(v)
                    except Exception:
                        continue
                    if vv in before_lines:
                        cnt[vv] = int(cnt.get(vv, 0) + 1)
                for v in (control_map.get(u, []) or []):
                    try:
                        vv = int(v)
                    except Exception:
                        continue
                    if vv in before_lines:
                        cnt[vv] = int(cnt.get(vv, 0) + 1)
            k = int(math.ceil(max(0.0, float(hub_top_pct)) * float(len(before_lines))))
            if k > 0 and len(cnt) > 0:
                items = sorted(cnt.items(), key=lambda x: (-int(x[1]), int(x[0])))
                hub_lines = set([int(x[0]) for x in items[:k]])
        except Exception:
            hub_lines = set()

    if hub_lines:
        dep_add_lines = [ln for ln in dep_add_lines if int(ln) not in hub_lines]

    n_lines = int(len(before_lines)) if len(before_lines) > 0 else max(1, int(len(dep_add_lines)))
    try:
        budget = int(min(int(budget_abs), int(math.ceil(float(budget_ratio) * float(n_lines)))))
    except Exception:
        budget = int(budget_abs)
    budget = max(0, int(budget))
    if budget == 0:
        return []
    return dep_add_lines[:budget]


def get_dep_removed_lines(
    filepath_before,
    removed_lines,
    hop: int = 1,
    budget_abs: int = 50,
    budget_ratio: float = 0.1,
    filter_hubs: bool = False,
    hub_top_pct: float = 0.01,
):
    before_cache_name = "_".join(str(filepath_before).split("/")[-3:])
    before_cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(before_cache_name).stem
    before_graph = pkl.load(open(before_cachefp, "rb"))[0]

    try:
        before_lines = set(before_graph.id.tolist())
    except Exception:
        before_lines = set()

    seeds: List[int] = []
    try:
        for x in (removed_lines or []):
            try:
                xx = int(x)
            except Exception:
                continue
            if xx in before_lines:
                seeds.append(xx)
    except Exception:
        seeds = []
    seeds = sorted(set(seeds))

    try:
        data_map = before_graph.set_index('id')['data'].to_dict()
    except Exception:
        data_map = {}
    try:
        control_map = before_graph.set_index('id')['control'].to_dict()
    except Exception:
        control_map = {}

    hub_lines = set()
    if bool(filter_hubs) and len(before_lines) > 0:
        try:
            cnt = {}
            for u in before_lines:
                for v in (data_map.get(u, []) or []):
                    try:
                        vv = int(v)
                    except Exception:
                        continue
                    if vv in before_lines:
                        cnt[vv] = int(cnt.get(vv, 0) + 1)
                for v in (control_map.get(u, []) or []):
                    try:
                        vv = int(v)
                    except Exception:
                        continue
                    if vv in before_lines:
                        cnt[vv] = int(cnt.get(vv, 0) + 1)

            k = int(math.ceil(max(0.0, float(hub_top_pct)) * float(len(before_lines))))
            if k > 0 and len(cnt) > 0:
                items = sorted(cnt.items(), key=lambda x: (-int(x[1]), int(x[0])))
                hub_lines = set([int(x[0]) for x in items[:k]])
        except Exception:
            hub_lines = set()

    def _neighbors(line: int) -> List[int]:
        out = set()
        try:
            out.update(list(data_map.get(line, []) or []))
        except Exception:
            pass
        try:
            out.update(list(control_map.get(line, []) or []))
        except Exception:
            pass
        out2: List[int] = []
        for y in out:
            try:
                yy = int(y)
            except Exception:
                continue
            if yy in before_lines:
                if hub_lines and yy in hub_lines:
                    continue
                out2.append(yy)
        return sorted(set(out2))

    n_lines = int(len(before_lines)) if len(before_lines) > 0 else max(1, int(len(seeds)))
    try:
        budget = int(min(int(budget_abs), int(math.ceil(float(budget_ratio) * float(n_lines)))))
    except Exception:
        budget = int(budget_abs)
    budget = max(0, int(budget))

    if budget == 0:
        return []

    selected: List[int] = []
    selected_set = set()
    for s in seeds:
        if len(selected) >= budget:
            break
        if s not in selected_set:
            selected.append(s)
            selected_set.add(s)

    frontier = list(seeds)
    visited = set(seeds)
    for _ in range(max(0, int(hop))):
        if len(selected) >= budget:
            break
        next_frontier: List[int] = []
        for u in frontier:
            for v in _neighbors(int(u)):
                if v in visited:
                    continue
                visited.add(v)
                next_frontier.append(v)
        next_frontier = sorted(set(next_frontier))
        for v in next_frontier:
            if len(selected) >= budget:
                break
            if v not in selected_set:
                selected.append(v)
                selected_set.add(v)
        frontier = next_frontier

    return selected


def helper_removed_depadd(
    row,
    hop: int = 1,
    budget_abs: int = 50,
    budget_ratio: float = 0.1,
    filter_hubs: bool = False,
    hub_top_pct: float = 0.01,
):
    before_path = str(utils.processed_dir() / f"bigvul/before/{row['id']}.c")
    after_path = str(utils.processed_dir() / f"bigvul/after/{row['id']}.c")
    try:
        removed_lines = row.get("removed", []) if isinstance(row, dict) else row["removed"]
    except Exception:
        removed_lines = []
    try:
        added_lines = row.get("added", []) if isinstance(row, dict) else row["added"]
    except Exception:
        added_lines = []

    # 同时计算两种 depadd：用于 TLC 和 FLC
    # depadd_removed: removed 行的依赖扩展 → TLC 的 VTS = removed ∪ depadd_removed
    # depadd_added:   added 行映射到 before 的依赖行 → FLC 的 VFS = depadd_added
    depadd_removed = []
    depadd_added = []

    try:
        depadd_removed = get_dep_removed_lines(
            before_path,
            removed_lines,
            hop=hop,
            budget_abs=budget_abs,
            budget_ratio=budget_ratio,
            filter_hubs=filter_hubs,
            hub_top_pct=hub_top_pct,
        )
    except Exception:
        depadd_removed = []

    try:
        depadd_added = get_dep_add_lines_bounded(
            before_path,
            after_path,
            added_lines,
            budget_abs=budget_abs,
            budget_ratio=budget_ratio,
            filter_hubs=filter_hubs,
            hub_top_pct=hub_top_pct,
        )
    except Exception:
        depadd_added = []

    return [row["id"], {
        "removed": removed_lines,
        "depadd_removed": depadd_removed,   # removed的依赖扩展 → 用于TLC
        "depadd_added": depadd_added,       # added映射到before → 用于FLC
    }]


def get_dep_add_lines_bigvul(
    cache=True,
    hop: int = 1,
    budget_abs: int = 50,
    budget_ratio: float = 0.1,
    filter_hubs: bool = False,
    hub_top_pct: float = 0.01,
    saved_path: str = None,
):
    """Cache dependent added lines for bigvul.

    返回格式 v2: {sample_id: {removed, depadd_removed, depadd_added}}
    旧格式 v1:   {sample_id: {removed, depadd, anchor_type}}  (已废弃，会自动重建)
    """
    if saved_path is None:
        saved = utils.get_dir(utils.processed_dir() / "bigvul/eval") / "statement_labels.pkl"
    else:
        saved = Path(str(saved_path))
        saved.parent.mkdir(exist_ok=True, parents=True)
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            data = pkl.load(f)
        # 检测旧格式缓存（含 depadd/anchor_type 而非 depadd_removed/depadd_added）
        if data and isinstance(data, dict):
            first_val = next(iter(data.values()), None)
            if isinstance(first_val, dict) and 'depadd' in first_val and 'depadd_removed' not in first_val:
                print("⚠️ 检测到旧格式 statement_labels.pkl (v1)，正在重建为 v2 格式...")
                cache = False  # 强制重新生成
            else:
                return data
    df = bigvul()
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    fn = functools.partial(
        helper_removed_depadd,
        hop=int(hop),
        budget_abs=int(budget_abs),
        budget_ratio=float(budget_ratio),
        filter_hubs=bool(filter_hubs),
        hub_top_pct=float(hub_top_pct),
    )

    lines_dict = utils.dfmp(df, fn, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
    return lines_dict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--cache', action='store_true')
    p.add_argument('--hop', type=int, default=1)
    p.add_argument('--budget_abs', type=int, default=50)
    p.add_argument('--budget_ratio', type=float, default=0.1)
    p.add_argument('--filter_hubs', action='store_true')
    p.add_argument('--hub_top_pct', type=float, default=0.01)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    get_dep_add_lines_bigvul(
        cache=bool(getattr(args, 'cache', False)),
        hop=int(getattr(args, 'hop', 1)),
        budget_abs=int(getattr(args, 'budget_abs', 50)),
        budget_ratio=float(getattr(args, 'budget_ratio', 0.1)),
        filter_hubs=bool(getattr(args, 'filter_hubs', False)),
        hub_top_pct=float(getattr(args, 'hub_top_pct', 0.01)),
        saved_path=getattr(args, 'out', None),
    )