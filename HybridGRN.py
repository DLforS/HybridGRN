import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
from sklearn import linear_model
import warnings
import networkx as nx
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestRegressor
import matplotlib

matplotlib.use('Agg')  # 在导入 pyplot 之前设置
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====================全局配置参数====================
CONFIG = {
    "n12": 0.3,  # 线性特征控制参数
    "n22": 0.8,  # 非线性特征控制参数
    "nM": 0.2,  # Bootstrap样本比例
    "nB": 100,  # Bootstrap抽样次数
    "alpha": 0.12,  # LASSO正则化强度
    "n_runs": 500,  # 稳定性分析重复次数
    "min_drop": 1,  # 最小删除节点数
    "max_drop": 3,  # 最大删除节点数
    "decay_factor": 0.7,  # 连接度衰减因子
    "penalty_threshold": 4,  # 开始惩罚的连接度阈值
    "output_dir": "results",
    "random_state": 40,
    "final_threshold_method": "dynamic",
    "top_fusion": {
        "n_range": [50, 100, 150, 200, 250],
        "visualize": True
    }
}


# ==================核心算法======================
class DegreePenalty:
    """节点连接度惩罚机制"""

    def __init__(self, decay_factor=0.8):
        self.decay_factor = decay_factor
        self.degree_history = {}

    def update_degrees(self, G):
        current_degrees = dict(G.degree())
        for node in current_degrees:
            hist_degree = self.degree_history.get(node, 0)
            self.degree_history[node] = (
                    hist_degree * self.decay_factor +
                    current_degrees[node] * (1 - self.decay_factor)
            )

    def get_penalty(self, node):
        avg_degree = self.degree_history.get(node, 0)
        return 1 / (1 + np.exp(-0.5 * (avg_degree - 5)))


def enhanced_polobag(X, genes, target_idx, config):
    n_genes = len(genes)
    valid_indices = [i for i in range(n_genes) if i != target_idx]
    drop_mask = np.random.rand(len(valid_indices)) < 0.1
    kept_indices = [valid_indices[i] for i in np.where(~drop_mask)[0]]

    min_features = max(2, int(0.1 * n_genes))
    if len(kept_indices) < min_features:
        kept_indices = np.random.choice(valid_indices, min_features, replace=False)

    X_sub = X[:, kept_indices]
    sub_genes = [genes[i] for i in kept_indices]
    y = X[:, target_idx]
    ntR = len(sub_genes)

    nlin = np.clip(int(config["n12"] * np.sqrt(ntR)), 1, 50)
    max_nonlin = max(0, (ntR - nlin) // 2)
    nnlin = np.clip(int(config["n22"] * np.sqrt(ntR)), 0, min(25, max_nonlin))

    wtM = np.zeros(ntR, dtype=np.float32)
    wtS = np.zeros(ntR, dtype=np.float32)
    stw = np.ones(ntR, dtype=np.float32) * 1e-10
    penalty_model = DegreePenalty(decay_factor=config.get("decay_factor", 0.8))

    for _ in range(config["nB"]):
        try:
            sample_size = max(int(config["nM"] * X_sub.shape[0]), 50)
            row_idx = np.random.choice(X_sub.shape[0], sample_size, replace=True)
            X_sample = X_sub[row_idx]
            y_sample = y[row_idx]

            lin_features = np.random.choice(ntR, nlin, replace=False) if nlin > 0 else []
            nonlin_pairs = []
            if nnlin > 0:
                available = list(set(range(ntR)) - set(lin_features))
                if len(available) >= 2 * nnlin:
                    nonlin_pairs = np.random.choice(available, 2 * nnlin, replace=False).reshape(-1, 2)

            Xtb = []
            if len(lin_features) > 0:
                Xtb.append(X_sample[:, lin_features])
            for pair in nonlin_pairs:
                inter_feature = X_sample[:, pair[0]] * X_sample[:, pair[1]]
                Xtb.append(inter_feature.reshape(-1, 1))
            if not Xtb:
                continue

            Xtb = np.hstack(Xtb)
            Xtb = (Xtb - np.mean(Xtb, axis=0)) / (np.std(Xtb, axis=0) + 1e-8)
            y_sample = (y_sample - np.mean(y_sample)) / (np.std(y_sample) + 1e-8)

            lasso = linear_model.Lasso(
                alpha=config["alpha"],
                fit_intercept=False,
                max_iter=5000,
                tol=1e-4,
                random_state=config["random_state"]
            )
            lasso.fit(Xtb, y_sample)
            coefs = lasso.coef_

            ptr = 0
            if len(lin_features) > 0:
                valid = slice(ptr, ptr + len(lin_features))
                penalties = [penalty_model.get_penalty(sub_genes[i]) for i in lin_features]
                wtM[lin_features] += np.abs(coefs[valid]) * penalties
                wtS[lin_features] += coefs[valid] * penalties
                stw[lin_features] += 1
                ptr += len(lin_features)

            for (idx1, idx2) in nonlin_pairs:
                if ptr >= len(coefs):
                    break
                c = coefs[ptr]
                sqrt_abs = np.sqrt(np.abs(c))
                sign = np.sign(c)
                penalty1 = penalty_model.get_penalty(sub_genes[idx1])
                penalty2 = penalty_model.get_penalty(sub_genes[idx2])
                wtM[idx1] += sqrt_abs * penalty1
                wtS[idx1] += sign * sqrt_abs * penalty1
                stw[idx1] += 1
                wtM[idx2] += sqrt_abs * penalty2
                wtS[idx2] += sign * sqrt_abs * penalty2
                stw[idx2] += 1
                ptr += 1

            current_G = nx.DiGraph()
            current_G.add_edges_from(
                [(sub_genes[i], sub_genes[j]) for i, j in np.argwhere(wtM > np.percentile(wtM, 90))]
            )
            penalty_model.update_degrees(current_G)
        except Exception as e:
            continue

    valid = stw > 1e-5
    wt = np.zeros(ntR)
    wt[valid] = np.sign(wtS[valid]) * wtM[valid] / stw[valid]

    return {sub_genes[i]: wt[i] for i in np.where(np.abs(wt) > 1e-4)[0]}


def genie3_parallel(X, genes, n_trees=100, n_jobs=4):
    def process_target(target_idx):
        y = X[:, target_idx]
        X_train = np.delete(X, target_idx, axis=1)
        regressor = RandomForestRegressor(
            n_estimators=n_trees,
            max_features='sqrt',
            n_jobs=1,
            random_state=42
        )
        regressor.fit(X_train, y)
        importances = regressor.feature_importances_
        return target_idx, importances

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_target)(i) for i in tqdm(range(len(genes)), desc="Running GENIE3")
    )

    importance_matrix = np.zeros((len(genes), len(genes)))
    for target_idx, importances in results:
        ptr = 0
        for i in range(len(genes)):
            if i == target_idx:
                importance_matrix[i, target_idx] = 0
            else:
                importance_matrix[i, target_idx] = importances[ptr]
                ptr += 1

    max_score = importance_matrix.max()
    min_score = importance_matrix.min()
    normalized = (importance_matrix - min_score) / (max_score - min_score)
    return pd.DataFrame(normalized, index=genes, columns=genes)


# ===================评估模块====================
class GRNEvaluator:
    def __init__(self, goldstandard_path):
        self.gt_edges = set()
        self.gt_non_edges = set()
        try:
            df = pd.read_csv(
                goldstandard_path,
                sep='\t',
                header=None,
                names=['source', 'target', 'label'],
                dtype={'source': str, 'target': str, 'label': int}
            )
            df = df.dropna()
            for src, tar, label in df.itertuples(index=False):
                src = src.strip()
                tar = tar.strip()
                if label == 1:
                    self.gt_edges.add((src, tar))
                elif label == 0:
                    self.gt_non_edges.add((src, tar))
        except Exception as e:
            raise ValueError(f"无法加载goldstandard: {str(e)}")

    def evaluate(self, stability_matrix, threshold=0.7, predicted_edges=None):
        if threshold is None:
            threshold = determine_threshold(stability_matrix)
        if predicted_edges is None:
            predicted_edges = set()
            for src in stability_matrix.index:
                for tar in stability_matrix.columns:
                    if src == tar:
                        continue
                    if stability_matrix.loc[src, tar] >= threshold:
                        predicted_edges.add((src, tar))
        else:
            predicted_edges = {(s, t) for s, t in predicted_edges if s != t}

        y_true = []
        y_score = []
        for src in stability_matrix.index:
            for tar in stability_matrix.columns:
                if src == tar:
                    continue
                score = stability_matrix.loc[src, tar]
                y_score.append(score)
                if (src, tar) in self.gt_edges:
                    y_true.append(1)
                elif (src, tar) in self.gt_non_edges:
                    y_true.append(0)
                if score >= threshold:
                    predicted_edges.add((src, tar))

        tp = len(predicted_edges & self.gt_edges)
        fp = len(predicted_edges - self.gt_edges)
        fn = len(self.gt_edges - predicted_edges)
        tn = len(self.gt_non_edges) - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) + 1e-5 if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall_curve, precision_curve)

        return {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4),
            'PR_AUC': round(pr_auc, 4),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }

    def evaluate_network_structure(self, G):
        degrees = np.array([d for _, d in G.degree()])
        return {
            'Degree Centralization': np.max(degrees) / (len(G) - 1),
            'Assortativity': nx.degree_assortativity_coefficient(G),
            'Avg Clustering': nx.average_clustering(G.to_undirected()),
            'Max Degree': np.max(degrees),
            'Min Degree': np.min(degrees)
        }

    def comprehensive_evaluate(self, adj_matrix, threshold=None):
        basic_metrics = self.evaluate(adj_matrix, threshold)
        G = nx.from_pandas_adjacency(adj_matrix >= (threshold or determine_threshold(adj_matrix)))
        structure_metrics = self.evaluate_network_structure(G)
        return {**basic_metrics, **structure_metrics}


# ===================Top边融合评估====================
def top_edge_fusion(lasso_scores, genie_scores, top_n=200):
    """Top边融合策略"""
    lasso_edges = []
    for src in lasso_scores.index:
        for tar in lasso_scores.columns:
            if src == tar:
                continue
            lasso_edges.append((src, tar, lasso_scores.loc[src, tar]))
    lasso_edges.sort(key=lambda x: -x[2])
    lasso_top = set((s, t) for s, t, _ in lasso_edges[:top_n])

    genie_edges = []
    for src in genie_scores.index:
        for tar in genie_scores.columns:
            if src == tar:
                continue
            genie_edges.append((src, tar, genie_scores.loc[src, tar]))
    genie_edges.sort(key=lambda x: -x[2])
    genie_top = set((s, t) for s, t, _ in genie_edges[:top_n])

    fused_edges = lasso_top.union(genie_top)
    fused_matrix = pd.DataFrame(0, index=lasso_scores.index, columns=lasso_scores.columns)
    for src, tar in fused_edges:
        fused_matrix.loc[src, tar] = 1
    return fused_matrix


def evaluate_top_fusion(lasso_scores, genie_scores, evaluator, top_n_range=None):
    """评估Top边融合策略"""
    if top_n_range is None:
        top_n_range = [50, 100, 200, 300]
    results = []
    for n in top_n_range:
        fused = top_edge_fusion(lasso_scores, genie_scores, top_n=n)
        metrics = evaluator.evaluate(fused, threshold=0.5)
        G = nx.from_pandas_adjacency(fused, create_using=nx.DiGraph)
        net_metrics = evaluator.evaluate_network_structure(G)
        metrics.update({
            'TopN': n,
            'Total_Edges': G.number_of_edges(),
            **net_metrics
        })
        results.append(metrics)
    return pd.DataFrame(results)


# ====================动态阈值调整====================
def get_dynamic_range(n_genes):
    if n_genes <= 10:
        return (10, 30)
    elif 10 < n_genes <= 100:
        return (max(100, n_genes), min(400, n_genes * 4))
    else:
        return (n_genes, n_genes * 4)


def iterative_threshold_adjust(stability_matrix, init_threshold, target_range, max_attempts=8):
    target_lower, target_upper = target_range
    threshold = init_threshold
    history = []
    base_lr = 0.3
    prev_direction = 0

    for attempt in range(max_attempts):
        G = nx.from_pandas_adjacency(stability_matrix >= threshold, create_using=nx.DiGraph)
        actual_edges = G.number_of_edges()
        history.append((threshold, actual_edges))
        if target_lower <= actual_edges <= target_upper:
            break

        if actual_edges < target_lower:
            error = target_lower - actual_edges
            adjust_ratio = error / target_lower
            direction = -1
        else:
            error = actual_edges - target_upper
            adjust_ratio = error / target_upper
            direction = 1

        lr = base_lr * (1 + np.tanh(adjust_ratio * 2))
        if direction == prev_direction:
            lr *= 1.2
        else:
            lr *= 0.5

        adjustment = 1 + direction * lr * np.sqrt(adjust_ratio)
        if direction == -1:
            adjustment = max(0.8, min(0.95, adjustment))
        else:
            adjustment = min(1.2, max(1.05, adjustment))

        threshold *= adjustment
        prev_direction = direction
        min_score = stability_matrix.values.min()
        max_score = stability_matrix.values.max()
        threshold = np.clip(threshold, min_score * 0.9, max_score * 1.1)

    scores = stability_matrix.values.flatten()
    scores = scores[scores > 1e-5]
    if len(scores) == 0:
        return 0.0

    if actual_edges < target_lower:
        required = min(target_lower, len(scores))
        final_thresh = np.partition(scores, -required)[-required]
    elif actual_edges > target_upper:
        required = min(len(scores) - target_upper, len(scores))
        final_thresh = np.partition(scores, required)[required]
    else:
        final_thresh = threshold

    return np.clip(final_thresh, scores.min(), scores.max())


def determine_threshold(stability_matrix, config=CONFIG):
    n_genes = len(stability_matrix)
    if n_genes <= 20:
        target_edges = (n_genes, 2 * n_genes)
        col_percentile = 70
        decay_factor = 0.85
    elif n_genes <= 100:
        target_edges = (int(n_genes * 0.8), int(n_genes * 2.5))
        col_percentile = 80
        decay_factor = 0.75
    else:
        target_edges = (n_genes * 2, n_genes * 4)
        col_percentile = 90
        decay_factor = 0.65

    penalty_model = DegreePenalty(decay_factor=decay_factor)
    G_init = nx.from_pandas_adjacency(stability_matrix, create_using=nx.DiGraph)
    penalty_model.update_degrees(G_init)
    adjusted_matrix = stability_matrix.copy()
    for gene in adjusted_matrix.columns:
        p = penalty_model.get_penalty(gene)
        adjusted_matrix[gene] *= np.exp(-p)

    scores = adjusted_matrix.values.flatten()
    scores = scores[scores > 1e-5]
    if len(scores) == 0:
        return 0.0

    density = len(scores) / (n_genes ** 2)
    init_percentile = 100 * (1 - min(0.2, target_edges[1] / len(scores)))
    initial_thresh = np.percentile(scores, init_percentile)
    prev_edges = np.inf
    for iter in range(15):
        current_edges = np.sum(scores >= initial_thresh)
        if target_edges[0] <= current_edges <= target_edges[1] or current_edges == prev_edges:
            break
        prev_edges = current_edges
        error = current_edges - ((target_edges[0] + target_edges[1]) // 2)
        lr = 0.3 * (1 + np.tanh(error / 50))
        if current_edges > target_edges[1]:
            initial_thresh *= (1 + lr * (current_edges / target_edges[1]) ** 0.5)
        else:
            initial_thresh *= (1 - lr * (1 - current_edges / target_edges[0]) ** 0.5)
        initial_thresh = np.clip(initial_thresh, np.percentile(scores, 10), np.percentile(scores, 95))

    final_thresh = initial_thresh
    sorted_scores = np.sort(scores)[::-1]
    if np.sum(scores >= final_thresh) < target_edges[0]:
        final_thresh = sorted_scores[min(target_edges[0], len(sorted_scores) - 1)]
    return np.clip(final_thresh, sorted_scores[-1], sorted_scores[0])


# ====================连通图保证====================
def ensure_node_connectivity(G, stability_matrix):
    G = G.copy()
    isolated_nodes = [
        node for node in G.nodes()
        if G.in_degree(node) == 0 and G.out_degree(node) == 0
    ]
    if not isolated_nodes:
        return G

    added_edges = set()
    for node in isolated_nodes:
        try:
            out_edges = stability_matrix.loc[node].nlargest(5)
            in_edges = stability_matrix[node].nlargest(5)
            candidates = pd.concat([
                out_edges.rename('score').reset_index().assign(type='out'),
                in_edges.rename('score').reset_index().assign(type='in')
            ])
            valid_candidates = []
            for _, row in candidates.iterrows():
                if row['index'] == node:
                    continue
                if row['type'] == 'out':
                    edge = (node, row['index'])
                else:
                    edge = (row['index'], node)
                if not G.has_edge(*edge):
                    valid_candidates.append((row['score'], edge))
            if not valid_candidates:
                continue
            best_score, best_edge = max(valid_candidates, key=lambda x: x[0])
            added_edges.add(best_edge)
        except KeyError:
            continue

    for edge in added_edges:
        G.add_edge(edge[0], edge[1])
    return G


# ====================稳定性分析====================
def stability_analysis(X, genes, config):
    os.makedirs(config["output_dir"], exist_ok=True)
    stability = pd.DataFrame(0.0, index=genes, columns=genes)

    def process_run(run_id):
        np.random.seed(config["random_state"] + run_id)
        drop_num = np.random.randint(config["min_drop"], config["max_drop"] + 1)
        dropped = np.random.choice(len(genes), drop_num, replace=False)
        kept = np.setdiff1d(np.arange(len(genes)), dropped)
        current_genes = [genes[i] for i in kept]
        X_sub = X[:, kept]
        edges = []
        for target_idx in range(len(current_genes)):
            res = enhanced_polobag(X_sub, current_genes, target_idx, config)
            edges.extend([(reg, current_genes[target_idx]) for reg in res])
        return edges

    results = Parallel(n_jobs=8, max_nbytes='512M', batch_size=2)(
        delayed(process_run)(run_id) for run_id in tqdm(range(config["n_runs"]))
    )
    for edges in results:
        for src, tar in edges:
            stability.loc[src, tar] += 1
    stability = stability / config["n_runs"]
    stability.to_csv(os.path.join(config["output_dir"], "stability_scores.tsv"), sep='\t')
    return pd.DataFrame(stability, index=genes, columns=genes)


# ===================网络分析模块====================
def network_analysis(stability_matrix, config):
    init_threshold = determine_threshold(stability_matrix, config)
    n_genes = len(stability_matrix)
    avg_degree = np.mean(stability_matrix.values.flatten())
    if avg_degree < 0.1:
        target_range = (n_genes // 2, n_genes * 2)
    else:
        target_range = get_dynamic_range(n_genes)

    final_threshold = iterative_threshold_adjust(
        stability_matrix,
        init_threshold,
        target_range,
        max_attempts=8
    )

    G = nx.from_pandas_adjacency(stability_matrix >= final_threshold, create_using=nx.DiGraph)
    current_edges = G.number_of_edges()
    if current_edges < target_range[0]:
        candidate_edges = []
        for src in stability_matrix.index:
            for tar in stability_matrix.columns:
                if src != tar and not G.has_edge(src, tar):
                    candidate_edges.append((src, tar, stability_matrix.loc[src, tar]))
        candidate_edges.sort(key=lambda x: -x[2])
        for src, tar, score in candidate_edges[:target_range[0] - current_edges]:
            G.add_edge(src, tar)

    return G, final_threshold


# ========权重分析========
def evaluate_weight_combinations(lasso_scores, genie_scores, goldstandard, weight_range=(0.0, 1.0, 0.1)):
    evaluator = GRNEvaluator(goldstandard)
    weights = np.linspace(weight_range[0], weight_range[1],
                          num=int((weight_range[1] - weight_range[0]) / weight_range[2]) + 1)
    weights = np.round(weights, decimals=2)

    def safe_division(numerator, denominator, default=0.0):
        return numerator / denominator if denominator != 0 else default

    def process_weight(w1, w2):
        try:
            if not (0 <= w1 <= 1) or not (0 <= w2 <= 1):
                return None
            fused = lasso_scores * w1 + genie_scores * w2
            max_score = fused.max().max()
            if max_score < 1e-5 or np.isclose(max_score, 0):
                return None
            metrics = evaluator.comprehensive_evaluate(fused)
            tp = metrics['TP']
            fp = metrics['FP']
            fn = metrics['FN']
            metrics['Precision'] = safe_division(tp, (tp + fp))
            metrics['Recall (Sensitivity)'] = safe_division(tp, (tp + fn))
            precision = metrics['Precision']
            recall = metrics['Recall (Sensitivity)']
            metrics['F1'] = safe_division(2 * precision * recall, (precision + recall))
            return {'w_lasso': w1, 'w_genie': w2, **metrics}
        except Exception:
            return None

    results = []
    for w in weights:
        result = process_weight(w, 1 - w)
        if result:
            results.append(result)
    valid_df = pd.DataFrame(results)
    if valid_df.empty:
        raise ValueError("所有权重组合评估失败，请检查输入数据")
    return valid_df


def select_optimal_weight(weight_results, metric='ROC_AUC'):
    if weight_results.empty:
        raise ValueError("权重结果为空，无法选择最优")
    if metric not in weight_results.columns:
        available = ', '.join(weight_results.columns)
        raise KeyError(f"指标{metric}不存在，可用指标: {available}")
    clean_df = weight_results.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    if clean_df.empty:
        raise ValueError(f"指标{metric}全部为无效值")
    best_idx = clean_df[metric].idxmax()
    return {
        'weight_lasso': clean_df.loc[best_idx, 'w_lasso'],
        'weight_genie': clean_df.loc[best_idx, 'w_genie'],
        'best_score': clean_df.loc[best_idx, metric],
        'all_results': clean_df
    }


def visualize_weight_performance(weight_df, output_dir, dataset_name=""):
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1)
    plt.plot(weight_df['w_lasso'], weight_df['ROC_AUC'], color='#2c7bb6', marker='o', label='ROC AUC')
    plt.plot(weight_df['w_lasso'], weight_df['PR_AUC'], color='#d7191c', marker='s', label='PR AUC')
    plt.plot(weight_df['w_lasso'], weight_df['F1'], color='#008837', marker='^', label='F1 Score')
    plt.xlabel('Lasso Model Weight')
    plt.ylabel('Score Value')
    plt.title('Main Performance Metrics Trend')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(weight_df['w_lasso'], weight_df['Degree Centralization'], color='#7b3294', marker='D',
             label='Degree Centralization')
    plt.plot(weight_df['w_lasso'], weight_df['Avg Clustering'], color='#fdae61', marker='X', label='Avg Clustering')
    plt.xlabel('Lasso Model Weight')
    plt.ylabel('Network Metric Value')
    plt.title('Network Structure Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(weight_df['w_lasso'], weight_df['Precision'], color='#1a9641', marker='P', label='Precision')
    plt.plot(weight_df['w_lasso'], weight_df['Recall (Sensitivity)'], color='#a6611a', marker='*', label='Recall')
    plt.xlabel('Lasso Model Weight')
    plt.ylabel('Rate Value')
    plt.title('Precision-Recall Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(weight_df['w_lasso'], weight_df['TP'] + weight_df['FP'], color='#0571b0', marker='o',
             label='Total Predicted Edges')
    plt.plot(weight_df['w_lasso'], weight_df['TP'], color='#ca0020', marker='s', label='True Positives')
    plt.xlabel('Lasso Model Weight')
    plt.ylabel('Edge Count')
    plt.title('Edge Distribution Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Model Weight Optimization Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f'weight_performance_{dataset_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===================新增边可视化====================
def visualize_network_growth(lasso_network, genie_network, fused_network, gold_edges, output_path):
    all_genes = set(lasso_network.nodes()) | set(genie_network.nodes()) | set(fused_network.nodes())

    # 处理节点数不足的情况
    if len(all_genes) < 2:
        print("警告：网络节点数不足2，无法进行可视化")
        return

    base_network = nx.DiGraph()
    base_network.add_nodes_from(all_genes)

    lasso_edges = set(lasso_network.edges())
    genie_edges = set(genie_network.edges())
    fused_edges = set(fused_network.edges())

    common_edges = lasso_edges & genie_edges & fused_edges
    fusion_only_edges = fused_edges - (lasso_edges | genie_edges)
    improved_gold_edges = fusion_only_edges & gold_edges

    # 处理没有边的情况
    if not (common_edges or fusion_only_edges or improved_gold_edges):
        print("警告：网络中没有边，跳过可视化")
        return

    plt.figure(figsize=(15, 12))

    try:
        # 尝试使用spring_layout
        pos = nx.spring_layout(base_network, seed=42)
    except nx.NetworkXException:
        # 如果失败，使用随机布局
        print("警告：spring_layout失败，使用随机布局")
        pos = nx.random_layout(base_network, seed=42)
    # 绘制基础节点
    nx.draw_networkx_nodes(base_network, pos, node_size=300,
                           node_color='lightgray', alpha=0.7)

    legend_handles = []

    # 绘制公共边
    if common_edges:
        nx.draw_networkx_edges(base_network, pos, edgelist=common_edges,
                               edge_color='gray', width=1.5, alpha=0.3)
        legend_handles.append(Line2D([0], [0], color='gray', lw=2,
                                     label=f"Common Edges ({len(common_edges)})"))

    # 绘制新增边
    if fusion_only_edges:
        nx.draw_networkx_edges(base_network, pos, edgelist=fusion_only_edges,
                               edge_color='green', width=2.5, alpha=0.7)
        legend_handles.append(Line2D([0], [0], color='green', lw=2,
                                     label=f"Fusion New Edges ({len(fusion_only_edges)})"))

    # 高亮正确的新增金标准边
    if improved_gold_edges:
        nx.draw_networkx_edges(base_network, pos, edgelist=improved_gold_edges,
                               edge_color='red', width=4.0, alpha=1.0)
        legend_handles.append(Line2D([0], [0], color='red', lw=3,
                                     label=f"Correct New Predictions ({len(improved_gold_edges)})"))

        # 标记这些边的端点
        involved_nodes = set()
        for u, v in improved_gold_edges:
            involved_nodes.add(u)
            involved_nodes.add(v)

        nx.draw_networkx_nodes(base_network, pos, nodelist=list(involved_nodes),
                               node_size=500, node_color='gold', alpha=0.9)

    # 添加标签 - 只添加涉及的节点标签
    nx.draw_networkx_labels(base_network, pos, font_size=10,
                            font_weight='bold', font_color='darkblue')

    # 只在有图例项时添加图例
    if legend_handles:
        plt.legend(handles=legend_handles, loc='upper right', fontsize=12)

    # 添加统计信息
    stats_text = f"Total Nodes: {len(all_genes)}\n"

    if fusion_only_edges:
        accuracy = len(improved_gold_edges) / len(fusion_only_edges) * 100 if len(fusion_only_edges) > 0 else 0
        stats_text += (
            f"Total Edges Added: {len(fusion_only_edges)}\n"
            f"Correct Predictions: {len(improved_gold_edges)} "
            f"({accuracy:.1f}% accuracy)"
        )
    else:
        stats_text += "No new edges added by fusion model"

    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12,
                bbox={'facecolor': 'lightyellow', 'alpha': 0.7, 'pad': 10})

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_edge_contribution(lasso_scores, genie_scores, fused_scores, gold_edges, config, evaluator):
    lasso_thresh = determine_threshold(lasso_scores, config)
    genie_thresh = determine_threshold(genie_scores, config)
    fused_thresh = determine_threshold(fused_scores, config)

    def get_predicted_edges(scores, threshold):
        edges = set()
        for src in scores.index:
            for tar in scores.columns:
                if src == tar:
                    continue
                if scores.loc[src, tar] >= threshold:
                    edges.add((src, tar))
        return edges

    def edges_to_matrix(edges, ref_matrix):
        matrix = pd.DataFrame(0.0, index=ref_matrix.index, columns=ref_matrix.columns)
        for src, tar in edges:
            if src in matrix.index and tar in matrix.columns:
                matrix.loc[src, tar] = 1.0
        return matrix

    lasso_pred = get_predicted_edges(lasso_scores, lasso_thresh)
    genie_pred = get_predicted_edges(genie_scores, genie_thresh)
    fused_pred = get_predicted_edges(fused_scores, fused_thresh)
    base_matrix = edges_to_matrix(lasso_pred | genie_pred, lasso_scores)
    fused_matrix = edges_to_matrix(fused_pred, fused_scores)
    base_metrics = evaluator.evaluate(base_matrix, threshold=0.5)
    enhanced_metrics = evaluator.evaluate(fused_matrix, threshold=0.5)
    new_edges = fused_pred - (lasso_pred | genie_pred)
    correct_new = new_edges & gold_edges
    incorrect_new = new_edges - gold_edges

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    sizes = [len(correct_new), len(incorrect_new)]
    labels = [f'Correct Predictions\n{len(correct_new)}', f'Incorrect Predictions\n{len(incorrect_new)}']
    colors = ['#4caf50', '#f44336']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True, explode=(0.1, 0))
    ax1.set_title('Classification of Newly Added Edges')

    metrics = ['Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']
    base_values = [base_metrics[m] for m in metrics]
    enhanced_values = [enhanced_metrics[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width / 2, base_values, width, label='Base (Lasso+GENIE3)', color='#2196f3')
    ax2.bar(x + width / 2, enhanced_values, width, label='With New Edges', color='#4caf50')
    for i in range(len(metrics)):
        ax2.text(i - width / 2, base_values[i] + 0.02, f'{base_values[i]:.3f}', ha='center')
        ax2.text(i + width / 2, enhanced_values[i] + 0.02, f'{enhanced_values[i]:.3f}', ha='center')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Improvement from Newly Added Edges')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1.15)
    plt.suptitle('Fusion Model Edge Contribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(config["output_dir"], 'edge_contribution_analysis.png')
    plt.savefig(output_path, dpi=300)
    plt.close()


# ===================主要流程====================
def safe_zscore(arr):
    """处理标准差为0的情况"""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std

def safe_zscore(arr):
    """处理标准差为0的情况"""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def single_analysis_pipeline(input_file, config):
    df = pd.read_csv(input_file, sep='\t')
    genes = df.index.tolist()
    X = np.array([safe_zscore(df.loc[g]) for g in genes]).T
    os.makedirs(config["output_dir"], exist_ok=True)

    print("正在执行topo-polobag模型分析...")
    lasso_result = stability_analysis(X, genes, config)
    lasso_network, lasso_threshold = network_analysis(lasso_result, config)

    print("正在执行GENIE3分析...")
    genie_result = genie3_parallel(X, genes, n_trees=100, n_jobs=4)
    genie_network, genie_threshold = network_analysis(genie_result, config)
    genie_result.to_csv(os.path.join(config["output_dir"], "genie_scores.tsv"), sep='\t')

    evaluator = GRNEvaluator(config["gold_standard"])
    lasso_metrics = evaluator.comprehensive_evaluate(lasso_result, lasso_threshold)
    genie_metrics = evaluator.comprehensive_evaluate(genie_result, genie_threshold)
    pd.DataFrame(lasso_metrics, index=['Lasso']).to_csv(os.path.join(config["output_dir"], "lasso_evaluation.tsv"),
                                                        sep='\t', index_label='Model')
    pd.DataFrame(genie_metrics, index=['GENIE3']).to_csv(os.path.join(config["output_dir"], "genie3_evaluation.tsv"),
                                                         sep='\t', index_label='Model')

    print("\n=== 开始权重优化 ===")
    weight_results = evaluate_weight_combinations(lasso_result, genie_result, config["gold_standard"])
    dataset_name = os.path.basename(input_file).split('.')[0]
    visualize_weight_performance(weight_results, config["output_dir"], dataset_name)
    optimal = select_optimal_weight(weight_results)
    print(f"最优权重：Lasso={optimal['weight_lasso']:.2f}, GENIE3={optimal['weight_genie']:.2f}")
    print(f"最佳ROC AUC：{optimal['best_score']:.4f}")
    weight_results.to_csv(os.path.join(config["output_dir"], "weight_comparison.tsv"), sep='\t')
    fused_scores = lasso_result * optimal['weight_lasso'] + genie_result * optimal['weight_genie']

    print("\n=== 开始Top边融合评估 ===")
    top_fusion_results = evaluate_top_fusion(lasso_result, genie_result, evaluator, [50, 100, 150, 200, 250])
    top_fusion_results.to_csv(os.path.join(config["output_dir"], "top_fusion_results.csv"), index=False)
    plt.figure(figsize=(10, 6))
    top_fusion_results.set_index('TopN')[['ROC_AUC', 'PR_AUC', 'F1']].plot()
    plt.title("Top Edge Fusion Performance")
    plt.savefig(os.path.join(config["output_dir"], "top_fusion_performance.png"))
    plt.close()

    final_threshold = determine_threshold(fused_scores, config) if config[
                                                                       "final_threshold_method"] == "dynamic" else 0.5
    final_network = nx.from_pandas_adjacency(fused_scores >= final_threshold, create_using=nx.DiGraph)
    final_network = ensure_node_connectivity(final_network, fused_scores)

    # 新增边可视化
    gold_edges = set(evaluator.gt_edges)
    visualize_network_growth(
        lasso_network,
        genie_network,
        final_network,
        gold_edges,
        os.path.join(config["output_dir"], 'network_growth.png')
    )
    plot_edge_contribution(
        lasso_result,
        genie_result,
        fused_scores,
        gold_edges,
        config,
        evaluator
    )

    # 保存最终结果
    fused_scores.to_csv(os.path.join(config["output_dir"], "fused_scores.tsv"), sep='\t')
    nx.write_gexf(final_network, os.path.join(config["output_dir"], "final_network.gexf"))

    # 模型比较
    eval_results = {
        "Lasso": evaluator.comprehensive_evaluate(lasso_result, lasso_threshold),
        "GENIE3": evaluator.comprehensive_evaluate(genie_result, genie_threshold),
        "Fused": evaluator.comprehensive_evaluate(fused_scores, final_threshold)
    }
    comparison_df = pd.DataFrame(eval_results).T
    metric_order = ['Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'TP', 'FP', 'FN', 'TN',
                    'Degree Centralization', 'Assortativity', 'Avg Clustering', 'Max Degree', 'Min Degree']
    comparison_df = comparison_df.reindex(columns=metric_order)
    comparison_df.to_csv(os.path.join(config["output_dir"], "model_comparison.csv"), float_format="%.3f")

    # 性能比较图
    plt.figure(figsize=(10, 6))
    comparison_df[['ROC_AUC', 'PR_AUC']].plot(kind='bar', rot=0)
    plt.title("Model Performance Comparison")
    plt.savefig(os.path.join(config["output_dir"], "auc_comparison.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    comparison_df[['Degree Centralization', 'Max Degree']].plot(kind='bar', title="Network Centralization Comparison")
    plt.savefig(os.path.join(config["output_dir"], "centralization_comparison.png"))


def batch_analysis(data_dir="input", gold_dir="gold_standard", output_base="results"):
    for i in range(1, 6):
        print(f"\n==== 正在处理数据集 size10_{i} ====")
        input_file = os.path.join(data_dir, f"insilico_size10_{i}.txt")
        gold_file = os.path.join(gold_dir, f"insilico_size10_{i}_goldstandard.tsv")
        output_dir = os.path.join(output_base, f"size10_{i}")
        os.makedirs(output_dir, exist_ok=True)
        config = CONFIG.copy()
        config.update({"gold_standard": gold_file, "output_dir": output_dir})
        try:
            single_analysis_pipeline(input_file, config)
            print(f"数据集 size10_{i} 处理完成！")
        except Exception as e:
            print(f"处理数据集 size10_{i} 时出错: {str(e)}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    batch_analysis(data_dir="input", gold_dir="gold_standard", output_base="results")