# -*- coding: utf-8 -*-
"""
rf_binary_classification.py

A self-contained Random Forest binary classification pipeline.

Features
--------
- Train/validation/test split (with stratification)
- Optional hyperparameter search via RandomizedSearchCV
- Train & test ROC curves, AUC, confusion matrices
- Feature importance plot
- SHAP-based model explanation (summary & waterfall)
- Returns a result dictionary: metrics, figures, model

Author: Kunbo / ChatGPT
Date: 2025-11-24
"""

import os
import time
import random
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

import shap  # noqa: E402
import scikitplot as skplt  # noqa: E402


# 全局绘图字体设置（Times + SimSun，方便中英文图）
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["ps.useafm"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42


def _ensure_dir(path: str) -> str:
    """Ensure a directory exists."""
    if path is None:
        return ""
    os.makedirs(path, exist_ok=True)
    if not path.endswith(os.sep):
        path = path + os.sep
    return path


def _save_fig(
    fig: plt.Figure,
    save_dir: str,
    prefix: str,
    fmt: str = "png",
    dpi: int = 300,
    timestamp: Optional[str] = None,
) -> str:
    """Save a matplotlib figure and return the filename."""
    save_dir = _ensure_dir(save_dir)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{timestamp}.{fmt}"
    fig.savefig(os.path.join(save_dir, filename), dpi=dpi, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return filename


def _binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute basic binary metrics given probs and a threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    # recall == sensitivity
    sensitivity = recall_score(y_true, y_pred)
    # specificity = recall on negative class
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1": float(f1),
    }


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    class_labels: List[Any],
    title: str,
    save_dir: Optional[str],
    pic_format: str,
    dpi: int,
    timestamp: str,
) -> Optional[str]:
    """Plot confusion matrix heatmap, return filename if saved."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[f"预测: {class_labels[0]}", f"预测: {class_labels[1]}"],
        yticklabels=[f"真实: {class_labels[0]}", f"真实: {class_labels[1]}"],
        ylabel="真实标签",
        xlabel="预测标签",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_dir is not None:
        fname = _save_fig(fig, save_dir, prefix=title.replace(" ", "_"), fmt=pic_format, dpi=dpi, timestamp=timestamp)
        return fname
    else:
        plt.close(fig)
        return None


def _plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_dir: Optional[str],
    pic_format: str,
    dpi: int,
    timestamp: str,
) -> Optional[str]:
    """Plot ROC curve, return filename if saved."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax.plot(fpr, tpr, lw=2, alpha=0.9, label=f"AUC = {auc:.3f}")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2, linestyle="-.")

    if save_dir is not None:
        fname = _save_fig(fig, save_dir, prefix=title.replace(" ", "_"), fmt=pic_format, dpi=dpi, timestamp=timestamp)
        return fname
    else:
        plt.close(fig)
        return None


def _plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    save_dir: Optional[str],
    pic_format: str,
    dpi: int,
    timestamp: str,
) -> Optional[str]:
    """Plot RandomForest feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(6, max(4, len(feature_names) * 0.3)), dpi=dpi)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    ax.barh(range(len(sorted_names)), sorted_importances[::-1])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("RandomForest Feature Importances")
    fig.tight_layout()

    if save_dir is not None:
        fname = _save_fig(
            fig,
            save_dir,
            prefix="rf_feature_importance",
            fmt=pic_format,
            dpi=dpi,
            timestamp=timestamp,
        )
        return fname
    else:
        plt.close(fig)
        return None


def _plot_ks(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Optional[str],
    pic_format: str,
    dpi: int,
    timestamp: str,
) -> Optional[str]:
    """Plot KS statistic curve using scikit-plot."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    skplt.metrics.plot_ks_statistic(y_true, np.vstack([1 - y_prob, y_prob]).T, ax=ax)
    ax.set_title("KS Statistic Plot")
    fig.tight_layout()

    if save_dir is not None:
        fname = _save_fig(fig, save_dir, prefix="ks_curve", fmt=pic_format, dpi=dpi, timestamp=timestamp)
        return fname
    else:
        plt.close(fig)
        return None


def _run_shap_explanation(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_dir: Optional[str],
    pic_format: str,
    dpi: int,
    timestamp: str,
    max_display: int = 20,
) -> Dict[str, Optional[str]]:
    """
    Run SHAP explanation for RandomForestClassifier.

    Returns a dict with filenames of:
        - shap_summary_beeswarm
        - shap_summary_bar
        - shap_waterfall_sample (for a single sample)
    """
    if save_dir is None:
        # 如果没有保存路径，只计算不画图
        return {"shap_summary_beeswarm": None, "shap_summary_bar": None, "shap_waterfall_sample": None}

    # 使用 TreeExplainer（对树模型更快）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)[1]  # 取正类的 shap 值

    # Beeswarm summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=max_display)
    fig = plt.gcf()
    beeswarm_name = _save_fig(
        fig,
        save_dir,
        prefix="shap_summary_beeswarm",
        fmt=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    # Bar importances
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=max_display)
    fig_bar = plt.gcf()
    bar_name = _save_fig(
        fig_bar,
        save_dir,
        prefix="shap_summary_bar",
        fmt=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    # Waterfall for one sample（选择第一个样本）
    sample_idx = 0
    sample_shap_values = shap_values[sample_idx]
    sample_features = X_test.iloc[sample_idx, :]

    explanation_obj = shap.Explanation(
        values=sample_shap_values,
        base_values=explainer.expected_value[1],
        data=sample_features.values,
        feature_names=X_test.columns.tolist(),
    )

    shap.waterfall_plot(explanation_obj, max_display=max_display, show=False)
    fig_wf = plt.gcf()
    waterfall_name = _save_fig(
        fig_wf,
        save_dir,
        prefix="shap_waterfall_sample",
        fmt=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    return {
        "shap_summary_beeswarm": beeswarm_name,
        "shap_summary_bar": bar_name,
        "shap_waterfall_sample": waterfall_name,
    }


def run_random_forest_classification(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits_cv: int = 5,
    do_hyperparam_search: bool = True,
    n_iter_search: int = 30,
    scoring: str = "roc_auc",
    save_dir: Optional[str] = None,
    pic_format: str = "png",
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Run a complete RandomForest binary classification pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target.
    target : str
        Name of the binary target column.
    features : list of str
        List of feature column names used for modeling.
    test_size : float, default=0.2
        Proportion of data held out for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    n_splits_cv : int, default=5
        Number of folds for internal CV when searching hyperparameters.
    do_hyperparam_search : bool, default=True
        Whether to run RandomizedSearchCV on RandomForest hyperparameters.
    n_iter_search : int, default=30
        Number of parameter settings that are sampled in RandomizedSearchCV.
    scoring : str, default="roc_auc"
        Scoring metric used in CV.
    save_dir : str or None
        Directory to save figures/model. If None, figures will not be saved.
    pic_format : str, default="png"
        Figure format: "png", "pdf", "svg", etc.
    dpi : int, default=300
        DPI for figures.

    Returns
    -------
    result : dict
        {
            "description": str,
            "metrics": {
                "train": {...},
                "test": {...},
            },
            "model": RandomForestClassifier,
            "figures": {
                "train_roc": filename or None,
                "test_roc": filename or None,
                "train_confusion_matrix": filename or None,
                "test_confusion_matrix": filename or None,
                "feature_importance": filename or None,
                "ks_curve": filename or None,
                "shap_summary_beeswarm": filename or None,
                "shap_summary_bar": filename or None,
                "shap_waterfall_sample": filename or None,
            },
        }
    """
    t0 = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(1, 999))

    # --- 数据准备 ---
    df = df[features + [target]].dropna().copy()
    y_raw = df[target].values
    # 二值化：如果标签不是 0/1，则按排序后映射为 0/1
    unique_labels = np.sort(np.unique(y_raw))
    if len(unique_labels) != 2:
        raise ValueError(f"仅支持二分类，当前 target={target} 有 {len(unique_labels)} 个取值: {unique_labels}")

    if set(unique_labels) != {0, 1}:
        y = label_binarize(y_raw, classes=unique_labels).ravel()
        label_map = dict(zip(unique_labels, [0, 1]))
    else:
        y = y_raw.astype(int)
        label_map = {0: 0, 1: 1}

    X = df[features].copy()

    # --- 训练/测试划分 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # --- 构建基础模型 ---
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )

    # --- 超参数搜索（可选） ---
    if do_hyperparam_search:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        search_info = f"RandomizedSearchCV 最佳参数：\n{search.best_params_}\n"
    else:
        model = base_model
        model.fit(X_train, y_train)
        search_info = "未进行超参数搜索，使用默认 RandomForest 参数。\n"

    # --- 预测与指标 ---
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # 简单用 0.5 作为阈值（也可以在需要时再加 Youden 优化）
    threshold = 0.5

    metrics_train = _binary_metrics(y_train, y_train_prob, threshold=threshold)
    metrics_test = _binary_metrics(y_test, y_test_prob, threshold=threshold)

    # --- 绘图 & SHAP ---
    figures: Dict[str, Optional[str]] = {}

    figures["train_roc"] = _plot_roc(
        y_train,
        y_train_prob,
        title="ROC curve (Train)",
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )
    figures["test_roc"] = _plot_roc(
        y_test,
        y_test_prob,
        title="ROC curve (Test)",
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    figures["train_confusion_matrix"] = _plot_confusion_matrix(
        y_train,
        y_train_prob,
        threshold=threshold,
        class_labels=list(label_map.keys()),
        title="Train Confusion Matrix",
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )
    figures["test_confusion_matrix"] = _plot_confusion_matrix(
        y_test,
        y_test_prob,
        threshold=threshold,
        class_labels=list(label_map.keys()),
        title="Test Confusion Matrix",
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    figures["feature_importance"] = _plot_feature_importance(
        model,
        feature_names=features,
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    figures["ks_curve"] = _plot_ks(
        y_test,
        y_test_prob,
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
    )

    shap_figs = _run_shap_explanation(
        model,
        X_train=X_train,
        X_test=X_test,
        save_dir=save_dir,
        pic_format=pic_format,
        dpi=dpi,
        timestamp=timestamp,
        max_display=min(20, len(features)),
    )
    figures.update(shap_figs)

    elapsed = time.time() - t0
    description = (
        f"使用 RandomForestClassifier 对 {target} 进行二分类建模。\n"
        f"样本总数 N={len(df)}, 特征数={len(features)}。\n"
        f"{search_info}"
        f"训练集 AUC={metrics_train['AUC']:.3f}，测试集 AUC={metrics_test['AUC']:.3f}。\n"
        f"总耗时：{elapsed:.2f} 秒。"
    )

    result = {
        "description": description,
        "metrics": {
            "train": metrics_train,
            "test": metrics_test,
        },
        "model": model,
        "figures": figures,
        "threshold": threshold,
        "features": features,
        "label_map": label_map,
    }
    return result


if __name__ == "__main__":
    """
    Example usage:

    df = pd.read_csv("your_data.csv")
    result = run_random_forest_classification(
        df=df,
        target="Outcome",
        features=[c for c in df.columns if c != "Outcome"],
        test_size=0.2,
        random_state=42,
        do_hyperparam_search=True,
        save_dir="./rf_results",
    )
    print(result["description"])
    print(result["metrics"])
    """
    pass
