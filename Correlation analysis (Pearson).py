"""
Correlation analysis (Pearson / Spearman / Kendall)

Requirement:
    pandas
    numpy
    matplotlib
    seaborn
    scipy

Example
-------
import pandas as pd
from Correlation_analysis_Pearson import correlation_analysis

df = pd.read_csv("your_dataset.csv")
result = correlation_analysis(
    df_input=df,
    features=["Alb", "Scr", "Urea"],   # or None to use all numeric cols
    method="pearson",
    plot_method="heatmap",
    output_dir="results"
)

print(result["str_result"])
corr_table = result["tables"]["correlation_matrix"]
p_table    = result["tables"]["pvalues"]
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _format_pvalue(p: float, decimal_places: int = 3):
    """Format p value with <0.001 style."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return np.nan
    threshold = 10 ** (-decimal_places)
    if 0 < p < threshold:
        return f"<{threshold:.{decimal_places}f}".rstrip("0").rstrip(".")
    return round(float(p), decimal_places)


def correlation_analysis(
    df_input: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "pearson",
    plot_method: str = "heatmap",
    hue: Optional[str] = None,
    decimal_places: int = 3,
    output_dir: Optional[str] = None,
    dpi: int = 600,
    image_format: str = "png",
) -> Dict:
    """
    相关性分析（Pearson / Spearman / Kendall）

    Parameters
    ----------
    df_input : DataFrame
        输入数据集.
    features : list of str or None
        纳入分析的变量名列表；若为 None，则自动选择所有数值型变量.
    method : {"pearson", "spearman", "kendall"}
        相关性方法.
    plot_method : {"heatmap", "clustermap", "pairplot"}
        绘图方式.
    hue : str or None
        pairplot 时的分组变量名（可选）.
    decimal_places : int
        相关系数和 p 值保留的小数位数.
    output_dir : str or None
        图片保存路径；若为 None，则不保存图片.
    dpi : int
        图片分辨率.
    image_format : str
        图片格式，例如 "png" 或 "jpeg".

    Returns
    -------
    result : dict
        {
          "str_result": 文本说明,
          "tables": {
                "correlation_matrix": DataFrame,
                "pvalues": DataFrame
          },
          "pics": {
                "heatmap"/"clustermap"/"pairplot": 文件路径（如有）
          }
        }
    """

    # 1. 选择变量
    if features is None:
        df_temp = df_input.select_dtypes(include="number").copy()
        features = df_temp.columns.tolist()
    else:
        missing = [f for f in features if f not in df_input.columns]
        if missing:
            raise ValueError(f"Columns not found in input dataframe: {missing}")
        df_temp = df_input[features].copy()

    # 去除缺失
    df_temp = df_temp.dropna()
    if df_temp.shape[0] < 3:
        raise ValueError(
            "Not enough non-missing observations for correlation analysis (need >= 3 rows)."
        )

    # 2. 相关系数矩阵
    corr = df_temp.corr(method=method)

    # 3. p 值矩阵
    pvals = np.zeros_like(corr.values, dtype=float)
    for i, col_i in enumerate(features):
        for j, col_j in enumerate(features):
            if method == "pearson":
                r, p = stats.pearsonr(df_temp[col_i], df_temp[col_j])
            elif method == "spearman":
                r, p = stats.spearmanr(df_temp[col_i], df_temp[col_j])
            elif method == "kendall":
                r, p = stats.kendalltau(df_temp[col_i], df_temp[col_j])
            else:
                raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
            pvals[i, j] = p

    corr_rounded = corr.round(decimal_places)
    pval_df = pd.DataFrame(pvals, index=features, columns=features)
    pval_fmt = pval_df.applymap(lambda x: _format_pvalue(x, decimal_places))

    # 4. 文本说明
    desc_lines = [
        f"Correlation method: {method}.",
        f"Number of observations after dropping missing values: {df_temp.shape[0]}.",
        "Correlation coefficients range from -1 (perfect negative) to 1 (perfect positive).",
        "P-values < 0.05 are typically considered statistically significant.",
    ]
    str_result = "\n".join(desc_lines)

    # 5. 绘图（可选）
    pics: Dict[str, str] = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if plot_method == "heatmap":
        plt.figure(figsize=(6, 5), dpi=dpi)
        sns.heatmap(
            corr,
            annot=True,
            fmt=f".{decimal_places}f",
            cmap=None,
            linewidths=0.3,
        )
        plt.tight_layout()
        if output_dir:
            fname = f"correlation_heatmap_{method}.{image_format}"
            path = os.path.join(output_dir, fname)
            plt.savefig(path, dpi=dpi, format=image_format, bbox_inches="tight")
            pics["heatmap"] = path
        plt.close()

    elif plot_method == "clustermap":
        g = sns.clustermap(
            corr,
            annot=True,
            fmt=f".{decimal_places}f",
            cmap=None,
        )
        if output_dir:
            fname = f"correlation_clustermap_{method}.{image_format}"
            path = os.path.join(output_dir, fname)
            g.savefig(path, dpi=dpi, format=image_format, bbox_inches="tight")
            pics["clustermap"] = path
        plt.close(g.fig)

    elif plot_method == "pairplot":
        cols_for_plot = features.copy()
        if hue is not None and hue in df_input.columns and hue not in cols_for_plot:
            cols_for_plot.append(hue)
        g = sns.pairplot(df_input[cols_for_plot].dropna(), hue=hue)
        if output_dir:
            fname = f"correlation_pairplot.{image_format}"
            path = os.path.join(output_dir, fname)
            g.savefig(path, dpi=dpi, format=image_format, bbox_inches="tight")
            pics["pairplot"] = path
        plt.close(g.fig)

    result = {
        "str_result": str_result,
        "tables": {
            "correlation_matrix": corr_rounded,
            "pvalues": pval_fmt,
        },
        "pics": pics,
    }
    return result
