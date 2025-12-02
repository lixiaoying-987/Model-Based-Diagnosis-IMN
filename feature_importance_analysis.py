import datetime
import pandas as pd
import matplotlib
import os

#解耦导入
from AnalysisFunction.celerytest.utils.analysis_dict import _analysis_dict
from AnalysisFunction.celerytest.utils.get_environment import _get_environment_info
from AnalysisFunction.celerytest.utils.data_standardization import data_standardization
from AnalysisFunction.celerytest.utils.fea_importance_models import _logistic2_cfeatures_importance, \
    _lasso_rfeatures_importance, _ridge_rfeatures_importance, _xgboost_rfeatures_importance, \
    _randomforest_rfeatures_importance, _adaboost_rfeatures_importance, _linear_rfeatures_importance, \
    _kneighb_rfeatures_importance, _LinearSVM_rfeatures_importance, _logisticL1_cfeatures_importance, \
    _xgboost_cfeatures_importance, _randomforest_cfeatures_importance, _adaboost_cfeatures_importance, \
    _gussnb_cfeatures_importance, _mlp_cfeatures_importance, _svm_cfeatures_importance, _kneighb_cfeatures_importance, \
    _cnb_cfeatures_importance, _lightgbm_cfeatures_importance, _DecisionTree_cfeatures_importance, \
    _GBDT_cfeatures_importance
from AnalysisFunction.celerytest.utils.round_dec import round_dec

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv  # noqa
import matplotlib as mpl
import traceback
mpl.rcParams['axes.grid'] = False
plt.rcParams["font.sans-serif"] = ['Times New Roman + SimSun']  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
plt.rcParams["ps.useafm"] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["pdf.fonttype"] = 42

random_model = ['LGBMClassifier', 'XGBClassifier', 'XGBRegressor', 'RandomForestClassifier', 'AdaBoostClassifier','DecisionTreeClassifier'
              'MLPClassifier', 'SVC', 'LogisticRegression', 'LogisticRegressionCV', 'RandomForestRegressor', 'GradientBoostingClassifier'
              'AdaBoostRegressor', 'LinearSVR', 'LassoCV', 'PCA', 'GaussianMixture', 'KMeans', 'SpectralClustering']

env_info = _get_environment_info(__file__)#环境导入

def features_importance(df_input, dependent_variable, independent_variable, top_features=10, model_type=1, standardization=True, output_dir=None, searching=True, dpi=600, image_format="jpeg", decimal_places=3, ):

    """
    重要度排序
    Input:
        df_input:DataFrame 输入的待处理数据
        dependent_variable:str 因变量
        independent_variable:list 自变量
        top_features:图表中展示的特征数量
        model_type:int 使用的模型编号
                    '_lasso_features_importance':LassoCV(),
                    '_ridge_features_importance':RidgeV(),
                    'xgboost_features_importance':XGBClassifier(),
        standardization:bool (2020/12/31) 是否先对数据进行标准化处理，默认为True,
        searching:bool 是否进行自动寻参，默认为True
        output_dir:str 图片存储路径
        image_format：图片保存格式
        decimal_places：结果保留位数
    """
    try:

        df_temp = df_input[independent_variable + [dependent_variable]].dropna().reset_index().drop(columns="index")
        if model_type != 15:
            value_max = 2000000
        else:
            value_max = 200000
        if df_temp.shape[0] * df_temp.shape[1] > value_max:
            return {
                "error": "数据量过大：样本数为"
                + str(df_temp.shape[0])
                + "，因子数："
                + str(df_temp.shape[1])
                + "。请减少样本量或者减少因子数量！(目前数据暂时只支持分析数据矩阵最大为"
                + str(value_max)
                +")"
            }

        if standardization:
            if model_type==17:
                mmethod = "MinMaxScaler"
            else:
                mmethod='StandardScaler'
            rresult_dict = data_standardization(df_temp, independent_variable, method=mmethod)
            _, df_temp, _, _, _, _ = _analysis_dict(rresult_dict)

        if model_type > 8 and len(pd.unique(df_temp[dependent_variable])) > 8:
            return {"error": "暂不允许类别数目大于5的情况。请检查因变量取值情况。"+"false-error"}

        model_name_map = {
            # 回归
            1: "Lasso Regression",
            2: "Ridge Regression",
            3: "XGBoost Regressor",
            4: "Random Forest Regressor",
            5: "AdaBoost Regressor",
            6: "Linear Regression",
            7: "K-Neighbors Regressor",
            8: "Linear SVM Regressor",
            # 分类
            9: "Logistic Regression (L1)",
            10: "XGBoost Classifier",
            11: "Random Forest Classifier",
            12: "AdaBoost Classifier",
            13: "Gaussian Naive Bayes",
            14: "MLP Classifier",
            15: "SVM Classifier",
            16: "K-Neighbors Classifier",
            17: "Complement Naive Bayes",
            18: "LightGBM Classifier",
            19: "Decision Tree Classifier",
            20: "Gradient Boosting Classifier",
            21: "Logistic Regression",
        }

        #模型字典
        model_dispatch = {
            # 回归
            1: (_lasso_rfeatures_importance, False),
            2: (_ridge_rfeatures_importance, False),
            3: (_xgboost_rfeatures_importance, True),
            4: (_randomforest_rfeatures_importance, True),
            5: (_adaboost_rfeatures_importance, True),
            6: (_linear_rfeatures_importance, True),
            7: (_kneighb_rfeatures_importance, True),
            8: (_LinearSVM_rfeatures_importance, True),
            # 分类
            9: (_logisticL1_cfeatures_importance, False),
            10: (_xgboost_cfeatures_importance, True),
            11: (_randomforest_cfeatures_importance, True),
            12: (_adaboost_cfeatures_importance, True),
            13: (_gussnb_cfeatures_importance, True),
            14: (_mlp_cfeatures_importance, True),
            15: (_svm_cfeatures_importance, True),
            16: (_kneighb_cfeatures_importance, True),
            17: (_cnb_cfeatures_importance, True),
            18: (_lightgbm_cfeatures_importance, True),
            19: (_DecisionTree_cfeatures_importance, True),
            20: (_GBDT_cfeatures_importance, True),
            21: (_logistic2_cfeatures_importance, False),
        }
        # plt.close()

        # 模型查找与执行
        model_config = model_dispatch.get(model_type)

        if not model_config:
            return {"error": f"指定的 model_type 无效: {model_type}"}

        model_function, requires_searching = model_config
        #映射赋值
        x_columns = independent_variable
        y_column = dependent_variable

        # 准备通用的函数参数
        call_args = {
            "df_input": df_temp,
            "x_columns": x_columns,
            "y_column": y_column,
            "top_features": top_features,
            "output_dir": output_dir,
            "dpi": dpi,
            "image_format": image_format,
        }
        #如果需要超参数搜索，添加searching
        if requires_searching:
            call_args["searching"] = searching

        # 使用字典解包 `**` 来调用函数，传入所有参数
        df_result, str_result, plot_name_dict, plot_name_dict_save = model_function(**call_args)
        plt.close()

        # 整合描述信息
        description_text = str_result  # str_result 是从模型函数返回的
        if model_type <= 5:
            description_text += "接下来可使用这些相关度较高的变量进行，利用左侧栏‘机器学习回归’进一步的细致化回归建模。"
        else:
            description_text += "接下来可使用这些相关度较高的变量进行，利用左侧栏‘机器学习分类’进一步的细致化分类建模。"

        model_name = model_name_map.get(model_type, f"未知模型 {model_type}") #模型名映射

        # 监控图片生成
        figures_list = []
        for i, (key, saved_file_name) in enumerate(plot_name_dict_save.items()):
            try:
                # 检查文件
                if os.path.exists(os.path.join(output_dir, saved_file_name)):
                    # 从文件名推断格式
                    base_name, ext = os.path.splitext(saved_file_name)
                    current_format = ext.strip('.')

                    figure_info = {
                        "name": f"重要度排序图_{i + 1}",
                        "file": {"png": f"{base_name}.png", current_format: saved_file_name},
                        "description": f"模型 {model_name} 生成的特征重要性排序图。"
                    }
                    figures_list.append(figure_info)
                else:
                    raise FileNotFoundError("模型函数报告已保存，但文件未找到。")
            except Exception as e:
                # 如果文件保存失败或路径处理出错，则记录失败信息
                failure_figure_info = {
                    "name": f"重要度排序图_{i + 1}",
                    "file": {},
                    "description": f"这个图生成失败: {str(e)}"
                }
                figures_list.append(failure_figure_info)

        tables_list = []
        df_result = df_result.applymap(lambda x: round_dec(x, Decimal_places=decimal_places))
        tables_list.append({
            "name": "重要度排序",
            "file": {"dataframe": df_result},
            "description": "包含各个特征的重要性得分及其排序的详细数据表。"
        })

        # 构建最终的返回字典
        result_dict = {
            "status": "success",
            "description": description_text,
            "tables": tables_list,
            "figures": figures_list,
            "environment": env_info
        }
        return result_dict
# 将这部分代码添加到 features_importance 函数的末尾，作为 try 块的补充
    except ValueError as e:
        # 捕获可预见的、主动抛出的错误
        return {
            "status": "error",
            "description": str(e),
            "tables": [],
            "figures": [],
            "environment": env_info
        }

    except Exception as e:
        # 捕获所有其他意外的运行时错误
        error_message = traceback.format_exc()
        print(error_message) # 在后台打印详细追溯信息，便于调试
        return {
            "status": "exception",
            "description": f"发生意外错误: {str(e)}\n\nTraceback:\n{error_message}",
            "tables": [],
            "figures": [],
            "environment": env_info
        }


    辅助工具函数，用于进行对应算法的特征重要性计算
"""

import pandas as pd
import numpy as np
import matplotlib

#解耦导入
from AnalysisFunction.celerytest.utils.Hyperparameters import GridDefaultRange
from AnalysisFunction.celerytest.utils.filter_dict_to_str import dictionary_string
from AnalysisFunction.celerytest.utils.horizontal_bar_plot import horizontal_bar_plot

matplotlib.use("AGG")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import RandomizedSearchCV
import matplotlib as mpl

mpl.rcParams['axes.grid'] = False
plt.rcParams["font.sans-serif"] = ['Times New Roman + SimSun']  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号)
plt.rcParams["ps.useafm"] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["pdf.fonttype"] = 42

random_model = ['LGBMClassifier', 'XGBClassifier', 'XGBRegressor', 'RandomForestClassifier', 'AdaBoostClassifier','DecisionTreeClassifier'
              'MLPClassifier', 'SVC', 'LogisticRegression', 'LogisticRegressionCV', 'RandomForestRegressor', 'GradientBoostingClassifier'
              'AdaBoostRegressor', 'LinearSVR', 'LassoCV', 'PCA', 'GaussianMixture', 'KMeans', 'SpectralClustering']


def _lasso_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    alpharange = np.logspace(-10, -2, 200, base=10)

    lasso_ = LassoCV(alphas=alpharange, cv=5,random_state=42).fit(x, y)
    param_dict = lasso_.get_params()
    param_dict["alpha"] = lasso_.alpha_
    str_result = "算法：lasso回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用lasso Regressor进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, lasso_.__class__.__name__
    )

    c1 = {"Variable": x_columns, "Weight Importance": abs(lasso_.coef_)}
    a1 = pd.DataFrame(c1)
    df_result = a1.sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    str_result += "\n注意：在使用Lasso进行重要度排序时，由于其线性特性，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。"

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _ridge_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    alpharange = np.logspace(-10, -2, 200, base=10)

    Ridge_ = RidgeCV(alphas=alpharange, cv=5).fit(x, y)
    param_dict = Ridge_.get_params()
    param_dict["alpha"] = Ridge_.alpha_
    str_result = "算法：岭回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Ridge Regressor进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, Ridge_.__class__.__name__
    )

    c1 = {"Variable": x_columns, "Weight Importance": abs(Ridge_.coef_)}
    a1 = pd.DataFrame(c1)
    df_result = a1.sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    str_result += "\n注意：在使用岭回归进行重要度排序时，由于其线性特性，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。"

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _xgboost_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    model: XGBOOST模型，如果不传则自动产生一个自动寻参后的XGBOOST模型
    searching: 是否自动寻参，默认为是
    output_dir:str 图片存储路径

    hyperparams: XGBClassifier params -- no selection yet
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Regression", XGBRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(XGBRegressor(), param_distributions=GridDefaultRange['XGBRegressor'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = XGBRegressor(**model.best_params_)
    else:
        model = XGBRegressor(random_state=42)
    # if searching:
    #     str_result = "采用XGBoost进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'XGBRegressor')
    # else:
    str_result = "算法：XGBoost回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用XGBoost进行变量重要度分析，模型参数为:\n" + dictionary_string(model.get_params(), model.__class__.__name__)
    model.fit(x, y)

    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]



def _randomforest_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Regression", RandomForestRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(RandomForestRegressor(), param_distributions=GridDefaultRange['RandomForestRegressor'],random_state=42, )
        #searcher = BayesSearchCV(RandomForestRegressor(),  search_spaces=BayesDefaultRange['RandomForestRegressor']  # 使用字符串键)
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = RandomForestRegressor(**model.best_params_).fit(x, y)
    else:
        model = RandomForestRegressor(random_state=42).fit(x, y)
    param_dict = model.get_params()

    # if searching:
    #     str_result = "采用Random Forrest Regressor进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'RandomForestRegressor')
    # else:
    str_result = "算法：随机森林回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Random Forrest Regressor进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _adaboost_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Regression", AdaBoostRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(AdaBoostRegressor(), param_distributions=GridDefaultRange['AdaBoostRegressor'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = AdaBoostRegressor(**model.best_params_).fit(x, y)
    else:
        model = AdaBoostRegressor(random_state=42).fit(x, y)
    param_dict = model.get_params()

    # if searching:
    #     str_result = "采用AdaBoost Regressor进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'AdaBoostRegressor')
    # else:
    str_result = "算法：AdaBoost回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用AdaBoost Regressor进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _linear_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    #获取模型参数
    param_dict = model.get_params()
    #将模型参数转化为字符串
    str_result = "算法：线性回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Linear Regression进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    #获取系数并排序
    df_result = pd.DataFrame({
        "Variable": x_columns,
        "Weight Importance": abs(model.coef_)
    }).sort_values(by="Weight Importance", ascending=False)
    #获取重要度最高的特征
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    str_result += "\n注意：在使用线性回归进行重要度排序时，由于其线性特性，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。"

    plot_name_dict = {}
    #绘制图形
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _kneighb_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna().sample(frac=1, random_state=42).reset_index(drop=True)
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", KNeighborsRegressor())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(KNeighborsRegressor(),
                                      param_distributions=GridDefaultRange['KNeighborsRegressor'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = KNeighborsRegressor(**model.best_params_).fit(x, y)
    else:
        model = KNeighborsRegressor().fit(x, y)
    #获取模型参数
    param_dict = model.get_params()
    #将模型参数转化为字符串
    str_result = "算法：K近邻回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用K近邻回归进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    #获取系数并排序
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    #获取重要度最高的特征
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    #绘制图形
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _LinearSVM_rfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", LinearSVR())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(LinearSVR(), param_distributions=GridDefaultRange['LinearSVR'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = LinearSVR(**model.best_params_).fit(x, y)
    else:
        model = LinearSVR(random_state=42).fit(x, y)
    param_dict = model.get_params()
    #将模型参数转化为字符串
    str_result = "算法：线性支持向量机回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用线性支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'LinearSVR')
    # else:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
    #         param_dict, model.__class__.__name__)
    #排序
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)
    #获取重要度最高的特征
    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    #画图
    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]



# ----------分类重要度排序-----------
def _logisticL1_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    crange = np.logspace(-9, 1, 200, base=10)

    logicv_ = LogisticRegressionCV(Cs=crange, cv=5, penalty="l1", solver="saga",random_state=42).fit(
        x, y
    )
    param_dict = logicv_.get_params()
    param_dict["C"] = logicv_.C_
    str_result = "算法：逻辑回归分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用L1正则化的Logistic回归进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, logicv_.__class__.__name__
    )

    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(logicv_.coef_[0])}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    str_result += "\n注意：在使用Logistiv+L1进行重要度排序时，由于其指数部分的线性形式，将默认所有变量具有相同量纲和值域。这一假设在勾选数据标准化后可认为成立。如果线性模型(与常识对比)在此数据集上表现较差，可考虑使用非线性模型如XGBoost。"

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _logistic2_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]
    # 对因变量进行有序编码
    encoder = OrdinalEncoder()
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

    logicv = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', random_state=42).fit(
        x, y_encoded
    )
    param_dict = logicv.get_params()
    param_dict["C"] = logicv.C
    str_result = "算法：有序逻辑回归模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用有序逻辑回归进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, logicv.__class__.__name__
    )

    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(logicv.coef_[0])}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    str_result += "\n注意：函数对结局变量进行了有序编码。这是有序逻辑回归所必需的。用户需要确保结局变量是有序的分类变量。"
    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _xgboost_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    model: XGBOOST模型，如果不传则自动产生一个自动寻参后的XGBOOST模型
    searching: 是否自动寻参，默认为是
    output_dir:str 图片存储路径

    hyperparams: XGBClassifier params -- no selection yet
    """
    x = df_input[x_columns]
    y = df_input[y_column]

    if searching:
        # searcher = RandSearcherCV("Classification", XGBClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(XGBClassifier(), param_distributions=GridDefaultRange['XGBClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = XGBClassifier(**model.best_params_).fit(x, y)
    else:
        model = XGBClassifier(random_state=42).fit(x, y)
    # if searching:
    #     str_result = "采用极端梯度提升树(XGBOOST)进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'XGBClassifier')
    # else:
    str_result = "算法：XGBoost分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用极端梯度提升树(XGBOOST)进行变量重要度分析，模型参数为:\n" + dictionary_string(
            model.get_params(), model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )
    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _randomforest_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("Classification", RandomForestClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(RandomForestClassifier(),param_distributions=GridDefaultRange['RandomForestClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = RandomForestClassifier(**model.best_params_).fit(x, y)
    else:
        model = RandomForestClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()


    # if searching:
    #     str_result = "采用Random Forrest Classifier进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'RandomForestClassifier')
    # else:
    str_result = "算法：随机森林模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用Random Forrest Classifier进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _adaboost_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", AdaBoostClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(AdaBoostClassifier(), param_distributions=GridDefaultRange['AdaBoostClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = AdaBoostClassifier(**model.best_params_).fit(x, y)
    else:
        model = AdaBoostClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()

    str_result = "算法：AdaBoost分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用AdaBoost Classifier进行变量重要度分析，模型参数为:\n" + dictionary_string(
                param_dict, model.__class__.__name__)


    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _DecisionTree_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:

        searcher = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=GridDefaultRange['DecisionTreeClassifier'],random_state=42, )
        model = searcher.fit(x, y)
        model = DecisionTreeClassifier(**model.best_params_).fit(x, y)
    else:
        model = DecisionTreeClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：决策树分类模型\n"
    str_result +="变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用DecisionTree进行变量重要度分析，模型参数为:\n" + dictionary_string(
                param_dict, model.__class__.__name__)


    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _GBDT_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        searcher = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=GridDefaultRange['GradientBoostingClassifier'],random_state=42, )
        model = searcher.fit(x, y)
        model = GradientBoostingClassifier(**model.best_params_).fit(x, y)
    else:
        model = GradientBoostingClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：GBDT分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用GBDT进行变量重要度分析，模型参数为:\n" + dictionary_string(
                param_dict, model.__class__.__name__)


    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": abs(model.feature_importances_)}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])

    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]

def _gussnb_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", GaussianNB())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(GaussianNB(), param_distributions=GridDefaultRange['GaussianNB'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = GaussianNB(**model.best_params_).fit(x, y)
    else:
        model = GaussianNB().fit(x, y)
    param_dict = model.get_params()


    # if searching:
    #     str_result = "采用高斯朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'GaussianNB')
    # else:
    str_result = "算法：高斯朴素贝叶斯分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用高斯朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _cnb_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", ComplementNB())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(ComplementNB(), param_distributions=GridDefaultRange['ComplementNB'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = ComplementNB(**model.best_params_).fit(x, y)
    else:
        model = ComplementNB().fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：补朴素贝叶斯分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用补朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用补朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'ComplementNB')
    # else:
    #     str_result = "采用补朴素贝叶斯分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _mlp_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", MLPClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(MLPClassifier(), param_distributions=GridDefaultRange['MLPClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = MLPClassifier(**model.best_params_).fit(x, y)
    else:
        model = MLPClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：神经网络分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用多层感知器（神经网络）分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用多层感知器（神经网络）分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'XGBRegressor')
    # else:
    #     str_result = "采用多层感知器（神经网络）分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _svm_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", SVC())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(SVC(), param_distributions=GridDefaultRange['SVC'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = SVC(**model.best_params_).fit(x, y)
    else:
        model = SVC(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：支持向量机分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'SVC')
    # else:
    #     str_result = "采用支持向量机分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _kneighb_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", KNeighborsClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=GridDefaultRange['KNeighborsClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = KNeighborsClassifier(**model.best_params_).fit(x, y)
    else:
        model = KNeighborsClassifier().fit(x, y)
    param_dict = model.get_params()

    # if searching:
    #     str_result = "采用K近邻分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'KNeighborsClassifier')
    # else:
    str_result = "算法：K近邻分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用K近邻分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
            param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]


def _lightgbm_cfeatures_importance(
    df_input,
    x_columns,
    y_column,
    top_features,
    searching=True,
    output_dir=None,
    dpi=600,
    image_format="jpeg",
):
    """
    df_input:Dataframe
    x_columns:自变量list
    y_column：因变量str
    top_features:图表中展示的特征数量
    (会剔除空值)
    output_dir:str 图片存储路径

    hyperparams: alpha(alpharange), cv, tol
    """
    dftemp = df_input[x_columns + [y_column]].dropna()
    x = dftemp[x_columns]
    y = dftemp[y_column]

    if searching:
        # searcher = RandSearcherCV("classification", LGBMClassifier())
        # model = searcher(x, y)  # ; searcher.report()
        searcher = RandomizedSearchCV(LGBMClassifier(), param_distributions=GridDefaultRange['LGBMClassifier'],random_state=42, )
        # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
        model = searcher.fit(x, y)
        model = LGBMClassifier(**model.best_params_).fit(x, y)
    else:
        model = LGBMClassifier(random_state=42).fit(x, y)
    param_dict = model.get_params()
    str_result = "算法：LinghtGBM分类模型\n"
    str_result += "变量：结局变量：{}，特征变量：{}\n".format(y_column, ', '.join(x_columns))
    str_result += "采用LightGBM分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
        param_dict, model.__class__.__name__)
    # if searching:
    #     str_result = "采用LightGBM分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(model.best_params_, 'LGBMClassifier')
    # else:
    #     str_result = "采用LightGBM分类算法进行变量重要度分析，模型参数为:\n" + dictionary_string(
    #         param_dict, model.__class__.__name__)
    weight_im = abs(
        permutation_importance(
            model, x, y, n_repeats=10, random_state=0
        ).importances_mean
    )
    weight_im = weight_im / sum(weight_im)
    df_result = pd.DataFrame(
        {"Variable": x_columns, "Weight Importance": weight_im}
    ).sort_values(by="Weight Importance", ascending=False)

    top_list = list(df_result.head(top_features)["Variable"])
    str_result += "\n重要度最高的前{0}个变量（由高到低）分别为：{1}。\n".format(
        len(top_list), str(top_list)[1:-1]
    )

    plot_name_dict = {}
    if output_dir is not None:
        plot_name_dict = horizontal_bar_plot(
            df_result.head(top_features).sort_values(
                by="Weight Importance", ascending=True
            ),
            "Variable",
            "Weight Importance",
            "Feature Importance (Coefficient)",
            output_dir,
            dpi=dpi,
            image_format=image_format,
        )
    return df_result, str_result, plot_name_dict["pics"], plot_name_dict["save_pics"]
