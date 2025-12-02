import os
import random
import datetime
import pickle
import math
import time
import pandas as pd
import numpy as np
import matplotlib
import joblib
from skopt import BayesSearchCV
from statsmodels.formula.api import ols
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import os
from joblib.externals.loky.backend.resource_tracker import ResourceTracker
from AnalysisFunction.utils_ml.GeneticSearch_ML import GeneticSearchCV
from AnalysisFunction.utils_ml.Hyper import HyperoptSearchCV
from AnalysisFunction.utils_ml.params import BayesDefaultRange, GeneticDefaultRange, HyperDefaultRange

matplotlib.use("AGG")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
#
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC

from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from AnalysisFunction.utils_ml.FeatrueSelect import mrmr_classif
from AnalysisFunction.utils_ml.FeatrueSelect import ReliefF

from sklearn.metrics import precision_recall_curve, make_scorer
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    mutual_info_score,
    v_measure_score,
    normalized_mutual_info_score,
    silhouette_samples,
)

from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.preprocessing import label_binarize

from xgboost import plot_importance
import AnalysisFunction.X_5_SmartPlot as x5
from AnalysisFunction.X_5_SmartPlot import plot_calibration_curve
from AnalysisFunction.X_5_SmartPlot import calculate_net_benefit
from AnalysisFunction.X_5_SmartPlot import plot_decision_curves
from AnalysisFunction.X_1_DataGovernance import data_standardization
from AnalysisFunction.X_1_DataGovernance import _analysis_dict
from AnalysisFunction.X_2_DataSmartStatistics import comprehensive_smart_analysis

from AnalysisFunction.utils_ml import filtering, dic2str, round_dec, save_fig
from AnalysisFunction.utils_ml import (
    classification_metric_evaluate,
    regression_metric_evaluate,
)
from AnalysisFunction.utils_ml import (
    make_class_metrics_dict,
    make_regr_metrics_dict,
    multiclass_metric_evaluate,
)
from AnalysisFunction.utils_ml import ci

# from AnalysisFunction.utils_ml import (
#     GridSearcherCV,
#     RandSearcherCV,
#     GridSearcherSelf,
#     RandSearcherSelf,
# )
from AnalysisFunction.utils_ml.params import GridDefaultRange, BayesDefaultRange
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from AnalysisFunction.utils_ml.params import RandDefaultRange
from AnalysisFunction.utils_ml.auc_delong import delong_roc_test

from functools import reduce
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.utils import check_fitted
from yellowbrick.classifier import ConfusionMatrix
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


def _send(self, cmd, name, rtype):
    if len(name) > 512:
        # posix guarantees that writes to a pipe of less than PIPE_BUF
        # bytes are atomic, and that PIPE_BUF >= 512
        raise ValueError("name too long")
    msg = f"{cmd}:{name}:{rtype}\n".encode()
    nbytes = os.write(self._fd, msg)
    assert nbytes == len(msg)

ResourceTracker._send = _send


def ML_Classfication(
    df,
    group,
    features,
    decimal_num=3,
    validation_ratio=0.15,
    test_ratio=0.15,
    scoring="roc_auc",
    method="KNeighborsClassifier",
    isKFold="cross",
    n_splits=10,
    explain=True,
    shapSet=2,
    explain_numvar=2,
    explain_sample=2,
    shap_catter=False,
    shap_catter_feas=[],
    searching="default",
    auto_model="RandomizedSearchCV",
    validationCurve=False,
    smooth=False,
    savePath=None,
    style='lancet',
    dpi=600,
    picFormat="jpeg",
    label="LABEL",#标签名
    testSet=False,
    trainSet=False,
    modelSave=True,
    datasave=False,
    testLabel=2,
    trainLabel=0,
    randomState=42,
    resultType=0,
    DCA_cut=0,
    shap_waterfall=True,
    **kwargs,
):
    def _plot_conffusion_matrix(clf, x_data, y_data, un_group, resThreshold, name, dic_name, str_time, savePath=None,
                                picFormat='jpeg', dpi=600):
        plot_name_dict, plot_name_dict_save = {}, {}
        from sklearn.metrics import confusion_matrix
        y_prob = clf.predict_proba(x_data)[:, 1]  # 获取属于正类的概率
        threshold = resThreshold
        # 将概率转换为类别标签
        y_pred_custom = (y_prob > threshold).astype(int)
        con_mx = confusion_matrix(y_data, y_pred_custom, labels=un_group)
        fig, axes = plt.subplots(figsize=(6, 6), dpi=dpi)

        cmap = plt.get_cmap('Blues')
        plt.imshow(con_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()

        #标签
        indices = range(len(con_mx))
        plt.xticks(indices, un_group)
        plt.yticks(indices, un_group)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        thresh = con_mx.max() / 2.
        for i, j in np.ndindex(con_mx.shape):
            plt.text(j, i, format(con_mx[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if con_mx[i, j] > thresh else 'black')

        plt.tight_layout()

        plot_name = name + str_time + ".png"
        plot_name_save = name + str_time + "." + picFormat
        plt.savefig(savePath + plot_name, dpi=dpi, format='png', bbox_inches='tight')
        plt.savefig(savePath + plot_name_save, dpi=dpi, format=picFormat, bbox_inches='tight')
        plt.close(fig)
        plot_name_dict.update({dic_name: plot_name})
        plot_name_dict_save.update({dic_name: plot_name_save})
        return plot_name_dict, plot_name_dict_save

    """
    机器学习分类分析

    Input:
        df_input:DataFrame 输入的待处理数据
        group_name:str 分组名
        validation_ratio:float 验证集比例 
        test_ratio:float 测试集比例 
        isKFold:str "cross"交叉验证，nest:嵌套交叉验证 resample：重采样 resample1：无 
        scoring:str 目标评价指标
        method:str 使用的机器学习分类方法/模型
                    'LogisticRegression':LogisticRegression(**kwargs),
                    'XGBClassifier':XGBClassifier(**kwargs),
                    'RandomForestClassifier':RandomForestClassifier(**kwargs),
                    'SVC':SVC(**kwargs),
                    'KNeighborsClassifier':KNeighborsClassifier(**kwargs),
        n_splits:int 交叉验证的子集数目
        explain:bool 是否进行模型解释
        explain_numvar:int 需要解释的变量数
        explain_sample:int 需要例释的样本数
        searching:str 是否进行自动寻参，默认为否 ("default", "handle", "auto")
        auto_model:str 自动寻参的方法 ("RandomizedSearchCV", "GridSearchCV", "BayesSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "GeneticSearchCV", "Hyperopt")
        validationCurve:bool 是否绘制交叉验证/重采样折叠的 ROC 曲线
        smooth:bool 是否对测试集 ROC 曲线进行平滑处理
        savePath:str 图片存储路径
        style:str 绘图风格 ("lancet", "nejm", "jama", "npg") # 保留自 ML_combine.py
        dpi:int 图片分辨率
        picFormat:str 图片格式 ("jpeg", "png", "pdf", "svg"等)
        label:str 数据集中用于划分测试集/训练集的标签列名 # 标签名
        testSet:bool 是否根据 label 列划分测试集
        trainSet:bool 当 testSet=True 且 isKFold='resample1' 时，是否根据 label 列划分训练集
        modelSave:bool 是否保存训练好的模型
        datasave:bool 是否保存带有预测结果的数据表
        testLabel:int/str 测试集的标签值 # 测试集标签
        trainLabel:int/str 训练集的标签值 (仅用于 testSet=True 且 isKFold='resample1') # 训练集标签
        randomState:int 随机状态种子，用于复现结果
        resultType:int 结果表格中置信区间/标准差的显示类型 (0: SD, 1: CI)
        DCA_cut:float 决策曲线分析图的横坐标截断值
        **kwargs:dict 使用机器学习分类方法的参数

    Return:
        result_dict: dict 包含分析结果的字典
            str_result: dict 分析结果描述
            tables: dict 结果数据表字典
            pics: dict 主要图片文件名字典 (用于显示)
            save_pics: dict 所有保存的图片文件名字典 (用于下载)
            model: dict 保存的模型信息
    """
    palette_dict = {
        'lancet': ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF",
                   "#ADB6B6FF","#1B1919FF"],
        'nejm': ["#BC3C29FF", "#0072B5FF", "#E18727FF", "#20854EFF", "#7876B1FF", "#6F99ADFF", "#FFDC91FF", "#EE4C97FF",
                 "#BC3C29FF"],
        'jama': ["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF", "#79AF97FF", "#6A6599FF", "#80796BFF", "#374E55FF",
             "#DF8F44FF"],
        'npg': ["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF",
                "#7E6148FF", "#B09C85FF"]}

    name_dict = { # 模型名称字典
        "LogisticRegression": "logistic",
        "XGBClassifier": "XGBoost",
        "RandomForestClassifier": "RandomForest",
        "LGBMClassifier": "LightGBM",
        "SVC": "SVM",
        "MLPClassifier": "MLP",
        "GaussianNB": "GNB",
        "ComplementNB": "CNB",
        "AdaBoostClassifier": "AdaBoost",
        "KNeighborsClassifier": "KNN",
        "DecisionTreeClassifier": "DecisionTree",
        "BaggingClassifier": "Bagging",
        "GradientBoostingClassifier": 'GBDT',
    }
    # colors = x5.CB91_Grad_BP # 用于决策曲线的颜色

    timing_results = {} ## 用于存储时间信息

    str_time = (
        str(datetime.datetime.now().hour)
        + str(datetime.datetime.now().minute)
        + str(datetime.datetime.now().second)
    )
    random_number = random.randint(1, 100)
    str_time = str_time + str(random_number)

    list_name = [group]

    save_str=""
    plot_name_dict_save = {}  ##存储所有图片文件名
    result_model_save = {}  ##模型存储信息
    resThreshold = 0  #用于存储最终的模型最佳阈值
    df_save_dic={} # 用于存储最终结果数据表字典
    conf_dic_train, conf_dic_valid, conf_dic_test = {}, {}, {} # 用于存储置信区间
    df_input= df.copy(deep=True)

    #_____数据处理计时_____
    start_time_data_prep = time.time()

    if testSet:
        df = df[features + [group] + [label]].dropna()
        if label in features or label == group:
            return {"error": "标签列不能在所在模型中，请重新选择数据划分标签列！"+"false-error"}
    else: # 如果未指定测试集划分标签，只选择特征列和分组列
        df = df[features + [group]].dropna()

    binary = True
    u = np.sort(np.unique(np.array(df[group]))) #
    if len(u) == 2 and set(u) != set([0, 1]):
        y_result = label_binarize(df[group], classes=[ii for ii in u])
        y_result_pd = pd.DataFrame(y_result, columns=[group])
        df = pd.concat([df.drop(group, axis=1), y_result_pd], axis=1)
    elif len(u) > 2:
        if len(u) > 10:
            return {"error": "暂不允许类别数目大于10的情况。请检查因变量取值情况。"+"false-error"}
        binary = False
        if scoring == "roc_auc":
            scoring = scoring + "_ovo"
        else:
            scoring = scoring + "_macro"
        return {"error": "暂时只支持二分类。请检查因变量取值情况。"+"false-error"}

    # 数据划分逻辑
    if testSet:
        if isinstance(df[label][0], str):
            testLabel = str(testLabel)
            trainLabel = str(trainLabel)
        df = df[features + [group] + [label]].dropna()
        if datasave:
            df_save=df_input.iloc[list(df.index)]

        if isKFold == 'resample1' and trainSet:
            test_a = df[df[label] == testLabel]
            train_a = df[df[label] != testLabel]
            train_t = train_a[train_a[label] == trainLabel]
            valid_t = train_a[train_a[label] != trainLabel]
            train_alls = train_t.drop(label, axis=1)
            valid_alls = valid_t.drop(label, axis=1)
            test_alls = test_a.drop(label, axis=1)
            Xtrain = train_alls.drop(group, axis=1)
            Ytrain = train_alls.loc[:, list_name].squeeze(axis=1)
            X_valid = valid_alls.drop(group, axis=1)
            Y_valid = valid_alls.loc[:, list_name].squeeze(axis=1)
            # Xtest = test_a.drop(label, axis=1)
            Xtest = test_alls.drop(group, axis=1)
            Ytest = test_a.loc[:, list_name].squeeze(axis=1)

            if datasave:
                 df_save['Label_ML'] = list(map(lambda x: int(x), np.zeros(len(df_save))))
                 df_save.loc[list(Xtest.index), 'Label_ML'] = 2
                 df_save.loc[list(X_valid.index), 'Label_ML'] = 1
                 df_save.loc[list(Xtrain.index), 'Label_ML'] = 0
                 save_str += '在对新数据进行保存中，新的数据划分标签为：Label_ML，便签数据中的2为测试集，1为验证集，0为训练集。'
        else: # 如果指定了测试集划分标签，不是 resample1 模式或未指定训练集划分标签
            test_a = df[df[label] == testLabel]
            train_a = df[df[label] != testLabel]
            train_all = train_a.drop(label, axis=1)
            test_all = test_a.drop(label, axis=1)
            df = df.drop(label, axis=1)
            Xtrain = train_all.drop(group, axis=1)
            Ytrain = train_all.loc[:, list_name].squeeze(axis=1)
            Xtest = test_all.drop(group, axis=1)
            Ytest = test_all.loc[:, list_name].squeeze(axis=1)

            if datasave:
                df_save['Label_ML'] = list(map(lambda x: int(x), np.zeros(len(df_save))))
                df_save.loc[list(Xtest.index),'Label_ML'] = 1 # 测试集标签为 1
                save_str += '在对新数据进行保存中，新的数据划分标签为：Label_ML，便签数据中的1为测试集，0为训练集。'

    else:  # 如果未指定测试集划分标签 (随机划分训练集和测试集)
        df = df[features + [group]].dropna()
        X = df.drop(group, axis=1)
        Y = df.loc[:, list_name].squeeze(axis=1)
        X_train_temp, Xtest, Y_train_temp, Ytest = TTS(
            X,
            Y,
            test_size=test_ratio,
            random_state=randomState,
            stratify=Y
        )
        if isKFold == 'resample1' and trainSet:
            # 在临时训练集上再次分割，创建出最终的训练集和验证集
            val_size_adjusted = validation_ratio / (1.0 - test_ratio)
            Xtrain, X_valid, Ytrain, Y_valid = TTS(
            X_train_temp,
                   Y_train_temp,
                   test_size=val_size_adjusted,
                   random_state=randomState,
                   stratify=Y_train_temp
            )
        else:
            Xtrain, Ytrain = X_train_temp, Y_train_temp

        if datasave:
            df_save=df_input.iloc[list(df.index)]
            df_save['Label_ML'] = list(map(lambda x: int(x), np.zeros(len(df_save))))
            if isKFold == 'resample1': # 如果是 resample1 模式，测试集标签为 2
                df_save.loc[list(Xtest.index), 'Label_ML'] = 2
                save_str+='在对新数据进行保存中，新的数据划分标签为：Label_ML，便签数据中的2为测试集，1为验证集，0为训练集。'
            else:
                df_save.loc[list(Xtest.index),'Label_ML'] = 1
                save_str += '在对新数据进行保存中，新的数据划分标签为：Label_ML，便签数据中的1为测试集，0为训练集。'



    df_dict = {}

    str_result = "采用%s机器学习方法进行分类，分类变量为%s，模型中的变量包括" % (method, group)
    str_result += "、".join(features)

    #数据预处理计算结束
    end_time_data_prep = time.time()
    timing_results['数据预处理与划分时间'] = end_time_data_prep - start_time_data_prep

    #占位符
    str_result += "\n[TIMING_PLACEHOLDER]\n"

    plot_name_dict1 = {} # 用于存储寻参相关的图片文件名

    #计时超参数调优
    start_time_tuning = time.time()

    if searching == "auto":
            # 网格寻参和随机寻参方法重写
            if auto_model == "RandomizedSearchCV":# 随机寻参
                searcher = RandomizedSearchCV(globals()[method](), param_distributions=RandDefaultRange[method], random_state=randomState,n_jobs=-1,cv=n_splits) # 添加 random_state

            elif auto_model == "GridSearchCV":# 网格寻参
                searcher = GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method],n_jobs=-1,cv=n_splits)

            elif auto_model == "BayesSearchCV":# 贝叶斯寻参
                searcher = BayesSearchCV(globals()[method](), search_spaces=BayesDefaultRange[method], random_state=randomState,n_jobs=-1,n_iter=10,cv=n_splits) # 添加 random_state

            elif auto_model == "HalvingGridSearchCV":# 连续二分网格搜索
                searcher = HalvingGridSearchCV(globals()[method](), param_grid=GridDefaultRange[method], random_state=randomState,n_jobs=-1,cv=n_splits) # 添加 random_state

            elif auto_model == "HalvingRandomSearchCV":# 连续二分随机搜索
                searcher = HalvingRandomSearchCV(globals()[method](), param_distributions=RandDefaultRange[method], random_state=randomState,n_jobs=-1,cv=n_splits) # 添加 random_state

            elif auto_model == "GeneticSearchCV":  # 遗传算法寻参
                searcher = GeneticSearchCV(
                    globals()[method](),
                    param_grid=GeneticDefaultRange[method],
                    n_generations=10,
                    population_size=50,
                    mutation_rate=0.1,
                    crossover_rate=0.8,
                    elite_size=2,
                    scoring=scoring,
                    cv=n_splits,
                    random_state=randomState,
                    n_jobs = -1
                )
            elif auto_model == "Hyperopt":     # Hyperopt 算法寻参
                searcher = HyperoptSearchCV(
                    estimator=globals()[method](),
                    param_space=HyperDefaultRange[method],
                    max_evals=100,
                    scoring=scoring,
                    cv=n_splits,
                    random_state=randomState,
                    n_jobs = -1
                )

            searcher.fit(Xtrain, Ytrain)  # 执行寻参包含了交叉验证
            clf = searcher.best_estimator_
            print(clf)

            if auto_model == "Hyperopt":
                fig = searcher.plot_trials()
                str_time = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
                    datetime.datetime.now().second)
                random_number = random.randint(1, 1000)
                str_time = str_time + str(random_number)
                savepath_temp = 'best_loss_plot' + str_time
                pic_name = savepath_temp + '.png'
                pic_name1 = savepath_temp + '.' + picFormat
                fig.savefig(savePath + pic_name, bbox_inches='tight', format='png', dpi=dpi)
                fig.savefig(savePath + pic_name1, bbox_inches='tight', format=picFormat, dpi=dpi)
                plt.close(fig)
                plot_name_dict1.update({'迭代最佳损失曲线（训练损失图）': pic_name})
                plot_name_dict_save.update({'迭代最佳损失曲线（训练损失图）': pic_name1})
            #最佳参数
            best_index = searcher.best_index_
            mean_valid_score = searcher.cv_results_['mean_test_score'][best_index]
            if 'std_test_score' in searcher.cv_results_:
                std_valid_score = searcher.cv_results_['std_test_score'][best_index]
            else:
                # 判断不存在，则将标准差设为 NaN
                std_valid_score = np.nan

            #从searcher中获取阈值
            _, _, full_train_metrics, _ = classification_metric_evaluate(clf, Xtrain, Ytrain, binary)
            resThreshold = full_train_metrics["cutoff"]
            # print(resThreshold)


    elif searching == "handle": # 如果手动指定参数
        if method == "SVC":
            kwargs["probability"] = True #
        if method == "RandomForestClassifier" and kwargs.get("max_depth") == "None":
            kwargs["max_depth"] = None
        if method == "DecisionTreeClassifier" and kwargs.get("max_depth") == "None":
            kwargs["max_depth"] = None
        if method == "MLPClassifier": # 神经网络参数处理
            hls_vals = str(kwargs["hidden_layer_sizes"]).split(",") # 获取隐藏层大小字符串并分割
            hls_value = () # 初始化隐藏层大小元组
            for hls_val in hls_vals:
                try:
                    if int(hls_val) >= 5 and int(hls_val) <= 200: # 检查隐藏层宽度范围
                        hls_value = hls_value + (int(hls_val),)
                    else:
                        return {"error": "请按照要求重新设置隐藏层宽度！"+"false-error"}
                except:
                    return {"error": "请重新设神经网络模型中的隐藏层宽度！"+"false-error"}
            kwargs["hidden_layer_sizes"] = hls_value
        if method == "GaussianNB" and kwargs.get("priors") == "None":
            kwargs["priors"] = None
        elif method == "GaussianNB":
            pri_vals = str(kwargs["priors"]).split(",")
            pri_value = ()
            pri_sum = 0.0
            for pri_val in pri_vals:
                try:
                    pri_sum = float(pri_val) + pri_sum
                    pri_value = pri_value + (float(pri_val),)
                except:
                    return {"error": "请重新设朴素贝叶斯模型中的先验概率！"+"false-error"}
            if len(pri_vals) == len(Y.unique()) and abs(pri_sum - 1.0) < 1e-6:
                kwargs["priors"] = pri_value
            else:
                return {"error": "请重新设朴素贝叶斯模型中的先验概率！"+"false-error"}
        if method in random_model:
            kwargs["random_state"]=42
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)

    elif searching == "default": # 如果使用默认参数
        if method == "SVC":
            kwargs["probability"] = True
        elif method == "MLPClassifier":
            kwargs["hidden_layer_sizes"] = (20, 10)
            kwargs["max_iter"] = 20
        elif method == "RandomForestClassifier":
            kwargs["n_estimators"] = 20
        if method in random_model:
            kwargs["random_state"]=42
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)

    if searching == 'auto':
        str_result += "\n选择的自动寻参方法为\n%s："+str(auto_model)
        str_result += "\n模型参数为:\n%s" % dic2str(searcher.best_params_, method) # 添加最佳参数到结果描述
    else: # 如果手动或使用默认参数
        str_result += "\n模型参数为:\n%s" % dic2str(clf.get_params(), clf.__class__.__name__) # 添加模型参数到结果描述
    str_result += "\n数据集样本数总计N=%d例，分类变量中包含的类别信息为：\n" % (df.shape[0]) # 添加数据集信息
    group_labels = df[group].unique()
    group_labels.sort()
    for lab in group_labels:
        n = sum(df[group] == lab)
        str_result += "\t 类别(" + str(lab) + ")：N=" + str(n) + "例\n" # 添加类别信息到结果描述

    #计时超参数调优
    end_time_tuning = time.time()
    timing_results['超参数调优及模型构建与训练时间'] = end_time_tuning - start_time_tuning


    # 学习曲线分割数
    if isKFold=='resample1':
        lc_splits=5
    else:
        lc_splits=n_splits
    plot_name_list = x5.plot_learning_curve(
        clf,
        Xtrain,
        Ytrain,
        cv=lc_splits,
        scoring=scoring,
        path=savePath,
        dpi=dpi,
        picFormat=picFormat,
    )
    plot_name_dict_save["学习曲线"] = plot_name_list[1]
    plot_name_list.pop(len(plot_name_list) - 1)
    #画校准曲线
    calibration_curve_name, _ = plot_calibration_curve(
        clf,
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        name=name_dict[method],
        path=savePath,
        smooth=smooth,
        picFormat=picFormat,
        dpi=dpi,
    )
    plot_name_list.append(calibration_curve_name[0])
    plot_name_dict_save["校准曲线"] = calibration_curve_name[1]


    if binary:
        fig_roc_valid = plt.figure(figsize=(4, 4), dpi=dpi) # 验证集 ROC 曲线图
        # 画对角线
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.08, color="grey") # 添加网格线

    data_all={}
    _, _, full_train_metrics, _ = classification_metric_evaluate(clf, Xtrain, Ytrain, binary)
    resThreshold = full_train_metrics["cutoff"]
    print(resThreshold)

    best_auc = 0.0
    tprs_train, tprs_valid = [], [] #
    fpr_train_alls, tpr_train_alls = [], []
    mean_fpr = np.linspace(0, 1, 100)
    list_evaluate_dic_train = make_class_metrics_dict()
    list_evaluate_dic_valid = make_class_metrics_dict()


    #计时验证
    start_time_cv = time.time()

    # 交叉验证/重采样逻辑
    if isKFold == 'cross' or isKFold == 'nest': # 交叉验证或嵌套交叉验证
        KF = StratifiedKFold(n_splits=n_splits, random_state=randomState, shuffle=True) # 分层 KFold 交叉验证
        for i, (train_index, valid_index) in enumerate(KF.split(Xtrain, Ytrain)): # 遍历各折
            X_train_fold, X_valid_fold = Xtrain.iloc[train_index], Xtrain.iloc[valid_index] # 获取当前折的训练集和验证集特征
            Y_train_fold, Y_valid_fold = Ytrain.iloc[train_index], Ytrain.iloc[valid_index] # 获取当前折的训练集和验证集目标变量
            data_all.update({ # 存储当前折的数据
                i: {"Xtrain": X_train_fold,"Ytrain": Y_train_fold,"Xvalid": X_valid_fold,"Yvalid": Y_valid_fold}}
            )

            if isKFold == "nest": # 嵌套交叉验证
                best_auc_inner = 0.0
                resThreshold_inner = 0
                inner_cv = StratifiedKFold(
                    n_splits=n_splits, random_state=randomState, shuffle=True
                )
                for j, (train_index_inner, test_index_inner) in enumerate(inner_cv.split(X_train_fold, Y_train_fold)): # 遍历内层各折
                    X_train_inner, X_test_inner = (X_train_fold.iloc[train_index_inner],X_train_fold.iloc[test_index_inner],)
                    y_train_inner, y_test_inner = (Y_train_fold.iloc[train_index_inner],Y_train_fold.iloc[test_index_inner],)
                    model_i = clone(clf).fit(X_train_inner, y_train_inner) # 在内层训练集上训练模型
                    _, _, metric_dic_train_i, _ = classification_metric_evaluate( # 计算内层训练集指标
                        model_i, X_train_inner, y_train_inner, binary
                    )
                    _, _, metric_dic_valid_i, _ = classification_metric_evaluate( # 计算内层测试集指标 (作为验证)
                        model_i,
                        X_test_inner,
                        y_test_inner,
                        binary,
                        Threshold=metric_dic_train_i["cutoff"], # 使用内层训练集阈值
                    )
                    if metric_dic_valid_i["AUC"] > best_auc_inner:
                        best_auc_inner = metric_dic_valid_i["AUC"]
                        model_inner = model_i # 更新内层最佳模型
                        resThreshold_inner = metric_dic_train_i["cutoff"]
                        data_all[i].update({'model':model_inner,'threshold':resThreshold_inner})

    else: # 如果是重采样模式 ('resample' 或 'resample1')
        if isKFold=='resample1' and trainSet: # 如果是 resample1 模式且指定了训练集划分标签
             data_all.update({
                 0: {"Xtrain": Xtrain, "Ytrain": Ytrain, "Xvalid": X_valid, "Yvalid": Y_valid}}
             )
             # 在 resample1 模式且 trainSet 为 True 的情况下，需要在这里训练模型
             model = clone(clf).fit(Xtrain, Ytrain)
             data_all[0].update({'model': model, 'threshold': 0.5}) # 存储模型并设置默认阈值 0.5

        else: # 如果是 'resample' 模式 或 ('resample1' 且未指定 trainSet)
            for i in range(n_splits): # 进行 n_splits 次重采样
                X_train_rs, X_valid_rs, Y_train_rs, Y_valid_rs = TTS(Xtrain, Ytrain, test_size=validation_ratio, random_state=randomState + i) # 随机划分训练集和验证集
                data_all.update({ # 存储当前重采样的数据
                    i: {"Xtrain": X_train_rs, "Ytrain": Y_train_rs, "Xvalid": X_valid_rs, "Yvalid": Y_valid_rs}}
                )
            if isKFold == 'resample1' and not testSet and datasave: #resample1 模式且未指定测试集划分标签，需要保存数据
                 df_save.loc[list(data_all[0]['Xvalid'].index), 'Label_ML'] = 1

    i=0
    for data_key, data_value in data_all.items():
        X_train_iter, Y_train_iter, X_valid_iter, Y_valid_iter = (data_value["Xtrain"],data_value["Ytrain"],data_value["Xvalid"],data_value["Yvalid"],) # 获取当前折/重采样的数据

        if isKFold!='nest':
            if not (isKFold == 'resample1' and trainSet):
                 model = clone(clf).fit(X_train_iter, Y_train_iter) # 克隆并训练模型
            fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(model, X_train_iter, Y_train_iter, binary) # 计算训练集指标
        else:
            model = data_all[data_key]['model']
            fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate( # 计算训练集指标，使用内层最佳阈值
                model, X_train_iter, Y_train_iter, binary, Threshold=data_all[data_key]['threshold'])

        fpr_valid, tpr_valid, metric_dic_valid, _ = classification_metric_evaluate( # 计算验证集指标
            model, X_valid_iter, Y_valid_iter, binary, Threshold=metric_dic_train["cutoff"] # 使用训练集计算出的阈值
        )
        metric_dic_valid.update({"cutoff": metric_dic_train["cutoff"]})

        # 计算所有评价指标并添加到列表中
        for key in list_evaluate_dic_train.keys():
            list_evaluate_dic_train[key].append(metric_dic_train[key])
            list_evaluate_dic_valid[key].append(metric_dic_valid[key])

        if binary:

            tprs_valid.append(np.interp(mean_fpr, fpr_valid, tpr_valid))
            tprs_valid[-1][0] = 0.0

            # 绘制验证集 ROC 曲线
            if validationCurve:
                plt.figure(fig_roc_valid.number)
                plt.plot(
                    fpr_valid,
                    tpr_valid,
                    lw=1,
                    alpha=0.4,
                    color=palette_dict[style][i % len(palette_dict[style])],
                    label="ROC fold %4d (auc=%0.3f 95%%CI (%0.3f-%0.3f))"
                    % (
                        i + 1,
                        metric_dic_valid["AUC"],
                        metric_dic_valid["AUC_L"],
                        metric_dic_valid["AUC_U"],
                    ) if resultType == 1 else "ROC fold %4d auc=%0.3f" % (i+1, metric_dic_valid["AUC"]),
                )

            #训练集 ROC 曲线
            fpr_train_alls.append(fpr_train)
            tpr_train_alls.append(tpr_train)
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_train[-1][0] = 0.0

        i+=1

    #验证
    end_time_cv = time.time()
    timing_results['验证时间（交叉验证/嵌套交叉验证/重采样）'] = end_time_cv - start_time_cv

    if modelSave:
        import pickle

        modelfile = open(savePath + method + str_time + ".pkl", "wb")
        pickle.dump(clf, modelfile)
        modelfile.close()
        result_model_save["modelFile"] = method + str_time + ".pkl"
        result_model_save["modelFeature"] = features
    if datasave:
        res_pro = clf.predict_proba(df_save[features])
        feas_Yprob = []
        for i in range(res_pro.shape[1]):
            feas_Yprob.append('Yprob_' + str(i)+'_'+str_time)

        pd_Yprob = pd.DataFrame(res_pro, columns=feas_Yprob, index=df_save.index)
        df_save = pd.concat([df_save, pd_Yprob], axis=1)
        df_save['Threshold'+'_'+str_time] = resThreshold
        df_dict.update({'存储数据表':df_save})
        str_result += '\n' + save_str


    if binary:
        mean_tpr_valid = np.mean(tprs_valid, axis=0)
        mean_tpr_valid[-1] = 1.0
        mean_auc_valid = np.mean(list_evaluate_dic_valid["AUC"])
        aucs_lower, aucs_upper = ci(list_evaluate_dic_valid["AUC"])
        plt.figure(fig_roc_valid.number)
        plt.plot(
            mean_fpr,
            mean_tpr_valid,
            color=palette_dict[style][0],
            lw=2,
            alpha=0.8,
            label=r"Mean (validation) ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f))"
            % (
                mean_auc_valid,
                np.mean(list_evaluate_dic_valid["AUC_L"]),
                np.mean(list_evaluate_dic_valid["AUC_U"]),
            ) if resultType == 1 else r"Mean (validation) ROC (auc=%0.3f SD (%0.3f))" % (mean_auc_valid, np.std(list_evaluate_dic_valid["AUC"])), # 根据 resultType 选择标签格式 (SD)
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC curve(Validation)")
        plt.legend(loc="lower right", fontsize=5)
        if savePath is not None:
            plot_name_list.append(
                save_fig(savePath, "ROC_curve", "png", fig_roc_valid, str_time=str_time)
            )
            plot_name_dict_save["验证集 ROC 曲线"] = save_fig(
                savePath, "ROC_curve", picFormat, fig_roc_valid, str_time=str_time
            )
        plt.close(fig_roc_valid)


    mean_dic_train, stdv_dic_train = {}, {}
    mean_dic_valid, stdv_dic_valid = {}, {}
    for key in list_evaluate_dic_valid.keys():
        mean_dic_train[key] = np.mean(list_evaluate_dic_train[key])
        mean_dic_valid[key] = np.mean(list_evaluate_dic_valid[key])
        if resultType == 0:
            stdv_dic_train[key] = np.std(list_evaluate_dic_train[key], axis=0)
            stdv_dic_valid[key] = np.std(list_evaluate_dic_valid[key], axis=0)
        elif resultType == 1:
            conf_dic_train[key] = list(ci(list_evaluate_dic_train[key]))
            conf_dic_valid[key] = list(ci(list_evaluate_dic_valid[key]))

    resThreshold_train = mean_dic_train['cutoff']

    (
        fpr_test,
        tpr_test,
        metric_dic_test,
        df_test_result,
    ) = classification_metric_evaluate( # 计算测试集指标
        clf, Xtest, Ytest, binary, Threshold=resThreshold # 使用最佳模型和阈值
    )
    metric_dic_test.update({"cutoff": resThreshold})

    # 混淆矩阵表格(测试集)
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict_proba(Xtest)[:, 1] # 获取测试集预测概率
    y_pred[y_pred > resThreshold] = 1
    y_pred[y_pred <= resThreshold] = 0
    cm = confusion_matrix(Ytest, y_pred) # 计算混淆矩阵(测试集)
    cm_df = pd.DataFrame(
        cm, index=['真实值: ' + str(group_labels[0]), '真实值: ' + str(group_labels[1])],
        columns=['预测值: ' + str(group_labels[0]), '预测值: ' + str(group_labels[1])]
    ).T # 转置数据框
    cm_df['Freq'] = cm_df.sum(axis=1)
    cm_df = cm_df.reset_index().rename(columns={'index': '预测值'})


    # 混淆矩阵表格(训练集)
    train_cutoff_numeric = np.mean(list_evaluate_dic_train["cutoff"])
    y_pred_train = clf.predict_proba(Xtrain)[:, 1]
    y_pred_train_classified = (y_pred_train > train_cutoff_numeric).astype(int)
    cm_train = confusion_matrix(Ytrain, y_pred_train_classified)
    cm_df_train = pd.DataFrame(
        cm_train, index=['真实值: ' + str(group_labels[0]), '真实值: ' + str(group_labels[1])],
        columns=['预测值: ' + str(group_labels[0]), '预测值: ' + str(group_labels[1])]
    ).T
    cm_df_train['Freq'] = cm_df_train.sum(axis=1)
    cm_df_train = cm_df_train.reset_index().rename(columns={'index': '预测值'})


    #画训练集 ROC
    if binary: #
        fig_roc_train = plt.figure(figsize=(4, 4), dpi=dpi)
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.08, color="grey")

        if validationCurve:
            for i in range(len(tpr_train_alls)):
                plt.plot(
                    fpr_train_alls[i],
                    tpr_train_alls[i],
                    lw=1,
                    color=palette_dict[style][i % len(palette_dict[style])],
                    alpha=0.4,
                    label="ROC fold %4d (auc=%0.3f 95%%CI (%0.3f-%0.3f)) "
                    % (
                        i + 1,
                        list_evaluate_dic_train["AUC"][i],
                        list_evaluate_dic_train["AUC_L"][i],
                        list_evaluate_dic_train["AUC_U"][i],
                    ) if resultType == 1 else "ROC fold %4d auc=%0.3f " % (i+1, list_evaluate_dic_train["AUC"][i]),
                )

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_train[-1] = 1.0
        mean_auc_train = np.mean(list_evaluate_dic_train["AUC"])
        plt.plot(
            mean_fpr,
            mean_tpr_train,
            color=palette_dict[style][0],
            lw=1.8,
            alpha=0.7,
            label=r"Mean (train) ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f))"
            % (
                mean_auc_train,
                np.mean(list_evaluate_dic_train["AUC_L"]),
                np.mean(list_evaluate_dic_train["AUC_U"]),
            ) if resultType == 1 else "Mean (train) ROC (auc=%0.3f SD (%0.3f))" % (mean_auc_train,np.std(list_evaluate_dic_train["AUC"])), # 根据 resultType 选择标签格式 (SD)
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC curve(Training)")
        plt.legend(loc="lower right", fontsize=5)
        if savePath is not None:
            plot_name_list.append(
                save_fig(savePath, "ROC_curve_train", "png", fig_roc_train, str_time=str_time)
            )
            plot_name_dict_save["训练集 ROC 曲线"] = save_fig( # 保存指定格式图片
                savePath, "ROC_curve_train", picFormat, fig_roc_train, str_time=str_time
            )
        plt.close(fig_roc_train)


        plot_name_list.reverse()

        #画测试集 ROC
        fig_roc_test = plt.figure(figsize=(4, 4), dpi=dpi)
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.08, color="grey")

        if smooth: # 如果需要平滑处理
            from scipy.interpolate import interp1d

            tpr_test_unique, tpr_test_index = np.unique(fpr_test, return_index=True)
            fpr_test_new = np.linspace(min(fpr_test), max(fpr_test), len(fpr_test))
            f = interp1d(
                tpr_test_unique, tpr_test[tpr_test_index], kind="linear"
            )
            tpr_test_new = f(fpr_test_new)
        else:
            fpr_test_new = fpr_test
            tpr_test_new = tpr_test
        plt.plot( # 绘制测试集 ROC 曲线
            fpr_test_new,
            tpr_test_new,
            lw=1.5,
            alpha=0.6,
            color=palette_dict[style][0],
            label="Test Set ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f)) " # 标签格式
            % (
                metric_dic_test["AUC"],
                metric_dic_test["AUC_L"],
                metric_dic_test["AUC_U"],
            ) if resultType == 1 else "Test Set ROC auc=%0.3f" % (metric_dic_test["AUC"]),
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC curve(Test)")
        plt.legend(loc="lower right", fontsize=5)
        if savePath is not None:
            plot_name_list.append(
                save_fig(savePath, "ROC_curve_test", "png", fig_roc_test, str_time=str_time)
            )
            plot_name_dict_save["测试集 ROC 曲线"] = save_fig(
                savePath, "ROC_curve_test", picFormat, fig_roc_test, str_time=str_time
            )
        plt.close(fig_roc_test)

        if testSet:
            df_count_c = Xtest.shape[0]
            df_count_r = (Xtest.shape[0] / df.shape[0]) * 100
            if isKFold == 'resample1':
                str_result += "其中在总体样本中根据标签%s为%s划分为测试集，标签%s为训练集，其余标签归为验证集。其中" % (
                    label, testLabel, trainLabel)
            else:
                str_result += "其中在总体样本中根据标签%s为%s划分为总体测试集，其余标签归为训练集。其中" % (label,
                                                                                                          testLabel)
        else:
            df_count_c = Ytest.shape[0]
            df_count_r = (Ytest.shape[0] / df.shape[0]) * 100
            str_result += "其中在总体样本中随机抽取"
        diff, ratio = 0, 0
        if isKFold == "cross" or isKFold == 'nest':
            str_splits = '剩余样本作为训练集进行%d折交叉验证，' % n_splits
        elif isKFold == 'resample':
            valid_count = round(Xtrain.shape[0] * validation_ratio)#计算验证集数量
            str_splits = '剩余样本每次随机抽取验证集N=%d例(%d%%)作为验证数据进行%d次重采样，' % (valid_count,validation_ratio * 100, n_splits)
        elif isKFold == 'resample1':
            if not testSet:
                valid_count = round(Xtrain.shape[0] * validation_ratio)#计算验证集数量
                str_splits = '剩余样本随机抽取验证集N=%d例(%d%%)作为验证数据进行%d次重采样，' % (valid_count, validation_ratio * 100, n_splits)
            else:
                if trainSet:  # 仅在 trainSet 为 True 时执行
                    train_count = Xtrain.shape[0]
                    valid_count = X_valid.shape[0]
                    str_splits = "训练集N=%d例，验证集N=%d例" % (train_count, valid_count)
                else:
                    str_splits = "，"
        if resultType == 1:  ##CI
            str_result += (
                    "测试集N=%d例(%3.2f%%)，%s并在验证集中得到AUC=%5.4f(%5.4f-%5.4f)。\n最终模型在测试集中的AUC=%5.4f，准确度=%5.4f。\n"
                    % (
                        df_count_c,
                        df_count_r,
                        str_splits,
                        mean_dic_valid["AUC"],
                        mean_dic_valid["AUC_L"],
                        mean_dic_valid["AUC_U"],
                        df_test_result["AUC"].values[0],
                        df_test_result["准确度"].values[0],
                    )
            )
            diff = mean_dic_valid["AUC"] - float(df_test_result.loc["Mean", "AUC"])
            ratio = diff / float(df_test_result.loc["Mean", "AUC"])
        elif resultType == 0:  ##SD
            str_result += (
                    "取测试集N=%d例(%3.2f%%)，%s，并在验证集中得到AUC=%5.4f±%5.4f。\n最终模型在测试集中的AUC=%5.4f，准确度=%5.4f。\n"
                    % (
                        df_count_c,
                        df_count_r,
                        str_splits,
                        mean_dic_valid["AUC"],
                        stdv_dic_valid["AUC"],
                        df_test_result["AUC"].values[0],
                        df_test_result["准确度"].values[0],
                    )
            )
            diff = float(stdv_dic_valid["AUC"]) - float(df_test_result.loc["Mean", "AUC"])
            ratio = diff / float(df_test_result.loc["Mean", "AUC"])

        if not np.isnan(float(diff)) and diff > 0 and (ratio > 0.1):
            str_result += "注意到AUC指标下验证集表现超出测试集{}，约{}%，可能存在过拟合现象。建议更换模型或重新设置参数。".format(
                round(diff, decimal_num), round(ratio * 100, decimal_num)
            )
        else:
            str_result += (
                "鉴于AUC指标下验证集表现未超出测试集或超出比小于10%，可认为拟合成功，{}模型可以用于此数据集的分类建模任务。".format(
                    name_dict[method]
                )
            )
        str_result += "\n需注意：在结果表格中训练集结果汇总和验证集结果汇总是基于验证方式进行计算的，而测试集结果汇总是基于完整测试集进行的验证。"
        str_result += "\n如果想进一步对比更多分类模型的表现，可使用左侧栏智能分析中的‘分类多模型综合分析’功能。在混淆矩阵表中如果样本总数与分析结果出现细微偏差是由于使用四舍五入计算导致，不影响实际结果。"

        df_test_result = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))


    if resultType == 1:
        for tem in ["AUC", "AUC_L", "AUC_U"]:
             if tem in mean_dic_train and tem != "AUC": del mean_dic_train[tem]
             if tem in mean_dic_valid and tem != "AUC": del mean_dic_valid[tem]
             if tem in conf_dic_train and tem != "AUC": del conf_dic_train[tem]
             if tem in conf_dic_valid and tem != "AUC": del conf_dic_valid[tem]


        for key in conf_dic_train.keys():
            mean_dic_train[key] = (
                str(round_dec(float(mean_dic_train[key]), d=decimal_num))
                + "("
                + str(round_dec(float(conf_dic_train[key][0]), d=decimal_num))
                + "-"
                + str(round_dec(float(conf_dic_train[key][1]), d=decimal_num))
                + ")"
            )
            mean_dic_valid[key] = (
                str(round_dec(float(mean_dic_valid[key]), d=decimal_num))
                + "("
                + str(round_dec(float(conf_dic_valid[key][0]), d=decimal_num))
                + "-"
                + str(round_dec(float(conf_dic_valid[key][1]), d=decimal_num))
                + ")"
            )

        df_train_result = pd.DataFrame([mean_dic_train], index=["Mean"])
        df_valid_result = pd.DataFrame([mean_dic_valid], index=["Mean"])

        df_train_result.rename(
            columns={
                "AUC": "AUC(95%CI)",
                "cutoff": "cutoff(95%CI)",
                "准确度": "准确度(95%CI)",
                "灵敏度": "灵敏度(95%CI)",
                "特异度": "特异度(95%CI)",
                "阳性预测值": "阳性预测值(95%CI)",
                "阴性预测值": "阴性预测值(95%CI)",
                "F1分数": "F1分数(95%CI)",
                "Kappa": "Kappa(95%CI)",
            },
            inplace=True,
        )
        df_valid_result.rename(
            columns={
                "AUC": "AUC(95%CI)",
                "cutoff": "cutoff(95%CI)",
                "准确度": "准确度(95%CI)",
                "灵敏度": "灵敏度(95%CI)",
                "特异度": "特异度(95%CI)",
                "阳性预测值": "阳性预测值(95%CI)",
                "阴性预测值": "阴性预测值(95%CI)",
                "F1分数": "F1分数(95%CI)",
                "Kappa": "Kappa(95%CI)",
            },
            inplace=True,
        )

        df_test_result["AUC (95%CI)"] = df_test_result["AUC"].astype(str) + " (" + df_test_result["AUC_L"].astype(str) + "-" + df_test_result["AUC_U"].astype(str) + ")"
        df_test_result.drop(columns=["AUC", "AUC_L", "AUC_U"], inplace=True)
        all_columns = df_test_result.columns.tolist()
        new_order = ['AUC (95%CI)'] + [col for col in all_columns if col != 'AUC (95%CI)']
        df_test_result = df_test_result[new_order]


    elif resultType == 0:
        for tem in ["AUC_L", "AUC_U"]:
            if tem in mean_dic_train: del mean_dic_train[tem]
            if tem in mean_dic_valid: del mean_dic_valid[tem]
            if tem in stdv_dic_train: del stdv_dic_train[tem]
            if tem in stdv_dic_valid: del stdv_dic_valid[tem]

        for key in stdv_dic_train.keys():
            mean_dic_train[key] = (
                str(round_dec(float(mean_dic_train[key]), d=decimal_num))
                + " ("
                + str(round_dec(float(stdv_dic_train[key]), d=decimal_num))
                + ")"
            )
            mean_dic_valid[key] = (
                str(round_dec(float(mean_dic_valid[key]), d=decimal_num))
                + " ("
                + str(round_dec(float(stdv_dic_valid[key]), d=decimal_num))
                + ")"
            )

        df_train_result = pd.DataFrame([mean_dic_train], index=["Mean"])
        df_valid_result = pd.DataFrame([mean_dic_valid], index=["Mean"])

        df_train_result.rename(
            columns={
                "AUC": "AUC(SD)",
                "cutoff": "cutoff(SD)",
                "准确度": "准确度(SD)",
                "灵敏度": "灵敏度(SD)",
                "特异度": "特异度(SD)",
                "阳性预测值": "阳性预测值(SD)",
                "阴性预测值": "阴性预测值(SD)",
                "F1分数": "F1分数(SD)",
                "Kappa": "Kappa(SD)",
            },
            inplace=True,
        )
        df_valid_result.rename(
            columns={
                "AUC": "AUC(SD)",
                "cutoff": "cutoff(SD)",
                "准确度": "准确度(SD)",
                "灵敏度": "灵敏度(SD)",
                "特异度": "特异度(SD)",
                "阳性预测值": "阳性预测值(SD)",
                "阴性预测值": "阴性预测值(SD)",
                "F1分数": "F1分数(SD)",
                "Kappa": "Kappa(SD)",
            },
            inplace=True,
        )
        df_test_result.drop(columns=["AUC_L", "AUC_U"], inplace=True)


    df_dictjq = { # 结果数据表字典
        "训练集结果汇总": df_train_result,
        "验证集结果汇总": df_valid_result,
        "测试集结果汇总": df_test_result,
        "测试集混淆矩阵": cm_df, # 混淆矩阵数据框
        "训练集混淆矩阵": cm_df_train,

    }
    df_dict.update(df_dictjq)

    plot_name_dict = {
        "训练集 ROC 曲线图": plot_name_list[0],
        "验证集 ROC 曲线图": plot_name_list[1],
        "测试集 ROC 曲线图": plot_name_list[4],
        "学习曲线图": plot_name_list[3],
        "模型校准曲线": plot_name_list[2],
    }

    if binary:  #画 DCA 曲线
        DCA_dict = {}
        (
            prob_pos,
            p_serie,
            net_benefit_serie,
            net_benefit_serie_All,
        ) = calculate_net_benefit(clf, Xtest, Ytest) # 计算净获益
        DCA_dict[name_dict[method]] = { # 存储当前模型的 DCA 结果
            "p_serie": p_serie,
            "net_b_s": net_benefit_serie,
            "net_b_s_A": net_benefit_serie_All,
        }
        decision_curve_p = plot_decision_curves( # 绘制决策曲线
            DCA_dict,
            colors=palette_dict[style],
            name="Test",
            savePath=savePath,
            dpi=dpi,
            picFormat=picFormat,
            DCA_cut=DCA_cut,
        )
        plot_name_dict["测试集 DCA 曲线图"] = decision_curve_p[0]
        plot_name_dict_save["测试集 DCA 曲线图"] = decision_curve_p[1]


    #画混淆矩阵图
    if isKFold =='resample1' and trainSet: # 如果是 resample1 模式且指定了训练集划分标签
        pic_matrix,pic_save_matrix=_plot_conffusion_matrix(clf, Xtrain, Ytrain, group_labels,resThreshold=resThreshold_train,
                                savePath=savePath,picFormat=picFormat, dpi=600)
        plot_name_dict.update(pic_matrix)
        plot_name_dict_save.update(pic_save_matrix)
        pic_matrix, pic_save_matrix = _plot_conffusion_matrix(clf, X_valid, Y_valid, group_labels,resThreshold=resThreshold_train,
                                                              name='valid_confusionMatrix_',
                                                              dic_name='最优模型验证集混淆矩阵', str_time=str_time,
                                                              savePath=savePath, picFormat=picFormat, dpi=600)
        plot_name_dict.update(pic_matrix)
        plot_name_dict_save.update(pic_save_matrix)
    else: # 包括 cross, nest, resample 模式，以及 resample1 模式但未指定 trainSet
        pic_matrix, pic_save_matrix = _plot_conffusion_matrix(clf, Xtrain, Ytrain, group_labels,resThreshold=resThreshold_train,
                                                              name='train_confusionMatrix_',
                                                              dic_name='最优模型训练集混淆矩阵', str_time=str_time,
                                                              savePath=savePath, picFormat=picFormat, dpi=600)
        plot_name_dict.update(pic_matrix)
        plot_name_dict_save.update(pic_save_matrix)

    # 测试集混淆矩阵图
    pic_matrix, pic_save_matrix = _plot_conffusion_matrix(clf, Xtest, Ytest, group_labels,resThreshold=resThreshold,
                                                          name='test_confusionMatrix_',
                                                          dic_name='最优模型测试集混淆矩阵', str_time=str_time,
                                                          savePath=savePath, picFormat=picFormat, dpi=600)
    plot_name_dict.update(pic_matrix)
    plot_name_dict_save.update(pic_save_matrix)

    # KS 曲线图
    import scikitplot as skplt
    plot_name = 'KS_' + str_time + ".png"
    plot_name_save = 'KS_' + str_time + "." + picFormat
    y_probas = clf.predict_proba(Xtest)
    skplt.metrics.plot_ks_statistic(Ytest, y_probas, title='KS Statistic Plot(Test)')
    plt.savefig(savePath + plot_name, dpi=dpi, format='png', bbox_inches='tight')
    plt.savefig(savePath + plot_name_save, dpi=dpi, format=picFormat, bbox_inches='tight')
    plt.close() # 关闭图形
    plot_name_dict.update({'测试集 KS 曲线': plot_name})
    plot_name_dict_save.update({'测试集 KS 曲线': plot_name_save})

    # 画 parallel_coordinates 图
    from yellowbrick.features import parallel_coordinates
    try:
        visualizer = parallel_coordinates(df[features], df[group], normalize=None)
        plot_name = "parallel_coordinates_" + str_time + ".png"
        plot_name_save = "parallel_coordinates_" + str_time + "." + picFormat
        pic_kwargs = {'dpi': dpi, 'format': 'png', 'bbox_inches': 'tight'}
        visualizer.show(outpath=savePath + plot_name, clear_figure=True, **pic_kwargs)
        pic_kwargs = {'dpi': dpi, 'format': picFormat, 'bbox_inches': 'tight'}
        plt.close()
        visualizer = parallel_coordinates(df[features], df[group], normalize=None)
        visualizer.show(outpath=savePath + plot_name_save,clear_figure=True, **pic_kwargs)
        plot_name_dict.update({'全部数据_parallel_coordinates图': plot_name})
        plot_name_dict_save.update({'全部数据_parallel_coordinates图': plot_name_save})
        plt.close()
    except Exception as e:
        print(f"Error plotting parallel coordinates: {e}")


    # 决策树图
    if method == 'DecisionTreeClassifier' and len(features) < 10:
        try:
            import pydotplus #
            from sklearn import tree #
            from sklearn.tree import export_graphviz
            plot_name = "DecisionTree_pic_" + str_time + ".png"
            plot_name_save = "DecisionTree_pic_" + str_time + "." + picFormat
            group_u=list(np.sort(np.unique(np.array(df[group]))))
            group_name = [str(num) for num in group_u]

            model_for_plot = clf
            if searching == "auto":
                if hasattr(searcher, 'best_estimator_'):
                    model_for_plot = searcher.best_estimator_

            dot_data = tree.export_graphviz(model_for_plot, out_file=None,
                                            feature_names=features,
                                            class_names=np.array(group_name),
                                            filled=True, rounded=True,
                                            special_characters=True)

            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_png(savePath+plot_name)
            if picFormat == 'pdf':
                graph.write_pdf(savePath+plot_name_save)
            else:
                graph.write_png(savePath + plot_name_save)
            plot_name_dict.update({'决策树图': plot_name})
            plot_name_dict_save.update({'决策树图': plot_name_save})
        except Exception as e:
            print(f"Error plotting Decision Tree: {e}")


    if explain or modelSave:
        import shap

        f = lambda x: clf.predict_proba(x)[:, 1]
        med = Xtrain.median().values.reshape((1, Xtrain.shape[1]))

        result_model_save["modelShapValue"] = [ # 存储 SHAP 解释所需的参考值（通常是中位数）
            float("{:.3f}".format(i)) for i in list(med[0])
        ]
        result_model_save["modelName"] = method
        result_model_save["modelClass"] = "机器学习分类"
        result_model_save["Threshold"] = resThreshold

    df_shapValue = Xtest # 用于 SHAP 解释的数据，默认为测试集
    df_shapValue_show = pd.DataFrame()
    shapValue_list = []
    shapValue_name = []
    if explain:
        if shapSet == 2:  ## Xtrain, Xtest, Ytrain, Ytest 中选择 SHAP 样本
            df_shapValue = Xtest # SHAP 解释数据为测试集
            if explain_sample == 4: # 解释 4 个代表性样本
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(Ytest)):
                    if ( # 预测为 1 实际为 1 的样本
                        flage1
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytest.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_list.append(i)
                        shapValue_name.append("shap_样本_预测值为1实际值为1")
                        flage1 = False
                    elif ( # 预测为 1 实际为 0 的样本
                        flage2
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytest.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_list.append(i)
                        shapValue_name.append("shap_样本_预测值为1实际值为0")
                        flage2 = False
                    elif ( # 预测为 0 实际为 1 的样本
                        flage3
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytest.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为0实际值为1")
                        shapValue_list.append(i)
                        flage3 = False
                    elif ( # 预测为 0 实际为 0 的样本
                        flage4
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytest.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为0实际值为0")
                        shapValue_list.append(i)
                        flage4 = False

                    if not flage1 and not flage2 and not flage3 and not flage4:
                        break
            else:
                df_shapValue_show = pd.concat(
                    [df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0
                )
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend(
                    "shap_样本_" + str(i) for i in range(explain_sample)
                )

        elif shapSet == 1: # 在 Xtrain 中选择 SHAP 样本
            df_shapValue = Xtrain
            if explain_sample == 4:
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(Ytrain)):
                    if (
                        flage1
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytrain.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为1实际值为1")
                        shapValue_list.append(i)
                        flage1 = False
                    elif (
                        flage2
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytrain.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为1实际值为0")
                        shapValue_list.append(i)
                        flage2 = False
                    elif (
                        flage3
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytrain.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为0实际值为1")
                        shapValue_list.append(i)
                        flage3 = False
                    elif (
                        flage4
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytrain.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为0实际值为0")
                        shapValue_list.append(i)
                        flage4 = False

                    if not flage1 and not flage2 and not flage3 and not flage4:
                        break
            else:
                df_shapValue_show = pd.concat(
                    [df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0
                )
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend(
                    "shap_样本_" + str(i) for i in range(explain_sample)
                )
        elif shapSet == 0: # 在全部数据中选择 SHAP 样本
            df_shapValue = pd.concat([Xtrain, Xtest], axis=0)
            df_shapValue_Y = pd.concat([Ytrain, Ytest], axis=0)
            if explain_sample == 4:
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(df_shapValue_Y)):
                    if (
                        flage1
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and df_shapValue_Y.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为1实际值为1")
                        shapValue_list.append(i)
                        flage1 = False
                    elif (
                        flage2
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and df_shapValue_Y.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为1实际值为0")
                        shapValue_list.append(i)
                        flage2 = False
                    elif (
                        flage3
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and df_shapValue_Y.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为0实际值为1")
                        shapValue_list.append(i)
                        flage3 = False
                    elif (
                        flage4
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and df_shapValue_Y.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_样本_预测值为0实际值为0")
                        shapValue_list.append(i)
                        flage4 = False

                    if not flage1 and not flage2 and not flage3 and not flage4:
                        break
            else:
                df_shapValue_show = pd.concat(
                    [df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0
                )
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend(
                    "shap_样本_" + str(i) for i in range(explain_sample)
                )
        explainer = shap.KernelExplainer(f, med)
        shap_values = explainer.shap_values(df_shapValue)

        if explain_numvar > 0:
            # SHAP beeswarm summary plot
            assert explain_numvar <= len(features)
            plt.clf()
            plt.ioff()
            shap.summary_plot(shap_values, df_shapValue, show=False, max_display=50) # 绘制 SHAP beeswarm 图
            fig = plt.gcf()

            if savePath is not None:
                plot_name_dict["SHAP_变量贡献度总结"] = save_fig( # 保存 SHAP beeswarm 图
                    savePath, "shap_summary", "png", fig, str_time=str_time
                )
                plot_name_dict_save["SHAP_变量贡献度总结"] = save_fig( # 保存指定格式 SHAP beeswarm 图
                    savePath, "shap_summary", picFormat, fig, str_time=str_time
                )
            plt.close(fig)

            shap.summary_plot(
                shap_values, df_shapValue, plot_type="bar", show=False,max_display=50
            )
            fig1 = plt.gcf()

            if savePath is not None:
                plot_name_dict["SHAP_重要性图"] = save_fig(
                    savePath, "shap_import", "png", fig1, str_time=str_time
                )
                plot_name_dict_save["SHAP_重要性图"] = save_fig(
                    savePath, "shap_import", picFormat, fig1, str_time=str_time
                )
            plt.close(fig1)
            if shap_catter and len(shap_catter_feas) > 0:
                for fea in shap_catter_feas:
                    shap.dependence_plot(
                        fea,
                        shap_values,
                        df_shapValue,
                        interaction_index=None,
                        show=False,
                    )
                    fig2 = plt.gcf()
                    if savePath is not None:
                        plot_name_dict["SHAP_点图_" + fea] = save_fig(
                            savePath,
                            "shap_catter_" + fea,
                            "png",
                            fig2,
                            str_time=str_time,
                        )
                        plot_name_dict_save["SHAP_点图_" + fea] = save_fig(
                            savePath,
                            "shap_catter_" + fea,
                            picFormat,
                            fig2,
                            str_time=str_time,
                        )
                        plt.close(fig2)


        if explain_sample > 0:
            for i in range(len(shapValue_list)):
                # SHAP explain
                shap.force_plot( # 绘制 SHAP force plot
                    explainer.expected_value,
                    shap_values[shapValue_list[i]],
                    df_shapValue_show.iloc[i, :],
                    show=False,
                    figsize=(15, 3),
                    matplotlib=True,
                )
                fig = plt.gcf()
                if savePath is not None:
                    plot_name_dict[shapValue_name[i]] = save_fig(
                        savePath,
                        "shap_sample_{}".format(i + 1),
                        "png",
                        fig,
                        str_time=str_time,
                    )
                    plot_name_dict_save[shapValue_name[i]] = save_fig(
                        savePath,
                        "shap_sample_{}".format(i + 1),
                        picFormat,
                        fig,
                        str_time=str_time,
                    )
                plt.close(fig)

                if shap_waterfall:
                    sample_idx = shapValue_list[i]
                    sample_shap_values = shap_values[sample_idx]
                    sample_features = df_shapValue_show.iloc[i, :]
                    explanation_object = shap.Explanation(
                        values=sample_shap_values,
                        base_values=explainer.expected_value,  # 模型的基线值
                        data=sample_features.values,  # 样本特征值
                        feature_names=df_shapValue_show.columns.tolist()  # 特征名
                    )

                    shap.waterfall_plot(explanation_object, max_display=50, show=False)

                    fig_waterfall = plt.gcf()

                    if savePath is not None:
                        waterfall_name = shapValue_name[i].replace("shap_样本", "shap_waterfall_樣本")
                        plot_name_dict[waterfall_name] = save_fig(
                            savePath,
                            "shap_waterfall_{}".format(i + 1),
                            "png",
                            fig_waterfall,
                            str_time=str_time,
                        )
                        plot_name_dict_save[waterfall_name] = save_fig(
                            savePath,
                            "shap_waterfall_{}".format(i + 1),
                            picFormat,
                            fig_waterfall,
                            str_time=str_time,
                        )
                    plt.close(fig_waterfall)


    # 将寻参相关的图片文件名字典更新到主图片字典
    plot_name_dict.update(plot_name_dict1)

    #总时间汇总
    time_summary_str = "\n\n--- 流程执行时间汇总 ---\n"
    for step_name, duration in timing_results.items():
        time_summary_str += f"{step_name}: {duration:.4f} 秒\n"

    #替换计时信息
    str_result = str_result.replace("[TIMING_PLACEHOLDER]", time_summary_str)

    result_dict = {
        "str_result": {"分析结果描述": str_result},
        "tables": df_dict,
        "pics": plot_name_dict,
        "save_pics": plot_name_dict_save,
        "model": result_model_save
    }
    return result_dict
