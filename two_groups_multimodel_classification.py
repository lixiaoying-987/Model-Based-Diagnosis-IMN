多模型比较
import os
import random
import datetime
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib
# import joblib
# from skopt import BayesSearchCV
# from statsmodels.formula.api import ols
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# from AnalysisFunction.utils_ml.params import BayesDefaultRange

matplotlib.use("AGG")
import matplotlib.pyplot as plt

from sklearn.base import clone
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# from sklearn.inspection import permutation_importance
# from skopt import BayesSearchCV
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split as TTS
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import RandomizedSearchCV

# from xgboost import XGBRegressor
# from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
# from sklearn.neighbors import KNeighborsRegressor
#
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import Birch
# from sklearn.cluster import KMeans
# from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import AgglomerativeClustering
#
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
#
# from sklearn.feature_selection import RFECV
# from sklearn.decomposition import PCA
# from AnalysisFunction.utils_ml.FeatrueSelect import mrmr_classif
# from AnalysisFunction.utils_ml.FeatrueSelect import ReliefF
#
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
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

# from xgboost import plot_importance
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
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV##,HalvingGridSearchCV
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

def two_groups_classfication_multimodels(
    df_input,
    group,
    features,
    methods=[],
    decimal_num=3,
    testsize=0.2,
    boostrap=5,
    randomState=42,
    smooth=False,
    searching="default",
    label='LABEL',
    trainSet=False,
    trainLabel=0,
    PR=False,
    style='lancet',
    dpi=600,
    picFormat="jpeg",
    isKFold="cross",
    savePath=None,
    resultType=0,
    delong=False,
    DCA_cut=0,
    **kwargs,
):
    """
    df_input:Dataframe
    features:自变量list
    group：因变量str
    testsize: 测试集比例
    boostrap：重采样次数
    searching:bool 是否进行自动寻参，默认为否
    savePath:str 图片存储路径
    ##交叉验证 cross ,重采样 resample,嵌套交叉验证nest
    """
    palette_dict = {
        'lancet': ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF",
                   "#ADB6B6FF",
                   "#1B1919FF"],
        'nejm': ["#BC3C29FF", "#0072B5FF", "#E18727FF", "#20854EFF", "#7876B1FF", "#6F99ADFF", "#FFDC91FF", "#EE4C97FF",
                 "#BC3C29FF"],
        'jama': ["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF", "#79AF97FF", "#6A6599FF", "#80796BFF", "#374E55FF",
                 "#DF8F44FF"],
        'npg': ["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF",
                "#7E6148FF", "#B09C85FF"]}
    str_time = (
        str(datetime.datetime.now().hour)
        + str(datetime.datetime.now().minute)
        + str(datetime.datetime.now().second)
    )
    random_number = random.randint(1, 100)
    str_time = str_time + str(random_number)

    features_flag = False
    if boostrap == 1 and trainSet:
        if label in features or label == group:
            return {"error": "标签列不能在所在模型中，请重新选择数据划分标签列！"+"false-error"}
        dftemp = df_input[features + [group]+[label]].dropna()
    else:
        dftemp = df_input[features + [group]].dropna()
    x = dftemp[features]
    y = dftemp[[group]]



    u = np.sort(np.unique(np.array(dftemp[group])))
    if len(u) == 2 and set(u) != set([0, 1]):
        y_result = label_binarize(dftemp[group], classes=[ii for ii in u])  # 将标签二值化
        y_result_pd = pd.DataFrame(y_result, columns=[group])
        df = pd.concat([dftemp.drop(group, axis=1), y_result_pd], axis=1)
        x = df[features]
        y = df[[group]]
    elif len(u) > 2:
        return {"error": "暂时只支持二分类。请检查因变量取值情况。"+"false-error"}

    name_dict = {
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
    if len(methods) == 0:
        methods = [
            "LogisticRegression",
            "XGBClassifier",
            "RandomForestClassifier",
            # 'SVC',
            # 'MLPClassifier',
            # 'AdaBoostClassifier',
            # 'KNeighborsClassifier',
            # 'DecisionTreeClassifier',
            # 'BaggingClassifier',
        ]
    str_result = "已采用多种机器学习模型尝试完成数据样本分类任务，包括：{}。各模型的参数值选取情况如下所示：\n\n".format(methods)

    plot_name_list = []
    plot_name_dict = {}
    plot_name_dict_save = {}

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    # 画对角线
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1,
        color="r",
        alpha=1.0,
    )
    ax.grid(which="major", axis="both", linestyle="-.", alpha=0.3, color="grey")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax.tick_params(top=False, right=False)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax.set_xlabel("1-Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC curve(Validation)")

    mean_fpr = np.linspace(0, 1, 100)
    colors = x5.CB91_Grad_BP

    df_0 = pd.DataFrame(columns=list(make_class_metrics_dict().keys()), index=[0])
    df_0_test = df_0.copy()

    df_plot = pd.DataFrame(columns=["method", "mean", "std"])



    fpr_train_alls, tpr_train_alls, train_method_alls, mean_auc_train_alls = (
        [],
        [],
        [],
        [],
    )
    fraction_of_positives_alls, mean_predicted_value_alls, clf_score_alls = [], [], []
    AUC_95CI_test, AUC_95CI_SD_test, AUC_95CI_train, AUC_95CI_SD_train = [], [], [], []
    brier_scores_all = []

    DCA_dict = {}
    model_test_data_all = {}
    sdorci = " SD " if resultType == 0 else " 95%CI "

    X_train_ps, Y_train_ps, model_train_s = [], [], []  ###PR曲线
    X_test_ps, Y_test_ps = [], []
    name = []

    for i, method in enumerate(methods):
        tprs_train, tprs_test = [], []

        name.append(name_dict[method])
        if searching == "auto":
            # if method == "LGBMClassifier":
            #     searcher = GridSearcherCV("Classification", globals()[method]())
            #     selected_model = searcher(x, y)
            # else:
            #     searcher = RandSearcherCV("Classification", globals()[method]())
            #     selected_model = searcher(x, y)  # ; searcher.report()
            searcher = RandomizedSearchCV(globals()[method](), param_distributions=GridDefaultRange[method])
            # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
            selected_model = searcher.fit(x, y)
        elif searching == "default":
            # selected_model = globals()[method]() if (method != 'SVC') else globals()[method](probability=True)
            if method == "SVC":
                selected_model = globals()[method](probability=True,random_state=42)
            elif method == "MLPClassifier":
                selected_model = globals()[method](
                    hidden_layer_sizes=(20, 10), max_iter=20,random_state=42
                )
            elif method == "RandomForestClassifier":
                selected_model = globals()[method](n_estimators=20,random_state=42)
            else:
                if method in random_model:
                    selected_model = globals()[method](random_state=42)
                else:
                    selected_model = globals()[method]()


        elif searching == "handle":
            method_dicts = kwargs
            if i == 0:
                me_count = True
                for me_list in methods:
                    if me_list in method_dicts.keys():
                        me_count = False
                        continue
                if me_count:
                    return {"error": "请设置要调参的模型！"+"false-error"}
            if method in method_dicts.keys():
                method_dict = {}
                if method == "SVC":
                    method_dict.update({"probability": True})
                method_dict.update(method_dicts[method])
                if (
                    method == "RandomForestClassifier"
                    and method_dict["max_depth"] == "None"
                ):
                    method_dict["max_depth"] = None
                if (
                    method == "DecisionTreeClassifier"
                    and method_dict["max_depth"] == "None"
                ):
                    method_dict["max_depth"] = None
                if method == "MLPClassifier":
                    hls_vals = str(method_dict["hidden_layer_sizes"]).split(",")
                    hls_value = ()
                    for hls_val in hls_vals:
                        try:
                            if int(hls_val) >= 5 and int(hls_val) <= 200:
                                hls_value = hls_value + (int(hls_val),)
                            else:
                                return {"error": "请按照要求重新设置隐藏层宽度！"+"false-error"}
                        except:
                            return {"error": "请重新设神经网络模型中的隐藏层宽度！"+"false-error"}
                    method_dict["hidden_layer_sizes"] = hls_value
                if method == "GaussianNB" and method_dict["priors"] == "None":
                    method_dict["priors"] = None
                elif method == "GaussianNB":
                    pri_vals = str(method_dict["priors"]).split(",")
                    pri_value = ()
                    pri_sum = 0.0
                    for pri_val in pri_vals:
                        try:
                            pri_sum = float(pri_val) + pri_sum
                            pri_value = pri_value + (float(pri_val),)
                        except:
                            return {"error": "请重新设朴素贝叶斯模型中的先验概率！"+"false-error"}
                    if len(pri_vals) == len(np.unique(y)) and pri_sum == 1.0:
                        method_dict["priors"] = pri_value
                    else:
                        return {"error": "请重新设朴素贝叶斯模型中的先验概率！"+"false-error"}
                if method in random_model:
                    method_dict["random_state"] = 42
                selected_model = globals()[method](**method_dict)
            else:
                # if method == "LGBMClassifier":
                #     searcher = GridSearcherCV("Classification", globals()[method]())
                #     selected_model = searcher(x, y)
                # else:
                #     searcher = RandSearcherCV("Classification", globals()[method]())
                #     selected_model = searcher(x, y)  # ; searcher.report()
                searcher = RandomizedSearchCV(globals()[method](), param_distributions=GridDefaultRange[method])
                # searcher=GridSearchCV(globals()[method](),param_grid=GridDefaultRange[method])
                selected_model = searcher.fit(x, y)
                selected_model = globals()[method](**selected_model.best_params_)

        list_evaluate_dic_train = make_class_metrics_dict()
        list_evaluate_dic_test = make_class_metrics_dict()

        clf_score = 1
        fraction_of_positives = np.array([1])
        mean_predicted_value = np.array([1])

        p_serie_s_te, net_benefit_serie_s_te, net_benefit_serie_All_s_te = [], [], []

        data_all = {}
        test_data_delong = {}
        conf_dic_train, conf_dic_test = {}, {}
        if isKFold == "cross":
            # KF = KFold(n_splits=boostrap, random_state=42,shuffle=True)
            KF = StratifiedKFold(
                n_splits=boostrap, random_state=randomState, shuffle=True
            )
            for i_k, (train_index, valid_index) in enumerate(KF.split(x, y)):
                # 划分训练集和验证集
                Xtrain, Xtest = x.iloc[train_index], x.iloc[valid_index]
                Ytrain, Ytest = y.iloc[train_index], y.iloc[valid_index]
                data_all.update(
                    {
                        i_k: {
                            "Xtrain": Xtrain,
                            "Ytrain": Ytrain,
                            "Xtest": Xtest,
                            "Ytest": Ytest,
                        }
                    }
                )
                test_data_delong.update({i_k: np.array(Ytest).T[0]})
        elif isKFold == "nest":
            KF = StratifiedKFold(
                n_splits=boostrap, random_state=randomState, shuffle=True
            )
            inner_cv = StratifiedKFold(
                n_splits=boostrap, random_state=randomState, shuffle=True
            )
            for i_k, (train_index, valid_index) in enumerate(KF.split(x, y)):
                # 划分训练集和验证集
                Xtrain, Xtest = x.iloc[train_index], x.iloc[valid_index]
                Ytrain, Ytest = y.iloc[train_index], y.iloc[valid_index]
                data_all.update(
                    {
                        i_k: {
                            "Xtrain": Xtrain,
                            "Ytrain": Ytrain,
                            "Xtest": Xtest,
                            "Ytest": Ytest,
                        }
                    }
                )
                test_data_delong.update({i_k: np.array(Ytest).T[0]})
        else:
            for index in range(0, boostrap):
                if boostrap == 1:
                    if trainSet:
                        if isinstance(dftemp[label][0], str):
                            trainLabel = str(trainLabel)
                        train_a  = dftemp[dftemp[label] == trainLabel]
                        test_a = dftemp[dftemp[label] != trainLabel]
                        train_all = train_a.drop(label, axis=1)
                        test_all = test_a.drop(label, axis=1)
                        # dftemp = dftemp.drop(label, axis=1)
                        Xtrain = train_all.drop(group, axis=1)
                        Ytrain = train_all.loc[:, [group]].squeeze(axis=1)
                        Xtest = test_all.drop(group, axis=1)
                        Ytest = test_all.loc[:, [group]].squeeze(axis=1)
                    else:
                        X = dftemp.drop(group, axis=1)
                        Y = dftemp.loc[:, [group]].squeeze(axis=1)
                        Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=testsize, random_state=randomState, )
                else:
                    if searching == "handle":
                        Xtrain, Xtest, Ytrain, Ytest = TTS(
                            x, y, test_size=testsize, random_state=index
                        )
                    else:
                        Xtrain, Xtest, Ytrain, Ytest = TTS(x, y, test_size=testsize)
                data_all.update(
                    {
                        index: {
                            "Xtrain": Xtrain,
                            "Ytrain": Ytrain,
                            "Xtest": Xtest,
                            "Ytest": Ytest,
                        }
                    }
                )
                test_data_delong.update({index: np.array(Ytest).T[0]})
        if method == methods[0]:
            model_test_data_all.update({"original": test_data_delong})
        # for index in range(0, boostrap):
        test_all_data_delong = {}
        X_train_p, Y_train_p, model_train = [], [], []  ##PR曲线
        X_test_p, Y_test_p = [], []
        brier_scores=[]
        for data_key, data_value in data_all.items():
            if isKFold == "nest":
                best_auc = 0.0
                resThreshold = 0
                Xtrain, Ytrain, Xtest, Ytest = (
                    data_value["Xtrain"],
                    data_value["Ytrain"],
                    data_value["Xtest"],
                    data_value["Ytest"],
                )
                for j, (train_index_inner, test_index_inner) in enumerate(
                    inner_cv.split(Xtrain, Ytrain)
                ):
                    X_train_inner, X_test_inner = (
                        Xtrain.iloc[train_index_inner],
                        Xtrain.iloc[test_index_inner],
                    )
                    y_train_inner, y_test_inner = (
                        Ytrain.iloc[train_index_inner],
                        Ytrain.iloc[test_index_inner],
                    )

                    model_i = clone(selected_model).fit(X_train_inner, y_train_inner)
                    # 利用classification_metric_evaluate函数获取在验证集的预测值
                    _, _, metric_dic_train_i, _ = classification_metric_evaluate(
                        model_i, X_train_inner, y_train_inner, True
                    )
                    _, _, metric_dic_valid_i, _ = classification_metric_evaluate(
                        model_i,
                        X_test_inner,
                        y_test_inner,
                        True,
                        Threshold=metric_dic_train_i["cutoff"],
                    )
                    # metric_dic_valid.update({'cutoff': metric_dic_train_i['cutoff']})
                    if metric_dic_valid_i["AUC"] > best_auc:
                        model = model_i
                        resThreshold = metric_dic_train_i["cutoff"]
            else:
                Xtrain, Ytrain, Xtest, Ytest = (
                    data_value["Xtrain"],
                    data_value["Ytrain"],
                    data_value["Xtest"],
                    data_value["Ytest"],
                )

                model = clone(selected_model).fit(Xtrain, Ytrain)
            ####################################
            X_train_p.append(Xtrain)
            Y_train_p.append(Ytrain)
            model_train.append(model)
            X_test_p.append(Xtest)
            Y_test_p.append(Ytest)
            ##########################################
            Yprob = model.predict_proba(Xtest)[:, 1]
            test_all_data_delong.update({data_key: Yprob})
            (
                prob_pos,
                p_serie,
                net_benefit_serie,
                net_benefit_serie_All,
            ) = calculate_net_benefit(model, Xtest, Ytest)
            p_serie_s_te.append(p_serie)
            net_benefit_serie_s_te.append(net_benefit_serie)
            net_benefit_serie_All_s_te.append(net_benefit_serie_All)
            """
            if hasattr(model, "predict_proba"):
                prob_pos = model.predict_proba(Xtest)[:, 1]
            else:  # use decision function
                prob_pos = model.decision_function(Xtest)
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            """
            clf_score1 = brier_score_loss(
                Ytest, prob_pos, pos_label=y[group].max()
            )  ##strategy='quantile',
            brier_scores.append(clf_score1)
            if clf_score > clf_score1:
                clf_score = clf_score1
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    Ytest, prob_pos, n_bins=10
                )

            # 利用classification_metric_evaluate函数获取在测试集的预测值
            try:
                if isKFold == "nest":
                    (
                        fpr_train,
                        tpr_train,
                        metric_dic_train,
                        _,
                    ) = classification_metric_evaluate(
                        model, Xtrain, Ytrain, Threshold=resThreshold
                    )
                else:
                    (
                        fpr_train,
                        tpr_train,
                        metric_dic_train,
                        _,
                    ) = classification_metric_evaluate(model, Xtrain, Ytrain)

                fpr_test, tpr_test, metric_dic_test, _ = classification_metric_evaluate(
                    model, Xtest, Ytest, Threshold=metric_dic_train["cutoff"]
                )
                metric_dic_test.update({"cutoff": metric_dic_train["cutoff"]})
            except Exception as e:
                return {
                    "error": "数据不均衡，至少有一组验证集中存在结局全部为0或者1的数据！请选择另外一种方法重采样（交叉验证）的方法处理！"
                }

            # interp:插值 把结果添加到tprs列表中
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
            tprs_train[-1][0] = 0.0
            tprs_test[-1][0] = 0.0

            # 计算所有评价指标
            for key in list_evaluate_dic_train.keys():
                list_evaluate_dic_train[key].append(metric_dic_train[key])
                list_evaluate_dic_test[key].append(metric_dic_test[key])

        model_test_data_all.update({method: test_all_data_delong})
        DCA_dict[name_dict[method]] = {
            "p_serie": p_serie_s_te,
            "net_b_s": net_benefit_serie_s_te,
            "net_b_s_A": net_benefit_serie_All_s_te,
        }

        X_train_ps.append(X_train_p)
        Y_train_ps.append(Y_train_p)
        model_train_s.append(model_train)

        X_test_ps.append(X_test_p)
        Y_test_ps.append(Y_test_p)
        ###画校准曲线
        # X_train, X_test, Y_train, Y_test = TTS(x, y, test_size=testsize, random_state=0)
        # model_CC = clone(selected_model).fit(X_train, Y_train)
        # y_pred = model.predict(Xtest)
        # if hasattr(model_CC, "predict_proba"):
        #    prob_pos = model_CC.predict_proba(X_test)[:, 1]
        # else:  # use decision function
        #    prob_pos = model_CC.decision_function(X_test)
        #    prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        # clf_score = brier_score_loss(Y_test, prob_pos, pos_label=y.max())##strategy='quantile',
        # fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos,
        #                                                                        n_bins=10)
        clf_score_alls.append(clf_score)
        brier_scores_all.append(brier_scores)
        fraction_of_positives_alls.append(fraction_of_positives)
        mean_predicted_value_alls.append(mean_predicted_value)

        for key in list_evaluate_dic_train.keys():
            metric_dic_train[key] = np.mean(list_evaluate_dic_train[key])
            metric_dic_test[key] = np.mean(list_evaluate_dic_test[key])

            if resultType == 0:  ##SD
                list_evaluate_dic_train[key] = np.std(
                    list_evaluate_dic_train[key], axis=0
                )
                list_evaluate_dic_test[key] = np.std(
                    list_evaluate_dic_test[key], axis=0
                )
            elif resultType == 1:  ##CI
                conf_dic_train[key] = list(ci(list_evaluate_dic_train[key]))
                conf_dic_test[key] = list(ci(list_evaluate_dic_test[key]))
        result_dic_train = metric_dic_train
        result_dic_test = metric_dic_test
        if resultType == 0:  ##SD
            for tem in ["AUC_L", "AUC_U"]:
                del list_evaluate_dic_train[tem]
                del list_evaluate_dic_test[tem]
            for key in list_evaluate_dic_train.keys():
                if key == "AUC":
                    result_dic_train["AUC(95%CI)"] = (
                        str(round_dec(float(metric_dic_train[key]), d=decimal_num))
                        + "("
                        + str(
                            round_dec(
                                float(list_evaluate_dic_train[key]), d=decimal_num
                            )
                        )
                        + ")"
                    )

                    result_dic_test["AUC(95%CI)"] = (
                        str(round_dec(float(metric_dic_test[key]), d=decimal_num))
                        + "("
                        + str(
                            round_dec(float(list_evaluate_dic_test[key]), d=decimal_num)
                        )
                        + ")"
                    )
                else:
                    result_dic_train[key] = (
                        str(round_dec(float(metric_dic_train[key]), d=decimal_num))
                        + "("
                        + str(
                            round_dec(
                                float(list_evaluate_dic_train[key]), d=decimal_num
                            )
                        )
                        + ")"
                    )
                    result_dic_test[key] = (
                        str(round_dec(float(metric_dic_test[key]), d=decimal_num))
                        + "("
                        + str(
                            round_dec(float(list_evaluate_dic_test[key]), d=decimal_num)
                        )
                        + ")"
                    )
        elif resultType == 1:
            for tem in ["AUC_L", "AUC_U"]:
                del conf_dic_train[tem]
                del conf_dic_test[tem]
            for key in conf_dic_train.keys():
                if key == "AUC":
                    result_dic_train["AUC(95%CI)"] = (
                        str(round_dec(float(metric_dic_train[key]), decimal_num))
                        + " ("
                        + str(round_dec(float(metric_dic_train["AUC_L"]), decimal_num))
                        + "-"
                        + str(round_dec(float(metric_dic_train["AUC_U"]), decimal_num))
                        + ")"
                    )

                    result_dic_test["AUC(95%CI)"] = (
                        str(round_dec(float(metric_dic_test[key]), decimal_num))
                        + " ("
                        + str(round_dec(float(metric_dic_test["AUC_L"]), decimal_num))
                        + "-"
                        + str(round_dec(float(metric_dic_test["AUC_U"]), decimal_num))
                        + ")"
                    )
                else:
                    result_dic_train[key] = (
                        str(round(float(metric_dic_train[key]), decimal_num))
                        + "("
                        + str(round_dec(float(conf_dic_train[key][0]), d=decimal_num))
                        + "-"
                        + str(round_dec(float(conf_dic_train[key][1]), d=decimal_num))
                        + ")"
                    )
                    result_dic_test[key] = (
                        str(round(float(metric_dic_test[key]), decimal_num))
                        + "("
                        + str(round_dec(float(conf_dic_test[key][0]), d=decimal_num))
                        + "-"
                        + str(round_dec(float(conf_dic_test[key][1]), d=decimal_num))
                        + ")"
                    )
        df_train_result = pd.DataFrame([result_dic_train], index=["Mean"])
        df_test_result = pd.DataFrame([result_dic_test], index=["Mean"])
        df_train_result["分类模型"] = name_dict[method]
        df_test_result["分类模型"] = name_dict[method]

        AUC_95CI_test.append(list(df_test_result.iloc[0, -4:-2]))
        AUC_95CI_train.append(list(df_train_result.iloc[0, -4:-2]))

        df_0 = pd.concat([df_0, df_train_result])
        df_0_test = pd.concat([df_0_test, df_test_result])

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_test = np.mean(tprs_test, axis=0)
        mean_tpr_train[-1] = 1.0
        mean_tpr_test[-1] = 1.0
        mean_auc_train = auc(mean_fpr, mean_tpr_train)  # 计算训练集平均AUC值
        mean_auc_test = auc(mean_fpr, mean_tpr_test)

        ###画训练集ROC
        fpr_train_alls.append(mean_fpr)
        tpr_train_alls.append(mean_tpr_train)
        train_method_alls.append(method)
        mean_auc_train_alls.append(mean_auc_train)

        std_value = 0
        if resultType == 0:
            std_value = list_evaluate_dic_test["AUC"]
        elif resultType == 1:
            std_value = (conf_dic_train["AUC"][1] - conf_dic_train["AUC"][0]) / 2
        df_plot = df_plot.append(
            {
                "method": name_dict[method],
                "mean": mean_auc_test,
                "std": std_value,
            },
            ignore_index=True,
        )
        print(df_plot)
        ax.plot(
            mean_fpr,
            mean_tpr_test,
            c=palette_dict[style][i],
            label=name_dict[method]
            + "(AUC = "
            + df_test_result['AUC(95%CI)'][0][:df_test_result['AUC(95%CI)'][0].find('(')].strip()
            + sdorci
            + df_test_result['AUC(95%CI)'][0][df_test_result['AUC(95%CI)'][0].find('('):].strip()
            + ")",
            lw=1.5,
            alpha=1,
        )
        if searching == 'auto':
            str_result += (
                    method
                    + ": AUC="
                    + str(round_dec(mean_auc_train, decimal_num))
                    + ";  模型参数:\n"
                    + dic2str(selected_model.best_params_, method)
                    + "\n"
            )
        else:
            str_result += (
                    method
                    + ": AUC="
                    + str(round_dec(mean_auc_train, decimal_num))
                    + ";  模型参数:\n"
                    + dic2str(selected_model.get_params(), method)
                    + "\n"
            )
    ###模型德龙检测
    if delong:
        delong_z, delong_p = [], []
        for i in range(boostrap):
            zzz, ppp = [], []
            for method1 in methods:
                zz, pp = [], []
                for method2 in methods:
                    z, p = delong_roc_test(
                        model_test_data_all["original"][i],
                        model_test_data_all[method1][i],
                        model_test_data_all[method2][i],
                    )
                    zz.append(z[0][0])
                    pp.append(p[0][0])
                zzz.append(zz)
                ppp.append(pp)
            delong_z.append(zzz)
            delong_p.append(ppp)
        if boostrap == 1:
            delong_zz1 = pd.DataFrame(
                reduce(lambda x, y: np.array(x) + np.array(y), delong_z),
                index=methods,
                columns=methods,
            )
            delong_pp1 = pd.DataFrame(
                reduce(lambda x, y: np.array(x) + np.array(y), delong_p),
                index=methods,
                columns=methods,
            )
        else:
            delong_zz1 = pd.DataFrame(
                reduce(lambda x, y: np.array(x) + np.array(y), delong_z)
                / len(delong_z),
                index=methods,
                columns=methods,
            )
            delong_pp1 = pd.DataFrame(
                reduce(lambda x, y: np.array(x) + np.array(y), delong_p)
                / len(delong_p),
                index=methods,
                columns=methods,
            )

        delong_zz = delong_zz1.applymap(lambda x: round_dec(x, d=decimal_num))
        delong_pp = delong_pp1.applymap(lambda x: round_dec(x, d=decimal_num))
        delong_zz_name = pd.DataFrame(
            list(delong_zz.index), columns=["name"], index=list(delong_zz.index)
        )
        delong_zz = pd.concat([delong_zz_name, delong_zz], axis=1, ignore_index=False)
        delong_pp_name = pd.DataFrame(
            list(delong_pp.index), columns=["name"], index=list(delong_pp.index)
        )
        delong_pp = pd.concat([delong_pp_name, delong_pp], axis=1, ignore_index=False)

    # ymin = min([y - dy for y, dy in zip(df_plot['mean'], df_plot['std'])])
    # ymax = max([y + dy for y, dy in zip(df_plot['mean'], df_plot['std'])])
    # ymin, ymax = ymin - (ymax - ymin) / 4.0, ymax + (ymax - ymin) / 10.0

    if boostrap != 1:
        ymax = (
            np.max(df_plot["mean"])
            + np.max(df_plot["std"])
            + (np.max(df_plot["mean"]) - np.min(df_plot["mean"])) / 4
        )
        ymin = (
            np.min(df_plot["mean"])
            - np.max(df_plot["std"])
            - (np.max(df_plot["mean"]) - np.min(df_plot["mean"])) / 4
        )

        ymax = math.ceil(ymax * 100) / 100
        ymin = int(ymin * 100) / 100

    ax.legend(loc="lower right", fontsize=5)
    ax.legend(loc="lower right", fontsize=5)

    df_test_auc = []
    if savePath is not None:
        plot_name_list.append(
            save_fig(savePath, "valid_ROC_curve", "png", fig, str_time=str_time)
        )
        plot_name_dict_save["验证集ROC曲线"] = save_fig(
            savePath, "valid_ROC_curve", picFormat, fig, str_time=str_time
        )
        plt.close(fig)

        # 画训练集ROC
        fig1 = plt.figure(figsize=(4, 4), dpi=dpi)
        # 画对角线
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.3, color="grey")

        for i in range(len(fpr_train_alls)):
            df_test_auc.append(df_0.iloc[i + 1]["AUC"])
            plt.plot(
                fpr_train_alls[i],
                tpr_train_alls[i],
                lw=1.5,
                alpha=0.9,
                c=palette_dict[style][i],
                label=name_dict[train_method_alls[i]]
                + "(AUC = "
                + df_0.iloc[i + 1]['AUC(95%CI)'][:df_0.iloc[i + 1]['AUC(95%CI)'].find('(')].strip()
                + sdorci
                + df_0.iloc[i + 1]['AUC(95%CI)'][df_0.iloc[i + 1]['AUC(95%CI)'].find('('):].strip()
                + ")",
            )

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC curve(Training)")
        plt.legend(loc="lower right", fontsize=5)

        plot_name_list.append(
            save_fig(savePath, "ROC_Train_curve", "png", fig1, str_time=str_time)
        )
        plot_name_dict_save["训练集ROC曲线图"] = save_fig(
            savePath, "ROC_Train_curve", picFormat, fig1, str_time=str_time
        )
        plt.close(fig1)
        plot_name_list.reverse()  ###所有图片倒置

        if boostrap != 1:
            # df_plot.drop('mean', axis=1)
            # df_plot.loc[:,'mean']=pd.Series(df_test_auc,name='mean')
            plot_name_list += x5.forest_plot(
                df_input=df_plot,
                name="method",
                value="mean",
                err="std",
                direct="horizontal",
                fig_size=[len(methods) + 3, 9],
                ylim=[ymin, ymax],
                title="Forest Plot of Each Model AUC Score ",
                path=savePath,
                dpi=dpi,
                picFormat=picFormat,
            )
            plot_name_dict_save["验证集多模型森林图"] = plot_name_list[len(plot_name_list) - 1]
            plot_name_list.pop(len(plot_name_list) - 1)
    plt.close()
    ###画校准曲线
    if savePath is not None:
        from scipy.optimize import curve_fit
        from scipy.interpolate import make_interp_spline

        def fit_f(x, a, b):
            return a * np.arcsin(x) + b

        def fit_show(x, y_fit):
            a, b = y_fit.tolist()
            return a * np.arcsin(x) + b

        fig, ax1 = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        for i in range(len(mean_predicted_value_alls)):
            if resultType == 0:
                BS_SD = np.std(brier_scores_all[i])
                BS_score=" (%1.3f SD(%1.3f))" % (np.mean(brier_scores_all[i]),BS_SD)
            else:
                brier_down, brier_up = ci(brier_scores_all[i])
                BS_score=" (%1.3f 95%%CI(%1.3f-%1.3f))" % (
                np.mean(brier_scores_all[i]), brier_down, brier_up)
            if smooth and len(fraction_of_positives_alls[i]) >= 3:
                x_new = np.linspace(
                    min(mean_predicted_value_alls[i]),
                    max(mean_predicted_value_alls[i]),
                    len(fraction_of_positives_alls[i]) * 10,
                )
                try:
                    p_fit, _ = curve_fit(
                        fit_f,
                        mean_predicted_value_alls[i],
                        fraction_of_positives_alls[i],
                        maxfev=10000,
                    )
                    y_smooth = fit_show(x_new, p_fit)
                    # y_fit = np.polyfit(mean_predicted_value_alls[i], fraction_of_positives_alls[i], 3)
                    # y_smooth = f_fit(x_new, y_fit)
                    # y_smooth = spline(mean_predicted_value_alls[i], fraction_of_positives_alls[i], x_new)

                    ax1.plot(
                        x_new,
                        y_smooth,
                        c=palette_dict[style][i],
                        label=name_dict[methods[i]]+BS_score,
                    )
                except Exception as e:
                    ax1.plot(
                        mean_predicted_value_alls[i],
                        fraction_of_positives_alls[i],
                        "s-",
                        c=palette_dict[style][i],
                        label=name_dict[methods[i]]+BS_score,
                    )
            else:
                ax1.plot(
                    mean_predicted_value_alls[i],
                    fraction_of_positives_alls[i],
                    "s-",
                    c=palette_dict[style][i],
                    label=name_dict[methods[i]]+BS_score,
                )

        ax1.set_xlabel("Mean predicted value")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title("Calibration curve(Validation)")##Calibration plots  (reliability curve)
        plt.gca()
        plt.close()
        plot_name = "Calibration_curve_" + str_time
        plot_name_list.append(
            save_fig(savePath, plot_name, "png", fig, str_time=str_time)
        )
        plot_name_dict_save["验证集多模型校准曲线"] = save_fig(
            savePath, plot_name, picFormat, fig, str_time=str_time
        )

    ###画DCA曲线
    if savePath is not None:
        decision_curve_p = plot_decision_curves(
            DCA_dict,
            colors=palette_dict[style],
            name="Validation",
            savePath=savePath,
            dpi=dpi,
            picFormat=picFormat,
            DCA_cut=DCA_cut
        )
        plot_name_list.append(decision_curve_p[0])
        plot_name_dict_save["验证集DCA曲线图"] = decision_curve_p[1]

    df_train_result1 = df_0.drop([0])
    df_test_result1 = df_0_test.drop([0])

    classfier = df_train_result1.pop("分类模型")

    df_train_result = df_train_result1.applymap(lambda x: round_dec(x, d=decimal_num))
    df_train_result.insert(0, "分类模型", classfier)

    df_test_result1.pop("分类模型")
    df_test_result = df_test_result1.applymap(lambda x: round_dec(x, d=decimal_num))
    df_test_result.insert(0, "分类模型", classfier)

    AUC_95CI_tr = df_train_result.pop("AUC(95%CI)")
    df_train_result.insert(1, "AUC(95%CI)", AUC_95CI_tr)
    AUC_95CI_te = df_test_result.pop("AUC(95%CI)")
    df_test_result.insert(1, "AUC(95%CI)", AUC_95CI_te)

    df_train_result = df_train_result.drop(["AUC_L", "AUC_U"], axis=1)
    df_test_result = df_test_result.drop(["AUC_L", "AUC_U"], axis=1)

    if features_flag:
        df_count_r = round_dec(Xtest.shape[0] / x.shape[0], decimal_num)
    else:
        df_count_r = round_dec(testsize, decimal_num)
    if isKFold:
        str_result += (
            "\n下示森林图展示了各模型进行"
            + group
            + "预测的ROC结果,图中的误差线为ROC均值及SD。\n"
            + "模型的ROC均值及SD的是通过"
            + str(boostrap)
            + "折交叉验证,"
            + "模型中的变量包括"
            + ",".join(features)
            + "。\n"
        )
    else:
        str_result += (
            "\n下示森林图展示了各模型进行"
            + group
            + "预测的ROC结果,图中的误差线为ROC均值及SD。\n"
            + "模型的ROC均值及SD的是通过多次重复采样计算，重复采样次数为"
            + str(boostrap)
            + "次,"
            + "每一次重采样训练的验证集占总体样本的"
            + str(df_count_r * 100)
            + "%,训练集占"
            + str((1 - df_count_r) * 100)
            + "%,"
            + "模型中的变量包括"
            + ",".join(features)
            + "。\n"
        )

    best_ = (
        df_train_result.loc[df_train_result.index == "Mean"]
        .sort_values(by="AUC", ascending=False)
        .head(1)
    )
    name_train = best_.iloc[0]["分类模型"]
    str_result += "在目前所有模型中，训练集表现最佳者为{}（依据AUC排序），在各评价标准中其在训练集对应分数分别为：\n".format(
        name_train
    )
    for col in best_.columns[1:]:
        str_result += "\t{}：{}\n".format(col, best_.iloc[0][col])

    best_ = (
        df_test_result.loc[df_test_result.index == "Mean"]
        .sort_values(by="AUC", ascending=False)
        .head(1)
    )
    name_test = best_.iloc[0]["分类模型"]
    str_result += "验证集表现最佳者为{}（依据AUC排序），在各评价标准中其在验证集对应分数分别为：\n".format(name_test)
    for col in best_.columns[1:]:
        str_result += "\t{}：{}\n".format(col, best_.iloc[0][col])

    if name_test == name_train:
        str_result += "二者吻合，可以认为{}是针对此数据集的最佳模型选择。".format(name_train)
    else:
        str_result += "二者不吻合，{}极可能存在过拟合现象，{}可能稳定性相对较好。具体模型选择可根据下表详细评分信息进行取舍。".format(
            name_train, name_test
        )

    if resultType == 0:
        df_train_result.rename(
            columns={
                "AUC(95%CI)": "AUC(SD)",
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
        df_test_result.rename(
            columns={
                "AUC(95%CI)": "AUC(SD)",
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
    elif resultType == 1:
        df_train_result.rename(
            columns={
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
        df_test_result.rename(
            columns={
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
    df_dict = {
        "多模型分类-训练集结果汇总": df_train_result.drop(["AUC"], axis=1),
        "多模型分类-验证集结果汇总": df_test_result.drop(["AUC"], axis=1),
    }
    if delong:
        for ii in range(delong_zz.shape[0]):
            delong_zz.iloc[ii, ii + 1] = "NA"
        for ii in range(delong_pp.shape[0]):
            delong_pp.iloc[ii, ii + 1] = "NA"
        df_dict.update({"delong检测Z值均值表": delong_zz})
        df_dict.update({"delong检测P值均值表": delong_pp})

    if boostrap != 1:
        plot_name_dict = {
            "训练集ROC曲线图": plot_name_list[0],
            "验证集ROC曲线图": plot_name_list[1],
            "验证集多模型森林图": plot_name_list[2],
            "验证集多模型校准曲线": plot_name_list[3],
            "验证集DCA曲线图": plot_name_list[4],
        }
    else:
        plot_name_dict = {
            "训练集ROC曲线图": plot_name_list[0],
            "验证集ROC曲线图": plot_name_list[1],
            "验证集多模型校准曲线": plot_name_list[2],
            "验证集DCA曲线图": plot_name_list[3],
        }

    ###画PR曲线

    if PR:
        from sklearn.metrics import plot_precision_recall_curve
        from AnalysisFunction.X_5_SmartPlot import plot_precision_recall_curve


        fig = plot_precision_recall_curve(
            model_train_s,
            X_train_ps,
            Y_train_ps,
            name=name,
            picname="PR Curve(Training)",
            resultType=resultType,
        )
        plot_name_dict["训练集多模型PR曲线"] = save_fig(
            savePath, "PR_train", "png", fig, str_time=str_time
        )
        plot_name_dict_save["训练集多模型PR曲线"] = save_fig(
            savePath, "PR_train", picFormat, fig, str_time=str_time
        )
        plt.close(fig)
        fig = plot_precision_recall_curve(
            model_train_s,
            X_test_ps,
            Y_test_ps,
            name=name,
            picname="PR Curve(Validation)",
            resultType=resultType,
        )
        plot_name_dict["验证集多模型PR曲线"] = save_fig(
            savePath, "PR_valid", "png", fig, str_time=str_time
        )
        plot_name_dict_save["验证集多模型PR曲线"] = save_fig(
            savePath, "PR_valid", picFormat, fig, str_time=str_time
        )
        plt.close(fig)
    result_dict = {
        "str_result": {"分析结果描述": str_result},
        "tables": df_dict,
        "pics": plot_name_dict,
        "save_pics": plot_name_dict_save,
    }
    return result_dict
