1  数据插补
import datetime
import os
import traceback # 新增：用于捕获详细的异常信息
#解耦导入
from AnalysisFunction.celerytest.utils.data_describe import _describe
from AnalysisFunction.celerytest.utils.feature_classification import _feature_classification
from AnalysisFunction.celerytest.utils.get_environment import _get_environment_info
from AnalysisFunction.celerytest.utils.knn_imputation import _knn_imputation
from AnalysisFunction.celerytest.utils.rf_imputation import _rf_imputation


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import datetime
import random
import warnings
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
warnings.filterwarnings('ignore')

# 新增：获取环境信息
env_info = _get_environment_info(__file__)

imputation_method = {'mean': '均值', 'median': '中位数', 'most_frequent': '众数',
                     'nan': '空值', 'randomforest': '随机森林', 'interpolate': '插值法', 'mul_interpolate': '多重插补', 'KNN': 'KNN'}

dict_method = dict(imputation_method)


"""
智能数据填补
df_input:Dataframe 处理数据
features:list 填补特征(列)
method:str 填补方式(一般数据填补：'mean': '均值', 'median': '中位数', 'most_frequent': '众数',constant:‘常数填补',
                 nan': '空值'，'interpolate': '插值法','mul_interpolate': '多重插补',
                  AI智能填补：'randomforest': '随机森林填补', 'KNN': 'KNN')
constant：object 常数填补参数
output_dir：图片保存路径
dpi：图片保存分辨率
image_format：图片另存格式
"""


def data_filling(df_input, features=None, method=None, constant=None, output_dir=None, dpi=600, image_format='jpeg'):
    try:
        # tables_list = []
        # figures_list = [] # 由 distribution_curve 填充

        if features is None:
            continuous_features, categorical_features, time_features = _feature_classification(df_input)
            features = continuous_features
            df_temp = df_input[features]
        else:
            df_temp = df_input[features]

        if (method == 'randomforest'):
            df_temp = _rf_imputation(df_temp)
            method_str = dict_method[method]

        elif method in ["mean", "median", "most_frequent"]:
            imp_ = SimpleImputer(missing_values=np.nan, strategy=method)
            df_temp = pd.DataFrame(data=imp_.fit_transform(df_temp), columns=features)
            method_str = dict_method[method]
        elif method == 'constant':
            imp_ = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=constant)
            df_temp = pd.DataFrame(data=imp_.fit_transform(df_temp), columns=features)
            method_str = '常数' + str(constant)
        elif method == 'interpolate':
            df_temp = df_temp.interpolate().fillna(method='bfill')  # 插值法不能对首行空值填充，后面加上向后填充方法
            method_str = dict_method[method]
        elif method =='mul_interpolate':
            from AnalysisFunction.X_5_R_SmartPlot import mul_interpolate_R_docker
            mul_str_result, df_temp = mul_interpolate_R_docker(df_temp, method='pmm', m=10, path=output_dir)
            if mul_str_result == "":
                method_str = dict_method[method]
                df_temp = pd.DataFrame(df_temp)
            else:
                raise ValueError(mul_str_result)

        elif method == 'KNN':
            df_temp = _knn_imputation(df_temp)
            method_str = dict_method[method]
        else:
            raise ValueError('请输入正确的方法')

        # 图片监控
        def distribution_curve(df_pre, df_pos, features, path, dpi=600, picFormat='jpeg'):
            figures_list = [] # 初始化
            for feature in features:
                str_time = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute) + str(
                    datetime.datetime.now().second)
                random_number = random.randint(1, 1000)
                str_time = str_time + str(random_number)
                
                plot_base_name = 'distribution_curve_' + feature.replace(' ', '_') + '_' + str_time
                pic_name_png = plot_base_name + '.png'
                pic_name_jpeg = plot_base_name + '.' + picFormat
                plt.figure(figsize=[6, 6],dpi=dpi)
                
                try:
                    temp1 = df_pre[feature]
                    temp2 = df_pos[feature]
                    temp1.plot(kind='kde', label='pre')
                    temp2.plot(kind='kde', label='post')
                    plt.title(feature)
                    plt.legend()
                    plt.savefig(os.path.join(path, pic_name_png), bbox_inches='tight', format='png')
                    plt.savefig(os.path.join(path, pic_name_jpeg), bbox_inches='tight',format=picFormat)

                    figures_list.append({
                        "name": f"分布曲线_{feature}",
                        "file": {"png": pic_name_png, picFormat: pic_name_jpeg},
                        "description": f"'{feature}' 填补前 (pre) 与填补后 (post) 的分布曲线。"
                    })

                except Exception as e:
                    figures_list.append({
                        "name": f"分布曲线_{feature}",
                        "file": {},
                        "description": f"图形生成失败: {str(e)}"
                    })
                finally:
                    plt.close('all') 
            
            return figures_list # 返回标准格式的列表

        df_result = df_input.copy()
        df_result = df_result.drop(features, axis=1)
        df_result.index = list(df_temp.index)
        
        if len(df_temp[features])!=len(df_result):
            raise ValueError("填补的数据中存在共有缺失的数据使得填补之后数据总样本小于整体样本量，请再添加特征重新填补！")
        
        df_result = pd.concat([df_result, df_temp[features]], axis=1)
        figures_list = distribution_curve(df_input, df_result, features, output_dir, dpi=dpi, picFormat=image_format)

        str_result = '采用' + method_str + '填补的方式对' + "、".join(features) + '进行数据填补'
        input_o_desc, input_n_desc = _describe(df_input[features])
        result_o_desc, result_n_desc = _describe(df_result[features])

        input_o_desc.fillna('', inplace=True)
        input_n_desc.fillna('', inplace=True)
        result_o_desc.fillna('', inplace=True)
        result_n_desc.fillna('', inplace=True)

        tables_list = [
            {
                "name": "填补后数据表",
                "file": {"dataframe": df_result},
                "description": "包含填补后特征的完整数据集。"
            },
            {
                "name": "原分类变量描述表",
                "file": {"dataframe": input_o_desc},
                "description": "填补前分类变量的描述性统计。"
            },
            {
                "name": "原连续变量描述表",
                "file": {"dataframe": input_n_desc},
                "description": "填补前连续变量的描述性统计。"
            },
            {
                "name": "分析之后分类变量描述表",
                "file": {"dataframe": result_o_desc},
                "description": "填补后分类变量的描述性统计。"
            },
            {
                "name": "分析之后连续变量描述表",
                "file": {"dataframe": result_n_desc},
                "description": "填补后连续变量的描述性统计。"
            }
        ]
        return {
            "status": "success",
            "description": str_result,
            "tables": tables_list,
            "figures": figures_list,
            "environment": env_info
        }

    except ValueError as e:
        return {
            "status": "error",
            "description": str(e),
            "tables": [],
            "figures": [],
            "environment": env_info
        }
    
    # 优化：捕获意外的运行时错误
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message) # 打印追溯信息
        return {
            "status": "exception",
            "description": f"发生意外错误: {str(e)}\n\nTraceback:\n{error_message}",
            "tables": [],
            "figures": [],
            "environment": env_info
        }


2 异常值处理
def abnormal_deviation_process(df_input, features, method='median', ratio=1.5, path=None,dpi=600,picFormat='jpeg'):
    df_feature = df_input[features]

    gl_res_str_adp = ""
    gl_index_adp=0
    # df_feature1 = df_feature.copy()
    def lower_upper_limit(c):
        lower_q = np.nanquantile(c, 0.25, interpolation='lower')  # 下四分位数
        higher_q = np.nanquantile(c, 0.75, interpolation='higher')  # 上四分位数
        int_r = higher_q - lower_q
        lower_limit = lower_q - ratio * int_r
        high_limit = higher_q + ratio * int_r
        return lower_limit, high_limit

    def set_Disposition(columns, Disposition):
        if Disposition == 'median':
            a = np.median(columns)
        elif Disposition == 'mean':
            a = np.mean(columns)
        elif Disposition == 'most_frequent':
            a = Counter(columns).most_common()[0][0]
        elif Disposition == 'zero':
            a = 0
        elif Disposition == 'nan':
            a = np.nan
        return a

    def replace_exception(column_list):
        # 将列中异常值进行替换
        nonlocal gl_res_str_adp
        nonlocal gl_index_adp
        column_list1 = column_list.copy()
        gl_res_str_adp += "对变量"+features[gl_index_adp]+"进行处理："
        lower_limit, high_limit = lower_upper_limit(column_list)
        re_str=""
        for i in range(len(column_list)):
            if i==0:
                gl_index_adp+=1
            if column_list[i] < lower_limit or column_list[i] > high_limit:
                column_list1[i] = set_Disposition(column_list, method)
                re_str+=str(i+1)+","
        if re_str!="":
            gl_res_str_adp += re_str+"行的数据存在偏离\n"
        else:
            gl_res_str_adp += "没有偏离的数据\n"
        return column_list1
    df_temp = df_feature.apply(replace_exception, axis=0)
    str_result = '采用' + dict_method[method] + '方法对' + "、".join(features) + '中异常偏离超过正常值范围' + str(ratio) + '倍值的数据进行处理。'
    df_result = df_input.drop(features, 1)
    df_result = pd.concat([df_result, df_temp], axis=1)
    input_o_desc, input_n_desc = _describe(df_input[features])
    result_o_desc, result_n_desc = _describe(df_result[features])
    input_o_desc.fillna('', inplace=True)
    input_n_desc.fillna('', inplace=True)
    result_o_desc.fillna('', inplace=True)
    result_n_desc.fillna('', inplace=True)
    ax1 = plt.figure()
    plt_dict_path = {}
    plt_dict_path_save = {}
    if len(features)<19:
        plt01 = x5.comparison_plot(df_input=df_feature, features=features, group=None, kind='box', concat_way='free',
                                   row_size=1, col_size=len(features), path=path, dpi=dpi, picFormat=picFormat)
        ax2 = plt.figure()
        plt02 = x5.comparison_plot(df_input=df_temp, features=features, group=None, kind='box', concat_way='free',
                                   row_size=1,
                                   col_size=len(features), path=path, dpi=dpi, picFormat=picFormat)

        plt_dict_path.update({'比较图1': plt01['pics']['比较图']})
        plt_dict_path_save.update({'比较图1': plt01['save_pics']['比较图']})
        plt_dict_path.update({'比较图2': plt02['pics']['比较图']})
        plt_dict_path_save.update({'比较图1': plt02['save_pics']['比较图']})
    # if len(features) == 1:
    #     df_box = pd.concat([df_feature, df_temp], axis=1)
    #     df_box.columns = [features[0], features[0] + "(处理后)"]
    #     df_box.boxplot(  # 指定绘图数据
    #                 patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充
    #                 showmeans=True,  # 以点的形式显示均值
    #                 boxprops={'color': 'black', 'facecolor': 'steelblue'},  # 设置箱体属性，如边框色和填充色
    #                 flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3},
    #                 # 设置均值点的属性，如点的形状、填充色和点的大小
    #                 meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'markersize': 4},
    #                 # 设置中位数线的属性，如线的类型和颜色
    #                 medianprops={'linestyle': '--', 'color': 'orange'},
    #                 # labels=['']  # 删除x轴的刻度标签，否则图形显示刻度标签为1
    #                 )
    result_dict = {'str_result': {'分析结果描述': str_result+gl_res_str_adp},
                   'tables': {'最终数据': df_result, '原分类变量描述表': input_o_desc, '原连续变量描述表': input_n_desc,
                              '分析之后分类变量描述表': result_o_desc, '分析之后连续变量描述表': result_n_desc, },
                   'pics': plt_dict_path, 'save_pics': plt_dict_path_save}
    return result_dict


 自动剔除缺失变量和缺失病例（行，列）
def miss_data_delete(df_input, miss_rate, miss_axis,features=None):
    str_result = '无缺失率大于' + str(miss_rate) + '的数据'
    list1=[]
    if (miss_axis == 1):
        miss_rowtotal, miss_rowper = _miss_row(df_input)
        list1 = miss_rowper[miss_rowper > miss_rate].index
        df_result = df_input.drop(list1)
        # df_miss_rate = pd.concat([df_input, miss_rowper], axis=1)
        if len(list1) > 0:
            list_new = map(lambda x: str(x+2), list1)
            #str_result = '剔除缺失率大于' + str(miss_rate) + '的案例，剔除案例的id为' + "、".join(list_new)
            str_result = '剔除缺失率大于' + str(miss_rate) + '的案例，剔除剔除的行号为' + "、".join(list_new)
    if (miss_axis == 0):
        col_total, col_percent = _miss_col(df_input)
        list1 = col_percent[col_percent > miss_rate].index
        df_result = df_input.drop(list1, axis=1)
        # df_miss_rate = df_input.append(col_percent, ignore_index=True)
        if len(list1) > 0:
            list_new = map(lambda x: str(x), list1)
            str_result = '剔除缺失率大于' + str(miss_rate) + '的变量，共剔除' + str(len(list1)) + '个变量,是' + "、".join(list_new)
            for i, v in col_percent.items():
                if v <= miss_rate:
                    break
                else:
                    str_result += '\n%s缺失率为%.2f%%,共缺失样本%d例' % (
                        i, (v * 100), col_total[i]
                    )
    if not features is None:
        if len(list1)>0:
            ff=list(set(features)-set(list1))
        else:
            ff=features
        input_o_desc, input_n_desc = _describe(df_input[ff])
        result_o_desc, result_n_desc = _describe(df_result[ff])
    else:
        input_o_desc, input_n_desc = _describe(df_input)
        result_o_desc, result_n_desc = _describe(df_result)


    input_o_desc.fillna('', inplace=True)
    input_n_desc.fillna('', inplace=True)
    result_o_desc.fillna('', inplace=True)
    result_n_desc.fillna('', inplace=True)

    df_result = df_result.reset_index(drop=True)


    result_dict = {'str_result': {'分析结果描述': str_result},
                   'tables': {'最终数据': df_result,'原分类变量描述表':input_o_desc,'原连续变量描述表':input_n_desc,'分析之后分类变量描述表':result_o_desc,'分析之后连续变量描述表':result_n_desc,},
                   'pics': None}
    return result_dict
