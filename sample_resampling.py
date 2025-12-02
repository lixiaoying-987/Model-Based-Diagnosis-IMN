def data_balance(df_input, group, ratio, method='SMOTE', randomState=42):
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
    from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from sklearn.cluster import KMeans
    from sklearn.utils import resample
    plot_name_dict_save = {}  ##存储图片
    df_input = df_input.dropna(subset=[group])
    Ratio = ratio
    list_name = [group]
    continuous_features, categorical_features, time_features = _feature_classification(df_input)
    features = continuous_features
    if group not in features:
        features += group
    df_temp = df_input[features].copy()
    df_temp = df_temp.fillna(0)
    Y = df_temp.loc[:, list_name]
    X = df_temp.drop(group, axis=1)
    column_nameX = X.columns

    try:
        if method in ['SMOTE', 'ADASYN', 'RandomOverSampler', 'BorderlineSMOTE']:
            if method == 'SMOTE':
                sampler = SMOTE(sampling_strategy=Ratio, random_state=randomState)
            elif method == 'ADASYN':
                sampler = ADASYN(sampling_strategy=Ratio, random_state=randomState)
            elif method == 'RandomOverSampler':
                sampler = RandomOverSampler(sampling_strategy=Ratio, random_state=randomState)
            elif method == 'BorderlineSMOTE':
                sampler = BorderlineSMOTE(sampling_strategy=Ratio, kind='borderline-1', random_state=randomState)

            X_resampled, Y_resampled = sampler.fit_resample(X, Y)

        elif method in ['RandomUnderSampler', 'ClusterCentroids', 'NearMiss']:
            if method == 'RandomUnderSampler':
                sampler = RandomUnderSampler(sampling_strategy=Ratio, random_state=randomState)
            elif method == 'ClusterCentroids':
                sampler = ClusterCentroids(sampling_strategy=Ratio, random_state=randomState)
            elif method == 'NearMiss':
                sampler = NearMiss(sampling_strategy=Ratio)

            X_resampled, Y_resampled = sampler.fit_resample(X, Y)

        elif method in ['SMOTEENN', 'SMOTETomek']:
            if method == 'SMOTEENN':
                sampler = SMOTEENN(sampling_strategy=Ratio, random_state=randomState)
            elif method == 'SMOTETomek':
                sampler = SMOTETomek(sampling_strategy=Ratio, random_state=randomState)

            X_resampled, Y_resampled = sampler.fit_resample(X, Y)

            # # 调整样本数量，确保总样本数与原始数据一致
            # total_samples = len(X)
            # if len(X_resampled) < total_samples:
            #     # 从重采样的数据中有放回地抽样，增加样本数量
            #     X_resampled, Y_resampled = resample(
            #         X_resampled, Y_resampled,
            #         replace=True,
            #         n_samples=total_samples,
            #         random_state=randomState
            #     )
            # elif len(X_resampled) > total_samples:
            #     # 随机下采样，减少样本数量
            #     X_resampled, Y_resampled = resample(
            #         X_resampled, Y_resampled,
            #         replace=False,
            #         n_samples=total_samples,
            #         random_state=randomState
            #     )

        elif method == 'BDSK':
            bsmote = BorderlineSMOTE(sampling_strategy=ratio, random_state=randomState)
            X_res, Y_res = bsmote.fit_resample(X, Y)

            # 检查过采样是否增加了数据集大小。如果是，则进行下采样。改进的下采样。
            if len(X_res) != len(X):
                rus = RandomUnderSampler(sampling_strategy=ratio, random_state=randomState)  # 使用 RandomUnderSampler
                X_resampled, Y_resampled = rus.fit_resample(X_res, Y_res)
            else:
                X_resampled = X_res
                Y_resampled = Y_res


        elif method == 'BalanceCascade':
            from imblearn.ensemble import RUSBoostClassifier
            from sklearn.linear_model import LogisticRegression
            balance_cascade = RUSBoostClassifier(sampling_strategy=ratio, random_state=randomState,n_estimators=10)
            balance_cascade.fit(X, Y.values.ravel())
            # 使用模型的预测结果模拟采样
            Y_pred = balance_cascade.predict(X)  # 获取预测标签
            X_resampled = X  # 原数据保持不变
            Y_resampled = Y.copy()
            Y_resampled.loc[:, group] = Y_pred



        elif method == 'EasyEnsemble':
            from imblearn.ensemble import EasyEnsembleClassifier  # 使用分类器
            easy_ensemble = EasyEnsembleClassifier(sampling_strategy=ratio,n_estimators=10, random_state=randomState)
            easy_ensemble.fit(X, Y.values.ravel())  # 训练分类器

            # 使用模型的预测结果模拟采样
            Y_pred = easy_ensemble.predict(X)  # 获取预测标签
            X_resampled = X  # 原数据保持不变
            Y_resampled = Y.copy()
            Y_resampled.loc[:, group] = Y_pred


        # 对于未包含的情况，返回错误
        else:
            return {'error': '不支持的方法：' + method + ' false-error'}

    except Exception as e:
        return {'error': str(e) + ' false-error'}

    # 将X和Y合并
    array_result = np.column_stack((X_resampled, Y_resampled))
    list_column_name = list(column_nameX)
    list_column_name.append(group)
    df_result = pd.DataFrame(array_result, columns=list_column_name)

    # 确保数据类型一致
    df_result = df_result.astype(df_temp.dtypes.to_dict())
    # if not keep_decimals:
    #     df_result = df_result.round(0)

    group_labels = np.unique(df_result[group])
    str_temp = ''
    for group_label in group_labels:
        str_temp += group + '(' + str(group_label) + ')=' + str((df_result[group] == group_label).sum()) + '例，'

    str_result = '采用' + dict_method[method] + '方法对数据进行平衡，使得少数类和多数类的比例为' + str(Ratio) + \
                 '，最终匹配结果为' + str_temp + \
                 '该方法会自动剔除包含空值的行，以及非数值变量的列。'


    input_o_desc, input_n_desc = _describe(df_input)
    result_o_desc, result_n_desc = _describe(df_result)
    input_o_desc.fillna('', inplace=True)
    input_n_desc.fillna('', inplace=True)
    result_o_desc.fillna('', inplace=True)
    result_n_desc.fillna('', inplace=True)
    result_dict = {'str_result': {'分析结果描述': str_result},
                   'tables': {'最终数据': df_result, '原分类变量描述表': input_o_desc, '原连续变量描述表': input_n_desc,
                              '分析之后分类变量描述表': result_o_desc, '分析之后连续变量描述表': result_n_desc, },
                   }
    # output_path = os.path.join(savePath, 'balanced_data.csv')
    # df_result.to_csv(output_path, index=False)
    return result_dict

