4.共线性分析
def get_var_vif(df_input, features=None,decimal_num=3):
    if features is not None:
        df_input = df_input[features]
    else:
        n,b=_feature_get_n_b(df_input)
        df_input = df_input[n+b]
    df_input=df_input.dropna()
    df_input[df_input.shape[1]]=1
    #vif
    vif=[]
    for i in range(df_input.shape[1] - 1):
        vif.append(variance_inflation_factor(df_input.values, i))
    #result_out
    df_result=pd.DataFrame(df_input.columns[:-1, ])
    df_result.rename(columns={0:"变量名"},inplace=True)
    df_result["vif"]=vif
    df_result[["vif"]]= df_result[["vif"]].applymap(lambda x: _round_dec(x, decimal_num))
    df_result=df_result.sort_values(['vif'],ascending = False)
    df_result.reset_index(drop = True,inplace=True)

    result_dict = {'str_result': None,
                   'tables': {'分析结果描述表': df_result},
                   'pics': None}
    return result_dict

