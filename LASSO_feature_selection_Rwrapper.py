6.lasso
def R_lasso(df_input, dependent_variable, feature, tim="", savePath=None, method='binomial', decimal_num=3, dpi=600,
            picFormat='jpeg'):
    """
    LASSO回归
    df_input：DataFrame 输入的待处理数据
    dependent_variable：str 应变量
    feature: strvector 自变量集合
    adjust: str 调整变量
    method:str 分类还是连续因变量?binomial(分类),gaussian(连续)
    savePath:str 图片路径
    decimal_num: int  小数点位数
    """
    LASSO = ro.r(
        '''
#V1.0.0
#Author YUANKE
#date 2021年8月18日11:40:35
LASSO <- function(mydata,target,feature,tim=NULL,fml="binomial",savePath,P=0.05,round= 3,dpi=600,picFormat='jpeg'){
  # input:
  # mydata:dataframe 需处理的数据：要求数据框只能是数值型，不能有字母汉字
  # target:str 应变量
  # feature: strvector 自变量集合
  # fml: str 分类还是连续因变量?binomial(分类),gaussian(连续)
  # savePath:str 图片保存路径
  # round:int 小数点位数
  # return:
  # results$p1: str 图1
  # results$p2: str 图2
  # results$descrip: str 描述结果
  library(glmnet)
  library(survival)
  set.seed(1234)
  result<-data.frame()
  results <- list() #结果
  descrip<-NULL
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  rand <- sample(1:100,1)
  mydata <- na.omit(mydata)
  mydata<-as.data.frame(lapply(mydata,as.numeric))#先要全部改成数值型
  if (fml=='cox'){
    for(i in 1:length(feature)){
      formula <- paste0("Surv","(",tim,",",target,")",'~',feature[i])
      fit_cox<-coxph(as.formula(formula), data = mydata,ties = "breslow")
      result[i,c(1:5)]<-summary(fit_cox)$coefficients[,c(1,3,4,5,2)]
    }

    #predictor <- rownames(result)
    orlow <- exp(result[,1]-1.96*result[,2])
    orup <- exp(result[,1]+1.96*result[,2])
    dtable <- cbind(orlow,orup)
    coefs <- cbind(result,dtable)
    coefs <- round(coefs,round)
    rownames(coefs)  <- NULL
    coefs <- data.frame(name = feature, coefs,row.names = feature)
    colnames(coefs) <- c("Predictor","Estimate","SE","Z","p","Hazard Ratio","Lower","Upper")
    results$cox_coefs <- coefs
    fea<-rownames(coefs[c(which(coefs$p<P)),])##选取显著因子
    if (length(fea)<2){
        x<-data.matrix(mydata[,feature])
        descrip<-paste0(descrip,"在进行Lasso-Cox 特征筛选时将用全部特征进行lasso特征筛选！@")
    }
    else{
        x<-data.matrix(mydata[,fea])
    }

    y<-data.matrix(Surv(as.double(mydata[,tim]),as.double(mydata[,target])))
  }else{
    y<-data.matrix(mydata[,target])#本来是as.matrix
    x<-data.matrix(mydata[,feature])#需要先改成数值型再改成matrix否则会将factor变成字符型
  }
  f1 = glmnet(x, y, family=fml, nlambda=100, alpha=1)
  cvfit=cv.glmnet(x,y, family=fml)#为了避免出错改成data.matrix
  tab<-data.frame(round(data.matrix(coef(cvfit)),round))
  colnames(tab)<-'coef'
  tab['name']<-row.names(tab)
  results$tab<-tab[c('name','coef')]
  #图1
  p1name <- paste0("p1pic",mytime,"_",rand,".png")
  p1name1 <- paste0("p1pic",mytime,"_",rand,".",picFormat)
  png(file=paste0(savePath,p1name),width=7,height=8,units="in",res=dpi)
  plot(f1, xvar="lambda", label=T)#出图1
  abline(v = log(cvfit$lambda.min), lty = 3)
  abline(v = log(cvfit$lambda.1se), lty = 3)
  dev.off()
  if ((picFormat=='svg')|(picFormat=='pdf')|(picFormat=='eps'))
  {
    if (picFormat=='eps'){picFormat='cairo_ps'}
    do.call(picFormat,list(file=paste0(savePath,p1name1),width=7,height=8))
    if (picFormat=='cairo_ps'){picFormat='eps'}
  }
  else
  {
    do.call(picFormat,list(file=paste0(savePath,p1name1),width=7,height=8,units="in",res=dpi))
  }
  plot(f1, xvar="lambda", label=T)#出图1
  abline(v = log(cvfit$lambda.min), lty = 3)
  abline(v = log(cvfit$lambda.1se), lty = 3)
  dev.off()

  results$p1 <- c(p1name,p1name1)
  #图2
  p2name <- paste0("p2pic",mytime,"_",rand,".png")
  p2name1 <- paste0("p2pic",mytime,"_",rand,".",picFormat)
  png(file=paste0(savePath,p2name),width=7,height=8,units="in",res=dpi)
  p2<-plot(cvfit)#出图2
  dev.off()
  if ((picFormat=='svg')|(picFormat=='pdf')|(picFormat=='eps'))
  {
    if (picFormat=='eps'){picFormat='cairo_ps'}
    do.call(picFormat,list(file=paste0(savePath,p2name1),width=7,height=8))
    if (picFormat=='cairo_ps'){picFormat='eps'}
  }
  else
  {
    do.call(picFormat,list(file=paste0(savePath,p2name1),width=7,height=8,units="in",res=dpi))
  }
  p2<-plot(cvfit)#出图2
  dev.off()
  results$p2 <- c(p2name,p2name1)
  #描述
  s1=cvfit$lambda.min#求出最小值
  s2=cvfit$lambda.1se#求出最小值一个标准误的λ值
  mod1<-coef(cvfit$glmnet.fit,s=s1,exact = F)#这两个模型就是最终筛选出的模型
  mod2<-coef(cvfit$glmnet.fit,s=s2,exact = F)#

  m1<-rownames(data.frame(which(mod1[-1,]!=0)))
  m2<-rownames(data.frame(which(mod2[-1,]!=0)))
  descrip<-paste0(descrip,'最小均方误差的λ为',round(s1,round),',对应模型的变量选择为:',paste(m1,collapse = '+'),'@',
                  '最小距离的标准误差的λ为',round(s2,round),',对应模型的变量选择为:',paste(m2,collapse = '+'))
  results$descrip<-enc2utf8(descrip)
  return(results)
}
           ''')

    if len(feature) < 2:
        return {'error': 'LASSO回归中特征组合最少需要两个，请继续添加！' + 'false-error'}
    dv_unique = pd.unique(df_input[dependent_variable])
    if method == 'binomial' and len(dv_unique) > 2:
        return {'error': 'LASSO回归中binomial方法因变量只能是二分类，请重新操作！' + 'false-error'}
    if method == 'cox':
        r_df_input = df_input[[dependent_variable] + [tim] + feature].dropna()
        r_df_input = r_df_input[r_df_input[tim] > 0]  ##cox 中删除时间小于等于0的数据
    else:
        r_df_input = df_input[[dependent_variable] + feature].dropna()

    _, cc, _ = _feature_classification(r_df_input[feature])
    if len(cc) > 0:
        return {'error': 'LASSO回归存在非数值型的分类数据，请将其转化为数值型的数据之后再进行分析！' + 'false-error'}

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        r_df = ro.conversion.py2rpy(r_df_input)
    target = dependent_variable
    feature = ro.StrVector(feature)

    result = LASSO(mydata=r_df, target=target, feature=feature, tim=tim,
                   fml=method, savePath=savePath, P=0.1, round=decimal_num, dpi=dpi, picFormat=picFormat)

    list_plot_path = []
    list_plot_dict = {}
    table_dic = {}
    list_plot_dict_save = {}
    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        list_plot_path = list(result.rx2('p1'))
        list_plot_dict.update({'图1': list_plot_path[0]})
        list_plot_dict_save.update({'图1': list_plot_path[1]})
        list_plot_path = list(result.rx2('p2'))
        list_plot_dict.update({'图2': list_plot_path[0]})
        list_plot_dict_save.update({'图2': list_plot_path[1]})
        if method == 'cox':
            df_result1 = ro.conversion.rpy2py(result.rx2('cox_coefs'))  # 表格
            table_dic.update({'cox回归系数表': df_result1})
        df_result = ro.conversion.rpy2py(result.rx2('tab'))  # 表格
        table_dic.update({'系数表': df_result})
        str_result = tuple(result.rx2('descrip'))[0]  # 描述
        str_result = str_result.replace('@', '\n')

    result_dict = {'str_result': {'分析结果描述': str_result}, 'tables': table_dic, 'pics': list_plot_dict,
                   'save_pics': list_plot_dict_save}
    return result_dict

