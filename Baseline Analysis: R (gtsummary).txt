1.Baseline Analysis: R (gtsummary)
baseline_ana_R<-function(df_input,
                         outcome,
                         N_features=NULL,
                         Q_features=NULL,
                         method="gtsummary",
                         savepath=NULL,
                         style=1,
                         #theme="nejm",
                         decimal_num=3){

  ### method  gtsummary或者compareGroups

  ### style gtsummary单用
  ### theme=c("jama", "lancet", "nejm", "qjecon") gtsummary单用

  library(rms)
  library(compareGroups)
  library(gtsummary)
  library(gt)
  library(flextable)
  results <- list() #结果
  results$error<-'success'
  table_name<-NULL
  table_path<-NULL
  table_savepath<-NULL
  descrip<-NULL
  head_descrip<-NULL
  str_res<-NULL
  res_scrip<-NULL
  mytime <- format(Sys.time(), "%b_%d_%H_%M_%S_%Y")  #时间
  rand <- sample(1:1000,1)
  head_descrip<-paste0(head_descrip,"\nR版本为：",getRversion())
  head_descrip<-paste0(head_descrip,",所涉及的的主要包版本gtsummary为：",packageVersion('gtsummary'),'\n')

  #features <- c(Q_features, N_features)

  if (method == "gtsummary"){

    #theme_gtsummary_journal(journal=theme)
    if(style!=1){
      compact=TRUE
      eda=TRUE
    }else{
      compact=FALSE
      eda=FALSE
    }
    #打印的表格将更紧凑，字体更小，单元格填充更少
    theme_gtsummary_compact(set_theme=compact)
    #默认显示中值、平均值、IQR、SD和范围
    theme_gtsummary_eda(set_theme=eda)

    if (length(outcome)==1){
      features <- c(Q_features, N_features)
      mydata <- df_input[,c(outcome,features)]

      res_tab<-mydata %>%
        tbl_summary(by=outcome,
                    type = list(Q_features ~ "continuous2"),
                    digits = all_continuous() ~ decimal_num,
                    missing_text = "(Missing)"
        )

        res_tab<-res_tab %>%
          add_p(pvalue_fun = ~ style_pvalue(.x, digits = decimal_num))
        res_tab<-res_tab %>%
          add_overall()
    }
    else{
      features <- c(Q_features, N_features)
      mydata <- df_input[,features]

      res_tab<-mydata %>%
        tbl_summary(type = list(Q_features ~ "continuous2"),
                    digits = all_continuous() ~ decimal_num,
                    missing_text = "(Missing)")
    }

    result = res_tab[["table_body"]]
    result1 = result[,c(1,2,14)]
    result1 = na.omit(result1)
    result1= as.data.frame(result1)
    rownames(result1) = result1$variable

    res_scrip=paste0("利用gtsummary包进行基线分析，研究不同",outcome,"分组中各个指标的差异是否具有统计学意义，总的有效样本为",nrow(df_input),'例，其中')

    for (la in unique(df_input[[outcome]])) {
      count <- sum(df_input[[outcome]] == la, na.rm = TRUE)
      res_scrip <- paste0(res_scrip, outcome, "=", la, "：病例数为", count, "；")
    }

    #paste0(strsplit(result1[i,2], "\\.")[[1]][1:2], collapse = ".")
    coefstat <- NULL
    for(i in 1:nrow(result1)){
      if(result1[i,3]<0.05) {
        coefstat[i] <- paste0(rownames(result1)[i],'经',paste0(strsplit(result1[i,2], "\\.")[[1]][1:2], collapse = "."),"检验p值<0.05,组间存在统计学意义。",'\n')
      }
      else{
        coefstat[i] <- paste0(rownames(result1)[i],'经',paste0(strsplit(result1[i,2], "\\.")[[1]][1:2], collapse = "."),"检验p值>0.05,组间不存在统计学意义。",'\n')
      }
      conclusion <- paste0(res_scrip, paste0(coefstat[1:nrow(result1)],collapse = " "))
    }



    res_tab<-res_tab %>%
      modify_header(label ~ "**Variable**") %>%
      modify_spanning_header(all_stat_cols() ~"**group**") %>%
      modify_footnote(
        all_stat_cols() ~ "Median (IQR) or Frequency (%)"
      ) %>%
      modify_caption("**Baseline Table**")%>%
      bold_labels()
    res_tab1<-res_tab %>% as_gt()
    # rres_tab<-as_tibble(res_tab)
    tab_html=paste0("Table_",mytime,"_",rand,".html")
    #   tab_pdf=paste0("Table_",mytime,"_",rand,".pdf")
    tab_doc=paste0("Table_",mytime,"_",rand,".docx")
    # write.csv(rres_tab,paste0(savepath,tab_name),row.names = FALSE)
    # res_tab %>%
    #   as_gt() %>%
    #   gt::gtsave(filename =paste0(savepath,tab_pdf))
    gtsave(res_tab1,filename =paste0(savepath,tab_html))
    # gtsave(res_tab,filename =paste0(savepath,tab_pdf))
    mytable <- as_flex_table(res_tab)
    save_as_docx(mytable, path = paste0(savepath,tab_doc))
    table_name<-c(table_name,'基线表')
    table_path<-c(table_path,tab_html)
    table_savepath<-c(table_savepath,tab_doc)
  }
  if (method=="compareGroups"){

    if (length(outcome)==1){

      features <- c(Q_features, N_features)
      mydata <- df_input[,c(outcome,features)]

      mydata[N_features] <- lapply(mydata[N_features], as.factor)

      formula <- paste0(outcome,'~',paste(features,collapse = '+'))

      if (decimal_num==3){
        res = descrTable( as.formula(formula), data = mydata,include.miss = TRUE,
                          lab.missing=TRUE,show.p.overall=TRUE,
                          method = NA,show.all=TRUE,digits=3,extra.labels=c("Mean (SD)","Median [Q1-Q3]", "N (%)" ))####.符号代表包括其他的变量
      }

      if (decimal_num==2){
          res = descrTable( as.formula(formula), data = mydata,include.miss = TRUE,
                            lab.missing=TRUE,show.p.overall=TRUE,
                            method = NA,show.all=TRUE,digits=2,extra.labels=c("Mean (SD)","Median [Q1-Q3]", "N (%)" ))####.符号代表包括其他的变量
      }
      if (decimal_num==1){
          res = descrTable( as.formula(formula), data = mydata,include.miss = TRUE,
                            lab.missing=TRUE,show.p.overall=TRUE,
                            method = NA,show.all=TRUE,digits=1,extra.labels=c("Mean (SD)","Median [Q1-Q3]", "N (%)" ))####.符号代表包括其他的变量
      }
    }else{
      features <- c(Q_features, N_features)
      mydata <- df_input[,features]

      mydata[N_features] <- lapply(mydata[N_features], as.factor)

      formula <- paste0('~',paste(features,collapse = '+'))

      res = descrTable(as.formula(formula), data = mydata,method = NA,
                       show.all=TRUE,digits=1,extra.labels=c("Mean (SD)","Median [Q1-Q3]", "N (%)" ))####.符号代表包括其他的变量
    }

    res_scrip <- paste0(res_scrip,"[","基线表","]如基线表所示，compareGroups包在统计检验时，对于数值变量，分组变量是二分类的默认使用Student‘s t检验（t-test），分组变量是多分类的默认使用方差分析（ANOVA）。")
    res_scrip <-  paste0(res_scrip,"\n对于分类变量，无论分组变量是二分类还是多分类，默认都使用卡方检验（Chi-squared test），当列联表中存在期望频数 < 5 的格子数超过 20%，或任一期望频数 < 1 时，软件会自动采用Fisher精确检验（Fisher‘s exact test）。")

    conclusion <- paste0(res_scrip)

    tab_html=paste0("Table_comparegroup_",mytime,"_",rand,".xlsx")
    export2xls(res, file =paste0(savepath,tab_html))
    table_name<-c(table_name,'基线表')
    table_path<-c(table_path,tab_html)
    table_savepath<-c(table_savepath,tab_html)
  }


  descrip<-paste0(descrip,conclusion,head_descrip)
  results$table_name<-table_name
  results$table_path<-table_path
  results$table_savepath<-table_savepath
  results$descrip<-enc2utf8(descrip)
  return(results)
}





