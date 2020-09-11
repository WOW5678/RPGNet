# -*- coding: utf-8 -*-
"""
 @Time    : 2019/4/18 0018 上午 11:08
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 将我们的方法与gamenet方法的结果 针对每个指标 做显著性试验
"""
import scipy.stats  as stats
import scipy.optimize as opt
import csv

def load_data_my():
    with open('data/myModel.csv','r',encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        jaccard = [float(row[0]) for row in data]
        micro_precision = [float(row[1]) for row in data]
        macro_precision = [float(row[2]) for row in data]
        micro_recall = [float(row[3]) for row in data]
        macro_recall = [float(row[4]) for row in data]

        micro_f = [float(row[5]) for row in data]
        macro_f = [float(row[6]) for row in data]
        micro_auc = [float(row[7]) for row in data]
        macro_auc = [float(row[8]) for row in data]
        return jaccard, micro_precision, macro_precision, micro_recall, macro_recall, micro_f, macro_f, micro_auc, macro_auc


def load_data_gamenet():
    with open('data/msatt-kg.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        data=[row for row in reader]
        jaccard=[float(row[0]) for row in data]
        micro_precision=[float(row[1]) for row in data]
        macro_precision = [float(row[2]) for row in data]
        micro_recall=[float(row[3]) for row in data]
        macro_recall = [float(row[4]) for row in data]

        micro_f = [float(row[5]) for row in data]
        macro_f = [float(row[6]) for row in data]
        micro_auc = [float(row[7]) for row in data]
        macro_auc = [float(row[8]) for row in data]
        return jaccard,micro_precision,macro_precision,micro_recall,macro_recall,micro_f,macro_f,micro_auc,macro_auc

jaccard,micro_precision,macro_precision,micro_recall,macro_recall,micro_f,macro_f,micro_auc,macro_auc=load_data_my()
jaccard_,micro_precision_,macro_precision_,micro_recall_,macro_recall_,micro_f_,macro_f_,micro_auc_,macro_auc_=load_data_gamenet()
#对jaccard分析差异
jaccard_stat_val,jaccard_p_val=stats.ttest_ind(jaccard,jaccard_,equal_var=False)
print('Jaccard: T-statistic D=%.5f,p-value=%.5f'%(jaccard_stat_val,jaccard_p_val))

microP_stat_val,microP_p_val=stats.ttest_ind(micro_precision,micro_precision_,equal_var=False)
print('precision: T-statistic D=%.5f,p-value=%.5f'%(microP_stat_val,microP_p_val))

macroP_stat_val,macroP_p_val=stats.ttest_ind(macro_precision,macro_precision_,equal_var=False)
print('macro-precision: T-statistic D=%.5f,p-value=%.5f'%(macroP_stat_val,macroP_p_val))


microR_stat_val,microR_p_val=stats.ttest_ind(micro_recall,micro_recall_,equal_var=False)
print('micro-recall: T-statistic D=%.5f,p-value=%.5f'%(microR_stat_val,microR_p_val))

macroR_stat_val,macroR_val=stats.ttest_ind(macro_recall,macro_recall_,equal_var=False)
print('macro-recall: T-statistic D=%.5f,p-value=%.5f'%(macroR_stat_val,macroR_val))

microF_stat_val,microF_val=stats.ttest_ind(micro_f,micro_f_,equal_var=False)
print('micro-F: T-statistic D=%.5f,p-value=%.5f'%(microF_stat_val,microF_val))

macroF_stat_val,macroF_val=stats.ttest_ind(macro_f,macro_f_,equal_var=False)
print('macro-F: T-statistic D=%.5f,p-value=%.5f'%(macroF_stat_val,macroF_val))


microAUC_stat_val,microAUC_val=stats.ttest_ind(micro_auc,micro_auc_,equal_var=False)
print('micro-AUC: T-statistic D=%.5f,p-value=%.5f'%(microAUC_stat_val,microAUC_val))

macroAUC_stat_val,macroAUC_val=stats.ttest_ind(macro_auc,macro_auc_,equal_var=False)
print('micro-AUC: T-statistic D=%.5f,p-value=%.5f'%(macroAUC_stat_val,macroAUC_val))
