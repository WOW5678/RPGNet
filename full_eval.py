import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, \
    roc_auc_score, precision_score, recall_score
from keras.utils import to_categorical

thres = 0.5

def f1_score(preds, labels, average='micro'):
    '''Returns (precision, recall, F1 score) from a batch of predictions (thresholded probabilities)
       given a batch of labels (for macro-averaging across batches)'''
    #preds = (probs >= thres).astype(np.int32)
    # print('probs:',probs)
    # print('labels:',labels)
    # print('preds:',preds)
    #preds=probs
    # print(preds)
    # print(labels)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=average,
                                                                 warn_for=())
    return p, r, f

def auc_pr(probs, labels, average='micro'):
    '''Precision integrated over all thresholds (area under the precision-recall curve)'''
    if average == 'macro' or average is None:
        sums = labels.sum(0)
        nz_indices = np.logical_and(sums != labels.shape[0], sums != 0)
        probs = probs[:, nz_indices]
        labels = labels[:, nz_indices]
    return average_precision_score(labels, probs, average=average)


def auc_roc(probs, labels, average='micro'):
    '''Area under the ROC curve'''
    if average == 'macro' or average is None:
        sums = labels.sum(0)
        nz_indices = np.logical_and(sums != labels.shape[0], sums != 0)
        probs = probs[:, nz_indices]
        labels = labels[:, nz_indices]
    # print('labels:',labels)
    # print('probs:',probs)
    return roc_auc_score(labels, probs, average=average)


def precision_at_k(probs, labels, k, average='micro'):
    indices = np.argpartition(-probs, k-1, axis=1)[:, :k]
    preds = np.zeros(probs.shape, dtype=np.int)
    preds[np.arange(preds.shape[0])[:, np.newaxis], indices] = 1
    return precision_score(labels, preds, average=average)


def recall_at_k(probs, labels, k, average='micro'):
    indices = np.argpartition(-probs, k-1, axis=1)[:, :k]
    preds = np.zeros(probs.shape, dtype=np.int)
    preds[np.arange(preds.shape[0])[:, np.newaxis], indices] = 1
    return recall_score(labels, preds, average=average)


def full_evaluate(preds, golds, labelNum,labelMask):
    batchJaccard=[]
    oneHot_predicted_labels=np.zeros((len(preds),labelNum))
    for i in range(len(preds)):
        pred = list(set(preds[i]))
        label=np.nonzero(golds[i])[0]
        print('pred:',pred)
        print('label:',label)
        jaccard = jaccrad(pred, label)
        batchJaccard.append(jaccard)

        pred = np.sum(to_categorical(pred, num_classes=labelNum), axis=0)
        oneHot_predicted_labels[i]=pred
    jaccard=sum(batchJaccard)*1.0/len(batchJaccard)

    print('oneHot_predicted_labels:',oneHot_predicted_labels.shape)
    print('golds:',golds.shape)

    #针对每一个标签计算
    try:
        pred=oneHot_predicted_labels[:,labelMask]
        gold=golds[:,labelMask]
        print('pred.shape,gold.shape:',pred.shape,gold.shape)
        micro_p, micro_r, micro_f1 = f1_score(pred, gold,  average='micro')
        macro_p,macro_r,macro_f1= f1_score(pred, gold,  average='macro')

        micro_auc_roc= auc_roc(pred, gold, average='micro')
        macro_auc_roc= auc_roc(pred, gold, average='macro')

    except ValueError:
        micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc=0,0,0,0,0,0,0,0
    # micro_ps.append(micro_p)
    # macro_ps.append(macro_p)
    # micro_rs.append(micro_r)
    # macro_rs.append(macro_r)
    # micro_f1s.append(micro_f1)
    # macro_f1s.append(macro_f1)
    # micro_auc_rocs.append(micro_auc_roc)
    # macro_auc_rocs.append(macro_auc_roc)

    # avg_micro_p=sum(micro_ps)/len(micro_ps)
    # avg_macro_p = sum(macro_ps) / len(macro_ps)
    # avg_micro_r = sum(micro_rs) / len(micro_rs)
    # avg_macro_r = sum(macro_rs) / len(macro_rs)
    # avg_micro_f1 = sum(micro_f1s) / len(micro_f1s)
    # avg_macro_f1 = sum(macro_f1s) / len(macro_f1s)
    # avg_micro_auc_roc = sum(micro_auc_rocs) / len(micro_auc_rocs)
    # avg_macro_auc_roc = sum(macro_auc_rocs) / len(macro_auc_rocs)

    return jaccard, micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc

def jaccrad(predList, referList):  # terms_reference为源句子，terms_model为候选句子
    grams_reference = set(predList)  # 去重；如果不需要就改为list
    grams_model = set(referList)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = temp*1.0 / fenmu  # 交集
    return jaccard_coefficient

