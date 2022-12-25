import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, average_precision_score
from prettytable import PrettyTable
from pathlib import Path
from torchmetrics import AveragePrecision
import torch

def indices_to_one_hot(data, onehot_classes):
    """convert an iterable of indices to onehot encoded labels

    Args:
        data (list of indices or single index): [description]
        onehot_classes (int): [number of classes]
    """
    targets = np.array(data).reshape(-1)
    return np.eye(onehot_classes, dtype=int)[targets][0]   


def get_metrics(y_true, y_pred, classes):
    performance ={'overall':{}, 'class':{}}
    metrics_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics_micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
    metrics_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred) * 100
    total = len(y_true)
    correct = accuracy_score(y_true, y_pred, normalize=False)
    wrong = total - correct
    eer = wrong / total* 100
    performance['overall']['accuracy'] = '{0:.2f}%[{1}/{2}]'.format(accuracy, correct, total)
    performance['overall']['EER'] = '{0:.2f}%[{1}/{2}]'.format(eer, wrong, total)
    performance['overall']['precision_weighted'] = '{:.2f}'.format(metrics_weighted[0]* 100)
    performance['overall']['recall_weighted'] = '{:.2f}'.format(metrics_weighted[1]* 100)
    performance['overall']['f1_weighted'] = '{:.2f}'.format(metrics_weighted[2]* 100)
    performance['overall']['precision_micro'] = '{:.2f}'.format(metrics_micro[0]* 100)
    performance['overall']['recall_micro'] = '{:.2f}'.format(metrics_micro[1]* 100)
    performance['overall']['f1_micro'] = '{:.2f}'.format(metrics_micro[2]* 100)
    performance['overall']['precision_macro'] = '{:.2f}'.format(metrics_macro[0]* 100)
    performance['overall']['recall_macro'] = '{:.2f}'.format(metrics_macro[1]* 100)
    performance['overall']['f1_macro'] = '{:.2f}'.format(metrics_macro[2]* 100)
    performance['overall']['num_samples'] = int(np.float64(len(y_true)))

    metrics = precision_recall_fscore_support(y_true, y_pred, average=None, 
                                              zero_division=0, labels=[i for i in range(len(classes))])
    
    for i in range(len(classes)):
        performance['class'][classes[i]] = {'precision':'{:.2f}'.format(metrics[0][i]* 100), 'recall':'{:.2f}'.format(metrics[1][i]* 100), 
                                                'f1':'{:.2f}'.format(metrics[2][i]* 100), 'num_samples': int(np.float64(metrics[3][i]))}
    
    return performance


def plotConfussionMatrix(y_true, y_pred, classes, labels=None, sample_weight=None, normalize=None):
    confusion_arr = confusion_matrix(y_true, y_pred, labels=labels, 
                                        sample_weight=sample_weight, normalize=normalize)
    table = PrettyTable(['true/predicted']+ classes)
    for class_, row in zip(classes, confusion_arr):
        table.add_row([class_]+ [item for item in row])

    header = 'Confusion Matrix, {} normalization'.format('with' if normalize else 'without')
    return table.get_string(title=header)


def create_table(performance_metric, dataset):
    keys = ['accuracy', 'EER', 'precision', 'recall', 'f1', 'num_samples']
    table = PrettyTable(['Class'] + keys)
    
    table.add_row([dataset]+ [performance_metric['overall'].get(key, '-') for key in keys[:2]] + 
                  ['-' for key in keys[2:-1]]+ [performance_metric['overall']['num_samples']])

    table.add_row(['weighted']+ ['-' for key in keys[:2]] + 
                  [performance_metric['overall'].get(f'{key}_weighted', '-') for key in keys[2:]])
    table.add_row(['micro']+ ['-' for key in keys[:2]] + 
                  [performance_metric['overall'].get(f'{key}_micro', '-') for key in keys[2:]])
    table.add_row(['macro']+ ['-' for key in keys[:2]] +
                  [performance_metric['overall'].get(f'{key}_macro', '-') for key in keys[2:]])

    for class_ in performance_metric['class'].keys():
        table.add_row([class_]+ [performance_metric['class'][class_].get(key, '-') for key in keys])
    return table.get_string()

    
def read_output_file(file: Path, classes: list):
    labels, preds = [], []
    with open(file, 'r') as fd:
        for line in fd:
            label, pred = line.strip().split()
            labels.append(classes.index(label))
            preds.append(classes.index(pred))
    return labels, preds
