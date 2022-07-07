#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from curses import tparm
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class MultitaskConfusionMatrix(object):
    """MultitaskConfusionMatrix
    
    Calculates a confusion matrix for multi-task.

    NOTE:
        The shape of confusion matrix:
            y-axis (vertical): true
            x-axis (horizontal): predicted
        I've referenced the following web-site: https://takake-blog.com/python-confusion-matrix/
    """

    def __init__(self, num_attributes, attr_name_list=None, verbose=False):
        self.num_attributes = num_attributes

        if attr_name_list is not None:
            assert len(attr_name_list) == self.num_attributes, "the number of attribute names (%d) is different with num_attributes (%d)" % (len(attr_name_list), num_attributes)
            self.attr_name_list = attr_name_list
        else:
            self.attr_name_list = list(range(self.num_attributes))

        self.true_pos  = np.zeros(self.num_attributes, dtype=np.int64)
        self.true_neg  = np.zeros(self.num_attributes, dtype=np.int64)
        self.false_pos = np.zeros(self.num_attributes, dtype=np.int64)
        self.false_neg = np.zeros(self.num_attributes, dtype=np.int64)

        if verbose:
            print('Multitask Confusion Matrix')
            print('    Number of attributes:', self.num_attributes)

    def update(self, label_trues, label_preds, use_cuda=False):
        """Compute and add TP, TN, FP, FN

        Parameters
        ----------
        label_trues : np.array or torch.Tensor (expected shape: [batch, num_attributes])
            true labels
        label_preds : np.array or torch.Tensor (expected shape: [batch, num_attributes])
            predicted labels
        """

        ### convert torch.Tensor --> numpy.array
        if isinstance(label_trues, torch.Tensor):
            if use_cuda:
                label_trues = label_trues.cpu()
            label_trues = label_trues.data.numpy()
        if isinstance(label_preds, torch.Tensor):
            if use_cuda:
                label_preds = label_preds.cpu()
            label_preds = label_preds.data.numpy()

        ### binarize label_preds
        label_preds[label_preds >= 0.5] = 1.0
        label_preds[label_preds < 0.5] = 0.0

        ### compute TP, TN, FP, FN
        self.true_pos  += np.sum(np.logical_and(label_preds == 1, label_trues == 1), axis=0)
        self.true_neg  += np.sum(np.logical_and(label_preds == 0, label_trues == 0), axis=0)
        self.false_pos += np.sum(np.logical_and(label_preds == 1, label_trues == 0), axis=0)
        self.false_neg += np.sum(np.logical_and(label_preds == 0, label_trues == 1), axis=0)

    def get_average_accuracy(self):
        score_list = []
        for i in range(self.num_attributes):
            score_tmp = self.get_single_attr_score(i)
            score_list.append(score_tmp['accuracy'])
        return sum(score_list) / self.num_attributes

    def get_single_attr_score(self, index):
        tp = self.true_pos[index]
        tn = self.true_neg[index]
        fp = self.false_pos[index]
        fn = self.false_neg[index]

        accuracy  = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f_measure = (2 * tp) / (2 * tp + fn + fp)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_measure}

    def get_attr_score(self):
        """return accuracies of all attributes

        Returns
        -------
        dict
            returned result has a following shape:
            {
                'attribute name 1: {'accuracy': **, 'precision': **, 'recall': **, 'f_measure': **},
                'attribute name 2: {'accuracy': **, 'precision': **, 'recall': **, 'f_measure': **},
                ...
            }
        """
        _acc_list = {}
        for i in range(self.num_attributes):
            _acc_list[self.attr_name_list[i]] = self.get_single_attr_score(i)
        return _acc_list

    def get_single_conf_mat(self, index):
        return {'TP': self.true_pos[index], 'TN': self.true_neg[index], 'FP': self.false_pos[index], 'FN': self.false_neg[index]}

    def get_conf_mat(self):
        """return confusion matrix of all attributes

        Returns
        -------
        dict
            returned result has a following shape:
            {
                'attribute name 1: {'TP': **, 'TN': **, 'FP': **, 'FN': **},
                'attribute name 2: {'TP': **, 'TN': **, 'FP': **, 'FN': **},
                ...
            }
        """
        _c_mat_results = {}
        for i in range(self.num_attributes):
            _c_mat_results[self.attr_name_list[i]] = self.get_single_conf_mat(i)
        return _c_mat_results

    def clear(self):
        """Clear TP, TN, FP, FN arrays"""
        self.true_pos  = np.zeros(self.num_attributes, dtype=np.int64)
        self.true_neg  = np.zeros(self.num_attributes, dtype=np.int64)
        self.false_pos = np.zeros(self.num_attributes, dtype=np.int64)
        self.false_neg = np.zeros(self.num_attributes, dtype=np.int64)
