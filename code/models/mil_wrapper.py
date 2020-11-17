__author__ = 'marvinler'

import torch
from torch import nn as nn


class MaxMinMIL(nn.Module):
    def __init__(self, classifier_model, alpha, beta=0., class_imbalance_weights=None, cuda=True):
        super().__init__()

        assert 1. >= alpha > 0., 'Meta-parameter alpha should be positive and lower or equal than 1.'
        assert 1. > beta, 'Meta-parameter beta should be lower than 1.'
        self.instance_model = classifier_model

        self.alpha = alpha
        self.beta = beta

        self.class_imbalance_weights = class_imbalance_weights

        self.loss_function = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_imbalance_weights)
        if cuda:
            self.loss_function.cuda()

        self.use_cuda = cuda

    def loss(self, predictions, computed_instances_labels, mask_instances_labels):
        """
        Computes instance-wise error signal with self.loss_function using instances predictions and computed
        proxy-labels, and use the input mask for averaging.
        :param predictions: tensor of instances predictions
        :param computed_instances_labels: tensor of computed proxy-labels, same shape
        :param mask_instances_labels: tensor of same shape than computed_instances_labels containing 1 if associated
        instance has an assigned proxy label, 0 otherwise
        :return: batch-averaged loss signal
        """
        instance_wise_loss = self.loss_function(predictions, computed_instances_labels)
        averaged_loss = (instance_wise_loss * mask_instances_labels).sum() / mask_instances_labels.sum()
        return averaged_loss

    def forward(self, instances, bag_label):
        assert instances.shape[0] == bag_label.shape[0] == 1, instances.shape
        instances = instances.squeeze(0)
        bag_label = bag_label.squeeze(0)
        n_instances = instances.size(0)

        current_device = instances.get_device()

        # Forwards each instance into optimized model
        instances_predictions = self.instance_model(instances)

        # Compute proxy-label based on bag label, alpha, beta, and predictions
        computed_instances_labels = torch.zeros(instances_predictions.shape, device=current_device).float()
        mask_instances_labels = torch.zeros(instances_predictions.shape, device=current_device).float()
        if bag_label == 0:  # if bag label is 0, then no pixel is positive (equivalent to ground-truth)
            computed_instances_labels[:] = 0.
            mask_instances_labels[:] = 1.
        else:  # otherwise, ensure alpha% of instances are positive, and beta% are negative, based on probabilities
            _, topk_idx = torch.topk(instances_predictions, k=int(self.alpha*n_instances), dim=0)
            computed_instances_labels[topk_idx] = 1.
            mask_instances_labels[topk_idx] = 1.
            if self.beta > 0.:
                _, bottomk_idx = torch.topk(instances_predictions, k=int(self.beta*n_instances), largest=False, dim=0)
                computed_instances_labels[bottomk_idx] = 0.
                mask_instances_labels[bottomk_idx] = 1.

        if self.use_cuda:
            computed_instances_labels = computed_instances_labels.cuda()
            mask_instances_labels = mask_instances_labels.cuda()

        # stop gradient flow in backprop
        computed_instances_labels = computed_instances_labels.detach()
        mask_instances_labels = mask_instances_labels.detach()

        return instances_predictions.unsqueeze(0), \
               computed_instances_labels.unsqueeze(0), mask_instances_labels.unsqueeze(0)
