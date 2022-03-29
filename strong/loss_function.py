from numpy import dtype
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    '''
    Contrastive Loss
    Args:
        margin: non-neg value, the smaller the stricter the loss will be, default: 0.2        
        
    '''
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pred_score, gt_score):
        if pred_score.dim() > 2:
            pred_score = pred_score.mean(dim=1).squeeze(1)
        # pred_score, gt_score: tensor, [batch_size]  
        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.margin) 
        loss = loss.mean().div(2)
        return loss


class ClippedMSELoss(nn.Module):
    """
    clipped MSE loss for listener-dependent model
    """
    def __init__(self, criterion,tau,mode='frame'):
        super(ClippedMSELoss, self).__init__()
        self.tau = torch.tensor(tau,dtype=torch.float)

        self.criterion = criterion
        self.mode = mode


    def forward_criterion(self, y_hat, label):

        y_hat = y_hat.squeeze(-1)
        loss = self.criterion(y_hat, label)
        threshold = torch.abs(y_hat - label) > self.tau
        loss = torch.mean(threshold * loss)
        return loss

    def forward(self, pred_score, gt_score):
        """
        Args:
            pred_mean, pred_score: [batch, time, 1/5]
        """
        # repeat for frame level loss
        time = pred_score.shape[1]
        if self.mode == 'utt':
            pred_score = pred_score.mean(dim=1)
        else:
            gt_score = gt_score.unsqueeze(1).repeat(1, time)
        main_loss = self.forward_criterion(pred_score, gt_score)
        return main_loss # lamb 1.0  

class CombineLosses(nn.Module):
    '''
    Combine losses
    Args:
        loss_weights: a list of weights for each loss
    '''
    def __init__(self, loss_weights:list, loss_instances:list):
        super(CombineLosses, self).__init__()
        self.loss_weights = loss_weights
        self.loss_instances = nn.ModuleList(loss_instances)
    def forward(self, pred_score, gt_score):
        loss = torch.tensor(0,dtype=torch.float).to(pred_score.device)
        for loss_weight, loss_instance in zip(self.loss_weights, self.loss_instances):
            loss += loss_weight * loss_instance(pred_score,gt_score)
        return loss
