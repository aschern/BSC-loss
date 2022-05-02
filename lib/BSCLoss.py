import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict
from sentence_transformers.SentenceTransformer import SentenceTransformer
from audtorch.metrics.functional import pearsonr


class BSCLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, tau=None, norm_dim=1):
        super(BSCLoss, self).__init__()
        self.model = model
        self.tau = tau
        self.norm_dim = norm_dim

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if self.tau is None:
            alpha = self.model._first_module().alpha
            print('tau: {}'.format(1 / self.model._first_module().alpha), flush=True)
        else:
            alpha = 1 / self.tau
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a = reps[0]
        rep_b = torch.cat(reps[1:])
        
        if self.norm_dim == 1: #or self.norm_dim == 0
            rep_an = rep_a / torch.norm(rep_a, p=2, dim=self.norm_dim, keepdim=True)
            rep_bn = rep_b / torch.norm(rep_b, p=2, dim=self.norm_dim, keepdim=True)
        elif self.norm_dim == 0:
            rep_an = rep_a / (rep_a.max(axis=0)[0] - rep_a.min(axis=0)[0])
            rep_bn = rep_b / (rep_b.max(axis=0)[0] - rep_b.min(axis=0)[0])
        else:
            rep_an = rep_a
            rep_bn = rep_b

        if labels is not None:
            dot_pr_mat = torch.mm(rep_an, rep_bn.T) * alpha
            
            loss_fct = -torch.mean(torch.diag(F.log_softmax(dot_pr_mat, dim=1)) * labels.float()) - torch.mean(torch.diag(F.log_softmax(dot_pr_mat, dim=0)) * labels.float())
            loss = loss_fct / 2
            return loss
        else:
            output = torch.cosine_similarity(rep_an, rep_bn)
            return reps, output
        
        
        
class ComboBSCLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, tau=None, norm_dim=1, bin_thr=0.5, mu=0.5):
        super(ComboBSCLoss, self).__init__()
        self.model = model
        self.tau = tau
        self.norm_dim = norm_dim
        self.bin_thr = 0.5
        self.mu = mu

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if self.tau is None:
            alpha = self.model._first_module().alpha
            print('tau: {}'.format(1 / self.model._first_module().alpha), flush=True)
        else:
            alpha = 1 / self.tau
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        
        if self.norm_dim == 1: #or self.norm_dim == 0
            rep_an = rep_a / torch.norm(rep_a, p=2, dim=self.norm_dim, keepdim=True)
            rep_bn = rep_b / torch.norm(rep_b, p=2, dim=self.norm_dim, keepdim=True)
        elif self.norm_dim == 0:
            rep_an = rep_a / (rep_a.max(axis=0)[0] - rep_a.min(axis=0)[0])
            rep_bn = rep_b / (rep_b.max(axis=0)[0] - rep_b.min(axis=0)[0])
        else:
            rep_an = rep_a
            rep_bn = rep_b

        if labels is not None:
            dot_pr_mat = torch.mm(rep_an, rep_bn.T) * alpha
            
            bin_labels = (labels >= self.bin_thr).float()
            loss_fps = (-torch.mean(torch.diag(F.log_softmax(dot_pr_mat, dim=1)) * bin_labels) - torch.mean(torch.diag(F.log_softmax(dot_pr_mat, dim=0)) * bin_labels)) / 2

            loss_fct = F.mse_loss(torch.cosine_similarity(rep_a, rep_b), labels.view(-1)) # -pearsonr(torch.cosine_similarity(rep_a, rep_b), labels)
            
            loss = self.mu * loss_fps + (1 - self.mu) * loss_fct
            return loss
        else:
            output = torch.cosine_similarity(rep_a, rep_b)
            return reps, output
