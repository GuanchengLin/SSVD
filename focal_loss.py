from torch import nn
import torch
from torch.nn import functional as F


class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes 
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  
            print(" --- Focal_loss alpha = {}--- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  

        self.gamma = gamma

    def forward(self, preds, labels, weight):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft) 
        loss = torch.mul(alpha, loss.t())

        return loss
