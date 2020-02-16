import torch
import torch.nn as nn


class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()),
                             int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction='none')[:, 0, :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / \
                       (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


if __name__ == '__main__':
    import numpy as np
    torch.set_printoptions(precision=9)
    gt = np.load('/home/adam/workspace/github/xuannianz/carrot/db/gt.npy')
    gt = np.transpose(gt, (2, 0, 1))
    gt = np.expand_dims(gt, axis=0)
    mask = np.load('/home/adam/workspace/github/xuannianz/carrot/db/mask.npy')
    mask = np.expand_dims(mask, axis=0)
    pred = np.load('pred.npy')
    gt = torch.tensor(gt)
    mask = torch.tensor(mask)
    pred = torch.tensor(pred)
    print(gt.shape, mask.shape, pred.shape)
    bce_loss = BalanceCrossEntropyLoss()
    print(bce_loss.forward(pred, gt, mask))
