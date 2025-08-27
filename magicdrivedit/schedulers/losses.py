import torch
from torch import nn


class AdaptivePseudoHuberLoss(nn.Module):
    def __init__(self, p=0.5, c=0.001, adaptive_weight=None):
        super(AdaptivePseudoHuberLoss, self).__init__()
        self.p = p
        self.c = c
        if adaptive_weight is not None:
            assert adaptive_weight in ("meanflow",)
        self.adaptive_weight = adaptive_weight

    def __call__(self, pred, target):
        error = pred - target.detach()
        l2 = error.pow(2)
        loss = l2

        if self.adaptive_weight == 'meanflow':
            w = 1 / self.pow(l2 + self.c, self.p)
            w = w.detach()
            loss *= w
 
        return loss


class LPIPSLoss(nn.Module):
    # import external
    import os.path as osp
    # EXTERNAL_TORCHHUB_DIR = osp.join(external.__path__[0], "torch", "hub")
    #torch.hub.set_dir(EXTERNAL_TORCHHUB_DIR)
    from piq import LPIPS

    def __init__(self):
        super(LPIPSLoss, self).__init__()
        lpips = LPIPS(replace_pooling=True, reduction="none")
        # LPIPS._weights_url = osp.join(EXTERNAL_TORCHHUB_DIR, 'lpips_weights.pt')
        self.lpips = lpips
        self.lpips_minibatch_size = 4

    def __call__(self, pred, target):
        B = pred.shape[0]
        dist_losses = []
        for bi in range(0, B, self.lpips_minibatch_size):
            dist_loss = self._dist_loss(pred[bi:bi + self.lpips_minibatch_size], target[bi:bi + self.lpips_minibatch_size])
            dist_losses.append(dist_loss)
        dist_loss = torch.cat(dist_losses, dim=0)
        return dist_loss
