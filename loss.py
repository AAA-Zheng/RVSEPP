import torch
import torch.nn as nn


class TripletSelHNLoss(nn.Module):

    def __init__(self, opt):
        super(TripletSelHNLoss, self).__init__()
        self.margin = opt.margin
        self.epsilon = opt.epsilon
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v, t):

        batch_size = v.size(0)
        if batch_size != self.batch_size:
            pos_mask = torch.eye(batch_size)
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        scores = get_sim(v, t)
        pos_scores = scores.diag().view(batch_size, 1)
        pos_scores_t = pos_scores.expand_as(scores)
        pos_scores_v = pos_scores.t().expand_as(scores)

        loss_t = (scores - pos_scores_t + self.margin).clamp(min=0)
        loss_v = (scores - pos_scores_v + self.margin).clamp(min=0)
        loss_t = loss_t * neg_mask
        loss_v = loss_v * neg_mask

        # Triplet
        avg_loss_t = loss_t.sum(1) / (batch_size - 1)
        avg_loss_v = loss_v.sum(0) / (batch_size - 1)

        # Triplet-HN
        max_loss_t = loss_t.max(1)[0]
        max_loss_v = loss_v.max(0)[0]

        # Calculate the difference in similarity between positive and hardest negative pairs
        scores_detach = scores.detach()
        pos_scores = scores_detach.diag()
        neg_scores = scores_detach * neg_mask - pos_mask
        hardest_neg_scores_t = neg_scores.max(1)[0]
        hardest_neg_scores_v = neg_scores.max(0)[0]
        diff_t = torch.abs(pos_scores - hardest_neg_scores_t)
        diff_v = torch.abs(pos_scores - hardest_neg_scores_v)

        # Selectively hard negative mining
        loss_t = torch.where(diff_t < self.epsilon, avg_loss_t, max_loss_t)
        loss_v = torch.where(diff_v < self.epsilon, avg_loss_v, max_loss_v)
        loss_t = loss_t.mean()
        loss_v = loss_v.mean()
        loss = (loss_t + loss_v) / 2

        return loss


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities
