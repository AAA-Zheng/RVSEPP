
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.gpo import GPO
from lib.mlp import MLP

import logging

logger = logging.getLogger(__name__)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class ImageEncoder(nn.Module):

    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.model = opt.model
        self.embed_size = opt.embed_size
        self.fc = nn.Linear(opt.img_dim, opt.embed_size)
        self.mlp = MLP(opt.embed_size, opt.embed_size // 2, opt.embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""

        features = self.fc(images)
        features = self.mlp(features) + features
        features, pool_weights = self.gpool(features, image_lengths)
        features = l2norm(features, dim=-1)

        return features


class TextEncoder(nn.Module):

    def __init__(self, opt):
        super(TextEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, batch_first=True, bidirectional=True)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features
