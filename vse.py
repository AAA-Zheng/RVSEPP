"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from encoders import ImageEncoder, TextEncoder
from loss import TripletLoss, TripletHNLoss, TripletSHNLoss, TripletSelHNLoss

import logging

logger = logging.getLogger(__name__)


class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        self.img_enc = ImageEncoder(opt)
        self.txt_enc = TextEncoder(opt)

        print(self.img_enc)
        print(self.txt_enc)

        self.img_enc.cuda()
        self.txt_enc.cuda()

        cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = TripletSelHNLoss(opt=opt)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        if self.opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)
        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, image_lengths, captions, caption_lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = images.cuda()
        captions = captions.cuda()
        image_lengths = image_lengths.cuda()
        img_emb = self.img_enc(images, image_lengths)

        caption_lengths = torch.Tensor(caption_lengths).cuda()
        cap_emb = self.txt_enc(captions, caption_lengths)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, diff = self.criterion(img_emb, cap_emb)
        self.logger.update('loss', loss.item(), img_emb.size(0))

        return loss

    def train_emb(self, images, image_lengths, captions, caption_lengths):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, image_lengths, captions, caption_lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and update
        loss.backward()

        clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

