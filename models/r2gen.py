import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.dwe_encoder_decoder import DWEEncoderDecoder
from modules.mlclassifier import GCNClassifier


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, submodel):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        # self.visual_extractor = VisualExtractor(args)
        self.feed_mode = args.feed_mode
        self.encoder_mode = args.encoder_mode
        # TODO
        if self.encoder_mode == 'dualwayencoder':
            assert self.feed_mode == 'both'  # "DualWayEncoder only accept feed_mode as both"
            self.encoder_decoder = DWEEncoderDecoder(args, tokenizer)
        elif self.encoder_mode == 'xdualwayencoder':
            assert self.feed_mode == 'both'
            self.encoder_decoder = DWEEncoderDecoder(args, tokenizer, lowrank = True)
        else:
            self.encoder_decoder = EncoderDecoder(args, tokenizer)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

        self.submodel = submodel

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        # att_feats torch.Size([16, 49, 2048])
        # node_feats torch.Size([16, 21, 2048])
        # fc_feats torch.Size([16, 2048])
        att_feats, node_feats, fc_feats = self.submodel(images[:, 0], images[:, 1])

        input_feats = self.feed_mode_controller(att_feats, node_feats)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, input_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, _, attention_scores= self.encoder_decoder(fc_feats, input_feats, mode='sample')
            return output,  attention_scores
        else:
            raise ValueError

    # edit
    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        if self.args.dataset_name == 'mimic_cxr_2images':
            att_feats, node_feats, fc_feats = self.submodel(images[:, 0], images[:, 1])
        else:
            # if only one image is inputted.
            att_feats, node_feats, fc_feats = self.submodel(images)

        input_feats = self.feed_mode_controller(att_feats, node_feats)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, input_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, _, attention_scores= self.encoder_decoder(fc_feats, input_feats, mode='sample')
            return output, attention_scores
        else:
            raise ValueError

    def feed_mode_controller(self, att_feats, node_feats):
        assert self.feed_mode != None

        if self.feed_mode == 'both':
            if self.encoder_mode in ['dualwayencoder', 'xdualwayencoder']:
                input_feats = [att_feats, node_feats]
            else:
                input_feats = torch.cat((att_feats, node_feats), dim=1)  # torch.Size([16, 70, 2048])
        # feed only CNN features
        elif self.feed_mode == 'cnn_only':
            input_feats = att_feats
        # feed only graph embedded features
        elif self.feed_mode == 'gcn_only':
            input_feats = node_feats
        return input_feats

