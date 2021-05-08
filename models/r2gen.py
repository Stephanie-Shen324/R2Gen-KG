import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.mlclassifier import GCNClassifier

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer, submodel):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        # self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

        #edit
        self.submodel = submodel

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):

        # att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        # att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        # print('att_feats_0',att_feats_0.shape)
        # print('fc_feats_0',fc_feats_0.shape)
        #att_feats0 ([16, 49, 1024])
        # fc_feats0 torch.Size([16, 1024])
        att_feats, node_feats, fc_feats = self.submodel(images[:,0], images[:,1])

        # feed both
        att_feats = torch.cat((att_feats, node_feats), dim = 1) #torch.Size([16, 70, 2048])

        # raise Exception('lol')

        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # att_feats ([16, 98, 1024])
        # fc_feats torch.Size([16, 2048])

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
     
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

