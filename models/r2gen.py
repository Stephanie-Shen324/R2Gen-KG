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
        # att_feats torch.Size([16, 49, 2048])
        # node_feats torch.Size([16, 21, 2048])
        # fc_feats torch.Size([16, 2048])
        att_feats, node_feats, fc_feats = self.submodel(images[:,0], images[:,1])
        
        feed_mode = self.args.feed_mode
        # feed both CNN features & graph embedded features
        if feed_mode == 'both':
            input_feats = torch.cat((att_feats, node_feats), dim = 1) #torch.Size([16, 70, 2048])
        # feed only CNN features 
        elif feed_mode == 'cnn_only':
            input_feats = att_feats
        # feed only graph embedded features
        elif feed_mode == 'gcn_only':
            input_feats = node_feats


        if mode == 'train':
            output = self.encoder_decoder(fc_feats, input_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, input_feats, mode='sample')
        else:
            raise ValueError
        return output
    #edit
    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        #att_feats, fc_feats = self.visual_extractor(images)
        att_feats, node_feats, fc_feats = self.submodel(images)
        
        feed_mode = self.args.feed_mode
        # feed both CNN features & graph embedded features
        if feed_mode == 'both':
            input_feats = torch.cat((att_feats, node_feats), dim = 1) #torch.Size([16, 70, 2048])
        # feed only CNN features 
        elif feed_mode == 'cnn_only':
            input_feats = att_feats
        # feed only graph embedded features
        elif feed_mode == 'gcn_only':
            input_feats = node_feats
     
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, input_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, input_feats, mode='sample')
        else:
            raise ValueError
        return output

