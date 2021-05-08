import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        #edit
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images) #patch_feats torch.Size([16, 1024, 7, 7])
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1)) #should be torch.Size([16, 1024])
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # print('patch_feats',patch_feats.shape) 
        # print('avg_feats',avg_feats.shape)
        # patch_feats torch.Size([16, 49, 1024])
        # avg_feats torch.Size([16, 1024])
        return patch_feats, avg_feats
