import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import time
import copy

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.args = args

        self.examples = self.ann[self.split]
        if args.dataset_name == 'mimic_cxr_2images':
            self.examples = self.convert_to_multi_images(self.examples)
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def convert_to_multi_images(self, dataset, print_num=True):
        t = time.time()
        n = 0
        if print_num:
            print('{} set: Converting to multiple image reports ... '.format(self.split), end='', flush=True)
        mergedDataset = []
        total = len(dataset)

        buffer = None
        for i in range(total):
            document = dataset[i]
            id = document['id']
            image_path = document['image_path'][0]
            # report = document['report']
            # split = document['split']
            study_id = document['study_id']
            # subject_id = document['subject_id']

            if study_id == buffer:
                mergedDataset[-1]['image_path'].append(image_path)
                mergedDataset[-1]['id'].append(id)
            else:
                newDocument = copy.deepcopy(document)
                newDocument['id'] = [newDocument['id']]
                mergedDataset.append(newDocument)
                n += 1
            buffer = study_id
        if print_num:
            print('done %d->%d (%.2fs)' % (total, n, time.time() - t), flush=True)
        return mergedDataset


    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
       
        image = torch.stack((image_1, image_2) , 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
    
class MimiccxrMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id'] # the id may varies
        image_path = example['image_path']
        folder_path = '/'.join(image_path[0].split('/')[:3])
        all_images_name = os.listdir(os.path.join(self.image_dir,folder_path))
        # print(len(all_images_name))
        if len(all_images_name) == 1:
          # print('only 1 image')
          image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
          image_2 = image_1 #duplicate the first
          # print(image_1 == image_2)

        else: # more than 1 images:
          # print('2 images')
          image_1 = Image.open(os.path.join(self.image_dir, folder_path, all_images_name[0])).convert('RGB')
          image_2 = Image.open(os.path.join(self.image_dir, folder_path, all_images_name[1])).convert('RGB')
          # print(image_1 == image_2)
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        image = torch.stack((image_1, image_2) , 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
