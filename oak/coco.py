# inspired by pycocotools
import json
import os
from collections import defaultdict

import pandas as pd


class COCO():

    def __init__(self, annotation_path):
        self.dataset = None
        if annotation_path is not None:
            dataset = json.load(open(annotation_path))
            assert type(dataset) == dict, f'annotation file format {type(dataset)} not supported'
            self.dataset = dataset
        self.create_index()

    def create_index(self):
        anns = {}
        cats = {}
        imgs = {}
        imgs_filenames = {}
        img_to_anns = defaultdict(list)
        cat_to_imgs = defaultdict(list)
        if self.dataset is not None:
            if 'annotations' in self.dataset:
                for ann in self.dataset['annotations']:
                    img_to_anns[ann['image_id']].append(ann)
                    anns[ann['id']] = ann

            if 'images' in self.dataset:
                for img in self.dataset['images']:
                    imgs[img['id']] = img
                    imgs_filenames[img['file_name']] = img

            if 'categories' in self.dataset:
                for cat in self.dataset['categories']:
                    cats[cat['id']] = cat

            if 'annotations' in self.dataset and 'categories' in self.dataset:
                for ann in self.dataset['annotations']:
                    cat_to_imgs[ann['category_id']].append(ann['image_id'])

        self.anns = anns
        self.img_to_anns = img_to_anns
        self.cat_to_imgs = cat_to_imgs
        self.imgs = imgs
        self.imgs_filenames = imgs_filenames
        self.cats = cats

    def as_df(self, data_dir):
        data = []
        for img_id, anns in self.img_to_anns.items():
            img = self.imgs[img_id]
            img_filename = os.path.basename(img['file_name'])
            width = img['width']
            height = img['height']

            for ann in anns:
                bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                cat_id = ann['category_id']
                data.append({
                    'filename': data_dir + img_filename,
                    'img_filename': img_filename,
                    'img_h': height,
                    'img_w': width,
                    'col_x': int(bbox_x),
                    'row_y': int(bbox_y),
                    'width': int(bbox_w),
                    'height': int(bbox_h),
                    'label': self.cats[cat_id]['name'],
                    'ann_id': ann['id']
                })

        if len(data):
            df = pd.DataFrame(data)
            return df
