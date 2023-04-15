import json
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple
from fastdup.fastdup_controller import FastdupController
from fastdup import Fastdup
from coco import COCO


class Oak(Fastdup):
    def __init__(self, data_dir: Union[str, Path], cache_folder: Union[str, Path], coco_annotations: Union[str, Path] = None):
        super().__init__(cache_folder, data_dir)
        self.data_dir = data_dir
        self._coco = COCO(coco_annotations)
    
    def run(self, print_summary: bool = True, overwrite: bool = False, **fastdup_kwargs):
        annotations_df = self._coco.as_df()
        super().run(annotations=annotations_df, overwrite=overwrite, **fastdup_kwargs)

    def tag_data(self, filters=None): # add filters class
        connected_components_df , _ = self.connected_components()
        outlier_df = self.outliers()
        stats_df = self.img_stats()
        duplicate_images = parse_duplicates(connected_components_df)
        outlier_images = outlier_df[outlier_df.distance < 0.68].filename_nearest.tolist()
        broken_images = self.invalid_instances()
        broken_images = broken_images['filename'].to_list()
        blurry_images = stats_df[stats_df['blur'] < 50]
        bright_images = stats_df[stats_df['mean'] > 220.5]
        dark_images = stats_df[stats_df['mean'] < 13]
        a = 1



def parse_duplicates(df, sort_by='count', min_count=2, ascending=False):
    # columns to aggregate
    agg_dict = {'filename': list, 'mean_distance': max, 'count': len}

    if 'label' in df.columns:
        agg_dict['label'] = list
    
    # filter by count
    df = df[df['count'] >= min_count]
    clusters_df = df.groupby('component_id').agg(agg_dict)
    clusters_df = clusters_df.sort_values(by=[sort_by], ascending=ascending)

    duplicates = []

    for cluster_file_list in clusters_df.filename:
        duplicates.extend(cluster_file_list[1:])

    return duplicates


if __name__ == '__main__':
    oak = Oak('./oak/small_coco/data', './oak/small_coco/fastdup', './oak/small_coco/coco.json')
    oak.run(overwrite=True, data_type='bbox')
    oak.tag_data()
    a = 1