import os
from pathlib import Path
from typing import Union, Tuple
from fastdup import Fastdup

from coco import COCO


class Oak(Fastdup):
    def __init__(self, data_dir: Union[str, Path], cache_folder: Union[str, Path], coco_annotations: Union[None, str, Path] = None):
        super().__init__(cache_folder, data_dir)
        self.data_dir = data_dir
        self._coco = COCO(coco_annotations)
    
    def run(self, print_summary: bool = True, overwrite: bool = False, **fastdup_kwargs):
        # annotations_df = self._coco.as_df()
        annotations_df = None
        super().run(annotations=annotations_df, overwrite=overwrite, print_summary=print_summary, **fastdup_kwargs)

    def tag_data(self, filters=None): # add filters class
        connected_components_df , _ = self.connected_components()
        outlier_df = self.outliers()
        stats_df = self.img_stats()
        invalid_df = self.invalid_instances()
        duplicate_images = parse_duplicates(connected_components_df)
        outlier_images = outlier_df[outlier_df.distance < 0.68].filename_nearest.tolist()
        broken_images = invalid_df['filename'].to_list()
        blurry_images = stats_df[stats_df['blur'] < 50]['filename'].to_list()
        bright_images = stats_df[stats_df['mean'] > 220.5]['filename'].to_list()
        dark_images = stats_df[stats_df['mean'] < 13]['filename'].to_list()

        tag_groups = {
            'duplicate': set(duplicate_images),
            'outlier': set(outlier_images),
            'broken': set(broken_images),
            'blurry': set(blurry_images),
            'bright': set(bright_images),
            'dark': set(dark_images),
        }

        if self._coco.dataset is not None:
            for tag_name, filenames in tag_groups.items():
                for filename in filenames:
                    filename = os.path.basename(filename)
                    if 'tag' not in self._coco.imgs_filenames[filename]:
                        self._coco.imgs_filenames[filename]['tag'] = [tag_name]
                    else:
                        self._coco.imgs_filenames[filename]['tag'].append(tag_name)



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
    oak = Oak('tiny-coco/small_coco/train_2017_small', 'tiny-coco/small_coco/train_2017_smallastdup', 'tiny-coco/small_coco/coco.json')
    oak.run(overwrite=True)
    oak.tag_data()
    a = 1