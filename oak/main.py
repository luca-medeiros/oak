import os
import json
from pathlib import Path
from typing import Union

from fastdup import Fastdup

from oak.coco import COCO


class Oak(Fastdup):

    def __init__(self,
                 data_dir: Union[str, Path],
                 cache_folder: Union[str, Path]):
                #  ):
        super().__init__(cache_folder, data_dir)
        self.data_dir = data_dir
        self.cache_folder = cache_folder
        
    def run(self, print_summary: bool = True, overwrite: bool = False, **fastdup_kwargs):
        super().run(overwrite=overwrite, print_summary=print_summary, **fastdup_kwargs)

    def run_coco(self, coco_annotations: Union[None, str, Path], print_summary: bool = True, overwrite: bool = True, **fastdup_kwargs):
        self._coco = COCO(coco_annotations)
        annotations_df = self._coco.as_df(self.data_dir)
        super().run(annotations=annotations_df, overwrite=overwrite, print_summary=print_summary, **fastdup_kwargs)

    def tag_data(self, filters=None):  # add filters class
        cols = 'filename'
        HAS_COCO = False
        if hasattr(self, '_coco') and self._coco.dataset is not None:
            HAS_COCO = True
            cols = ['filename', 'ann_id']
        connected_components_df, _ = self.connected_components()
        outlier_df = self.outliers()
        outlier_df = outlier_df.rename(columns={"filename_outlier": "filename", "ann_id_outlier": "ann_id"})
        stats_df = self.img_stats()
        invalid_df = self.invalid_instances()
        # duplicate_images = parse_duplicates(connected_components_df, cols)
        duplicate_images = connected_components_df[connected_components_df['count'] >= 2][cols]
        outlier_images = outlier_df[outlier_df.distance < 0.68][cols]
        broken_images = invalid_df[cols]
        blurry_images = stats_df[stats_df['blur'] < 50][cols]
        bright_images = stats_df[stats_df['mean'] > 220.5][cols]
        dark_images = stats_df[stats_df['mean'] < 13][cols]

        tag_groups = {
            'duplicate': duplicate_images,
            'outlier': outlier_images,
            'broken': broken_images,
            'blurry': blurry_images,
            'bright': bright_images,
            'dark': dark_images,
        }

        if HAS_COCO:
            for tag_name, df in tag_groups.items():
                for ann_id in df.ann_id:
                    if 'tag' not in self._coco.anns[ann_id]:
                        self._coco.anns[ann_id]['tag'] = [tag_name]
                    else:
                        self._coco.anns[ann_id]['tag'].append(tag_name)
        return tag_groups
    
    def visualize_coco(self):
        import fiftyone as fo
        json.dump(self._coco.dataset, open(f'{self.cache_folder}/coco.json', 'w'))
        dataset = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset,
                                      data_path=self.data_dir,
                                      labels_path=f'{self.cache_folder}/coco.json')
        fo.launch_app(dataset)


if __name__ == '__main__':
    oak = Oak('/home/zeus/content/oak/small_coco/data/', '/home/zeus/content/oak/small_coco/fastdup/',
              )
    # oak.run(overwrite=True)
    # tags = oak.tag_data()
    oak.run_coco('oak/small_coco/coco.json', overwrite=True)
    oak.tag_data()
    oak.visualize_coco()
    a = 1
