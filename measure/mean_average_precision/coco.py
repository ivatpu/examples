"""Annotator class."""

import json
from typing import Generator, List, Dict, Tuple, Any
import os


# pylint: disable=too-many-instance-attributes
class Annotator:
    """Class for get annotations for test Nets."""

    def __init__(self, instances_path: str, images_path: str,
                 class_names: List[str] = None):
        """Init of class."""
        self.paths = dict(data=instances_path,
                          images=images_path)
        self._class_names = class_names or []
        raw_data = self.extract_json()
        self.extract_categories(raw_data)
        self.extract_images(raw_data)
        self.extract_annotations(raw_data)

    def generator(self) -> Generator[Tuple[str, List[Tuple[int, float, float, float, float]]], None, None]:
        """
        Generate images paths with corresponding ground truth values.

        Boxes parameters are generated in the form [class, xmin, ymin, xmax, ymax]
        """
        images_list = self.list_images()
        for filename in images_list:
            if 'jpg' in filename.lower() or 'jpeg' in filename.lower():
                image_path = os.path.join(self.paths['images'], filename)
                image_id = self.name_to_image_id(filename)
                real_categories = []
                if self._bboxes.get(image_id):
                    for bbox, image_cat in zip(self._bboxes[image_id],
                                               self.categories_by_image_id(
                                                   image_id)):
                        class_id = self._class_names.index(self.cat_to_name(
                            image_cat))
                        xmin = bbox[0]
                        ymin = bbox[1]
                        xmax = bbox[0] + bbox[2]
                        ymax = bbox[1] + bbox[3]
                        _bbox = xmin, ymin, xmax, ymax
                        real_categories.append((class_id, *_bbox))
                    yield image_path, real_categories

    def list_images(self) -> List[str]:
        """Return a list with dataset."""
        return os.listdir(self.paths['images'])

    def extract_json(self) -> Dict[str, Dict[Any, Any]]:
        """Load annotations from source."""
        with open(self.paths['data']) as file:
            return json.load(file)

    def extract_categories(self, data: dict) -> None:
        """Extract categories name and ids from json Dict."""
        cats = data['categories']
        self._cats: Dict[str, List[int]] = {}
        self._cats_r: Dict[int, str] = {}
        for cat in cats:
            if cat['name'] in self._class_names:
                classificator = cat['name']
            else:
                classificator = cat['supercategory']
            if not self._cats.get(classificator):
                self._cats[classificator] = []
            self._cats[classificator].append(cat['id'])
            self._cats_r[cat['id']] = classificator
        assert 1 in self._cats['person']
        assert self._cats_r[1] == 'person'

    def extract_images(self, data: dict) -> None:
        """Extract image ids and filename from json Dict."""
        imgs = data['images']
        self._imgs: Dict[str, int] = {img['file_name']: img['id'] for img in imgs}
        self._imgs_r = dict(zip(self._imgs.values(), self._imgs.keys()))
        assert self._imgs['000000397133.jpg'] == 397133
        assert self._imgs_r[397133] == '000000397133.jpg'

    def extract_annotations(self, data: dict) -> None:
        """Extract class annotation for images."""
        ants = data['annotations']
        self._ants: Dict[int, List[int]] = {}
        self._bboxes: Dict[int, List[List[float]]] = {}
        for ant in ants:
            if not self._ants.get(ant['image_id']):
                self._ants[ant['image_id']] = []
            self._ants[ant['image_id']].append(ant['category_id'])
            if not self._bboxes.get(ant['image_id']):
                self._bboxes[ant['image_id']] = []
            self._bboxes[ant['image_id']].append(ant['bbox'])
        assert 18 in self._ants[289343]

    def name_to_image_id(self, name: str) -> int:
        """Get image id from filename."""
        return self._imgs.get(name, -1)

    def image_id_to_name(self, image_id: int) -> str:
        """Get filename from image id."""
        return self._imgs_r.get(image_id, 'Import error')

    def cat_to_name(self, cat_id: int) -> str:
        """Get category name from category id."""
        return self._cats_r.get(cat_id, 'Import error')

    def categories_by_image_id(self, image_id: int) -> List[int]:
        """Get class by image id from annotations."""
        return self._ants.get(image_id, [-1])
