import glob
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from albumentations.augmentations.crops import functional as F
from catalyst import data
from catalyst.contrib.datasets import misc
import numpy as np
from torch.utils.data import Dataset

def images_in_dir(args):
    """ Searches for all images in the directory """
    path = Path(*args)
    if not path.exists():
        idx = path.name.rfind("_")
        path = path.parent / path.name[:idx] / path.name[dix + 1:]

    fles = glob.iglob(f"{path}/**/*", recursive=True)
    images = sorted(filter(has_image_extension, files))

    return images

def paired_random_crop(images: Iterable[np.ndarray], crop_sizes: Iterable[Tuple[int, int]]) -> Iterable[np.ndarray]:
    """ Crop a random part of the input images """
    h_start, w_start = random.random(), random.random()
    crops = [
        F.random_crop(image, height, width, h_start, w_start) for image, (height, width) in zip(images, crop_sizes)
    ]
    return crops

class PairedImagesDataset(Dataset):
    """
    Base Dataset for the Image Super-Resolution Task

    Args:
     train - True , creates dataset for the training task, otherwise for the validation task
     target_types : Type of target to use, ``'bicubic_X2'``, ``'unknown_X4'``, ``'X8'``, ``'mild'``, ...
     patch_size: If train == True, define sizes of patches to produce, return full image otherwise, Tuple of height and width
     transform : A function/ transform that takes in dictionary (with low and high res images) and returns a transformed version
     low_res_image_key: Key to use to store images of low_res
     hight_res_image_key : Key to use to store images for high_res
    """

    def __init__(
        self,
        train: bool = True,
        target_type: str = "bicubic_X4",
        patch_size: Tuple[int, int] = (96, 96),
        transform: Optional[Callable[[Any], Dict]] = None,
        low_res_image_key: str = "low_res",
        high_res_image_key: str = "high_res"
    ):
        super().__init__()

        self.train = train

        self.lr_key = low_res_image_key
        self.hr_key = high_res_image_key

        self.data: List[Dict[str, str]] = []
        self.open_fn = data.ReaderCompose([
            data.ImageReader(input_key="lr_image", output_key=self.lr_key),
            data.ImageReader(input_key="hr_image", output_key=self.hr_key),
        ])

        _, downscaling = target_type.split("_")
        self.scale = int(downscaling) if downscaling.isdigit() else 4
        height, width = patch_size
        self.target_patch_size = patch_size
        self.input_patch_size = (height // self.scale, width // self.scale)

        self.transform = utils.Augmentor(transform)

    def __getitem__(self, index: int) -> Dict:
        """ Gets element of the dataset """
        record = self.data[index]
        sample_dict = self.open_fn(record)

        if self.train:
            lr_crop, hr_crop = paired_random_crop(
                (sample_dict[self.lr_key], sample_dict[self.hr_key]),
                (self.input_patch_size, self.target_patch_size),
            )
            sample_dict.update({self.lr_key: lr_crop, self.hr_key: hr_crop})

        sample_dict = self.transform(sample_dict)

        return sample_dict

    def __len__(self) -> int:
        return len(self.data)


class FFHQDataset(PairedImagesDataset):
    """ FFHQ DAtaset """

    def __init__(
        self,
        root: str,
        train: bool = True,
        target_type: str = "bicubic_X4",
        patch_size: Tuple[int, int] = (96, 96),
        transform: Optional[Callable[[Any], Dict]] = None,
        low_res_image_key: str = "lr_image",
        high_res_image_key: str = "hr_image",
        download: bool = False,
    ) -> None:
        super().__init__(
            train=train,
            target_type=target_type,
            patch_size=patch_size,
            transform=transform,
            low_res_image_key=low_res_image_key,
            high_res_image_key=high_res_image_key,
        )

        mode = "train" if train else "valid"
        filename_hr = f"FFHQ_{mode}_HR.zip"
        filename_lr = f"FFHQ_{mode}_LR_{target_type}.zip"
        if download:
            misc.download_and_extract_archive(
                f"{self.url}{filename_hr}",
                download_root=root,
                filename=filename_hr,
                md5=self.resources[filename_hr]
            )

            misc.download_and_extract_archive(
                f"{self.url}{filename_lr}",
                download_root=root,
                filename=filename_lr,
                md5=self.resources[filename_lr]
            )

        lr_images = images_in_dir(root, Path(filename_lr).stem)
        hr_images = images_in_dir(root, Path(filename_hr).stem)
        assert len(lr_images) == len(hr_images)

        self.data = [
            {"lr_image": lr_image, "hr_image": hr_image} for lr_image, hr_image in zip(lr_images, hr_images)
        ]


class ImageFolderDataset(data.ListDataset):
    """ A generic data loader where the samples are arranged in this way
    Args:
        pathname : Root directory of dataset
        image_key: key to use to store imag
        image_name_key: Key to use to store name of the image
        transform: A function / transform that takes in dictionary and returns
            its transformed version
    """

    def __init__(
        self,
        pathname: str,
        image_key: str = "image",
        image_name_key: str = "filename",
        transform: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        files = glob.iglob(pathname, recursive=True)
        images = sorted(filter(has_image_extension, files))

        list_data = [{"image": filename} for filename in images]
        open_fn = data.ReaderCompose([
            data.ImageReader(input_key="image", output_key=image_key),
            data.LambdaReader(input_key="image", output_key=image_name_key),
        ])

        transform = utils.Augmentor(transform)

        super().__init__(
            list_data=list_data, open_fn=open_fn, dict_transform=transform
        )