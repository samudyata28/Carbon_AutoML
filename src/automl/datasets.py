"""
This module contains the datasets used in the AutoML exam.
If you want to edit this file be aware that we will later 
  push the test set to this file which might cause problems.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import tempfile
import shutil


class BaseVisionDataset(VisionDataset):
    """A base class for all vision datasets.

    Args:
        root: str or Path
            Root directory of the dataset (should contain the extracted datasets).
        split: string (optional)
            The dataset split, supports `train` (default), `val`, or `test`.
        transform: callable (optional)
            A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, `transforms.RandomCrop`.
        target_transform: callable (optional)
            A function/transform that takes in the target and transforms it.
        download: bool (optional)
            If true, downloads the dataset zip and extracts it into the root directory.
            If dataset is already downloaded, it is not downloaded again.
    """
  
    _dataset_name: str
    width: int
    height: int
    channels: int
    num_classes: int

    PHASE1_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-vision/vision-phase1.zip "
    PHASE2_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-vision/vision-phase2.zip "

    def __init__(
        self,
        root: Union[str, Path] = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ["train", "test"], f"Split {split} not supported"
        self._split = split
        self._base_folder = Path(self.root) / self._dataset_name

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download required files, "
                "or download them manually from https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-vision/"
            )

        data = pd.read_csv(self._base_folder / f"{self._split}.csv")
        self._labels = data['label'].tolist()
        self._image_files = data['image_file_name'].tolist()

    def _check_integrity(self) -> bool:
        train_images_folder = self._base_folder / "images_train"
        test_images_folder = self._base_folder / "images_test"
        if not (train_images_folder.exists() and train_images_folder.is_dir()) or \
           not (test_images_folder.exists() and test_images_folder.is_dir()):
            return False
        if not (self._base_folder / "train.csv").exists() or not (self._base_folder / "test.csv").exists():
            return False
        return True

    def download(self) -> None:
        """Download dataset if missing, choosing phase1 or phase2 based on dataset name."""
        data_path = Path(self.root)
        data_path.mkdir(exist_ok=True)

        if self._base_folder.exists():
            print(f"{self._dataset_name} dataset already exists. Skipping download.")
            return

        if self._dataset_name == "skin_cancer":
            phase_url = self.PHASE2_URL
            phase_name = "phase2"
        else:
            phase_url = self.PHASE1_URL
            phase_name = "phase1"

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading and extracting {phase_name} dataset...")
            zip_name = f"{phase_name}.zip"

            download_and_extract_archive(
                url=phase_url,
                download_root=temp_dir,
                extract_root=temp_dir,
                filename=zip_name,
                remove_finished=True,
            )

            phase_folder = next(Path(temp_dir).glob("phase*"))
            for item in phase_folder.iterdir():
                destination = data_path / item.name
                if destination.exists():
                    print(f"Skipping existing item: {destination}")
                    continue
                shutil.move(str(item), str(destination))

            print(f"{phase_name} download completed.")

    def extra_repr(self) -> str:
        """String representation of the dataset."""
        return f"split={self._split}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image_path = self._base_folder / f"images_{self._split}" / image_file
        image = PIL.Image.open(image_path)
        if self.channels == 1:
            image = image.convert("L")
        elif self.channels == 3:
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._image_files)


class EmotionsDataset(BaseVisionDataset):
    """ Emotions Dataset.

    "This dataset contains images of faces displaying one of seven emotions
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    """
    _dataset_name = "emotions"
    width = 48
    height = 48
    channels = 1
    num_classes = 7


class FlowersDataset(BaseVisionDataset):
    """Flower Dataset.

    This dataset contains images of 102 types of flowers. The task is to classify the flower type.
    """
    _dataset_name = "flowers"
    width = 512
    height = 512
    channels = 3
    num_classes = 102


class FashionDataset(BaseVisionDataset):
    """Fashion Dataset.

    This dataset contains images of fashion items. The task is to classify what kind of fashion item it is.
    """
    _dataset_name = "fashion"
    width = 28
    height = 28
    channels = 1
    num_classes = 10


class SkinCancerDataset(BaseVisionDataset):
    """SkinCancer Dataset.
    
    The SkinCancer dataset contains images of skin lesions. The task is to classify what kind of skin lesion it is.

    This is the test dataset for the AutoML exam. It does not contain the labels for the test split.
    You are expected to predict these labels and save them to a file called `final_test_preds.npy` for your
    final submission.
    """
    _dataset_name = "skin_cancer"
    width = 450
    height = 450
    channels = 3
    num_classes = 7