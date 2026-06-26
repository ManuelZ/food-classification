import os

import cv2
from torch.utils.data import Dataset


class KenyanFood13Dataset(Dataset):
    """
    Generic Dataset class for semantic segmentation datasets.
    """

    def __init__(
        self,
        data_path,
        images_folder,
        image_ids,
        labels=None,
        transforms=None,
    ):

        self.data_path = data_path
        self.images_folder = images_folder
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx] if self.labels is not None else None

        # Get image
        image_path = os.path.join(self.data_path, self.images_folder, f"{image_id}.jpg")

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # For the prediction step
        if label is None:
            label = str(image_id)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            return transformed["image"], label

        return image, label
