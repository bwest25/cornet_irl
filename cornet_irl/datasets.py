# datasets.py

import torch
from torch.utils.data import Dataset
from skimage import io
import os
import pickle as p
from tqdm import tqdm
from PIL import Image
import torchvision


class TDWSurfaceNormalsDataset(Dataset):
    """ThreeDWorld surface normals dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with img, normals, and names.txt
        """
        self.root_dir = root_dir
        names_txt_location = os.path.join(root_dir, "names.txt")
        names_txt = open(names_txt_location, 'r')
        self.names = names_txt.read().splitlines()

    def __len__(self):
        length = len(self.names)
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.names[idx]
        img = self.__get_img(name)
        downsampled_normals = self.__get_downsampled_normals(name)

        sample = {'img': img, 'downsampled_normals': downsampled_normals}

        return sample

    def __get_img(self, name):
        img_location = os.path.join(self.root_dir, "img",
                                    "img_{}.png".format(name))

        img = io.imread(img_location).astype('float32')
        img = torchvision.transforms.ToTensor()(img)

        return img

    def __get_normals(self, name):
        normals_location = os.path.join(self.root_dir, "normals",
                                        "normals_{}.png".format(name))

        normals = io.imread(normals_location).astype('float32')
        normals = torchvision.transforms.ToTensor()(normals)
        normals = torch.flatten(normals)

        return normals

    def __get_downsampled_normals(self, name):
        downsampled_normals_location =\
            os.path.join(self.root_dir, "downsampled_normals",
                         "downsampled_normals_{}.png".format(name))

        downsampled_normals = io.imread(downsampled_normals_location).\
            astype('float32')
        downsampled_normals =\
            torchvision.transforms.ToTensor()(downsampled_normals)
        downsampled_normals = torch.flatten(downsampled_normals)

        return downsampled_normals

    def generate_saved_downsampled_normals(self):
        for name in tqdm(self.names,
                         desc="Generating downsampled normals images"):
            normals_location = os.path.join(self.root_dir, "normals",
                                            "normals_{}.png".format(name))
            normals = Image.open(normals_location)
            downsampled_normals = normals.resize((56, 56))

            downsampled_normals_location =\
                os.path.join(self.root_dir, "downsampled_normals",
                             "downsampled_normals_{}.png".format(name))
            downsampled_normals.save(downsampled_normals_location)
