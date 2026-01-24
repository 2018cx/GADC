from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import copy

def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

class CustomDataset_REPA(Dataset):
    def __init__(self, root, image_size, transform=None, 
                 select_info=None, select_type="random", 
                 select_num=10000, interval=1, cpu_load=True,
                 embedding_root = None,
                 embedding_json=None,
                 target_mean=0.):
        self.root = root
        self.cpu_load = cpu_load
        self.embedding_root = embedding_root
        self.embedding_json = embedding_json
        self.target_mean = target_mean
        
        import json
        with open(self.embedding_json, "r") as f:
            self.embedding_json = json.load(f)
        self.embedding_json = {k: v for k, v in self.embedding_json.items()}

        if select_info is not None:
            print(type(select_info))
            assert isinstance(select_info, dict), "select_info must be a dict"
            _tmp = [[key, value] for key, value in select_info.items()][::interval]
     
        
        self.image_size = image_size
        self.transform = transform

        self.image_folder = ImageFolder(root=root, transform=self.transform)
        new_tmp = copy.deepcopy(self.image_folder.samples)

        self.class_to_idx = self.image_folder.class_to_idx
        self.num_classes = len(self.class_to_idx)
        self.class_names = list(self.class_to_idx.keys())
        
        self.class_images = []
        for i in range(1000):
            self.class_images.append(self.get_class_images(i))

        if select_info is not None:
            pre_select_num = int(select_num / self.num_classes)
            json_path = "../selection"
            if select_num == 10000:
                json_path = json_path + "/test_10.json"
            elif select_num == 50000:
                json_path = json_path + "/test_50.json"
            elif select_num == 100000:
                json_path = json_path + "/test_100.json"
            else:
                raise ValueError("select_num must be 10000 or 50000 or 100000")
            with open(json_path, "r") as f:
                select_training_samples = json.load(f)
            dict_path2class = {self.image_folder.samples[i][0]: self.image_folder.samples[i][1] for i in range(len(self.image_folder.samples))}
            new_samples = []
            for d in select_training_samples:
                _cls = dict_path2class[d]
                new_samples.append((d, _cls))
            self.image_folder.samples = new_samples
           
        if self.cpu_load:
            print("CPU loading ...")
            new_image_folder = []
            for img_path, class_idx in tqdm(self.image_folder.samples):
                try:
                    image = self.image_folder.loader(img_path)
                    if self.transform:
                        image = self.transform(image)
                    pt = torch.load(os.path.join("visual", 
                                    self.embedding_json[img_path]), map_location="cpu", weights_only=True)["gt_zs"]
                    new_image_folder.append((image, class_idx, pt))
                except Exception as e:
                    print(e)
                    continue
            self.image_folder.samples = new_image_folder

            

    def __len__(self):
        return len(self.image_folder.samples)

    def __getitem__(self, index):
        if self.cpu_load:
            image, class_idx, pt = self.image_folder.samples[index]
        else:
            img_path, class_idx = self.image_folder.samples[index]
            image = self.image_folder.loader(img_path)
            if self.transform:
                image = self.transform(image)
            pt = torch.load(os.path.join("visual", 
                                     self.embedding_json[img_path]), map_location="cpu", weights_only=True)["gt_zs"]
        
        return image, class_idx, pt

    def get_class_images(self, class_idx):
        class_images = []
        for img_path, _class_idx in self.image_folder.samples:
            if _class_idx == class_idx:
                class_images.append((img_path, class_idx))
        return class_images
