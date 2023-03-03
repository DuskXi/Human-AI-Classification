import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import Config

import os


class HADataset(Dataset):
    def __init__(self, config: Config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), test_data=False):
        self.config = config
        self.human_path = config["human_path"]
        self.ai_path = config["ai_path"]
        self.random_data = config["random_data"]
        self.human_set = self.read_data(self.human_path)
        self.ai_set = self.read_data(self.ai_path)
        self.device = device
        self.train_data_percentage = config["train_data_percentage"]
        self.transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        if self.random_data:
            random.shuffle(self.human_set)
            random.shuffle(self.ai_set)
        if len(self.human_set) > len(self.ai_set):
            self.human_set = self.human_set[:len(self.ai_set)]
        elif len(self.human_set) < len(self.ai_set):
            self.ai_set = self.ai_set[:len(self.human_set)]
        if not test_data:
            self.human_set = self.human_set[:int(len(self.human_set) * self.train_data_percentage)]
            self.ai_set = self.ai_set[:int(len(self.ai_set) * self.train_data_percentage)]
        else:
            self.human_set = self.human_set[int(len(self.human_set) * self.train_data_percentage):]
            self.ai_set = self.ai_set[int(len(self.ai_set) * self.train_data_percentage):]
        self.merged_set = []
        flag = False
        index_human = 0
        index_ai = 0
        for i in range(self.__len__()):
            if flag:
                if index_human < len(self.human_set):
                    self.merged_set.append((True, self.human_set[index_human]))
                    index_human += 1
            else:
                if index_ai < len(self.ai_set):
                    self.merged_set.append((False, self.ai_set[index_ai]))
                    index_ai += 1
            flag = not flag

    @staticmethod
    def read_data(path):
        images = [x for x in os.listdir(path) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]
        result = []
        for image in images:
            result.append({
                "image_path": os.path.join(path, image),
                "filename": image,
            })
        return result

    def __len__(self):
        return len(self.human_set) + len(self.ai_set)

    def __getitem__(self, index):
        (flag, data) = self.merged_set[index]
        label = np.array([1, 0]) if flag else np.array([0, 1])
        image = Image.open(data["image_path"]).convert('RGB')
        image = self.transformer(image)
        label = torch.from_numpy(label).float()
        return image.to(self.device), label.to(self.device)
