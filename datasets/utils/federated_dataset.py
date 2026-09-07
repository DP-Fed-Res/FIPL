from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple
from torchvision.datasets import ImageFolder, DatasetFolder
import numpy as np
import torch.optim
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root_path, train=True, transform=None,
                 target_transform=None, label_dict=None,
                 subset_train_num=8, subset_capacity=10):
        self.data_name = data_name
        self.root_path = root_path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.imagefolder_obj = ImageFolder(root_path, self.transform, self.target_transform)

        if label_dict is not None:
            class_names = self.imagefolder_obj.classes
            valid_classes = set(label_dict.keys())
            filtered_samples = []
            for path, label in self.imagefolder_obj.samples:
                class_name = class_names[label]
                if class_name in valid_classes:
                    filtered_samples.append((path, label_dict[class_name]))
            self.imagefolder_obj.samples = filtered_samples

        all_data = self.imagefolder_obj.samples
        self.train_index_list = []
        self.test_index_list = []
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

        if self.train:
            self.labels = [all_data[i][1] for i in self.train_index_list]
        else:
            self.labels = [all_data[i][1] for i in self.test_index_list]

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):
        if self.train:
            used_index_list = self.train_index_list
        else:
            used_index_list = self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FederatedDataset:
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders = []
        self.test_loader = []
        self.args = args

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list) -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def partition_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                  setting: FederatedDataset) -> Tuple[list, list]:
    from client_data.client_data_cache import load_cached_indices, save_cached_indices
    from utils.best_args import get_domain_dict

    seed = int(getattr(setting.args, 'seed', 0))
    np.random.seed(seed)

    percent_dict = getattr(setting, 'percent_dict', {})

    num_clients = getattr(setting.args, 'num_clients', 0)
    domain_dict = get_domain_dict(setting.NAME, num_clients)
    selected_domains = [ds.data_name for ds in train_datasets]

    ini_len_dict = {}
    for ds in train_datasets:
        name = ds.data_name
        if name not in ini_len_dict:
            ini_len_dict[name] = len(ds)

    cached_indices = load_cached_indices(
        setting.NAME, selected_domains, domain_dict, percent_dict, seed)

    for test_ds in test_datasets:
        n_limit = getattr(setting, 'TEST_SAMPLES_PER_DOMAIN', None)
        if n_limit is not None and len(test_ds) > n_limit:
            indices = np.random.choice(len(test_ds), n_limit, replace=False)
            test_ds = Subset(test_ds, indices)
        test_loader = DataLoader(test_ds,
                                 batch_size=setting.args.local_batch_size, shuffle=False)
        setting.test_loader.append(test_loader)

    for idx, train_ds in enumerate(train_datasets):
        name = train_ds.data_name
        percent = percent_dict.get(name, 1.0)
        num_samples = int(percent * ini_len_dict[name])

        if cached_indices is not None:
            selected_idx = cached_indices[idx]
        else:
            selected_idx = np.random.choice(ini_len_dict[name], size=num_samples, replace=False)

        subset = Subset(train_ds, selected_idx)
        train_loader = DataLoader(subset,
                                  batch_size=setting.args.local_batch_size,
                                  shuffle=True, drop_last=True)
        setting.train_loaders.append(train_loader)

        cache_tag = ' [cache hit]' if cached_indices is not None else ''
        print(f'  Client {idx} [{name}]: {num_samples}/{ini_len_dict[name]} samples ({percent:.0%}){cache_tag}')

    if cached_indices is None:
        all_indices = []
        for train_loader in setting.train_loaders:
            subset = train_loader.dataset
            if hasattr(subset, 'indices'):
                all_indices.append(np.array(subset.indices))
            else:
                all_indices.append(np.arange(len(subset)))

        cache_key = save_cached_indices(
            setting.NAME, selected_domains, domain_dict, percent_dict, seed,
            all_indices,
            extra_meta={'ini_len_dict': ini_len_dict},
        )
        print(f'  -> client data indices cached: client_data/{setting.NAME}/{cache_key}.pkl')

    return setting.train_loaders, setting.test_loader
