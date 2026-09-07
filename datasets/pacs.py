import torchvision.transforms as transforms
from utils.conf import data_path
from datasets.utils.federated_dataset import (FederatedDataset, ImageFolder_Custom,
                                               partition_domain_skew_loaders)
from datasets.transforms.denormalization import DeNormalize
from backbone.LightResNet import LightResNet


class FedLeaPACS(FederatedDataset):
    NAME = 'fl_pacs'
    SETTING = 'domain_skew'
    N_CLASS = 7

    DOMAINS_LIST = ['photo', 'art_painting', 'cartoon', 'sketch']

    Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    def get_data_loaders(self, selected_domain_list=[]):
        using_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list

        nor_transform = self.Nor_TRANSFORM

        train_dataset_list = []
        test_dataset_list = []

        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        data_root = data_path() + 'PACS/'

        for _, domain in enumerate(using_list):
            domain_path = data_root + domain + '/'
            train_dataset = ImageFolder_Custom(data_name=domain, root_path=domain_path,
                                               train=True, transform=nor_transform)
            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            domain_path = data_root + domain + '/'
            test_dataset = ImageFolder_Custom(data_name=domain, root_path=domain_path,
                                              train=False, transform=test_transform)
            test_dataset_list.append(test_dataset)

        traindls, testdls = partition_domain_skew_loaders(train_dataset_list, test_dataset_list, self)

        return traindls, testdls

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaPACS.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {'lightresnet': lambda c: LightResNet(3, c)}
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(LightResNet(3, FedLeaPACS.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaPACS.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
