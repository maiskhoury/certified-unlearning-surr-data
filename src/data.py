from torchvision.datasets import MNIST, USPS, CIFAR10, ImageFolder, CIFAR100
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset, TensorDataset
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import InterpolationMode

from typing import Tuple
import logging
import os
from tqdm import tqdm


def get_dataloaders(datasets,
                    batch_size=64,
                    shuffle=True):
    if isinstance(datasets, Dataset):
        return DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)
    else:
        dataloaders: list[DataLoader] = []
        for idx, dataset in enumerate(datasets):
            if isinstance(dataset, Dataset):
                dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
        return dataloaders


def get_train_test_datasets(idx: str, transform=None, target_transform=None,
                            train_path=None, test_path=None, device=None, save_path=None) -> Tuple[Dataset, Dataset]:
    train_dataset, test_dataset = None, None

    if idx == 'mnist':
        train_dataset = MNIST('./data', train=True, transform=transform,
                              target_transform=target_transform, download=True)
        test_dataset = MNIST('./data', train=False, transform=transform,
                             target_transform=target_transform, download=True)
    elif idx == 'usps':
        train_dataset = USPS('./data', train=True, transform=transform,
                             target_transform=target_transform, download=True)
        test_dataset = USPS('./data', train=False, transform=transform,
                            target_transform=target_transform, download=True)
    elif idx == 'cifar10':
        train_dataset = CIFAR10('./data', train=True, transform=transform,
                                target_transform=target_transform, download=True)
        test_dataset = CIFAR10('./data', train=False, transform=transform,
                               target_transform=target_transform, download=True)
    elif idx == 'caltech256':
        # Load the dataset using ImageFolder
        dataset_path = './data/Caltech256'
        dataset = ImageFolder(root=dataset_path, transform=transform, target_transform=target_transform)

        # Split into train and test datasets (if needed)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    elif idx == 'stanforddogs':
        dataset_path = './data/StanfordDogs'
        dataset = ImageFolder(root=dataset_path, transform=transform, target_transform=target_transform)

        # Split into train and test datasets (if needed)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    elif idx == 'cifar100':
        train_dataset = CIFAR100('./data', train=True, transform=transform,
                                target_transform=target_transform, download=True)
        test_dataset = CIFAR100('./data', train=False, transform=transform,
                               target_transform=target_transform, download=True)
    elif 'act' in idx:
        dataset_idx = idx.split('-')[0]

        if train_path is not None and test_path is not None:
            etrain_data = torch.load(train_path, weights_only=False)
            eval_data = torch.load(test_path, weights_only=False)
            train_dataset = TensorDataset(etrain_data['data'], etrain_data['label'])
            test_dataset = TensorDataset(eval_data['data'], eval_data['label'])
            logging.info('train and test datasets are loaded from {} and {}'.format(train_path, test_path))
        else:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
            train_dataset, val_dataset = get_train_test_datasets(dataset_idx, transform)
            train_loader, val_loader = get_dataloaders([train_dataset, val_dataset], batch_size=512,
                                                       shuffle=False)
            etrain_data, etrain_label, eval_data, eval_label = [], [], [], []
            with torch.no_grad():
                for data, label in tqdm(train_loader, desc='train data extraction', leave=True):
                    data = data.to(device)
                    edata = feature_extractor(data).to('cpu')
                    etrain_data.append(edata.view(edata.shape[0], -1))
                    etrain_label.append(label)
                for data, label in tqdm(val_loader, desc='test data extraction', leave=True):
                    data = data.to(device)
                    edata = feature_extractor(data).to('cpu')
                    eval_data.append(edata.view(edata.shape[0], -1))
                    eval_label.append(label)
            etrain_data = torch.cat(etrain_data, dim=0)
            etrain_label = torch.cat(etrain_label, dim=0)
            eval_data = torch.cat(eval_data, dim=0)
            eval_label = torch.cat(eval_label, dim=0)
            if save_path is not None:
                tfname = '_'.join([dataset_idx, 'linear', 'resnet18', 'train.pth'])
                tefname = '_'.join([dataset_idx, 'linear', 'resnet18', 'test.pth'])
                tfname = os.path.join(save_path, tfname)
                tefname = os.path.join(save_path, tefname)
                torch.save(
                    {
                        'data': etrain_data,
                        'label': etrain_label
                    }, tfname
                )

                torch.save(
                    {
                        'data': eval_data,
                        'label': eval_label
                    }, tefname
                )

                logging.info('train and test datasets {}-act are saved under {}'.format(dataset_idx, save_path))
            train_dataset = TensorDataset(etrain_data, etrain_label)
            test_dataset = TensorDataset(eval_data, eval_label)
    try:
        assert (test_dataset is not None and train_dataset is not None), 'the given dataset id is not recognized'
    except AssertionError as error:
        print(error)

    return train_dataset, test_dataset


def get_retain_forget_datasets(dataset, forget):
    retain_dataset, forget_dataset = None, None
    if isinstance(forget, float):
        # selective unlearning
        forget_size = int(len(dataset) * forget)
        retain_size = len(dataset) - forget_size
        retain_dataset, forget_dataset = random_split(dataset, [retain_size, forget_size])
    elif isinstance(forget, int):
        # class unlearning
        retain_dataset, forget_dataset = [], []
        for idx, (_, label) in enumerate(dataset):
            if label != forget:
                retain_dataset.append(idx)
            else:
                forget_dataset.append(idx)
        retain_dataset = Subset(dataset, retain_dataset)
        forget_dataset = Subset(dataset, forget_dataset)

    return retain_dataset, forget_dataset


def get_class_ratios(dataset, num_class):
    ratios = np.zeros(num_class, dtype=int)
    for _, label in dataset:
        ratios[label] += 1
    return ratios / len(dataset)


def _partite_by_class(dataset, num_class):
    idxs = [[] for _ in range(num_class)]
    for idx, (_, label) in enumerate(dataset):
        idxs[label].append(idx)
    return [Subset(dataset, idx) for idx in idxs]


def __check_max_reached(sizes, max_sizes):
    max_reached = []
    for idx, max_size in enumerate(max_sizes):
        if sizes[idx] >= max_size:
            max_reached.append(idx)
            sizes[idx] = max_size
    return sizes, np.asarray(max_reached)


# def _get_sizes(size, ratios, max_sizes):
#     num_class = len(ratios)
#     sizes = (ratios * size).astype(int)
#
#     sizes, max_reached = __check_max_reached(sizes, max_sizes)
#     remainder = size - np.sum(sizes)
#     for _ in range(remainder):
#         if len(max_reached) > 0:
#             available_classes = np.delete(np.arange(num_class), np.asarray(max_reached))
#         else:
#             available_classes = np.arange(num_class)
#         sizes[np.random.choice(available_classes)] += 1
#         sizes, max_reached = __check_max_reached(sizes, max_sizes)
#         if len(max_reached) == num_class:
#             break
#
#     assert np.sum(sizes) == size, 'size could not achieved'
#     return sizes

def _get_sizes(size, ratios, max_sizes):
    sizes = (ratios * size).astype(int)

    decisive_idx = np.argmin(max_sizes - sizes)
    dratio = ratios[decisive_idx]
    dmax_size = max_sizes[decisive_idx]
    dsize = dmax_size // dratio
    sizes = (ratios * dsize).astype(int)
    return sizes


def get_exact_surr_datasets(dataset,
                            target_size=None, target_ratios=None,
                            starget_size=None, starget_ratios=None,
                            surr_dataset=None, dirichlet=None, num_class=None):
    # TODO: errors might be explained later
    if surr_dataset is None and (target_ratios is not None and target_size is not None):
        num_class = len(target_ratios)
        # partite all classes
        class_partitions = _partite_by_class(dataset, num_class)

        # find target sizes for each class
        max_sizes = [len(partition) for partition in class_partitions]
        target_sizes = _get_sizes(target_size, target_ratios, max_sizes)
        logging.info('target sizes: {}'.format(target_sizes))
        starget_sizes = None
        if starget_size is not None and starget_ratios is not None:
            starget_sizes = _get_sizes(starget_size, starget_ratios, max_sizes)
            logging.info('starget sizes: {}'.format(starget_sizes))

        # randomly select specified number of samples from each class
        first_class_partitions, second_class_partitions = [], []
        for class_idx, target_size_by_class in enumerate(target_sizes):
            first_idx = np.random.choice(len(class_partitions[class_idx]), target_size_by_class, replace=False)
            if starget_sizes is not None:
                second_idx = np.random.choice(len(class_partitions[class_idx]), starget_sizes[class_idx], replace=False)
            else:
                second_idx = np.delete(np.arange(len(class_partitions[class_idx])), first_idx)
            first_class_partitions.append(Subset(class_partitions[class_idx], first_idx))
            second_class_partitions.append(Subset(class_partitions[class_idx], second_idx))
        return ConcatDataset(first_class_partitions), ConcatDataset(second_class_partitions)
    elif surr_dataset is not None:
        return dataset, surr_dataset
    elif dirichlet is not None:
        # generate class proportions for both exact and surrogate datasets
        client_class_distributions = np.random.dirichlet(dirichlet * np.ones(num_class), size=2)
        logging.info('dirichlet: {}'.format(dirichlet))
        logging.info('client class distributions: {}\n{}'.format(client_class_distributions[0],
                                                                 client_class_distributions[1]))
        # print(client_class_distributions)
        return get_exact_surr_datasets(dataset, target_size=target_size, target_ratios=client_class_distributions[0],
                                       starget_size=starget_size, starget_ratios=client_class_distributions[1])


def get_transforms(idx, model_type='resnet18'):
    if idx == 'cifar10':
        if model_type == 'mlp':
            # For MLP: keep original 32x32 size, simple normalization
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            # For ResNet18: resize to 224x224, use ImageNet normalization
            return v2.Compose([
                v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),  # ResNet18 expects 224x224 input size
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pretrained models
            ])
    elif idx == 'caltech256' or idx == 'stanforddogs' or idx == 'cifar100':
        return v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),  # ResNet18 expects 224x224 input size
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization for pretrained models
        ])
    else:
        return None
