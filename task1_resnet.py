"""
Script which was used to train the models using the Resnet-50 architecture or other single model for both tasks.
"""
import math
import glob
from torchvision.models import resnet50
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.utils.data as data

import pandas as pd
import os
import matplotlib.pyplot as plt
import copy
import csv
from torch.optim import lr_scheduler

ROOT = '/home/bianca/Poli/Master/ATAI/Hw2/'
SUBDIR_TRAIN = 'task2/train_data/'

ANNO_FILE = 'task2/train_data/annotations.csv'
TRAIN_FILE = 'train_anno_labeled.csv'
VALID_FILE = 'valid_anno_labeled.csv'

MERGED_ANNO_FILE = 'task2/train_data/merged_annotations.csv'
MERGED_TRAIN_FILE = 'merged_train_anno_labeled.csv'
MERGED_VALID_FILE = 'merged_valid_anno_labeled.csv'


# ------------- Dataset loading ------------- #
class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.filenames = [filename for filename in self.data.loc[:, "sample"].values]
        self.labels = self.data["label"]
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(os.path.join(self.root_dir, self.filenames[idx]))
        img = img.convert('RGB')  # convert to rgb

        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if math.isnan(label):
            return {"image": img, "image_name": self.filenames[idx]}
        else:
            return img, label

    def __len__(self):
        return len(self.data)


def get_transform():
    transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomVerticalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def split_dataset(root, anno_file, subdir, train_file, valid_file):
    """
    Method used for splitting the annotation file and creating the train and validation datasets.
    :param root: the path to the root directory where the files are found
    :param anno_file: the annotation file from which the data will be split
    :param subdir: the subdirectory where the files are found (different for each task)
    :param train_file: the name of the generated file containing the data which will be used for training
    :param valid_file: the name of the generated file containing the data which will be used for validation
    :return:
    """
    df = pd.read_csv(os.path.join(root, anno_file))
    train_set_size = int(len(df) * 0.8)
    valid_set_size = len(df) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = data.random_split(df, [train_set_size, valid_set_size], generator=seed)
    train_dataset = df.loc[train_dataset.indices]
    validation_dataset = df.loc[validation_dataset.indices]

    # Create new files for train and validation data
    # train
    train_path = train_dataset['sample']
    train_labels = train_dataset['label']
    df_train = pd.concat([train_path, train_labels], axis=1)
    df_train.to_csv(os.path.join(subdir, train_file))

    # validation
    validation_path = validation_dataset['sample']
    validation_labels = validation_dataset['label']
    df_validation = pd.concat([validation_path, validation_labels], axis=1)
    df_validation.to_csv(os.path.join(subdir, valid_file))


def merge_csvs(path_in, path_out, merged_file_name):
    csv_files = glob.glob(os.path.join(path_in, '*.{}'.format('csv')))
    df_append = pd.DataFrame()
    # append all files together
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_append = df_append.append(df_temp, ignore_index=True)
    df_append.to_csv(os.path.join(path_out, merged_file_name))


def get_dataloaders(batches, subdir, train_file, valid_file):
    # Initialize train dataset and dataloader
    transform = get_transform()
    train_dataset = MyDataset(csv_file=os.path.join(subdir, train_file), root_dir=ROOT, transform=transform)
    valid_dataset = MyDataset(csv_file=os.path.join(subdir, valid_file), root_dir=ROOT, transform=transform)

    loader_train = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=0)
    loader_val = DataLoader(valid_dataset, batch_size=batches, shuffle=True, num_workers=0)

    return loader_train, loader_val


def train_model(epochs, model, optimizer, csv_train_name, csv_valid_name, modelName,
                csv_train_accuracy_name, csv_valid_accuracy_name):
    best_model_wts = copy.deepcopy(model.state_dict())

    train_loss_dict = {}
    val_loss_dict = {}
    min_valid_loss = np.inf

    valid_accuracy_dict = {}
    train_accuracy_dict = {}

    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        train_loss = 0.0
        valid_loss = 0.0

        count_train = 0
        count_valid = 0

        train_accuracy = 0.0
        valid_accuracy = 0.0

        total_train = 0.0
        total_valid = 0.0

        model.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.mean().backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            count_train += 1
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_accuracy += (predicted == labels).sum().item()

        train_accuracy = train_accuracy / total_train
        train_accuracy_dict[epoch] = train_accuracy
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f} \tCount: {:.6f}  \tTraining Accuracy: {:.6f}'.format(
            epoch, train_loss, count_train, train_accuracy))

        model.eval()
        for i, data in enumerate(validation_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            # print statistics
            valid_loss += loss.item()
            count_valid += 1
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            valid_accuracy += (predicted == labels).sum().item()
        valid_accuracy = valid_accuracy / total_valid
        valid_accuracy_dict[epoch] = valid_accuracy
        valid_loss = valid_loss / len(validation_loader)
        print('Epoch: {} \tValidation Loss: {:.6f} \tCount: {:.6f}  \tValidation Accuracy: {:.6f}'.format(
            epoch, valid_loss, count_valid, valid_accuracy))

        train_loss_dict[epoch] = train_loss
        val_loss_dict[epoch] = valid_loss

        if valid_loss < min_valid_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), modelName)

    # -------------------------------------------------------------------------------------------- #
    # Generating csv file containing the training and validation losses and accuracy for each epoch.
    # open file for writing, "w" is writing
    w = csv.writer(open(csv_train_name, "w"))
    w.writerow(['epoch', 'loss'])
    # loop over dictionary keys and values
    for key, val in train_loss_dict.items():
        # write every key and value to file
        w.writerow([key, val])

    # open file for writing, "w" is writing
    w = csv.writer(open(csv_valid_name, "w"))
    w.writerow(['epoch', 'loss'])
    # loop over dictionary keys and values
    for key, val in val_loss_dict.items():
        # write every key and value to file
        w.writerow([key, val])

    # open file for writing, "w" is writing
    w = csv.writer(open(csv_train_accuracy_name, "w"))
    w.writerow(['epoch', 'accuracy'])
    # loop over dictionary keys and values
    for key, val in train_accuracy_dict.items():
        # write every key and value to file
        w.writerow([key, val])

    # open file for writing, "w" is writing
    w = csv.writer(open(csv_valid_accuracy_name, "w"))
    w.writerow(['epoch', 'accuracy'])
    # loop over dictionary keys and values
    for key, val in valid_accuracy_dict.items():
        # write every key and value to file
        w.writerow([key, val])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # merge_csvs('/home/bianca/Poli/Master/ATAI/Hw2/task1/csvs_to_merge',
    #            '/home/bianca/Poli/Master/ATAI/Hw2/task1/train_data', 'merged_annotations.csv')
    # split_dataset(ROOT, ANNO_FILE, SUBDIR_TRAIN, TRAIN_FILE, VALID_FILE)
    # split_dataset(ROOT, MERGED_ANNO_FILE, SUBDIR_TRAIN, MERGED_TRAIN_FILE, MERGED_VALID_FILE)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    num_epochs = 50  # 20

    train_loader, validation_loader = get_dataloaders(batch_size, SUBDIR_TRAIN, TRAIN_FILE, VALID_FILE)

    model_resnet = resnet50(pretrained=True)

    for param in model_resnet.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 100)
    model_conv = model_resnet.to(device)
    print(model_resnet)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    best_model = train_model(num_epochs, model_resnet, optimizer_conv, 'train_losses_resnet.csv',
                             'valid_losses_resnet.csv', 'best_model_labeled_data.pth',
                             'train_accuracy_resnet.csv', 'valid_accuracy_resnet.csv')

