"""
Script which was used to generate final submission file for the approach based on Resnet-50 and other single model
architectures (or stacking).
"""
import csv
import os
import torch
from torch import nn
from torchvision.models import resnet50, resnet152
from task1_resnet import MyDataset, get_transform
from torch.utils.data import DataLoader
import glob

import pandas as pd

ROOT = '/home/bianca/Poli/Master/ATAI/Hw2/'
SUBDIR_UNLABELED = 'task1/train_data/images/unlabeled'
SUBDIR_ROOT_TRAIN = 'task1/train_data'
BEST_MODEL_LABELED = 'best_model_labeled_data.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
UNLABELED_FILE_TRAIN = 'annotations_unlabeled_train_resnet152.csv'

SUBDIR_ROOT_VAL = 'task2/val_data'
MERGED_BEST_MODEL_LABELED = 'merged_best_model_labeled_data.pth'
UNLABELED_FILE_VAL = 'annotations_unlabeled_val.csv'


def create_anno_file_without_labels(data_type_path, anno_file_name_type, subdir_root_data):
    """
    This method was used to generate annotation files for the unlabeled and val data for using before predicting.
    This is used at the beginning, when no labels are generated and only saves the file names.
    :param data_type_path: The path to the specific folder (unlabeled/ val data).
    :param anno_file_name_type: The name of the file which will be generated.
    :param subdir_root_data: The subdirectory in which the images are found.
    :return:
    """
    path = os.path.join(ROOT, subdir_root_data)
    # Read file names
    # filenames = os.listdir(os.path.join(path, 'images', 'unlabeled'))
    filenames = os.listdir(path)

    header = ['sample', 'label']
    with open(os.path.join(path, 'annotations_' + anno_file_name_type + '.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for file in filenames:
            writer.writerow([os.path.join(data_type_path, file), ''])


def create_csv(path, file_name, data_rows):
    header = ['sample', 'label']
    with open(os.path.join(path, file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)


def load_trained_model(model_name):
    """
    Method which creates a model. The model is used to later load the best weights of the previously trained model.
    :param model_name: The name of the model.
    :return:
    """
    # Load trained model on labeled data
    model_resnet = resnet152(pretrained=True)

    for param in model_resnet.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 100)
    model_conv = model_resnet.to(device)
    model_conv = model_conv.load_state_dict(torch.load(os.path.join(ROOT, model_name)))

    return model_conv


def get_dataloader_unlabeled_data(batches, subdir, file):
    """
    Method which returns the data loader for the unlabeled data.
    :param batches: the number of batches
    :param subdir: the directory where the images are found
    :param file: the path to the annotation file
    :return:
    """
    # Initialize train dataset and dataloader
    transform = get_transform()
    unlabeled_dataset = MyDataset(csv_file=os.path.join(subdir, file), root_dir=ROOT, transform=transform)

    loader_unlabeled = DataLoader(unlabeled_dataset, batch_size=batches, shuffle=True, num_workers=0)

    return loader_unlabeled


def sort_csv(path, filename):
    # assign dataset
    csvData = pd.read_csv(os.path.join(path, filename))

    # displaying unsorted data frame
    print("\nBefore sorting:")
    print(csvData)

    # sort data frame
    csvData.sort_values(["sample"],
                        axis=0,
                        ascending=[True],
                        inplace=True)

    for idx, row in csvData.iterrows():
        name = row['sample']
        splitted_name = name.split(".")
        splitted_name[0] = str(int(splitted_name[0]))
        name = ".".join(splitted_name)
        csvData.at[idx, 'sample'] = name

    csvData.to_csv(os.path.join(path, filename), index=False)


def merge_csvs(path_in, path_out, merged_file_name):
    csv_files = glob.glob(os.path.join(path_in, '*.{}'.format('csv')))
    df_append = pd.DataFrame()
    # append all files together
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_append = df_append.append(df_temp, ignore_index=True)
    df_append.to_csv(os.path.join(path_out, merged_file_name))


if __name__ == '__main__':
    # create_anno_file_without_labels(SUBDIR_UNLABELED, "unlabeled_train", SUBDIR_ROOT_TRAIN)
    # create_anno_file_without_labels(SUBDIR_ROOT_VAL, "unlabeled_val", SUBDIR_ROOT_VAL)

    # merge_csvs('/home/bianca/Poli/Master/ATAI/Hw2/task1/csvs_to_merge',
    #            '/home/bianca/Poli/Master/ATAI/Hw2/task1/train_data', 'merged_annotations.csv')

    model = load_trained_model(BEST_MODEL_LABELED)
    unlabeled_loader = get_dataloader_unlabeled_data(64, SUBDIR_ROOT_TRAIN, UNLABELED_FILE_TRAIN)
    # unlabeled_loader = get_dataloader_unlabeled_data(64, SUBDIR_ROOT_VAL, UNLABELED_FILE_VAL)
    model.eval()

    rows = []
    for i, data in enumerate(unlabeled_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, img_names = data['image'].to(device), data['image_name']

        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted_numpy_list = predicted.to('cpu').numpy().tolist()
        for name, label in zip(img_names, predicted_numpy_list):
            # name = name.replace('task1/train_data/', '')
            # splitted_name = name.split(".")
            # splitted_name[0] = f"{int(splitted_name[0]):06d}"
            # name = '.'.join(splitted_name)
            rows.append([name, label])

    create_csv(SUBDIR_ROOT_VAL, 'annotations_unlabeled_resnet152.csv', rows)

    # sort_csv(SUBDIR_ROOT_VAL, 'annotations_unlabeled.csv')



