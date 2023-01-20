"""
Script used for training the models for ensemble methods (voting) for the second task - noisy labels
"""
import torch
from torch import nn
import torchvision
import numpy as np
import timm
import os
from datetime import datetime
import pandas as pd
import copy
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from PIL import Image
import torch.utils.data as data


BATCH_SIZE = 64
EPOCHS = 7
CLASSES = 100
metric = 0
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

ROOT = '/home/bianca/Poli/Master/ATAI/Hw2/'
SUBDIR_TRAIN = 'task2/train_data/'

ANNO_FILE = 'task2/train_data/annotations.csv'
TRAIN_FILE = 'train_anno_labeled.csv'
VALID_FILE = 'valid_anno_labeled.csv'

SUBDIR_ROOT_VAL = 'task2/val_data'
UNLABELED_FILE_VAL = 'annotations_unlabeled_val.csv'

MERGED_ANNO_FILE = 'task1/train_data/merged_annotations.csv'
MERGED_TRAIN_FILE = 'merged_train_anno_labeled.csv'
MERGED_VALID_FILE = 'merged_valid_anno_labeled.csv'


def model_adjustment(p_model):
    """
    Function to freeze layers.
    :param p_model: the model for which the freezing is applied
    :return:
    """
    for param in p_model.parameters():
        param.requires_grad = False
    return p_model


def model_assets(base_model, seed):
    """
    Function to return loss, optimizer and schedulers based on the model.
    :param base_model:
    :param seed:
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
    lrp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    if seed == 0 or seed == 1:
        mls_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=0.1)
        return criterion, optimizer, lrp_scheduler, mls_scheduler
    else:
        return criterion, optimizer, lrp_scheduler


def model_archive(seed):
    """
    Defining the models.
    :param seed:
    :return:
    """
    if seed == 0:
        model = torchvision.models.vgg19_bn(pretrained=True)
        model = model_adjustment(model)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4096, out_features=CLASSES),
        )
        model.to(device)
        criterion, optimizer, lrp_scheduler, mls_scheduler = model_assets(model, seed)
        return model, criterion, optimizer, lrp_scheduler, mls_scheduler

    elif seed == 1:
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = model_adjustment(model)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, out_features=CLASSES, bias=True)
        model.to(device)
        criterion, optimizer, lrp_scheduler, mls_scheduler = model_assets(model, seed)
        return model, criterion, optimizer, lrp_scheduler, mls_scheduler

    elif seed == 2:
        # model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        # model = model_adjustment(model)
        # in_features = model.classif.in_features
        # model.classif = nn.Linear(in_features, out_features=CLASSES, bias=True)
        # model.to(device)
        # criterion, optimizer, lrp_scheduler = model_assets(model, seed)
        # return model, criterion, optimizer, lrp_scheduler
        model = torchvision.models.resnet152(pretrained=True)
        model = model_adjustment(model)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features=CLASSES, bias=True)
        model.to(device)
        criterion, optimizer, lrp_scheduler = model_assets(model, seed)
        return model, criterion, optimizer, lrp_scheduler

    elif seed == 3:
        model = torchvision.models.densenet201(pretrained=True)
        model = model_adjustment(model)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, out_features=CLASSES, bias=True)
        model.to(device)
        criterion, optimizer, lrp_scheduler = model_assets(model, seed)
        return model, criterion, optimizer, lrp_scheduler

    elif seed == 4:
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model = model_adjustment(model)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features=CLASSES, bias=True)
        model.to(device)
        criterion, optimizer, lrp_scheduler = model_assets(model, seed)
        return model, criterion, optimizer, lrp_scheduler


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def create_csv(path, file_name, data_rows):
    header = ['sample', 'label']
    with open(os.path.join(path, file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)


def create_csv_soft_hard_predictions(path, file_name, data_rows):
    header = ['sample', 'hard', 'soft']
    with open(os.path.join(path, file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)


def predictions_df_unlabeled(dl_model, unlabeled_loader):
    """
    The following function returns predictions based on our two methods of Hard Voting and Soft Voting.
    class predictions are returned in case of hard voting, whereas, in the case of soft voting, probabilities of
    individual classes are returned.
    :param dl_model:
    :param unlabeled_loader:
    :return:
    """
    pred_hard, pred_soft = [], []
    dl_model.eval()
    img_names_predictions_final = []

    with torch.no_grad():
        for i, data in enumerate(unlabeled_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, img_names = data['image'].to(device), data['image_name']
            for idx in range(len(images)):
                img = images[idx]
                img = img.unsqueeze(0)
                # soft voting
                output = dl_model(img)
                sm = nn.Softmax(dim=1)
                probabilities = sm(output)
                prob_arr = (probabilities.detach().cpu().numpy())[0]
                # hard voting
                logps = dl_model(img)
                ps = torch.exp(logps)
                probab = list(ps.cpu()[0])
                pred_label = probab.index(max(probab))
                # exporting to dataframe
                pred_hard.append(pred_label)
                pred_soft.append(prob_arr)
                # process the name
                name = img_names[idx].replace('task2/val_data/', '')
                splitted_name = name.split(".")
                splitted_name[0] = f"{int(splitted_name[0]):06d}"
                name = '.'.join(splitted_name)
                img_names_predictions_final.append(name)

    return pred_hard, pred_soft, img_names_predictions_final


def batch_gd(model, criterion, optimizer, train_loader, valid_loader, epochs, seed, model_name, lrp_scheduler,
             mls_scheduler=None):
    """
    Training function.
    :param model:
    :param criterion:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :param seed:
    :param model_name:
    :param lrp_scheduler:
    :param mls_scheduler:
    :return:
    """
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    min_valid_loss = np.inf

    for it in range(epochs):
        t0 = datetime.now()
        model.train()
        train_loss = []
        train_total = 0
        train_correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            _, train_predict = torch.max(outputs.data, 1)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_total += targets.size(0)
            train_correct += (train_predict == targets).sum().item()

        else:
            model.eval()
            val_loss = []
            val_total = 0
            val_correct = 0
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, test_predict = torch.max(outputs.data, 1)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
                val_total += targets.size(0)
                val_correct += (test_predict == targets).sum().item()
            #get train and test loss
            val_loss = np.mean(val_loss)
            train_loss = np.mean(train_loss)
            #scheduler ReduceLROnPlateau and MultiStepLR
            lrp_scheduler.step(metric)
            if seed == 0 or seed == 1:
                mls_scheduler.step()
            ###
            print('learning_rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            # Save losses
            train_losses[it] = train_loss
            val_losses[it] = val_loss

            if val_loss < min_valid_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                min_valid_loss = val_loss
                # Saving State Dict
                torch.save(model.state_dict(), model_name)

            dt = datetime.now() - t0
            print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Acc: {(100*train_correct/train_total):.4f}, \
                    Test Loss: {val_loss:.4f}, Test Acc: {(100*val_correct/val_total):.4f}, Duration: {dt}')
        model.load_state_dict(best_model_wts)
    return train_losses, val_losses, model


# ------------------------------- DATA LOADING ------------------------------- #
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


def split_dataset(root, anno_file, subdir, train_file, valid_file):
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


def get_dataloaders(batches, subdir, train_file, valid_file):
    # Initialize train dataset and dataloader
    transform_train = get_transform()
    train_dataset = MyDataset(csv_file=os.path.join(subdir, train_file), root_dir=ROOT, transform=transform_train)
    valid_dataset = MyDataset(csv_file=os.path.join(subdir, valid_file), root_dir=ROOT, transform=transform_train)

    loader_train = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=0)
    loader_val = DataLoader(valid_dataset, batch_size=batches, shuffle=True, num_workers=0)

    return loader_train, loader_val


def get_transform():
    transform_train_v1 = torchvision.transforms.Compose([
        transforms.Resize(size=256, interpolation=transforms.functional.InterpolationMode('bicubic'), max_size=None,
                          antialias=None),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=([0.4850, 0.4560, 0.4060]), std=([0.2290, 0.2240, 0.2250])),
        transforms.RandomApply([AddGaussianNoise(0., 0.156)], p=0.5),
    ])

    return transform_train_v1


def get_dataloader_unlabeled_data(batches, subdir, file):
    # Initialize train dataset and dataloader
    transform = get_transform()
    unlabeled_dataset = MyDataset(csv_file=os.path.join(subdir, file), root_dir=ROOT, transform=transform)

    loader_unlabeled = DataLoader(unlabeled_dataset, batch_size=batches, shuffle=True, num_workers=0)

    return loader_unlabeled


if __name__ == "__main__":
    train_loader, validation_loader = get_dataloaders(BATCH_SIZE, SUBDIR_TRAIN, TRAIN_FILE, VALID_FILE)
    unlabeled_loader = get_dataloader_unlabeled_data(BATCH_SIZE, SUBDIR_ROOT_VAL, UNLABELED_FILE_VAL)

    # VGG19 with batch-normalization
    # model_vgg, criterion_vgg, optimizer_vgg, lrp_scheduler_vgg, mls_scheduler_vgg = model_archive(seed=0)

    # ViT (Vision Transformer)
    model_vit, criterion_vit, optimizer_vit, lrp_scheduler_vit, mls_scheduler_vit = model_archive(seed=1)

    # Ensemble Adversarial Inception-ResNet v2
    # model_eairv, criterion_eairv, optimizer_eairv, lrp_scheduler_eairv = model_archive(seed=2)

    # Resnet 152
    # model_rnt152, criterion_rnt152, optimizer_rnt152, lrp_scheduler_rnt152= model_archive(seed=2)
    # DenseNet201 classifier
    model_dn, criterion_dn, optimizer_dn, lrp_scheduler_dn = model_archive(seed=3)

    # ResNeXt50_32 Classifier
    model_rnx, criterion_rnx, optimizer_rnx, lrp_scheduler_rnx = model_archive(seed=4)

    config_vit = resolve_data_config({}, model=model_vit)
    transform_vit = create_transform(**config_vit)

    # config_eairv = resolve_data_config({}, model=model_eairv)
    # transform_eairv = create_transform(**config_eairv)

    val_transform = get_transform()

    # -------------------------------------- TRAINING -------------------------------------- #
    # print('\nResnet 152')
    # loss_train_rnt152, loss_test_rnt152, model_rnt152 = batch_gd(model_rnt152, criterion_rnt152, optimizer_rnt152, train_loader,
    #                                              validation_loader, EPOCHS, 3, 'rnt152_dict.pth', lrp_scheduler_rnt152)

    # print('Model: VGG-19-BN')
    # loss_train_vgg, loss_test_vgg, model_vgg = batch_gd(model_vgg, criterion_vgg, optimizer_vgg, train_loader, validation_loader,
    #                                          EPOCHS, 0, 'vgg_dict.pth', lrp_scheduler_vgg, mls_scheduler_vgg)
    print('\nModel: DenseNet201')
    loss_train_dn, loss_test_dn, model_dn = batch_gd(model_dn, criterion_dn, optimizer_dn, train_loader, validation_loader,
                                           EPOCHS, 2, 'dn_dict.pth', lrp_scheduler_dn)
    print('\nModel: ResNeXt50')
    loss_train_rnx, loss_test_rnx, model_rnx = batch_gd(model_rnx, criterion_rnx, optimizer_rnx, train_loader, validation_loader,
                                             EPOCHS, 4, 'rnx_dict.pth', lrp_scheduler_rnx)

    print('\nModel: Vision Transformer - ViT')
    loss_train_vit, loss_test_vit, model_vit = batch_gd(model_vit, criterion_vit, optimizer_vit, train_loader, validation_loader,
                                             EPOCHS, 1, 'vit_dict.pth', lrp_scheduler_vit, mls_scheduler_vit)

    # print('\nModel: Ensemble Adverserial Inception-Resnet-V2')
    # loss_train_eairv, loss_test_eairv, model_eairv = batch_gd(model_eairv, criterion_eairv, optimizer_eairv, train_loader,
    #                                              validation_loader, EPOCHS, 3, 'eairv_dict.pth', lrp_scheduler_eairv)


    # --------------------------------------- LOAD TRAINED MODELS ------------------------------------- #
    # model_vgg = torch.load(os.path.join(ROOT, 'submission_1/vgg_dict.pth'))
    # model_vit = model_vit.load_state_dict(torch.load(os.path.join(ROOT, 'submission_1/vit_dict.pth')))
    # model_dn = model_dn.load_state_dict(torch.load(os.path.join(ROOT, 'submission_1/dn_dict.pth')))
    # model_eairv = model_eairv.load_state_dict(torch.load(os.path.join(ROOT, 'submission_1/eairv_dict.pth')))
    # model_rnx = model_rnx.load_state_dict(torch.load(os.path.join(ROOT, 'submission_1/rnx_dict.pth')))

    # predictions for each model (pred_hard, pred_soft)
    # rnx_hard, rnt152_soft, rnt152_img_names = predictions_df_unlabeled(
    #     model_rnx, unlabeled_loader, val_transform)
    # vgg19_bn_hard, vgg19_bn_soft, vgg19_bn_img_names = predictions_df_unlabeled(model_vgg, unlabeled_loader, val_transform)
    vit_hard, vit_soft, vit_img_names = predictions_df_unlabeled(model_vit, unlabeled_loader)
    densenet201_hard, densenet201_soft, densenet201_img_names = predictions_df_unlabeled(model_dn, unlabeled_loader)
    # ensemble_adv_incres_v2_hard, ensemble_adv_incres_v2_soft, ensemble_adv_incres_v2_img_names = predictions_df_unlabeled(model_eairv, unlabeled_loader,
    #                                                                           transform_eairv)

    resnext50_hard, resnext50_soft, resnext50_img_names = predictions_df_unlabeled(model_rnx, unlabeled_loader)

    # rows_vgg = zip(vgg19_bn_img_names, vgg19_bn_hard, vgg19_bn_soft)
    rows_vit = zip(vit_img_names, vit_hard, vit_soft)
    rows_densenet = zip(densenet201_img_names, densenet201_hard, densenet201_soft)
    # rows_ensemble_adv = zip(ensemble_adv_incres_v2_img_names, ensemble_adv_incres_v2_hard, ensemble_adv_incres_v2_soft)
    # rows_rnt152 = zip(rnt152_img_names, rnt152_hard, rnt152_soft)
    rows_resnext = zip(resnext50_img_names, resnext50_hard, resnext50_soft)

    # create_csv_soft_hard_predictions(ROOT, "vgg_predictions.csv", rows_vgg)
    create_csv_soft_hard_predictions(ROOT, "vit_predictions.csv", rows_vit)
    create_csv_soft_hard_predictions(ROOT, "densenet_predictions.csv", rows_densenet)
    # create_csv_soft_hard_predictions(ROOT, "ensemble_adv.csv", rows_ensemble_adv)
    # create_csv_soft_hard_predictions(ROOT, "rnt152.csv", rows_rnt152)
    create_csv_soft_hard_predictions(ROOT, "resnext.csv", rows_resnext)
