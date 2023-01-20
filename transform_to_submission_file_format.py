"""
Script used for generating the final submission files for the approach based on ensemble methods - voting.
"""
import numpy as np
import os
import pandas as pd
import operator
import csv
import glob

ROOT = '/home/bianca/Poli/Master/ATAI/Hw2/'


def get_soft_voting(df_soft_voting):
    """
    This method returns the actual prediction (the label) generated through soft voting.
    :param df_soft_voting: The dataframe containing the array of probabilities for each image.
    :return: a list of predictions.
    """
    preds = []
    for x in range(len(df_soft_voting)):
        sample = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for y in range(len(df_soft_voting.columns)):
            sample = tuple(map(operator.add, sample, (tuple(df_soft_voting.iloc[x, y]))))
        sample = tuple(ti / len(sample) for ti in sample)
        element = max(sample)
        idx = sample.index(element)
        preds.append(idx)
    return preds


def get_weighted_average(df_soft_voting):
    """
    This method returns the actual prediction (the label) generated through  weighted soft voting.
    :param df_soft_voting: The dataframe containing the array of probabilities for each image.
    :return: a list of predictions.
    """
    preds = []
    weights = [0.2315, 0.973, 0.221, 0.189, 0.123]
    for x in range(len(df_soft_voting)):
        sample = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for y in range(len(df_soft_voting.columns)):
            ##
            k = tuple(float(weights[y]) * element for element in (tuple(df_soft_voting.iloc[x,y])))
            ##
            sample = tuple(map(operator.add, sample, k))
        sample = tuple(ti/len(sample) for ti in sample)
        element = max(sample)
        idx = sample.index(element)
        preds.append(idx)
    return preds


def sort_csv(path, filename):
    """
    Method which sorts in ascending order the rows of a csv file by the number of each image.
    :param path: The path to the csv file
    :param filename: The name of the csv file
    :return:
    """
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


def voting():
    df_hard_voting = pd.DataFrame.from_dict({#'vgg19_bn': vgg19_bn_hard,
                                             'densenet201': densenet201_hard,
                                             # 'rxnt': resnext50_hard,
                                            'vit': vit_hard,
                                             # 'en_adv_incresv2': ensemble_adv_incres_v2_hard,
                                             'resnext50': resnext50_hard})

    # df_soft_voting = pd.DataFrame.from_dict({#'vit_soft': vit_soft,
    #                                         'vgg19_bn_soft': vgg19_bn_soft,
    #                                         'densenet201_soft': densenet201_soft,
    #                                          # 'en_adv_incresv2_soft': ensemble_adv_incres_v2_soft,
    #                                          'resnext50_soft': resnext50_soft})

    ensemble_hard_predictions = np.asarray(df_hard_voting.mode(axis=1)[0])

    # ensemble_soft_preds = get_soft_voting(df_soft_voting)  # list
    #
    # weighted_soft_preds = get_weighted_average(df_soft_voting)  # list

    return ensemble_hard_predictions #, ensemble_soft_preds, weighted_soft_preds


def merge_csvs(path_in, path_out, merged_file_name):
    """
    This method is used to merge two csvs. This method is used for the first task when after predicting the pseudo-labels
    for the unlabeled data, the annotations and the generated file with the pseudo-labels are merged for training the
    final model.
    :param path_in: The path to the folder which contains the csv file which need to be merged.
    :param path_out: The path to the folder where the merged file will be saved.
    :param merged_file_name: The name of the resulting csv file (together with the extension)
    :return:
    """
    csv_files = glob.glob(os.path.join(path_in, '*.{}'.format('csv')))
    df_append = pd.DataFrame()
    # append all files together
    for file in csv_files:
        df_temp = pd.read_csv(file)
        df_append = df_append.append(df_temp, ignore_index=True)
    df_append.to_csv(os.path.join(path_out, merged_file_name))


def read_csv(csv_path):
    """
    Method which is used for reading the csv files which contain the name of each image, the hard predictions and the
    soft predictions.
    :param csv_path: The path to the csv file.
    :return: Separate lists containing all the names, hard predictions and soft predictions.
    """
    soft, hard, names = [], [], []
    # The format of the files is: img_name,
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            names.append(row[0])
            hard.append(row[1])
            soft.append(row[2])
    return names, soft, hard


def create_csv(path, file_name, data_rows):
    """
    Method which saves the data in the list given as argument in a csv file.
    :param path: The path where the csv file will be created.
    :param file_name: The name of the csv file (with the extension).
    :param data_rows: A list containing lists with all the data rows which will be written in the file.
    :return:
    """
    header = ['sample', 'label']
    with open(os.path.join(path, file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)


def plot_data_distribution(img_name, anno_file):
    """
    Method which saves a figure of the label distribution from the given annotation file.
    :param img_name: The name of the image which will be saved.
    :param anno_file: The annotation file from where the data is collected.
    :return:
    """
    df = pd.read_csv(os.path.join(ROOT, anno_file))
    df['label'].value_counts().sort_values().plot(figsize=(18, 9), kind='bar').get_figure().savefig(img_name)


if __name__ == "__main__":
    # plot_data_distribution()
    # First sort the csvs
    extension = ".csv"
    task_1 = "_task1_unlabeled"
    # vgg_csv_file = "vgg_predictions"
    vit_csv_file = "vit_predictions"
    densenet_csv_file = "densenet_predictions"
    # ensemble_csv_file = "ensemble_adv"
    resnext_csv_file = "resnext"
    # rnt152_csv_file = "rnt152"

    # sort_csv(ROOT, vgg_csv_file + extension)
    sort_csv(ROOT, vit_csv_file + extension)
    sort_csv(ROOT, densenet_csv_file + extension)
    # sort_csv(ROOT, ensemble_csv_file + sorted + extension)
    sort_csv(ROOT, resnext_csv_file + extension)
    # sort_csv(ROOT, rnt152_csv_file + extension)


    # Then read them and form the submission file with hard, soft and weighted soft predictions
    # df_vgg = pd.read_csv(os.path.join(ROOT, vgg_csv_file + extension))
    # vgg19_bn_img_names, vgg19_bn_hard, vgg19_bn_soft = df_vgg["sample"].values.tolist(), df_vgg["hard"].values.tolist(), df_vgg["soft"].values.tolist()

    df_vit = pd.read_csv(os.path.join(ROOT, vit_csv_file + extension))
    vit_img_names, vit_hard, vit_soft = df_vit["sample"].values.tolist(), df_vit["hard"].values.tolist(), \
                                                       df_vit["soft"].values.tolist()

    # vgg19_bn_img_names, vgg19_bn_hard, vgg19_bn_soft = read_csv(os.path.join(ROOT, vgg_csv_file + extension))
    # vit_img_names, vit_hard, vit_soft = read_csv(os.path.join(ROOT, vit_csv_file + sorted + extension))
    df_densenet = pd.read_csv(os.path.join(ROOT, densenet_csv_file + extension))
    densenet201_img_names, densenet201_hard, densenet201_soft = df_densenet["sample"].values.tolist(), df_densenet["hard"].values.tolist(), df_densenet["soft"].values.tolist()
    # densenet201_img_names, densenet201_hard, densenet201_soft = read_csv(os.path.join(ROOT, densenet_csv_file + extension))
    # ensemble_adv_incres_v2_img_names, ensemble_adv_incres_v2_hard, ensemble_adv_incres_v2_soft = read_csv(os.path.join(ROOT, ensemble_csv_file + sorted + extension))
    df_resnext = pd.read_csv(os.path.join(ROOT, densenet_csv_file + extension))
    resnext50_img_names, resnext50_hard, resnext50_soft = df_resnext["sample"].values.tolist(), df_resnext["hard"].values.tolist(), df_resnext[
        "soft"].values.tolist()

    # df_rnt152 = pd.read_csv(os.path.join(ROOT, rnt152_csv_file + extension))
    # rnt152_img_names, rnt152_hard, rnt152_soft = df_rnt152["sample"].values.tolist(), df_rnt152[
    #     "hard"].values.tolist(), df_rnt152[ "soft"].values.tolist()
    # resnext50_img_names, resnext50_hard, resnext50_soft = read_csv(os.path.join(ROOT, resnext_csv_file + extension))

    # At this point all the img_names lists should be identical
    # Save all the predictions and create the submission files.
    final_hard_predictions = voting() #, final_soft_preds, final_soft_weighted_preds = voting()

    data_hard = []
    for pred, img in zip(final_hard_predictions, densenet201_img_names):
        data_hard.append([img, int(pred)])

    # data_soft = []
    # for pred, img in zip(final_soft_preds, vgg19_bn_img_names):
    #     data_soft.append([img, pred])
    #
    # data_weighted = []
    # for pred, img in zip(final_soft_weighted_preds, vgg19_bn_img_names):
    #     data_weighted.append([img, pred])

    create_csv(ROOT, "submission5_task2_hard.csv", data_hard)
    # # create_csv(ROOT, "submission_task2_soft.csv", data_soft)
    # # create_csv(ROOT, "submission_task2_weighted_soft.csv", data_weighted)

