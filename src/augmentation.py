from string import Template
from zipfile import ZipFile
from os import path, mkdir, makedirs
import pandas as pd
from shutil import copy
import matplotlib.pyplot as plt
import numpy as np
import Augmentor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns

plt.style.use('seaborn')

COMPETITION_NAME = "galaxy-zoo-the-galaxy-challenge"
DATA_PATH = "./data/"

# For the classification of galaxies, the dataset provided by the galaxy challenge comes with 37 classes.
#
# To reduce the number of classes, we filter the classes we want and copy each class evenly
# your proper folder. We will only use images with response rates greater than 90%.
# - completely-rounded: Class7.1
# - in-between: 7.2
# - cigar-shaped: Class7.3
# - on-edge: Class2.1
# - spiral-barred: Class3.1 && Class4.1
# - spiral: Class3.2 && Class4.1

#%% Loading csv and adjusting the dataframe
original_training_data = pd.read_csv(DATA_PATH + "training_solutions_rev1.csv")

# Pandas read GalaxyID has float, converts it back to string.
original_training_data["GalaxyID"] = original_training_data["GalaxyID"].astype(str)

# Better column naming
columns_mapper = {
    "GalaxyID": "GalaxyID",
    "Class7.1": "completely_round",
    "Class7.2": "in_between",
    "Class7.3": "cigar_shaped",
    "Class2.1": "on_edge",
    "Class4.1": "has_signs_of_spiral",
    "Class3.1": "spiral_barred",
    "Class3.2": "spiral",
}

columns = list(columns_mapper.values())
galaxies_df = original_training_data.rename(columns=columns_mapper)[columns]
galaxies_df.set_index("GalaxyID", inplace=True)
#galaxies_df.head(10)


def plot_distribution(df, column):
    print("Items: " + str(df.shape[0]))
    sns.distplot(df[column])
    plt.xlabel("% Votes")
    plt.title('Distribution - ' + column)
    plt.show()


completely_round_df = galaxies_df.sort_values(by="completely_round", ascending=False)[0:7000]
completely_round_df["type"] = "completely_round"
completely_round_df = completely_round_df[["type", "completely_round"]]

#plot_distribution(completely_round_df, "completely_round")
########################
in_between_df = galaxies_df.sort_values(by="in_between", ascending=False)[0:6000]
in_between_df["type"] = "in_between"

# filters
bigger_than_completely_round = (
    in_between_df["in_between"] > in_between_df["completely_round"]
)
bigger_than_cigar_shaped = in_between_df["in_between"] > in_between_df["cigar_shaped"]

in_between_df = in_between_df[bigger_than_completely_round & bigger_than_cigar_shaped]
in_between_df = in_between_df[["type", "in_between"]]
#plot_distribution(in_between_df, "in_between")
#######################
cigar_shaped_df = galaxies_df.sort_values(by="cigar_shaped", ascending=False)[0:1550]
cigar_shaped_df["type"] = "cigar_shaped"

# filters
bigger_than_in_between = cigar_shaped_df["cigar_shaped"] > cigar_shaped_df["in_between"]
bigger_than_on_edge = cigar_shaped_df["cigar_shaped"] > cigar_shaped_df["on_edge"]

cigar_shaped_df = cigar_shaped_df[bigger_than_in_between & bigger_than_on_edge]
cigar_shaped_df = cigar_shaped_df[["type", "cigar_shaped"]]

#plot_distribution(cigar_shaped_df, "cigar_shaped")

on_edge_df = galaxies_df.sort_values(by="on_edge", ascending=False)[0:5000]
on_edge_df["type"] = "on_edge"
on_edge_df = on_edge_df[["type", "on_edge"]]
#plot_distribution(on_edge_df, "on_edge")
#######################
spiral_barred_df = galaxies_df.sort_values(
    by=["spiral_barred", "has_signs_of_spiral"], ascending=False
)[0:4500]

spiral_barred_filter = spiral_barred_df['spiral'] < spiral_barred_df['spiral_barred']
spiral_barred_df = spiral_barred_df[spiral_barred_filter]
spiral_barred_df["type"] = "spiral_barred"
spiral_barred_df = spiral_barred_df[["type", "spiral_barred"]]
#plot_distribution(spiral_barred_df, "spiral_barred")
###########################
spiral_df = galaxies_df.sort_values(
    by=["spiral", "has_signs_of_spiral"], ascending=False
)[0:8000]
spiral_df["type"] = "spiral"
spiral_df = spiral_df[["type", "spiral"]]
#plot_distribution(spiral_df, "spiral")
##########################
#%% Generate a single dataframe with all galaxies from each class
dfs = [
    completely_round_df,
    in_between_df,
    cigar_shaped_df,
    on_edge_df,
    spiral_barred_df,
    spiral_df,
]


# Merge and drop and possible duplicates
merged_dfs = pd.concat(dfs, sort=False)
merged_dfs.reset_index(inplace=True)
merged_dfs.drop_duplicates(subset="GalaxyID", inplace=True)
#merged_dfs.head(5)


#############################################
# Split the datafrane between train and test
train_df, validation_df = train_test_split(merged_dfs, test_size=0.2)
#%% plot distribuition
def plot_info_set(df, name):
    countings = df.groupby("type").count().to_dict()["GalaxyID"]
    labels = list(countings.keys())
    values = list(countings.values())
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.title(name)
    plt.xticks(index, labels, rotation=30)
    plt.show()


#plot_info_set(train_df, "Train dataset")
#plot_info_set(validation_df, "Test dataset")


ZOOM_FACTOR=1.6
DIMEN=70
FILTERED_DATA_PATH = "/data/filtered/"
DATASETS_PATH = "/data/sets/"
#DATASETS_PATH = "/data/filtered/"

import tensorflow as tf
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

def copy_files_of_set(df, dataset):
    print("Copying filtered files of " + dataset)
    if path.isdir(FILTERED_DATA_PATH + dataset) is False:
        makedirs(FILTERED_DATA_PATH + dataset, exist_ok=True)

    src_path = Template(DATA_PATH + "images_training_rev1/$name.jpg")

    for index, image in df.iterrows():
        dest_path = FILTERED_DATA_PATH + dataset + '/' + image['type']
        source_img = src_path.substitute(name=image["GalaxyID"])

        if path.isdir(dest_path) is False:
            mkdir(dest_path)

        copy(source_img, dest_path)

copy_files_of_set(train_df, "training")
copy_files_of_set(validation_df, "validation")


def resize_and_zoom(dataset):
    p = Augmentor.Pipeline(FILTERED_DATA_PATH + dataset, DATASETS_PATH + dataset)
    p.zoom(probability=1, max_factor=ZOOM_FACTOR, min_factor=ZOOM_FACTOR)
    p.resize(probability=1, width=DIMEN, height=DIMEN)
    p.process()
    
def augment_set(n, dataset = ""):
    p = Augmentor.Pipeline(FILTERED_DATA_PATH + "training/" + dataset, DATASETS_PATH + "training/" + dataset)
    p.zoom(probability=1, max_factor=ZOOM_FACTOR, min_factor=ZOOM_FACTOR)
    p.rotate_random_90(probability=0.2)
    p.flip_top_bottom(probability=0.5)
    p.flip_left_right(probability=0.5)
    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.5)
    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.8)
    p.resize(probability=1, width=DIMEN, height=DIMEN)
    p.sample(n)

resize_and_zoom("training")
resize_and_zoom("validation")
augment_set(n = 2000)

