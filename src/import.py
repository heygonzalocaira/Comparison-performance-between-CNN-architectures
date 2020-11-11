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