import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from keras import Dense, Flatten, Layer
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from loggings import logger

data_dir_1 = 