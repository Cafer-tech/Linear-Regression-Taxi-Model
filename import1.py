# data
import numpy as np
import pandas as pd

# machine learning
import keras
import ml_edu.experiment
import ml_edu.results

# data visualization
import plotly.express as px

#Load the dataset

chicago_taxi_dataset=pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
def get_full_dataset():
    chicago_taxi_dataset=pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
    return chicago_taxi_dataset
    