import pandas as pd

import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def recuperacion_descr(df):
    barrios = df["l3"].dropna().unique()
    for l in barrios:
        
        df["l_descr"] = df["description"].str.lower().str.contains()