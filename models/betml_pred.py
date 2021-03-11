#!/usr/bin/env python
# coding: utf-8

import dill as pickle
import sys
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')

def betml_pred(df, roll, prev):
    '''
    '''

    # Hyperparam 1
    df.loc[:, "Home_Goals_lastN"] = df["Home_Goals"].rolling(window=prev).mean().shift()
    df.loc[:, "Away_Goals_lastN"] = df["Away_Goals"].rolling(window=prev).mean().shift()
    print(df.tail(70))

    g = df.groupby( ["Home_Team", "Away_Team" ] )

    # Hyperparam 2 (roll. mean)
    t = g['Home_Goals'].rolling(roll).mean().shift().reset_index()
    T = pd.merge(df.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])
    T.index = T["Date"]
    T = T.drop("Date", axis=1)
    T = T.rename({"Home_Goals_x" : "Home_Goals", "Home_Goals_y" : "Home_Goals_hist"}, axis=1)

    t = g['Away_Goals'].rolling(roll).mean().shift().reset_index()
    T = pd.merge(T.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])

    # Set Date as index and drop Date column
    T.index = T["Date"]
    T = T.drop("Date", axis=1)
    T = T.rename({"Away_Goals_x" : "Away_Goals", "Away_Goals_y" : "Away_Goals_hist"}, axis=1)

    T.loc[T["Result"] == "1", "Class"] = 1
    T.loc[T["Result"] == "X", "Class"] = 0
    T.loc[T["Result"] == "2", "Class"] = -1

    T.loc[:, ["Home_Goals_hist", "Away_Goals_hist"]] = T.loc[:, ["Home_Goals_hist", "Away_Goals_hist"]].apply(lambda x: x.fillna(x.mean()))
    T.loc[:, ["Home_Goals_lastN", "Away_Goals_lastN"]] = T.loc[:, ["Home_Goals_lastN", "Away_Goals_lastN"]].apply(lambda x: x.fillna(x.mean()))
    X = T[ [ "Home_Goals_lastN", "Away_Goals_lastN", "Home_Goals_hist", "Away_Goals_hist" ] ]
    y = T[ [ "Class" ] ]

    # Final
    endtr = '2021-01-19'
    inits = '2021-01-20'

    Xtr = X[:endtr]
    ytr = y[:endtr]
    Xts = X[inits:]
    yts = y[inits:]

    # Set the parameters by cross-validation
    with open('betml.rf.pkl','rb') as f:
        RF = pickle.load(f)

    RF.fit(Xtr, ytr.values.reshape(-1))
    preds = RF.predict(Xts)

    # For printing
    yts.loc[:, 'Pred'] = preds
    ytr.loc[:, 'Pred'] = np.NaN
    df.loc[:, 'Pred'] = pd.concat( [ ytr, yts ] )['Pred']

    tmpdf = df[ ["Home_Team", "Away_Team", "Result", "Pred"] ]
    print(tmpdf[-30:])

    return

if __name__ == "__main__":

    csv1 = '../data/matches.csv'
    csv2 = '../data/current_matches_19Jan.csv'
    csv3 = 'data.csv'

    pastcsv = pd.read_csv(csv1, parse_dates=True, dayfirst=True, index_col=0)
    pastcsv.index = pd.to_datetime(pastcsv.index)
    pastcsv = pastcsv.sort_index()

    currcsv = pd.read_csv(csv2, parse_dates=True, dayfirst=True, index_col=0)
    currcsv.index = pd.to_datetime(currcsv.index)
    currcsv = currcsv.sort_index()

    predcsv = pd.read_csv(csv3, parse_dates=True, dayfirst=True, index_col=0)
    predcsv.index = pd.to_datetime(predcsv.index)
    predcsv = predcsv.sort_index()

    df = pd.concat([ pastcsv, currcsv, predcsv ])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    scores = []
    scores.append(betml_pred(df, 2, 2))
