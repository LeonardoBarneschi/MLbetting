#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import dill as pickle
from collections import Counter, OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

outcomes = [ "1", "2", "X" ]
cols = [ "Date", "Home_Team", "Away_Team", "Home_Goals", "Away_Goals", "Result" ]

def betml_train(df, roll=5, prev=3):
    '''
    '''

    # Hyperparam 1
    df.loc[:, "Home_Goals_lastN"] = df["Home_Goals"].rolling(window=prev).mean().shift()
    df.loc[:, "Away_Goals_lastN"] = df["Away_Goals"].rolling(window=prev).mean().shift()



    g = df.groupby( ["Home_Team", "Away_Team" ] )

    # Hyperparam 2 (roll. mean)
    t = g['Home_Goals'].rolling(roll).mean().reset_index()
    T = pd.merge(df.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])
    T.index = T["Date"]
    T = T.drop("Date", axis=1)
    T = T.rename({"Home_Goals_x" : "Home_Goals", "Home_Goals_y" : "Home_Goals_hist"}, axis=1)

    t = g['Away_Goals'].rolling(roll).mean().reset_index()
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

    # Until beginning (excluded) 2019-2020 season.
    endtr = '2019-08-24'

    # Season 2019-2020
    inicv = '2019-08-25'
    endcv = '2020-08-02'

    # Current Season
    # inits = '2020-08-03'

    Xtr = X[:endtr]
    ytr = y[:endtr]
    Xcv = X[inicv:endcv]
    ycv = y[inicv:endcv]

    RF = RandomForestClassifier()

    # Set the parameters by cross-validation
    grid_values = { 'n_estimators' : np.arange(10, 61, 10),
                    'max_depth'    : np.arange(1, 11),
                  }

    grid = GridSearchCV(RF, param_grid=grid_values, n_jobs=-1)
    grid.fit(Xtr, ytr.values.reshape(-1))
    RF = RandomForestClassifier(**grid.best_params_)

    RF.fit(Xtr, ytr.values.reshape(-1))
    score = RF.score(Xcv, ycv.values.reshape(-1))
    preds = RF.predict(Xcv)

    cm = confusion_matrix(ycv, preds)
    cm = cm / cm.astype(np.float).sum(axis=1, keepdims=True)

    return score, cm.diagonal(), grid.best_estimator_

if __name__ == "__main__":

    csv1 = '../data/matches.csv'
    csv2 = '../data/current_matches.csv'

    pastcsv = pd.read_csv(csv1, index_col=0)
    pastcsv.index = pd.to_datetime(pastcsv.index)
    pastcsv = pastcsv.sort_index()

    currcsv = pd.read_csv(csv2, index_col=0)
    currcsv.index = pd.to_datetime(currcsv.index)
    currcsv = currcsv.sort_index()

    df = pd.concat([ pastcsv, currcsv ])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    roll = np.arange(2, 6)
    prev = np.arange(2, 6)

    scores = []
    estims = []
    for i in roll:
        for j in prev:
            score, cm, beste = betml_train(df, roll=j, prev=i)
            scores.append([i, j, score, cm])
            estims.append(beste)

    scores = np.array(scores)
    idx = np.argmax(scores[:, -2])
    beste = estims[idx]

    with open('betml.rf.pkl', 'wb') as f:
        pickle.dump(beste, f)

    # BEST PARAMS

    # roll = 2
    # prev = 2
    # max_depth = 8
    # n_estimators = 40

    print(*scores, sep='\n')

