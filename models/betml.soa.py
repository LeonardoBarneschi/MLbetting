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
feats = [ "Home_Goals_lastN", "Away_Goals_lastN", "Home_Goals_hist", "Away_Goals_hist" ]
lbls = [ "Class" ]


def preprocess_matches(df, prevm=5, prevs=3):
    '''
    '''

    # Team performance in last prevm matches at home or away

    # Get prevm matches rolling avg home goals
    ghome = df.groupby("Home_Team")
    home_goals = ghome["Home_Goals"].rolling(prevm).mean().shift().reset_index()
    df = pd.merge(df.reset_index(), home_goals.reset_index(drop=True),
                  how='inner', on=["Date", "Home_Team"])

    df.index = df["Date"]
    df = df.drop("Date", axis=1)
    df = df.rename({"Home_Goals_x" : "Home_Goals", "Home_Goals_y" : "Home_Goals_lastN"}, axis=1)

    # Get prevm matches rolling avg away goals
    gaway = df.groupby("Away_Team")
    away_goals = gaway['Away_Goals'].rolling(prevm).mean().shift().reset_index()
    df = pd.merge(df.reset_index(), away_goals.reset_index(drop=True),
                  how='inner', on=["Date", "Away_Team"])

    df.index = df["Date"]
    df = df.drop("Date", axis=1)
    df = df.rename({"Away_Goals_x" : "Away_Goals", "Away_Goals_y" : "Away_Goals_lastN"}, axis=1)

    # Team performance in last prevs face to face matches

    # Get last prevs matches rolling avg home and away goals
    g = df.groupby( ["Home_Team", "Away_Team" ] )

    t = g['Home_Goals'].rolling(prevs).mean().shift().reset_index()
    T = pd.merge(df.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])
    T.index = T["Date"]
    T = T.drop("Date", axis=1)
    T = T.rename({"Home_Goals_x" : "Home_Goals", "Home_Goals_y" : "Home_Goals_hist"}, axis=1)

    t = g['Away_Goals'].rolling(prevs).mean().shift().reset_index()
    T = pd.merge(T.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])

    T.index = T["Date"]
    T = T.drop("Date", axis=1)
    T = T.rename({"Away_Goals_x" : "Away_Goals", "Away_Goals_y" : "Away_Goals_hist"}, axis=1)

    # Fill NaN values with average
    T.loc[:, ["Home_Goals_hist", "Away_Goals_hist"]] = T.loc[:, ["Home_Goals_hist", "Away_Goals_hist"]].apply(lambda x: x.fillna(x.mean()))
    T.loc[:, ["Home_Goals_lastN", "Away_Goals_lastN"]] = T.loc[:, ["Home_Goals_lastN", "Away_Goals_lastN"]].apply(lambda x: x.fillna(x.mean()))

    # Assign classes
    T.loc[T["Result"] == "1", "Class"] = 1
    T.loc[T["Result"] == "X", "Class"] = 0
    T.loc[T["Result"] == "2", "Class"] = -1

    return T


def scan_hyperparams(df, **kwargs):
    '''
    '''

    X = df[feats]
    y = df[lbls]

    # Until beginning (excluded) 2019-2020 season.
    lastr = kwargs.pop('lastr', '2019-08-24')
    firstcv = kwargs.pop('firstcv', '2019-08-25')
    lastcv = kwargs.pop('lastcv', '2020-08-02')

    Xtr = X[:lastr]
    ytr = y[:lastr]
    Xcv = X[firstcv:lastcv]
    ycv = y[firstcv:lastcv]

    RF = RandomForestClassifier()

    # Set the hyperparams grid
    grid_values = {
            'n_estimators' : np.arange(10, 61, 10),
            'max_depth'    : np.arange(1, 11),
            }

    grid = GridSearchCV(RF, param_grid=grid_values, n_jobs=-1)
    grid.fit(Xtr, ytr.values.reshape(-1))
    RF = RandomForestClassifier(**grid.best_params_)

    RF.fit(Xtr, ytr.values.reshape(-1))
    score = RF.score(Xcv, ycv.values.reshape(-1))
    preds = RF.predict(Xcv)

    cm = confusion_matrix(ycv, preds, normalize='true')

    return score, cm.diagonal(), grid.best_estimator_


if __name__ == "__main__":

    csv1 = '../data/matches.csv'
    csv2 = '../data/current_matches_19Jan.csv'

    pastcsv = pd.read_csv(csv1, index_col=0)
    pastcsv.index = pd.to_datetime(pastcsv.index)
    pastcsv = pastcsv.sort_index()

    currcsv = pd.read_csv(csv2, index_col=0)
    currcsv.index = pd.to_datetime(currcsv.index)
    currcsv = currcsv.sort_index()

    df = pd.concat([ pastcsv, currcsv ])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    matches = np.arange(2, 11)
    seasons = np.arange(2, 6)

    scores = []
    hypers = []
    accs = []
    estims = []
    for m in matches:
        for s in seasons:
            dfpreproc = preprocess_matches(df, prevm=m, prevs=s)
            score, classacc, beste = scan_hyperparams(dfpreproc)
            scores.append(score)
            accs.append(classacc)
            estims.append(beste)
            hypers.append([ m, s ])
            print("%-3d %-3d %6.2f %6.2f" % ( m, s, score * 100, classacc[-1] * 100 ))

    scores = np.array(scores)
    accs = np.array(accs)
    idx = np.argmax(accs[:,-1])
    beste = estims[idx]
    m, s = hypers[idx]

    print()
    print("Matches and Seasons for Best Model")
    print(m, s)
    with open('betml.rf.pkl', 'wb') as f:
        pickle.dump(beste, f)
