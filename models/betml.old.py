#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

outcomes = [ "1", "2", "X" ]
cols = [ "Date", "Home_Team", "Away_Team", "Home_Goals", "Away_Goals", "Result" ]

def mlmatch(df, ht, at, seas=3, prevm=5):
    '''
    Predict event final score (football) using a Poisson
    distribution model based on average scored/conceded
    gol of host and away team.

    Parameters
    ----------
    df: Pandas.DataFrame.
        past matches data.
    ht: str.
        Name of the home team.
    at: str
        Name of the away team.
    seas: int (default: 5).
        Number of seasons to take into
        account starting from present.
    prevm: int (default: 5).
        Previous matches to take into account.
    nsim: float  (default: 1e5).
        Number of simulated matches.
        An high number is required to better
        approximate the Poisson distribution.
    nres: int (default: 5).
        Number of results to include in the
        output dictionary.

    Returns
    -------
    res: Pandas.DataFrame.
        data frame of top nres results.
    '''

    # Sort by date
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Subset of past seasons matches between HT and AT
    dfhist = df[(df["Home_Team"] == ht) & (df["Away_Team"] == at)].tail(seas)

    # Subset of last home matches of HT
    dfh = df[df["Home_Team"] == ht].tail(prevm)

    # Subset of last away matches of AT
    dfa = df[df["Away_Team"] == at].tail(prevm)

    # Concat history with last matches
    dfh = pd.concat([ dfhist, dfh ]).drop_duplicates()
    dfa = pd.concat([ dfhist, dfa ]).drop_duplicates()

    # compute average scored goals (full time)
    avg_hs = dfh["Home_Goals"].mean()
    avg_as = dfa["Away_Goals"].mean()
    X = np.array( [avg_hs, avg_as] )
    y = dfhist["Result"].tail(1)

    return X, y



def betml(csvf):

    #csvf = '../data/matches.csv'
    pastcsv = pd.read_csv(csvf, index_col=0)
    pastcsv.index = pd.to_datetime(pastcsv.index)
    pastcsv = pastcsv.sort_index()

    #csvf = '../data/current_matches.csv'
    currcsv = pd.read_csv(csvf, index_col=0)
    currcsv.index = pd.to_datetime(currcsv.index)
    currcsv = currcsv.sort_index()

    df = pd.concat([ pastcsv, currcsv ])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Hyperparam 1
    df.loc[:, "Home_Goals_lastN"] = df["Home_Goals"].rolling(window=3).mean()
    df.loc[:, "Away_Goals_lastN"] = df["Away_Goals"].rolling(window=3).mean()


    g = df.groupby( ["Home_Team", "Away_Team" ] )

    # Hyperparam 2
    t = g['Home_Goals'].rolling(5).mean().reset_index()
    T = pd.merge(df.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])
    T.index = T["Date"]
    T = T.drop("Date", axis=1)
    T = T.rename({"Home_Goals_x" : "Home_Goals", "Home_Goals_y" : "Home_Goals_hist"}, axis=1)

    t = g['Away_Goals'].rolling(5).mean().reset_index()
    T = pd.merge(T.reset_index(), t.reset_index(drop=True), how='inner', on=["Date", "Home_Team", "Away_Team"])
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

    Xtr = X[:3500]
    ytr = y[:3500]
    Xts = X[3500:]
    yts = y[3500:]

    print(Xtr)
    RF = RandomForestClassifier()
    RF.fit(Xtr, ytr.values.reshape(-1))
    score = RF.score(Xts, yts.values.reshape(-1))
    preds = RF.predict(Xts)

    #print(np.where(yts == 1)[0].shape)
    #print(np.where(yts == 0)[0].shape)
    #print(np.where(yts ==-1)[0].shape)
    #print()
    cm = confusion_matrix(yts, preds)
    #print(cm)
    #sys.exit()

    #matches = [
    #        [ "LAZIO", "ROMA" ],
    #        [ "BOLOGNA", "HELLAS VERONA" ],
    #        [ "TORINO", "SPEZIA" ],
    #        [ "SAMPDORIA", "UDINESE" ],
    #        [ "NAPOLI", "FIORENTINA" ],
    #        [ "CROTONE", "BENEVENTO" ],
    #        [ "SASSUOLO", "PARMA" ],
    #        [ "ATALANTA", "GENOA" ],
    #        [ "INTER", "JUVENTUS" ],
    #        [ "CAGLIARI", "MILAN" ]
    #    ]

    #matches = [
    #        [ "FIORENTINA", "INTER" ],
    #        [ "NAPOLI", "EMPOLI" ],
    #        [ "JUVENTUS", "GENOA" ],
    #        [ "SASSUOLO", "SPAL" ],
    #        [ "ATALANTA", "CAGLIARI" ],
    #    ]

    #mlmatch(df, "LAZIO", "ROMA")
    #print(df.iloc[:,:6])

if __name__ == "__main__":

    csv1 = '../data/matches.csv'
    csv2 = '../data/current_matches.csv'

