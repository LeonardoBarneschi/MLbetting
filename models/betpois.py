#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

outcomes = [ "1", "2", "X" ]
cols = [ "Date", "Home_Team", "Away_Team", "Home_Goals", "Away_Goals", "Result" ]

def poismatch(df, ht, at, seas=3, prevm=5, nsim=100000, nres=5):
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

    # Simulate nsim matches
    h = np.random.poisson(lam=avg_hs, size=nsim).tolist()
    a = np.random.poisson(lam=avg_as, size=nsim).tolist()

    # Get top nres results
    res = list(zip(h, a))
    freq = Counter(res)

    # Make results summary
    res = pd.DataFrame.from_dict(freq, orient='index').reset_index()
    res["Home_Goals"] = res["index"].map(lambda x: x[0])
    res["Away_Goals"] = res["index"].map(lambda x: x[1])
    res["Probability"] = res[0].map(lambda x: np.round(100 * x / float(nsim),
                                                       decimals=2))
    res = res.drop([ "index", 0 ], axis=1)
    res.loc[res["Home_Goals"] > res["Away_Goals"], "Result"] = "1"
    res.loc[res["Home_Goals"] == res["Away_Goals"], "Result"] = "X"
    res.loc[res["Home_Goals"] < res["Away_Goals"], "Result"] = "2"

    # Sort by Probability and return top nres results
    res = res.sort_values("Probability", ascending=False).reset_index(drop=True)

    return res.head(nres)


def compute_confidence(df, pred=None):

    f = df.loc[:,outcomes].apply(lambda r: r.nlargest(2).values[0], axis=1)
    s = df.loc[:,outcomes].apply(lambda r: r.nlargest(2).values[-1], axis=1)
    df.loc[:,"Diff"] = f - s

    return df


if __name__ == "__main__":

    csvf = '../data/matches.csv'
    pastcsv = pd.read_csv(csvf, index_col=0)
    pastcsv.index = pd.to_datetime(pastcsv.index)
    pastcsv = pastcsv.sort_index()

    csvf = '../data/current_matches.csv'
    currcsv = pd.read_csv(csvf, index_col=0)
    currcsv.index = pd.to_datetime(currcsv.index)
    currcsv = currcsv.sort_index()

    df = pd.concat([ pastcsv, currcsv ])

    matches = [[ "TORINO"    , "FIORENTINA" ] ,
               [ "BOLOGNA"   , "MILAN"      ] ,
               [ "SAMPDORIA" , "JUVENTUS"   ] ,
               [ "INTER"     , "BENEVENTO"  ] ,
               [ "SPEZIA"    , "UDINESE"    ] ,
               [ "ATALANTA"  , "LAZIO"      ] ,
               [ "CAGLIARI"  , "SASSUOLO"   ] ,
               [ "CROTONE"   , "GENOA"      ] ,
               [ "NAPOLI"    , "PARMA"      ] ,
               [ "ROMA"      , "HELLAS VERONA"     ]]

    preds = []
    for ht, at in matches:
        dfres = poismatch(df, ht, at, seas=4, prevm=10, nres=9)
        aggp = dfres.groupby("Result").agg({'Probability' : 'sum'}).sort_values('Probability', ascending=False).sort_index()
        topr = dfres.groupby("Result").max().stack().droplevel(0)

        # Add missing outcomes so that all matches have the same
        # number of columns
        if len(aggp) < len(outcomes):
            chunks = [ topr[p:p+len(outcomes)] for p in range(0, len(topr), len(outcomes)) ]
            ass = list(zip(aggp.index.tolist(), chunks))
            missing = pd.Index(outcomes).difference(aggp.index).tolist()
            for m in missing:
                aggp = aggp.append(pd.Series(name=m, dtype=float)).sort_index()
                new_chunk = chunks[0]
                new_chunk.loc[:] = np.nan
                ass.append((m, new_chunk))

            ass = sorted(ass, key=lambda x: x[0])
            topr = pd.concat([ x[1] for x in ass ])

        data = pd.concat([ aggp, topr ])
        d1 = data.iloc[:3,0]
        d2 = data.iloc[3:,1]
        res = pd.concat([ d1, d2 ])
        res["Prediction"] = res[outcomes].idxmax(axis=1)
        res = pd.concat([ pd.Series([ at ], index=[ "Away_Team" ]), res ])
        res = pd.concat([ pd.Series([ ht ], index=[ "Home_Team" ]), res ])
        preds.append(res.to_frame().T)

    preds = pd.concat(preds)
    preds.to_csv("preds_round_20.csv", index=False)
