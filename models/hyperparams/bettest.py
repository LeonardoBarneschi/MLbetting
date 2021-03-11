#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

from betpois import poismatch

outcomes = [ "1", "2", "X" ]
cols = [ "Date", "Home_Team", "Away_Team", "Home_Goals", "Away_Goals", "Result" ]

if __name__ == '__main__':

    # Past seasons
    csvf = '../../data/matches.csv'
    pastcsv = pd.read_csv(csvf, index_col=0)
    pastcsv.index = pd.to_datetime(pastcsv.index)
    pastcsv = pastcsv.sort_index()

    # Current season
    csvf = '../../data/current_matches.csv'
    currcsv = pd.read_csv(csvf, index_col=0)
    currcsv.index = pd.to_datetime(currcsv.index)
    currcsv = currcsv.sort_index()

    # Group by Round
    grps = currcsv.groupby("Round")
    rnds = list(grps.groups.keys())

    # Get hyperparams grid
    seasons = np.arange(1, 6)
    prevms = np.arange(1, 11)
    topn = np.arange(1, 16)

    accs = []
    for i in seasons:
        for j in prevms:
            for k in topn:

                # Loop over rounds to predict their result and compare it to actual
                # results.
                comps = []
                for rnd in rnds:

                    rnd_comps = []
                    test = grps.get_group(rnd)
                    curr_seas_past = [ x for x in rnds if x < rnd ]
                    try:
                        curr_seas_data = pd.concat([ grps.get_group(grp) for grp in curr_seas_past ])
                        df = pd.concat([ pastcsv, curr_seas_data ])
                    except:
                        curr_seas_data = None
                        df = pastcsv

                    # Define matches to predict
                    matches = list(zip(test["Home_Team"], test["Away_Team"]))

                    for ht, at in matches:
                        try:
                            # Compute N most probable outcomes, group by result and compute aggregate prob
                            # dfres = poismatch(df, ht, at, seas=3, prevm=5, nres=10)
                            dfres = poismatch(df, ht, at, seas=i, prevm=j, nres=k)
                            aggp = dfres.groupby("Result").agg({
                                        'Probability' : 'sum'}).sort_values('Probability',
                                                                     ascending=False).sort_index()

                            # Get most probable result for each outcome
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
                        except:
                            continue

                        dfact = test[(test["Home_Team"] == ht) & (test["Away_Team"] == at)]
                        dfact = dfact.rename_axis("Date").reset_index()
                        dfact = dfact[cols]
                        data = pd.concat([ dfact.T.squeeze(), res ]).to_frame().T
                        data.index = data["Date"]
                        data = data.drop("Date", axis=1)
                        comps.append(data)

                comps = pd.concat(comps)
                comps.to_csv("predictions_%d_%d_%d.csv" % ( i, j, k ))
                t = comps[comps["Prediction"] == comps["Result"]]
                acc = 100 * float(len(t)) / len(comps)
                accs.append([ i, j, k, acc ])
                print("%3d %3d %3d %6.2f" % (i, j, k, acc))

    accs = pd.DataFrame({
        "NSeasons" : [ x[0] for x in accs ],
        "NMatches" : [ x[1] for x in accs ],
        "NTop" : [ x[2] for x in accs ],
        "Acc" : [ x[3] for x in accs ]
        })
    
    accs.to_csv("hyperparams.csv", index=False)
