#!/usr/bin/env python3

import argparse
import pandas as pd
import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("input", help="The csv file containing the simulation output")
parser.add_argument("--tmin", type=float, default=0.0,
                    help="Data with t<tmin are ignored")
parser.add_argument("--var", type=str, nargs="*",
                    default = None,
                    help="Variables drawn in plot")
parser.add_argument("--order", type=str, nargs="*",
                    default = None,
                    help="Orders drawn in plot")
args = parser.parse_args()

print("Reading: {:}".format(args.input))

df = pd.read_csv(args.input)
df = df[df.t >= args.tmin]
if args.var is not None:
    df = df[df.variable.isin(args.var)]
if args.order is not None:
    df = df[df.order.isin(args.order)]

#TODO check if df is empty

fig = px.scatter(df, x="t",y="value",color="source",facet_row="order", facet_col="variable")
fig = px.line(df, x="t",y="value",color="source",facet_row="order", facet_col="variable")
fig.update_layout(font=dict(size=14))
# Free some space if there is an unique source
if len(pd.unique(df.source)) == 1:
    fig.update_layout(showlegend=False)
# Free 'y' scale are suited, but only when there is a single variable
if len(pd.unique(df.variable)) == 1:
    fig.update_yaxes(matches=None,title="")
fig.show()
