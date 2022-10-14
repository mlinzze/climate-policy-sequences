#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script calculates the empirical conditional frequencies of the first policy of a certain instrument type preceding the first policy of another instrument type, within the same country/sector. See the Methods section of the article for more details.

"""

__author__ = "Manuel Linsenmeier"
__email__ = "m.linsenmeier@lse.ac.uk"
__version__ = "0.9"

import sys
import os
import copy

import numpy as np
import pandas as pd
import itertools

## ============================================================
## read in the data
## ============================================================

df = pd.read_csv('./data/climatepolicies_long_instruments.csv')

## ============================================================
## sequencing regarding instruments
## ============================================================

df = df.sort_values(by=['iso', 'year'], ascending=True)

instrument_columns = df.columns[3:]

# cumulative count of policies
# ... that is binary
for col in instrument_columns:
	df[col] = df.groupby(['iso', 'sector'])[col].transform('cumsum')
	df[col] = (df[col] >= 1.).astype(float)

# shorten column names
df.columns = list(df.columns.values[:3]) + [col.split('_')[0] for col in df.columns.values[3:]]

# select countries with a national carbon price and comprehensive coverage
df_cp = pd.read_csv('./data/carbonpricing_firstyear.csv')
sample = df_cp['iso'].unique()

## ============================================================
## by country
## ============================================================

cols = df.columns[3:]
df_results = pd.DataFrame()

nfail = 0
failed = []
for country in sample:

	# col is instrument type A
	for col in cols:

		dfr = df.loc[df['iso'] == country, :]

		# lag all other columns by at least one
		explanatory = [c for c in cols if c != col]

		# ex is instrument type B
		for ex in explanatory:
			dfr[ex] = dfr.groupby(['iso', 'sector'])[ex].shift(1)

		# drop all rows after first adoption of instrument type A
		dfr = dfr.loc[dfr[col] <= 1., :] # not required anymore because of binary format of column
		indices_drop = (dfr[col] == 1.) & dfr.duplicated(subset=['iso', 'sector', col], keep='first')
		dfr = dfr.loc[~indices_drop, :]
		dfr = dfr.dropna()

		dx = {}
		for ex in explanatory:
			n_yes_yes = dfr.groupby(['sector']).apply(lambda x: (x[col].max() == 1.) & (x[ex].max() > 0.)).sum()
			n_any_yes = dfr.groupby(['sector']).apply(lambda x: x[ex].max() > 0.).sum()
			n_yes_any = dfr.groupby(['sector']).apply(lambda x: x[col].max() == 1.).sum()
			dx[ex] = n_yes_yes / n_yes_any

		dx = pd.DataFrame.from_dict(dx, orient='index')
		dx.columns = ['conditional_probability']
		dx['dependent'] = col
		dx['country'] = country

		df_results = df_results.append(dx)

df_results = df_results.reset_index()
df_results.columns = ['instrument_tminus1', 'conditional_probability', 'instrument_t', 'country']
df_results = df_results.loc[:, ['country', 'instrument_t', 'instrument_tminus1', 'conditional_probability']]
df_results.to_csv('./results/probabilities_sequencing_by_country.csv', index=False)

## ============================================================
## by sector
## ============================================================

sectors = ['all'] + list(df['sector'].unique())

cols = df.columns[3:]
df_results = pd.DataFrame()

nfail = 0
failed = []
for sector in sectors:

	# col is instrument type A
	for col in cols:

		if sector == 'all':
			dfr = copy.deepcopy(df)
		else:
			dfr = copy.deepcopy(df.loc[df['sector'] == sector, :])

		# lag all other columns by at least one
		explanatory = [c for c in cols if c != col]

		# ex is instrument type B
		for ex in explanatory:
			dfr[ex] = dfr.groupby(['iso', 'sector'])[ex].shift(1)

		# drop all rows after first adoption of instrument type A
		dfr = dfr.loc[dfr[col] <= 1., :] # not required anymore because of binary format of column
		indices_drop = (dfr[col] == 1.) & dfr.duplicated(subset=['iso', 'sector', col], keep='first')
		dfr = dfr.loc[~indices_drop, :]
		dfr = dfr.dropna()

		dx = {}
		for ex in explanatory:
			n_yes_yes = dfr.groupby(['iso', 'sector']).apply(lambda x: (x[col].max() == 1.) & (x[ex].max() > 0.)).sum()
			n_any_yes = dfr.groupby(['iso', 'sector']).apply(lambda x: x[ex].max() > 0.).sum()
			n_yes_any = dfr.groupby(['iso', 'sector']).apply(lambda x: x[col].max() == 1.).sum()
			dx[ex] = n_yes_yes / n_yes_any

		dx = pd.DataFrame.from_dict(dx, orient='index')
		dx.columns = ['conditional_probability']
		dx['instrument_t'] = col
		dx['sector'] = sector
		dx = dx.reset_index().rename(columns={'index': 'instrument_tminus1'})

		df_results = df_results.append(dx)

df_results = df_results.loc[:, ['sector', 'instrument_t', 'instrument_tminus1', 'conditional_probability']]
df_results.to_csv('./results/probabilities_sequencing_by_sector.csv', index=False)
