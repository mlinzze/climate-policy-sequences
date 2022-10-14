#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script uses the empirical conditional frequencies of the previous script to derive a sequence of instrument types for a specific country/sector.

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
## function definitions
## ============================================================

def sequence(df_prob):
	cols = list(df_prob.columns)
	rows = list(df_prob.index)
	order = []
	for n in range(df_prob.shape[0]):
		counters = [0] * len(rows)
		for i, dim1 in enumerate(rows):
			for j, dim2 in enumerate([c for c in cols if c != dim1]):
				if ((dim1 in df_prob.index) and (dim2 in df_prob.columns) and 
					(dim1 in df_prob.columns) and (dim2 in df_prob.index)):
					if df_prob.loc[dim1, dim2] > df_prob.loc[dim2, dim1]:
						counters[i] += 1
		first = list(reversed([x for _, x in sorted(zip(counters, rows))]))[0]
		order.append(first)
		rows.remove(first)
	return order

## ============================================================
## by country
## ============================================================

df_probabilities = pd.read_csv('./results/probabilities_sequencing_by_country.csv')
instruments = df_probabilities['instrument_t'].unique()
countries = df_probabilities['country'].unique()

df_sequences = pd.DataFrame(index=countries, columns=['no_' + str(i) for i in range(1, len(instruments)+1, 1)])

for country in countries:
	df_prob = df_probabilities.loc[df_probabilities['country'] == country, :]
	df_prob = df_prob.pivot_table(index=['instrument_t'], columns=['instrument_tminus1'], values=['conditional_probability'], dropna=False)
	df_prob = df_prob.T
	df_prob.index = [s[1] for s in df_prob.index.values]
	df_prob = df_prob * 100.

	if df_prob.sum(axis=0).max() == 0.:
		continue
	seq = sequence(df_prob)
	print(country, seq)

	df_sequences.loc[country, :] = seq

# select countries with a national carbon price and comprehensive coverage
df_cp = pd.read_csv('./data/carbonpricing_firstyear.csv')
sample = df_cp['iso'].unique()
df_sequences = df_sequences.loc[sample, :]

df_sequences.to_csv('./results/sequences_instruments_by_country.csv')

## ============================================================
## by sector
## ============================================================

df_probabilities = pd.read_csv('./results/probabilities_sequencing_by_sector.csv')
instruments = df_probabilities['instrument_t'].unique()
sectors = df_probabilities['sector'].unique()

for sector in sectors:
	df_prob = df_probabilities.loc[df_probabilities['sector'] == sector, :]
	df_prob = df_prob.pivot_table(index=['instrument_t'], columns=['instrument_tminus1'], values=['conditional_probability'], dropna=False)
	df_prob = df_prob.T
	df_prob.index = [s[1] for s in df_prob.index.values]
	df_prob = df_prob * 100.

	instruments_order = ['policy', 'information', 'research', 'procurement', 'voluntary', 'grants', 'regulatory', 'carbon']
	df_prob = df_prob.loc[instruments_order, instruments_order]
	
	instruments_names = ['support', 'information', 'research', 'procurement', 'voluntary', 'grants', 'regulatory', 'pricing']
	df_prob.index = instruments_names
	df_prob.columns = instruments_names

	seq = sequence(df_prob)
	print(sector, seq)
	pd.DataFrame(np.array(seq), columns=['sector']).to_csv('./results/sequences_instruments_by_sector_{0:s}.csv'.format(sector.replace('_', '-')), index=False)

## ============================================================
## by country, by sector
## ============================================================

dfp = pd.read_csv('./data/climatepolicies_long_instruments.csv')

cols = dfp.columns[3:]
dfp.loc[:, cols] = dfp.groupby(['iso', 'sector'])[cols].transform(lambda x: (x.diff() > 0.).cumsum() == 1.)
for col in cols:
	dfp.loc[:, col] = dfp.loc[:, col].astype(int) * dfp['year'].values
dfm = dfp.groupby(['iso', 'sector'])[cols].max().reset_index()
dfm = dfm.replace(0., np.nan)
dfm.loc[:, cols] = dfm.loc[:, cols].rank(axis=1, method='dense')

dfm.to_csv('./results/sequences_instruments_by_country_by_sector.csv', index=False)
