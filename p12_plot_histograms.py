#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script visualises the length of policy sequences at the time countries adopted carbon prices as cumulative histograms.

"""

__author__ = "Manuel Linsenmeier"
__email__ = "m.linsenmeier@lse.ac.uk"
__version__ = "0.9"

import sys
import os
import copy
from copy import deepcopy
from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

## ============================================================
## read in the data
## ============================================================

df_length = pd.read_csv('./data/climatepolicies_long_length.csv')
df_length = df_length.sort_values(by=['iso', 'year'], ascending=True, ignore_index=True)
df_length = df_length.loc[df_length['carbon_pricing'] >= 1., :]
df_length = df_length.groupby(['iso', 'sector']).first().reset_index()

countries = df_length['iso'].unique()
sectors = ['electricity_and_heat_production', 
		'industry', 
		'buildings', 
		'transport']
labels = ['Energy', 'Industry', 'Buildings', 'Transport']

colors = ['#EE7733', '#33BBEE', '#EE3377', '#009988', '#BBBBBB']
#https://personal.sron.nl/~pault/#sec:qualitative '#0077BB','#CC3311',

## ============================================================

for i, sector in enumerate(sectors):
	fig, ax = plt.subplots(figsize=(4,3))
	ax.hist(df_length.loc[df_length['sector'] == sector, 'length_sequence'].values, bins=range(0, 8+1, 1), color='grey', cumulative=True, label=labels[i], density=True, align='left')
	ax.set_xlabel('Length of policy sequence')
	ax.set_ylabel('Percentage of countries\n(cumulative histogram)')
	sns.despine(ax=ax, offset=1., right=True, top=True)
	ax.set_xlim(-0.6, 7.6)
	ax.set_xticks(range(0, 7+1, 1))
	ax.set_xticklabels(['[{0:1d}, {1:1d})'.format(b, b+1) for b in range(0, 7+1, 1)])
	ax.set_yticklabels(['{0:2.0f}'.format(l * 100.) for l in ax.get_yticks()])
	ax.legend(loc='upper left')
	plt.xticks(rotation = 30)
	fig.savefig('./figures/histogram_policysequences_{0:s}.pdf'.format(sector), bbox_inches='tight')
