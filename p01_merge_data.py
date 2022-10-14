#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script merges three sources of the data: a dataset on climate policies (`carbonpricing_firstyear.csv`), a dataset on carbon pricing policies (`climatepolicies_long_length.csv`), and a dataset on country characteristics (`covariates.csv`).

"""

__author__ = "Manuel Linsenmeier"
__email__ = "m.linsenmeier@lse.ac.uk"
__version__ = "0.9"

import sys
import os

import numpy as np
import pandas as pd
from ast import literal_eval

## ============================================================
## prepare data for longitudinal analyses
## ============================================================

df_policies = pd.read_csv('./data/climatepolicies_long_length.csv')
df_policies = df_policies.sort_values(by=['iso', 'year'], ascending=True, ignore_index=True)

## =============
## average length of sequence for those sectors with a price in first year of carbon pricing

df_priced = pd.read_csv('./data/carbonpricing_firstyear.csv')
df_priced['sectors'] = df_priced['sectors'].apply(lambda x: literal_eval(x)).apply(lambda x: [a.lower().replace(' ', '_').split('se_')[-1] for a in x])
sectors_priced_dict = dict(zip(df_priced['iso'].values, df_priced['sectors'].values))

df_policies['priced'] = df_policies.apply(lambda x: x['sector'] in list(sectors_priced_dict.get(x['iso'], ['transport', 'buildings', 'industry', 'electricity_and_heat_production', 'afolu'])), axis=1)
df_length = df_policies.loc[df_policies['priced'], :].groupby(['iso', 'year'])['length_sequence'].mean().reset_index()
df_length = df_length.rename(columns={0: 'length_sequence'})
df_length['carbon_pricing'] = df_policies.groupby(['iso', 'year'])['carbon_pricing'].max().values

## =============

## covariates
df_cov = pd.read_csv('./data/covariates.csv')
covariates = ['log_gdp_pc_ppp', 'coc', 'hdi', 'reserves_coal', 'reserves_gas', 'reserves_oil']
df_cov = df_cov.loc[:, ['iso', 'year'] + covariates]

## =============

## merge
df_all = df_cov.merge(df_length, on=['iso', 'year'], how='left')

df_all = df_all.loc[df_all['iso'].notnull(), :]
df_all = df_all.sort_values(by=['iso', 'year'], ascending=True, ignore_index=True)
df_all = df_all.loc[df_all['year'] >= 1988., :]

df_all.to_csv('./data/data_merged.csv', index=False)
