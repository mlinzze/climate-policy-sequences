#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script uses a matching algorithm to compare the length of climate policy sequences of countries that adopted carbon pricing in a given year with the length of the sequences of countries that did not do so; first for all sectors jointly, and then by sector.

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

## ============================================================
## set params
## ============================================================

n_montecarlo = 1000

## ============================================================
## prepare data for longitudinal analyses
## ============================================================

df_policies = pd.read_csv('./data/climatepolicies_long_length.csv')
df_policies = df_policies.sort_values(by=['iso', 'year'], ascending=True, ignore_index=True)

def length_of_sequence(df, country, year, sectors):
	if sectors == 'all':
		return df.loc[(df['iso'] == country) & (df['year'] == year), 'length_sequence'].mean()
	else:
		return df.loc[(df['iso'] == country) & (df['year'] == year) & df['sector'].isin(sectors), 'length_sequence'].mean()

## sectors with a price in first year
df_priced = pd.read_csv('./data/carbonpricing_firstyear.csv')
df_priced['sectors'] = df_priced['sectors'].apply(lambda x: literal_eval(x)).apply(lambda x: [a.lower().replace(' ', '_').split('se_')[-1] for a in x])
sectors_priced_dict = dict(zip(df_priced['iso'].values, df_priced['sectors'].values))

## countries with carbon pricing
df_carbon = df_policies.groupby(['iso', 'year'])['carbon_pricing'].max().reset_index()

## ========================================
## same with randomly selected controls

df_firstyear = pd.read_csv('./data/carbonpricing_firstyear.csv')
treatments = df_firstyear.loc[:, ['iso', 'year_adoption']].set_index('iso').rename(columns={'year_adoption': 'year'})

bt_results = pd.DataFrame()
bt_results['average_treated'] = np.zeros(n_montecarlo)
bt_results['average_control'] = np.zeros(n_montecarlo)
bt_results['ATE'] = np.zeros(n_montecarlo)
for i, bt in enumerate(bt_results.index.values):

	treatments['control'] = np.nan
	treatments['length_treated'] = np.nan
	treatments['length_control'] = np.nan

	for j, iso in enumerate(treatments.index):
		year = treatments.loc[iso].values[0]
		dfs = df_carbon.loc[df_carbon['year'] == year, :]
		controls = dfs.loc[(dfs['iso'] != iso) & (dfs['carbon_pricing'] != 1.), :]
		treatments.loc[iso, 'control'] = controls['iso'].sample(1, random_state=(i*j)).values[0]

	for iso in treatments.index.values:
		year = treatments.loc[iso, 'year']
		control = treatments.loc[iso, 'control']
		sectors = sectors_priced_dict[iso]
		treatments.loc[iso, 'length_treated'] = length_of_sequence(df_policies, iso, year, sectors)
		treatments.loc[iso, 'length_control'] = length_of_sequence(df_policies, control, year, sectors)

	bt_results.loc[i, 'average_treated'] = treatments['length_treated'].mean()
	bt_results.loc[i, 'average_control'] = treatments['length_control'].mean()
	bt_results.loc[i, 'ATE'] = treatments['length_treated'].mean() - treatments['length_control'].mean()

te_random = bt_results['ATE'].mean()
print(te_random)
bt_results.to_csv('./results/matching_random_ATE_priced.csv')

## ========================================
## bootstrapping for CI

bt_results = pd.DataFrame()
bt_results['ATE'] = np.zeros(n_montecarlo)
treatments_bootstrap = pd.DataFrame()
for i, bt in enumerate(bt_results.index.values):
	
	## treatments_bootstrap mirrors treatments data frame, but randomly chosen countries
	nsample = treatments.shape[0]
	treatments_bootstrap['iso'] = dfs['iso'].sample(nsample, random_state=i).values
	treatments_bootstrap['year'] = treatments['year'].sample(nsample, random_state=i).values
	treatments_bootstrap['control'] = np.nan
	treatments_bootstrap['length_treated'] = np.nan
	treatments_bootstrap['length_control'] = np.nan
	treatments_bootstrap = treatments_bootstrap.set_index('iso')

	for j, iso in enumerate(treatments_bootstrap.index.values):
		year = treatments_bootstrap.loc[iso, 'year']
		dfs = df_carbon.loc[df_carbon['year'] == year, :]
		treatments_bootstrap.loc[iso, 'control'] = dfs.loc[(dfs['iso'] != iso), 'iso'].sample(1, random_state=(i*j)).values[0]

	for iso in treatments_bootstrap.index.values:
		year = treatments_bootstrap.loc[iso, 'year']
		control = treatments_bootstrap.loc[iso, 'control']
		sectors = sectors_priced_dict[df_priced['iso'].sample(1).values[0]]
		treatments_bootstrap.loc[iso, 'length_treated'] = length_of_sequence(df_policies, iso, year, sectors)
		treatments_bootstrap.loc[iso, 'length_control'] = length_of_sequence(df_policies, control, year, sectors)

	bt_results.loc[i, 'ATE'] = treatments_bootstrap['length_treated'].mean() - treatments_bootstrap['length_control'].mean()

ci_lower99 = bt_results['ATE'].quantile(0.005)
ci_upper99 = bt_results['ATE'].quantile(0.995)

ci_lower95 = bt_results['ATE'].quantile(0.025)
ci_upper95 = bt_results['ATE'].quantile(0.975)

ci_lower90 = bt_results['ATE'].quantile(0.05)
ci_upper90 = bt_results['ATE'].quantile(0.95)
bt_results.to_csv('./results/matching_random_ATE-placebo_priced.csv')

## ==============================================
## results by sector
## ==============================================

for sector in ['transport', 'buildings', 'industry', 'electricity_and_heat_production']:

	print(sector)

	df_firstyear = pd.read_csv('./data/carbonpricing_firstyear.csv')
	treatments = df_firstyear.loc[:, ['iso', 'year_adoption']].rename(columns={'year_adoption': 'year'})
	treatments = treatments.loc[treatments['iso'].apply(lambda x: sector in sectors_priced_dict[x]), :].set_index('iso')

	bt_results = pd.DataFrame()
	bt_results['average_treated'] = np.zeros(n_montecarlo)
	bt_results['average_control'] = np.zeros(n_montecarlo)
	bt_results['ATE'] = np.zeros(n_montecarlo)
	for i, bt in enumerate(bt_results.index.values):

		treatments['control'] = np.nan
		treatments['length_treated'] = np.nan
		treatments['length_control'] = np.nan

		for j, iso in enumerate(treatments.index):
			year = treatments.loc[iso].values[0]
			dfs = df_carbon.loc[df_carbon['year'] == year, :]
			controls = dfs.loc[(dfs['iso'] != iso) & (dfs['carbon_pricing'] != 1.), :]
			treatments.loc[iso, 'control'] = controls['iso'].sample(1, random_state=(i*j)).values[0]

		for iso in treatments.index.values:
			year = treatments.loc[iso, 'year']
			control = treatments.loc[iso, 'control']
			treatments.loc[iso, 'length_treated'] = length_of_sequence(df_policies, iso, year, [sector])
			treatments.loc[iso, 'length_control'] = length_of_sequence(df_policies, control, year, [sector])

		bt_results.loc[i, 'average_treated'] = treatments['length_treated'].mean()
		bt_results.loc[i, 'average_control'] = treatments['length_control'].mean()
		bt_results.loc[i, 'ATE'] = treatments['length_treated'].mean() - treatments['length_control'].mean()

	te_random = bt_results['ATE'].mean()
	print(te_random)
	bt_results.to_csv('./results/matching_random_ATE_{0:s}.csv'.format(sector))

	## ========================================
	## bootstrapping for CI

	bt_results = pd.DataFrame()
	bt_results['ATE'] = np.zeros(n_montecarlo)
	treatments_bootstrap = pd.DataFrame()
	for i, bt in enumerate(bt_results.index.values):
		
		## treatments_bootstrap mirrors treatments data frame, but randomly chosen countries
		nsample = treatments.shape[0]
		treatments_bootstrap['iso'] = dfs['iso'].sample(nsample, random_state=i).values
		treatments_bootstrap['year'] = treatments['year'].sample(nsample, random_state=i).values
		treatments_bootstrap['control'] = np.nan
		treatments_bootstrap['length_treated'] = np.nan
		treatments_bootstrap['length_control'] = np.nan
		treatments_bootstrap = treatments_bootstrap.set_index('iso')

		for j, iso in enumerate(treatments_bootstrap.index.values):
			year = treatments_bootstrap.loc[iso, 'year']
			dfs = df_carbon.loc[df_carbon['year'] == year, :]
			treatments_bootstrap.loc[iso, 'control'] = dfs.loc[(dfs['iso'] != iso), 'iso'].sample(1, random_state=(i*j)).values[0]

		for iso in treatments_bootstrap.index.values:
			year = treatments_bootstrap.loc[iso, 'year']
			control = treatments_bootstrap.loc[iso, 'control']
			treatments_bootstrap.loc[iso, 'length_treated'] = length_of_sequence(df_policies, iso, year, [sector])
			treatments_bootstrap.loc[iso, 'length_control'] = length_of_sequence(df_policies, control, year, [sector])

		bt_results.loc[i, 'ATE'] = treatments_bootstrap['length_treated'].mean() - treatments_bootstrap['length_control'].mean()

	ci_lower99 = bt_results['ATE'].quantile(0.005)
	ci_upper99 = bt_results['ATE'].quantile(0.995)

	ci_lower95 = bt_results['ATE'].quantile(0.025)
	ci_upper95 = bt_results['ATE'].quantile(0.975)

	ci_lower90 = bt_results['ATE'].quantile(0.05)
	ci_upper90 = bt_results['ATE'].quantile(0.95)
	bt_results.to_csv('./results/matching_random_ATE-placebo_{0:s}.csv'.format(sector))
