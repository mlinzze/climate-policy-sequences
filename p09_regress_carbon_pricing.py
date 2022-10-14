#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script conducts a series of ordinary least squares regressions to explore heterogeneity among countries that adopted a carbon price in terms of the length of their prior climate policy sequence, the year of the adoption of carbon pricing, and the initial average carbon price.

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

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from sklearn.linear_model import LassoCV, Lasso

## ============================================================
## read in the data
## ============================================================

df_carbon = pd.read_csv('./data/carbonpricing_firstyear.csv').drop(columns=['year'])
df_length = pd.read_csv('./data/data_merged.csv')
df_all = df_carbon.merge(df_length, left_on=['iso', 'year_adoption'], right_on=['iso', 'year'], how='left')
df_all['intensity'] = df_all['price_avg'] * df_all['coverage']

## ============================================================
## regression - average price on sequences
## ============================================================

explanatory_list = ['log_gdp_pc_ppp', 'reserves_coal', 'reserves_gas', 'reserves_oil', 'coc', 'hdi']#,
treatment = 'length_sequence'
dependent = 'intensity'

all_vars = explanatory_list + [treatment, dependent, 'year_adoption']
dfr = deepcopy(df_all)
dfr[dependent] = np.log(1 + dfr[dependent])
dfr.loc[:, all_vars] = (dfr.loc[:, all_vars] - dfr.loc[:, all_vars].mean()) / dfr.loc[:, all_vars].std()

for subset in ['sample', 'sample_noCAN', 'sample_noEU']:

	if subset == 'sample':
		dfz = dfr.loc[(dfr['price_avg'] > 0.), :]
		controls_sets = ['none', 'all', 'lasso1', 'handpicked']
	elif subset == 'sample_noCAN':
		dfz = dfr.loc[(dfr['price_avg'] > 0.) & dfr['iso'] != 'CAN', :]
		controls_sets = ['all']
	elif subset == 'sample_noEU':
		dfz = dfr.loc[(dfr['price_avg'] > 0.) & (~dfr['iso'].isin(['FRA', 'GBR', 'ITA', 'DEU', 'ESP'])), :]
		controls_sets = ['none', 'handpicked']

	for controls in controls_sets:

		if controls == 'all':
			explanatory = [c for c in explanatory_list]
		elif controls == 'none':
			explanatory = []
		elif controls == 'handpicked':
			explanatory = [v for v in explanatory_list if v not in ['reserves_gas', 'coc', 'hdi']] 
		else:
			## with Lasso
			X = dfz.loc[:, explanatory].values
			y = dfz.loc[:, dependent].values
			clf = linear_model.Lasso(alpha=0.1)
			clf.fit(X, y)
			print(explanatory)
			print(clf.coef_)
			explanatory = [e for ie, e in enumerate(explanatory) if (clf.coef_ != 0.)[ie]]

			if len(explanatory) == 0:
				print('Lasso selected 0 variables, use full list of variables instead')
				explanatory = [c for c in explanatory_list]

		independent = explanatory + [treatment]
		formula = dependent + '~ 1 + ' + ' + '.join(independent)
		res = sm.OLS.from_formula(formula, dfz).fit(missing='drop').get_robustcov_results()
		dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
		dx['nobs'] = res.nobs
		dx['rsq'] = res.rsquared
		dx['rsq_adj'] = res.rsquared_adj

		if (subset == 'sample') & (controls == 'all'):
			dfn = deepcopy(dfz)
			dfn['length_sequence'] = 0.
			residuals = pd.DataFrame(dfz[dependent] - res.predict(dfn.loc[:, independent])).rename(columns={0: dependent})
			residuals['iso'] = dfz['iso'].values
			residuals['length_sequence'] = residuals['iso'].apply(lambda x: df_all.loc[df_all['iso'] == x, 'length_sequence'].values[0])
			residuals.to_csv('./results/residuals_pricelevel_{0:s}_{1:s}_length.csv'.format(subset, controls))
		dx.to_csv('./results/coeffs_pricelevel_{0:s}_{1:s}.csv'.format(subset, controls))

## ============================================================
## regress on time of adoption
## ============================================================

explanatory_list = ['log_gdp_pc_ppp', 'reserves_coal', 'reserves_gas', 'reserves_oil', 'coc', 'hdi']#,
treatment = 'year_adoption'
dependent = 'length_sequence'

for subset in ['sample', 'sample_noCAN', 'sample_noEU']:

	if subset == 'sample':
		dfz = dfr.loc[(dfr['price_avg'] > 0.), :]
		controls_sets = ['none', 'all', 'lasso1', 'handpicked']
	elif subset == 'sample_noCAN':
		dfz = dfr.loc[(dfr['price_avg'] > 0.) & dfr['iso'] != 'CAN', :]
		controls_sets = ['all']
	elif subset == 'sample_noEU':
		dfz = dfr.loc[(dfr['price_avg'] > 0.) & (~dfr['iso'].isin(['FRA', 'GBR', 'ITA', 'DEU', 'ESP'])), :]
		controls_sets = ['none', 'handpicked']

	for controls in controls_sets:

		if controls == 'all':
			explanatory = [c for c in explanatory_list]
		elif controls == 'none':
			explanatory = []
		elif controls == 'handpicked':
			explanatory = [v for v in explanatory_list if v not in ['reserves_gas', 'coc', 'hdi']] 
		else:
			## with Lasso
			X = dfz.loc[:, explanatory].values
			y = dfz.loc[:, dependent].values
			clf = linear_model.Lasso(alpha=0.1)
			clf.fit(X, y)
			print(explanatory)
			print(clf.coef_)
			explanatory = [e for ie, e in enumerate(explanatory) if (clf.coef_ != 0.)[ie]]

			if len(explanatory) == 0:
				print('Lasso selected 0 variables, use full list of variables instead')
				explanatory = [c for c in explanatory_list]

		independent = explanatory + [treatment]
		formula = dependent + '~ 1 + ' + ' + '.join(independent)
		res = sm.OLS.from_formula(formula, dfz).fit(missing='drop').get_robustcov_results()
		dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
		dx['nobs'] = res.nobs
		dx['rsq'] = res.rsquared
		dx['rsq_adj'] = res.rsquared_adj

		if (subset == 'sample') & (controls == 'all'):
			dfn = deepcopy(dfz)
			dfn['year_adoption'] = 0.
			residuals = pd.DataFrame(dfz[dependent] - res.predict(dfn.loc[:, independent])).rename(columns={0: dependent})
			residuals['iso'] = dfz['iso'].values
			residuals['year_adoption'] = residuals['iso'].apply(lambda x: df_all.loc[df_all['iso'] == x, 'year_adoption'].values[0])
			residuals.to_csv('./results/residuals_sequence_{0:s}_{1:s}_year.csv'.format(subset, controls))
		dx.to_csv('./results/coeffs_sequence_{0:s}_{1:s}.csv'.format(subset, controls))

## ============================================================
## regression - average price on sequencing - by sector
## ============================================================

df_carbon = pd.read_csv('./data/carbonpricing_firstyear.csv').drop(columns=['year', 'sectors'])

df_length = pd.read_csv('./data/climatepolicies_long_length.csv')
df_length = df_length.sort_values(by=['iso', 'year'], ascending=True, ignore_index=True)
df_length = df_length.loc[df_length['carbon_pricing'] >= 1., :]
df_length = df_length.groupby(['iso', 'sector']).first().reset_index()
df_all = df_length.merge(df_carbon, left_on=['iso'], right_on=['iso'], how='left')

df_cov = pd.read_csv('./data/covariates.csv')
covariates = ['log_gdp_pc_ppp', 'coc', 'hdi', 'reserves_coal', 'reserves_gas', 'reserves_oil']
df_cov = df_cov.loc[:, ['iso', 'year'] + covariates]

df_all = df_all.merge(df_cov, on=['iso', 'year'], how='left')
df_all['intensity'] = df_all['price_avg'] * df_all['coverage']

explanatory_list = ['log_gdp_pc_ppp', 'reserves_coal', 'reserves_gas', 'reserves_oil', 'coc', 'hdi']#,
treatment = 'length_sequence'
dependent = 'intensity'

all_vars = explanatory_list + [treatment, dependent, 'year_adoption']
dfr = deepcopy(df_all)
dfr[dependent] = np.log(1 + dfr[dependent])
dfr.loc[:, all_vars] = (dfr.loc[:, all_vars] - dfr.loc[:, all_vars].mean()) / dfr.loc[:, all_vars].std()

for sector in ['electricity_and_heat_production', 'transport', 'buildings', 'industry']:

	df_sector = dfr.loc[dfr['sector'] == sector, :]

	for subset in ['sample', 'sample_noEU']:

		if subset == 'sample':
			dfz = df_sector.loc[(df_sector['price_avg'] > 0.), :]
		elif subset == 'sample_noEU':
			dfz = df_sector.loc[(df_sector['price_avg'] > 0.) & (~df_sector['iso'].isin(['FRA', 'GBR', 'ITA', 'DEU', 'ESP'])), :]

		for controls in ['none', 'handpicked']:

			if controls == 'all':
				explanatory = [c for c in explanatory_list]
			elif controls == 'none':
				explanatory = []
			elif controls == 'handpicked':
				explanatory = [v for v in explanatory_list if v not in ['reserves_gas', 'coc', 'hdi']] 
			else:
				## with Lasso
				X = dfz.loc[:, explanatory].values
				y = dfz.loc[:, dependent].values
				clf = linear_model.Lasso(alpha=0.1)
				clf.fit(X, y)
				print(explanatory)
				print(clf.coef_)
				explanatory = [e for ie, e in enumerate(explanatory) if (clf.coef_ != 0.)[ie]]

				if len(explanatory) == 0:
					print('Lasso selected 0 variables, use full list of variables instead')
					explanatory = [c for c in explanatory_list]

			independent = explanatory + [treatment]
			formula = dependent + '~ 1 + ' + ' + '.join(independent)
			res = sm.OLS.from_formula(formula, dfz).fit(missing='drop').get_robustcov_results()
			dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
			dx['nobs'] = res.nobs
			dx['rsq'] = res.rsquared
			dx['rsq_adj'] = res.rsquared_adj

			if (subset == 'sample') & (controls == 'all'):
				dfn = deepcopy(dfz)
				dfn['length_sequence'] = 0.
				residuals = pd.DataFrame(dfz[dependent] - res.predict(dfn.loc[:, independent])).rename(columns={0: dependent})
				residuals['iso'] = dfz['iso'].values
				residuals['length_sequence'] = residuals['iso'].apply(lambda x: df_sector.loc[df_sector['iso'] == x, 'length_sequence'].values[0])
				residuals.to_csv('./results/residuals_pricelevel_{0:s}_{1:s}_{2:s}_length.csv'.format(sector, subset, controls))
			dx.to_csv('./results/coeffs_pricelevel_{0:s}_{1:s}_{2:s}.csv'.format(sector, subset, controls))

## ============================================================
## regress on time of adoption - by sector
## ============================================================

explanatory_list = ['log_gdp_pc_ppp', 'reserves_coal', 'reserves_gas', 'reserves_oil', 'coc', 'hdi']#,
treatment = 'year_adoption'
dependent = 'length_sequence'

for sector in ['electricity_and_heat_production', 'transport', 'buildings', 'industry']:

	df_sector = dfr.loc[dfr['sector'] == sector, :]

	for subset in ['sample', 'sample_noEU']:

		if subset == 'sample':
			dfz = df_sector.loc[(df_sector['price_avg'] > 0.), :]
		elif subset == 'sample_noEU':
			dfz = df_sector.loc[(df_sector['price_avg'] > 0.) & (~df_sector['iso'].isin(['FRA', 'GBR', 'ITA', 'DEU', 'ESP'])), :]

		for controls in ['none', 'handpicked']:

			if controls == 'all':
				explanatory = [c for c in explanatory_list]
			elif controls == 'none':
				explanatory = []
			elif controls == 'handpicked':
				explanatory = [v for v in explanatory_list if v not in ['reserves_gas', 'coc', 'hdi']] 
			else:
				## with Lasso
				X = dfz.loc[:, explanatory].values
				y = dfz.loc[:, dependent].values
				clf = linear_model.Lasso(alpha=0.1)
				clf.fit(X, y)
				print(explanatory)
				print(clf.coef_)
				explanatory = [e for ie, e in enumerate(explanatory) if (clf.coef_ != 0.)[ie]]

				if len(explanatory) == 0:
					print('Lasso selected 0 variables, use full list of variables instead')
					explanatory = [c for c in explanatory_list]

			independent = explanatory + [treatment]
			formula = dependent + '~ 1 + ' + ' + '.join(independent)
			res = sm.OLS.from_formula(formula, dfz).fit(missing='drop').get_robustcov_results()
			dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
			dx['nobs'] = res.nobs
			dx['rsq'] = res.rsquared
			dx['rsq_adj'] = res.rsquared_adj

			if (subset == 'sample') & (controls == 'all'):
				dfn = deepcopy(dfz)
				dfn['year_adoption'] = 0.
				residuals = pd.DataFrame(dfz[dependent] - res.predict(dfn.loc[:, independent])).rename(columns={0: dependent})
				residuals['iso'] = dfz['iso'].values
				residuals['year_adoption'] = residuals['iso'].apply(lambda x: df_sector.loc[df_sector['iso'] == x, 'year_adoption'].values[0])
				residuals.to_csv('./results/residuals_sequence_{0:s}_{1:s}_{2:s}_year.csv'.format(sector, subset, controls))
			dx.to_csv('./results/coeffs_sequence_{0:s}_{1:s}_{2:s}.csv'.format(sector, subset, controls))
