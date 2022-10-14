#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This script conducts logistical regressions with the adoption of carbon pricing as dependent variable and the length of climate policy sequences and country characteristics as independent variables.

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

import statsmodels
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

## ============================================================
## logistic regression
## ============================================================

df_all = pd.read_csv('./data/data_merged.csv')

covariates = ['log_gdp_pc_ppp', 'coc', 'hdi', 'reserves_coal', 'reserves_gas', 'reserves_oil']
outcome = 'carbon_pricing'
treatment = 'length_sequence'

df = df_all.loc[:, covariates + [outcome] + [treatment] + ['iso', 'year']]
df = df.sort_values(by=['iso', 'year'], ascending=True, ignore_index=True)
df = df.loc[df['year'] >= 1988., :]

## =====
## with all observations
dfr = df.loc[df.isnull().sum(axis=1) == 0, :]
formula = outcome + '~ 1 + ' + ' + '.join([treatment])
res = sm.Logit.from_formula(formula, dfr).fit(method='newton', maxiter=1000, missing='drop', cov_type='cluster', cov_kwds={'groups': dfr['iso']})
dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
dx['nobs'] = res.nobs
dx['ncountries'] = dfr['iso'].unique().size
dx['AIC'] = res.aic
dx['LLR_p'] = res.llr_pvalue
dx['Prsq'] = res.prsquared
dx.to_csv('./results/coeffs_adoption_logit_nocontrol.csv')

dfr = df.loc[df.isnull().sum(axis=1) == 0, :]
formula = outcome + '~ 1 + ' + ' + '.join([treatment] + covariates)
res = sm.Logit.from_formula(formula, dfr).fit(method='newton', maxiter=1000, missing='drop', cov_type='cluster', cov_kwds={'groups': dfr['iso']})
dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
dx['nobs'] = res.nobs
dx['ncountries'] = dfr['iso'].unique().size
dx['AIC'] = res.aic
dx['LLR_p'] = res.llr_pvalue
dx['Prsq'] = res.prsquared
dx.to_csv('./results/coeffs_adoption_logit_allcontrols.csv')

dfr = df.loc[df.isnull().sum(axis=1) == 0, :]
formula = outcome + '~ 1 + ' + ' + '.join([treatment] + [c for c in covariates if c not in ['coc', 'hdi', 'reserves_oil']])
res = sm.Logit.from_formula(formula, dfr).fit(method='newton', maxiter=1000, missing='drop', cov_type='cluster', cov_kwds={'groups': dfr['iso']})
dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
dx['nobs'] = res.nobs
dx['ncountries'] = dfr['iso'].unique().size
dx['AIC'] = res.aic
dx['LLR_p'] = res.llr_pvalue
dx['Prsq'] = res.prsquared
dx.to_csv('./results/coeffs_adoption_logit_handpicked.csv')

explanatory = [treatment] + covariates
X = dfr.loc[:, explanatory].values
y = dfr.loc[:, outcome].values
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X, y)
print(explanatory)
print(clf.coef_)
explanatory_selected = [e for ie, e in enumerate(explanatory) if (clf.coef_ != 0.)[ie]]

formula = outcome + '~ 1 + ' + ' + '.join(explanatory_selected)
res = sm.Logit.from_formula(formula, dfr).fit(method='newton', maxiter=1000, missing='drop', cov_type='cluster', cov_kwds={'groups': dfr['iso']})
dx = res.summary2(float_format="%.5f").tables[1].iloc[:, [0, 1, 3]]
dx['nobs'] = res.nobs
dx['ncountries'] = dfr['iso'].unique().size
dx['AIC'] = res.aic
dx['LLR_p'] = res.llr_pvalue
dx['Prsq'] = res.prsquared
dx.to_csv('./results/coeffs_adoption_logit_lasso.csv')
