clean:
	rm ./data/*
	rm ./results/*
	rm ./tables/*
	rm ./figures/*

all:
	python3 ./p00a_prepare_data_policies.py
	python3 ./p00b_prepare_data_covariates.py
	python3 ./p00c_prepare_data_carbon_pricing.py
	python3 ./p00d_transform_data_policies_long.py
	python3 ./p01_merge_data.py
	python3 ./p02_calculate_probabilities_sequences.py
	python3 ./p03_derive_sequences.py
	python3 ./p04_plot_sequences.py
	python3 ./p05_matching.py
	python3 ./p06_matching_plot.py
	python3 ./p07_regress_adoption.py
	python3 ./p08_make_latextable_regression_adoption.py
	python3 ./p09_regress_carbon_pricing.py
	python3 ./p10_make_latextable_regressions_carbon_pricing.py
	python3 ./p11_plot_scatter.py
	python3 ./p12_plot_histograms.py