Overview
--------

The code in this replication package conducts the statistical analysis and produces the results presented in [Linsenmeier et al. 2022](https://mlinsenmeier.com/research/). The package consists of several scripts written in Python 3 that merge datasets, apply statistical methods, and generate all figures and tables with results that are included in the manuscript and SI of the paper. The replicator should expect the code to run for about 1 hour.

The structure of this README follows the template of the Social Science Data Editors available here: [https://social-science-data-editors.github.io/template_README/](https://social-science-data-editors.github.io/template_README/) (available with a CC-BY-NC).

Data Availability and Provenance Statements
----------------------------

### Statement about Rights

The authors have legitimate access to and permission to use all data used in this project. 

### Summary of Availability

All data are publicly available.

### Details on each Data Source

The original data were obtained from the following sources:

- Climate policy data: [climatepolicydatabase.org](https://climatepolicydatabase.org) (CC BY-NC 4.0).
- Data on carbon pricing policies: [https://carbonpricingdashboard.worldbank.org/](https://carbonpricingdashboard.worldbank.org/) (CC-BY 4.0: [https://datacatalog.worldbank.org/public-licenses](https://datacatalog.worldbank.org/public-licenses)).
- Data on education from the HDI: [https://hdr.undp.org/en/data](https://hdr.undp.org/en/data) (Creative Commons Attribution 3.0 IGO: [https://hdr.undp.org/terms-use](https://hdr.undp.org/terms-use)).
- Data on GDP per capita and control of corruption: [https://data.worldbank.org/](https://data.worldbank.org/) (CC-BY 4.0: [https://datacatalog.worldbank.org/public-licenses](https://datacatalog.worldbank.org/public-licenses)).
- Data on fossil fuel reserves: [https://www.eia.gov/](https://www.eia.gov/) (Public Domain: [https://www.eia.gov/about/copyrights_reuse.php](https://www.eia.gov/about/copyrights_reuse.php)).

Computational requirements
---------------------------

### Software Requirements

- Python 3.8.4
  - `numpy` 1.19.0
  - `pandas` 1.0.5
  - `scipy` 1.5.2
  - `statsmodels` 0.11.1
  - `sklearn` 0.23.1
  - `matplotlib` 3.2.2
  - `seaborn` 0.10.0

The file `requirements.txt` lists these dependencies, please run `pip install -r requirements.txt` as the first step. See [https://pip.readthedocs.io/en/1.1/requirements.html](https://pip.readthedocs.io/en/1.1/requirements.html) for further instructions on using the `requirements.txt` file.

### Memory and Runtime Requirements

Approximate time needed to reproduce the analyses on a standard (2022) desktop machine: 1 hour.

Description of individual scripts
----------------------------

- `p01_merge_data.py`: This script merges three sources of the data: a dataset on climate policies (`carbonpricing_firstyear.csv`), a dataset on carbon pricing policies (`climatepolicies_long_length.csv`), and a dataset on country characteristics (`covariates.csv`). 
- `p02_calculate_probabilities_sequences.py`: This script calculates the empirical conditional frequencies of the first policy of a certain instrument type preceding the first policy of another instrument type, within the same country/sector. See the Methods section of the article for more details.
- `p03_derive_sequences.py`: This script uses the empirical conditional frequencies of the previous script to derive a sequence of instrument types for a specific country/sector.
- `p04_plot_sequences.py`: This script visualises the sequences of instrument types that are generated by the previous script.
- `p05_matching.py`: This script uses a matching algorithm to compare the length of climate policy sequences of countries that adopted carbon pricing in a given year with the length of the sequences of countries that did not do so; first for all sectors jointly, and then by sector.
- `p06_matching_plot.py`: This script visalises the results of the matching algorithm generated by the previous script.
- `p07_regress_adoption.py`: This script conducts logistical regressions with the adoption of carbon pricing as dependent variable and the length of climate policy sequences and country characteristics as independent variables.
- `p08_make_latextable_regression_adoption.py`: This script produces regression tables formatted with Latex from the results generated by the previous script.
- `p09_regress_carbon_pricing.py`: This script conducts a series of ordinary least squares regressions to explore heterogeneity among countries that adopted a carbon price in terms of the length of their prior climate policy sequence, the year of the adoption of carbon pricing, and the initial average carbon price.
- `p10_make_latextable_regressions_carbon_pricing.py`: This script produces regression tables formatted with Latex from the results generated by the previous script.
- `p11_plot_scatter.py`: This script visualises results generated by the script p09 as scatter plots.
- `p12_plot_histograms.py`: This script visualises the length of policy sequences at the time countries adopted carbon prices as cumulative histograms.

### License for Code

The code in this repository is licensed under a CC-BY-NC license.

Instructions to Replicators
---------------------------

- Run all scripts in the order indicated by the file names (i.e. `p01`, `p02`, `p03`, ...). This can also be achieved with the Makefile in the repository (`make clean; make all`).
- Some of the scripts store intermediate results in the folder `results`.
- Once all scripts have finished, all tables and figures can be found in the respective folders `tables` and `figures`.

List of tables and scripts
---------------------------

| Figure/Table #         | Script                   | Output file                                                               |
|------------------------|--------------------------|---------------------------------------------------------------------------|
| Figure 1 (first row)   | `p04_plot_sequences.py`  | `./figures/sequences-instruments_all.pdf`                                 |
| Figure 1 (second row)  | `p04_plot_sequences.py`  | `./figures/sequences-instruments_by-sector.pdf`                           |
| Figure 1 (third row)   | `p04_plot_sequences.py`  | `./figures/sequences-instruments_by-country.pdf`                          |
| Figure 2b              | `p06_matching_plot.py`   | `./figures/matching_result.pdf`                                           |
| Figure 2c              | `p11_plot_scatter.py`    | `./figures/scatterplot_sequence.pdf`                                      |
| Figure 2d              | `p11_plot_scatter.py`    | `./figures/scatterplot_pricelevel.pdf`                                    |
| ED Figure ED2a         | `p12_plot_histograms.py` | `./figures/histogram_policysequences_electricity_and_heat_production.pdf` |
| ED Figure ED2b         | `p12_plot_histograms.py` | `./figures/histogram_policysequences_transport.pdf`                       |
| ED Figure ED2c         | `p12_plot_histograms.py` | `./figures/histogram_policysequences_buildings.pdf`                       |
| ED Figure ED2d         | `p12_plot_histograms.py` | `./figures/histogram_policysequences_industry.pdf`                        |
| SI Table S5            | `p08_make_latextable_regression_adoption.py`        | `./tables/table_results_adoption_01.tex`       |
| SI Table S6            | `p10_make_latextable_regressions_carbon_pricing.py` | `./tables/table_results_firstyear_01.tex`      |
| SI Table S7            | `p10_make_latextable_regressions_carbon_pricing.py` | `./tables/table_results_firstyear_02.tex`      |
| SI Table S8            | `p10_make_latextable_regressions_carbon_pricing.py` | `./tables/table_results_firstyear_03.tex`      |
| SI Table S9            | `p10_make_latextable_regressions_carbon_pricing.py` | `./tables/table_results_firstyear_04.tex`      |
| SI Table S10           | `p10_make_latextable_regressions_carbon_pricing.py` | `./tables/table_results_firstyear_05.tex`      |