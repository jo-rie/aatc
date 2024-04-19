# Accompanying code for the paper "The assessment of the ability of measurements, nowcasts, and forecasts to track changes"

The main files for the assessment of the ability to track changes are in the `aatc.py` file.

The code in this repository uses three external data sources; the [COVID-19-Nowcasting-Hub](https://github.com/KITmetricslab/hospitalization-nowcast-hub/), [forecasting data of patient admissions](https://github.com/bahmanrostamitabar/hourly-emergency-care), and the [MIMIC-III waveform database](https://physionet.org/content/mimic3wdb/1.0/).
The corresponding repositories for the first two have to pulled and, if necessary, the paths in the scripts have to adjusted.
The scripts starting with 'application' generate the corresponding plots in the paper. 
They are divided into functions to separate the different aspects of analysis.
The scripts starting with 'utils' contain the corresponding functions used in the application scripts.
The R script `preprocess_eda_data.R` can be copied into the 'hourly-emergency-care' folder and then extract the data from the respective `R` files.
