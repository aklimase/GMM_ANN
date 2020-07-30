# GMM_ANN


multiple setups of ANNs for creating simple ground motion prediction equations

includes hyperparameter grid search and plotting

workflow:

Compute_pga.py
•	Reads in signal seismogram, filters
•	Takes gradient for acceleration and finds the maximum value

Compile_database.py:
•	Script that takes anza record .pckl database and combines with event information from the USGS catalog
•	Matches by event id and converts magnitude
•	Write to pickle database

Create_ANN.py:
•	Contains fitANN function
•	Builds and fits model

Mixed_effects_run.py
•	Runs mixed effect maximum likelihood codes

ANN_run.py
•	Contains functions runANN and runANN_k
•	Format and scale model inputs and outputs
•	Calls residual plotting function from plot_gmpe

Grid_search.py
•	Grid search for hyperparameters for models with 1,2, or 3 hidden layers
•	Computes and saves number of hyperparameters and Akaike information criterion (AIC) for model selection

ANN_plot_top10models.py:
•	Plots top 10 AIC models for comparing behavior/functional form

Map_plot.py
•	Map plot of events and stations

Plot_gmpe.py
•	Contains plotting functions
•	plotting, plot_az,  plot_AIC, plotsiteterms, plot_dist_curves, plot_mag_curves, residual_histo, setup_test_curves, setup_test_curves_scatter, setup_curves_compare

Compare_plot.py
•	Plots site terms and model residuals for paper figures
