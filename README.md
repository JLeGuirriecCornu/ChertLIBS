# ChertLIBS
This repository hosts the framework for the preprocessing and classification of LIBS spectra develloped by the ERC SPEGEOCHERT, to reconstruct human mobility in the Pyrenees. Develloped by Julien Le Guirriec, pre-doc researcher and Jonàs Alcaina-Mateos, post-doc researcher.

Usage :
Control_card.py : standalone script to build control charts, optimised for the Z-903 SciAps handheld LIBS. Requires to be modified for paths to a reference dataset folder, and experimental dataset folder, and calls for LIBS_class.py

LIBS_class.py : backbone of the projects, contains class and functions to read, process, plot and export large quantities of LIBS spectra. 
Example usage to load several thousand spectra, remove the background, normalised them using SNV and plot them in the browser :


import plotly.io as pio
from LIBS_spectra_class import *

# render 
pio.renderers.default = "browser"

# load data
Data = LIBS_dataset()
Data.loadData("/home/spegeochert/Data/LIBS/Central_Pyrennees/MONTGAILLARD")
Data.loadData("/home/spegeochert/Data/LIBS/Central_Pyrennees/MONTSAUNES")
Data.loadData("/home/spegeochert/Data/LIBS/Central_Pyrennees/BUALA")
Data.matchMetadata(filePath="/home/spegeochert/Data/LIBS/Central_Pyrennees/Metadata.csv", matchName="N_test", getColumns = ["Jour","Type Ech","Nom","Role","Formation","# Of Loc","Cleaning Shots per Loc","Data_Shot Per Loc","# Shots to Avg","Use Gating","Intg Period","Intg Delay","Argon","Argon Flush"])

# remove the background and normalize
for i in tqdm(Data.spectra):
    i.background_removal(processing="base")
for i in tqdm(Data.spectra): 
    i.areanorm(processing='base/AirPLS', split=True) ##Split = True or False

# plot using pio
Data.plot_spectra()

Broadband_modelling.jl and Feature_modelling.jl : standalone julia script to process several classification models on a LIBS dataframe, exported from the python code
example usage :
julia Feature_modelling.jl dataframes/Peaks/peaks_AirPLS_area -p Peaks_airPLS_area -v
julia Broadband_modelling.jl dataframes/AirPLS -p AirPLS -m -w -v -r
