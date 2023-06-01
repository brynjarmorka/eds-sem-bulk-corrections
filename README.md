# Master's thesis - Improving SEM EDS analysis

This is parts of the code developed in my master's thesis.
The code is written in Python 3.7.4 and uses the following libraries, and their dependencies:
- numpy
- pandas
- hyperspy
- plotly
- jupyterlab


This repository is for the exploration of SEM EDS bulk corrections.

One notebook (quant_of_SEM_EDS_with_Cliff-Lorimer.ipynb) is for the Cliff-Lorimer quantification method in HyperSpy, which is based on the thin film approximation (formally incorrect for bulk specimens).
In the CL quantification, absorption corrections is tested.


Another notebook (ZAF_absorption_correction_model.ipynb) is an implementation of a simple ZAF absorption correction.

The main part of this repo is directed towards XPP bulk corrections, which is a simplification of the PAP model.
The XPP corrections are used in AZtec, the EDS software from Oxford Instruments (see: https://www.oxinst.com/blogs/how-to-overcome-challenges-in-publishing-eds-results).


The XPP model is described in (PAP_algorithm.ipynb), and applied to the EDS data.
Plots from the model are given in (PAP_plots.ipynb).
The equations in the model are avaliable in Python files in the folder "PAP_functions".

The EDS data acquired in this work on GaAs and GaSb is avaliable in the folder "data/exports".