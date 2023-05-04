# Author: Brynjar Morka Mæhlum, @brynjarmorka
# Date created: May the 4th be with you, 2023

# This is a Python implementation of the XPP algorithm, as described in:
# Quantitative analysis of homogeneous or stratified microvolumes applying the model "PAP"
# by Pouchou, Jean-Louis and Pichoir, Françoise, 1991

# Made as a part of my master thesis in nanotechnology at NTNU, spring 2023

import hyperspy.api as hs
import numpy as np
# import plotly.graph_objects as go
# import pandas as pd


# helper functions

def theoretical_energy(line: str) -> float:
    """
    Returns the theoretical energy of a given X-ray line, using HyperSpy.

    Parameters
    ----------
    line : str
        X-ray line, e.g. 'Cu_Ka'.

    Returns
    -------
    float
        Line energy in keV.
    """    
    element = line.split('_')[0]
    line_name = line.split('_')[1]
    return hs.material.elements[element]['Atomic_properties']['Xray_lines'][line_name]['energy (keV)']




