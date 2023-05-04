# Author: Brynjar Morka Mæhlum, @brynjarmorka
# Date created: May the 4th be with you, 2023

# This is a Python implementation of the XPP algorithm for bulk corrections, as described in:
# Quantitative analysis of homogeneous or stratified microvolumes applying the model "PAP"
# by Pouchou, Jean-Louis and Pichoir, Françoise, 1991

# All equations are referenced by their number in the paper, e.g. (1) for equation 1.
# The functions are adapted to work with HyperSpy, a Python library for analysis of multi-dimensional data.

# Made as a part of my master thesis in nanotechnology at NTNU, spring 2023

###################################################################################################

import hyperspy.api as hs
import numpy as np


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
    element = line.split("_")[0]
    line_name = line.split("_")[1]
    return hs.material.elements[element]["Atomic_properties"]["Xray_lines"][line_name][
        "energy (keV)"
    ]


def get_C_A_Z_arrays(
    *, elements: list, concentrations: list, concentration_type: str = "wt"
) -> tuple:
    """
    Give arrays of C_i, A_i and Z_i for a list of elements and concentrations.

    Parameters
    ----------
    elements : list of str
        List of elements, e.g. ['Ga', 'As'].
    concentrations : list
        List of concentrations, in either wt% or at%, e.g. [0.5, 0.5].
    concentration_type : str, optional
        Specifying wt% or at%, by default 'wt'. Either 'wt' or 'at'.

    Returns
    -------
    Tuple of three arrays
        List of C_i, A_i and Z_i for the specimen.
    """
    # """Returns a list of C, A, and Z for a list of elements and concentrations"""
    if concentration_type == "wt":
        list_C = concentrations
    elif concentration_type == "at":
        list_C = (
            hs.material.atomic_to_weight(
                atomic_percent=concentrations, elements=elements
            )
            / 100
        )
    else:
        raise ValueError(
            f'Concentration type {concentration_type} not supported. Use either "wt" or "at".'
        )
    list_Z = []
    list_A = []
    for element in elements:
        list_Z.append(hs.material.elements[element].General_properties.Z)
        list_A.append(hs.material.elements[element].General_properties.atomic_weight)
    return np.array(list_C), np.array(list_A), np.array(list_Z)
