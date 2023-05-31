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


def wt2at(*, wt: np.ndarray, atwt: np.ndarray) -> np.ndarray:
    # at%_1 = (wt%_1 / at_wt_1) / (wt%_1 / at_wt_1 + wt%_2 / at_wt_2)
    at1 = (wt[0] / atwt[0]) / (wt[0] / atwt[0] + wt[1] / atwt[1])
    at2 = 1 - at1
    return np.array([at1, at2])


def at2wt(*, at: np.ndarray, atwt: np.ndarray) -> np.ndarray:
    # wt%_1 = (at%_1 * at_wt_1) / (at%_1 * at_wt_1 + at%_2 * at_wt_2)
    wt1 = (at[0] * atwt[0]) / (at[0] * atwt[0] + at[1] * atwt[1])
    wt2 = 1 - wt1
    return np.array([wt1, wt2])


def calculate_atom_percent(*, wt_list, elements) -> np.ndarray:
    atwt = [
        hs.material.elements[element].General_properties["atomic_weight"]
        for element in elements
    ]
    at = wt2at(wt=wt_list, atwt=atwt)
    # print(f'{elements[0]} {at[0]:.2f} {elements[1]} {at[1]:.2f}')
    return np.array(at)
