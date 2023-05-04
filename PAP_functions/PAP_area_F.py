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

from PAP_functions.PAP_helper_functions import theoretical_energy


# calculating F, the area of the phi(rho z) curve

## (1) deceleration factor, 1/S

### Q_l^A, the ionization cross section


def set_m_small(*, line: str) -> float:
    """
    Gives the value of m_small for a given line.
    The value is dependent on the line type, i.e. K, L or M.

    Parameters
    ----------
    line : str
        HyperSpy line name, e.g. 'Cu_Ka'.

    Returns
    -------
    float
        Value of m_small.

    Notes
    -----
    See page 36 in the PAP-paper.
    The value of small_m is taken from the PROZA96 model for the K-level.
    Originally, in the PAP-paper m_small for K-lines is 0.86+0.12exp(-(Z_A/5)^2), but 0.9 is used here.
    """
    line_type = line.split("_")[1][0]
    if line_type == "K":
        # This value for m at a K-line is taken from the PROZA96 model.
        m_small = 0.9
    elif line_type == "L":
        m_small = 0.82
    elif line_type == "M":
        m_small = 0.78
    else:
        raise ValueError(f"Line type {line_type} in {line} not supported.")
    return m_small


def ionization_cross_section_Q(*, e0: float, line: str) -> float:
    """
    Gives the ionization cross section for a given line and nominal beam energy, Q.

    Parameters
    ----------
    e0 : float
        Nominal beam energy in keV.
    line : str
        X-ray line, e.g. 'Cu_Ka'. Must be in the HyperSpy database.

    Returns
    -------
    float
         Q_l^A(U), the ionization cross section.

    Notes
    -----
    It is "proportional to", but the constants of proportionality are the same for all lines.

    .. math:: Q_l^A(U) \propto ln(U) / (U^m_small * E_c^2)

    Equation (10) in the PAP-paper.
    """
    e_c = theoretical_energy(line=line)
    m_small = set_m_small(line=line)
    u = e0 / e_c
    # return np.exp(np.log(u) / (u**m_small * e_c**2))
    # I do not remember why I had the np.exp(...)
    return np.log(u) / (u**m_small * e_c**2)


### dE/drhos, deceleration of electrons, or average energy loss


def mean_atomic_mass_M(
    *, array_C: np.array, array_Z: np.array, array_A: np.array
) -> float:
    """
    Calculate M, the mean atomic mass of the material, from the arrays of atomic information.

    Parameters
    ----------
    array_C : np.array of floats
        The array of mass concentrations, in wt%.
    array_Z : np.array of ints
        The array of atomic numbers.
    array_A : np.array of floats
        The array of atomic masses, in Da (Dalton)

    Returns
    -------
    float
        M, the mean atomic mass of the material, in Da.

    Notes
    -----
    The units of the returned value is Da (Dalton), which is the same as the unit of the input array_A.
    I am not sure if this is correct implementation.

    Definded on page 35 in the PAP-paper.
    """
    return sum([array_C[i] * array_Z[i] / array_A[i] for i in range(len(array_C))])


def ionization_potential_Ji(*, z_i: float) -> float:
    """
    Calculate Ji, the ionization potential of element i.

    Parameters
    ----------
    z_i : float
        Atomic number of element i.

    Returns
    -------
    float
        The ionization potential of element i, in keV.

    Notes
    -----
    Equation (7) in the PAP-paper.
    """
    return z_i * 1e-3 * (10.04 + 8.25 * np.exp(-z_i / 11.22))


def mean_ionzation_potential_J(
    *, array_C: np.array, array_Z: np.array, array_A: np.array
) -> float:
    """
    Calulate J, the mean ionization potential of the specimen.

    Parameters
    ----------
    array_C : np.array of floats
        Mass concentrations, in wt%.
    array_Z : np.array of ints
        Atomic numbers.
    array_A : np.array of floats
        Atomic masses, in Da (Dalton).

    Returns
    -------
    float
        J, the mean ionization potential of the specimen, in keV.
    """
    array_Ji = []
    for z in array_Z:
        array_Ji.append(ionization_potential_Ji(z_i=z))
    m = mean_atomic_mass_M(array_C=array_C, array_Z=array_Z, array_A=array_A)
    j = 0
    for i in range(len(array_C)):
        j += array_C[i] * array_Z[i] / array_A[i] * np.log(array_Ji[i]) / m
    return np.exp(j)


def energy_dependent_terms_f_of_v(*, e: float, j: float) -> float:
    """
    Calculate f(V) in the PAP algorithm, which is the energy dependent terms in dE/drhos.
    V = E/J

    Parameters
    ----------
    e : float
        Electron energy, in keV.
    j : float
        Mean ionization potential of the material, in keV.

    Returns
    -------
    float
        f(V), the energy dependent terms in dE/drhos.
    """
    v = e / j
    return (
        6.6e-6 * v**0.78
        + 1.12e-5 * (1.35 - 0.45 * j**2) * v**0.1
        + 2.2e-6 / j * v ** (-0.5 + 0.25 * j)
    )


def energy_loss_dE_drhos(*, m: float, j: float, f_of_v: float) -> float:
    """
    Calculate dE/drhos, the energy loss of the electron beam.

    Parameters
    ----------
    m : float
        M, the mean atomic mass of the material, in Da.
    j : float
        J, the mean ionization potential of the specimen, in keV.
    f_of_v : float
        f(V), the energy dependent terms in dE/drhos.

    Returns
    -------
    float
        dE/drhos, the energy loss of the electron beam, in keV cm^2 / g.

    Notes
    -----
    Equation (5) in the PAP-paper.

    The PAP paper have a minus, but my plot is inversed with this minus
    return (- m / j / f_of_v)
    """
    return m / j / f_of_v


def whole_dE_drhos(
    *, array_C: np.array, array_Z: np.array, array_A: np.array, e0: float
) -> float:
    """
    Calculate dE/drhos for a array of elements at a given beam energy.

    Parameters
    ----------
    array_C : np.array of floats
        Mass concentrations, in wt%.
    array_Z : np.array of ints
        Atomic numbers.
    array_A : np.array of floats
        Atomic masses, in Da (Dalton).
    e0 : float
        Beam energy, in keV.

    Returns
    -------
    float
        dE/drhos, the energy loss of the electron beam, in keV cm^2 / g.
    """
    m = mean_atomic_mass_M(array_C=array_C, array_Z=array_Z, array_A=array_A)
    j = mean_ionzation_potential_J(array_C=array_C, array_Z=array_Z, array_A=array_A)
    f_of_v = energy_dependent_terms_f_of_v(e=e0, j=j)
    return energy_loss_dE_drhos(m=m, j=j, f_of_v=f_of_v)


def deceleration_factor_one_over_S(
    *, u0: float, e_c: float, j: float, m_big: float, m_small: float
) -> float:
    """
    Calculates 1/S, the deceleration factor for a given energy and material.

    Parameters
    ----------
    u0 : float
        Overvoltage
    e_c : float
        Critical ioniztion energy
    j : float
        J, mean ionization potential
    m_big : float
        M, mean atomic mass
    m_small : float
        m, K-, L-, or M-shell constant (0.9, 0.82, or 0.78)

    Returns
    -------
    float
        1/S, the deceleration factor

    Notes
    -----
    Equation (8) in the PAP-paper.
    """
    v0 = e_c / u0
    return (
        u0
        / (v0 * m_big)
        * (
            6.6e-6
            * (v0 / u0) ** 0.78
            * (
                (1 + 0.78 - m_small) * u0 ** (1 + 0.78 - m_small) * np.log(u0)
                - u0 ** (1 + 0.78 - m_small)
                + 1
            )
            / (1 + 0.78 - m_small) ** 2
            + (1.12e-5 * (1.35 - 0.45 * j**2))
            * (v0 / u0) ** 0.1
            * (
                (1 + 0.1 - m_small) * u0 ** (1 + 0.1 - m_small) * np.log(u0)
                - u0 ** (1 + 0.1 - m_small)
                + 1
            )
            / (1 + 0.1 - m_small) ** 2
            + (2.2e-6 / j)
            * (v0 / u0) ** (-0.5 + 0.25 * j)
            * (
                (1 + (-0.5 + 0.25 * j) - m_small)
                * u0 ** (1 + (-0.5 + 0.25 * j) - m_small)
                * np.log(u0)
                - u0 ** (1 + (-0.5 + 0.25 * j) - m_small)
                + 1
            )
            / (1 + (-0.5 + 0.25 * j) - m_small) ** 2
        )
    )
