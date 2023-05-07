# Author: Brynjar Morka Mæhlum, @brynjarmorka
# Date created: May the 4th be with you, 2023

# This is a Python implementation of the XPP algorithm for bulk corrections, as described in:
# Quantitative analysis of homogeneous or stratified microvolumes applying the model "PAP"
# by Pouchou, Jean-Louis and Pichoir, Françoise, 1991

# All equations are referenced by their number in the paper in the Notes section of the docstring.

# Made as a part of my master thesis in nanotechnology at NTNU, spring 2023

###################################################################################################

# This file is for:
# F, the area of the phi(rho z) curve. This is the generated intensity.

import numpy as np

from PAP_functions.PAP_helper_functions import theoretical_energy


# (1) 1/S - deceleration factor

## (1a) Q(U) - the ionization cross section


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
    Gives the ionization cross section for a given line and nominal beam energy, Q(U).

    Parameters
    ----------
    e0 : float
        Nominal beam energy in keV.
    line : str
        X-ray line, e.g. 'Cu_Ka'. Must be in the HyperSpy database.

    Returns
    -------
    float
         Q(U), the ionization cross section.

    Notes
    -----
    It is "proportional to", but the constants of proportionality are the same for all lines.
    The PAP paper writes "Q_l^A(E_0)", where l is the level, A is the atomic number and E_0 is the nominal beam energy.
    However, writing Q(U) is more efficient.

    .. math:: Q(U) \propto ln(U) / (U^m_small * E_c^2)

    Equation (10) in the PAP-paper.
    """
    e_c = theoretical_energy(line=line)
    m_small = set_m_small(line=line)
    u = e0 / e_c
    # return np.exp(np.log(u) / (u**m_small * e_c**2))
    # I do not remember why I had the np.exp(...)
    return np.log(u) / (u**m_small * e_c**2)


### (1b) dE/drhos - energy loss for the electrons


def mean_atomic_mass_M(
    *, array_C: np.ndarray, array_Z: np.ndarray, array_A: np.ndarray
) -> float:
    """
    Calculate M, the mean atomic mass of the material, from the arrays of atomic information.

    Parameters
    ----------
    array_C : np.ndarray of floats
        The array of mass concentrations, in wt%.
    array_Z : np.ndarray of ints
        The array of atomic numbers.
    array_A : np.ndarray of floats
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
    *, array_C: np.ndarray, array_Z: np.ndarray, array_A: np.ndarray
) -> float:
    """
    Calulate J, the mean ionization potential of the specimen.

    Parameters
    ----------
    array_C : np.ndarray of floats
        Mass concentrations, in wt%.
    array_Z : np.ndarray of ints
        Atomic numbers.
    array_A : np.ndarray of floats
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
    *, array_C: np.ndarray, array_Z: np.ndarray, array_A: np.ndarray, e0: float
) -> float:
    """
    Calculate dE/drhos for a array of elements at a given beam energy.

    Parameters
    ----------
    array_C : np.ndarray of floats
        Mass concentrations, in wt%.
    array_Z : np.ndarray of ints
        Atomic numbers.
    array_A : np.ndarray of floats
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


## (2) R - backscattering loss factor

# - R is the backscatter loss factor
# - $\bar{Z}_b$ is the mean atomic number of the backscattered electrons, weighted
# - $\bar{\eta}$ is the mean backscattering coefficient

# - $\bar{W}$ is
# - $G(U_0)$ is from Coulon and Zeller (28)
# - $U_0$ is the overvoltage, $E_0/E_c$


def mean_atomic_number_Zb(*, array_C: np.ndarray, array_Z: np.ndarray) -> float:
    """
    Calculate the (weighted) mean atomic number of the backscattered electrons.

    Parameters
    ----------
    array_C : np.ndarray
        Concentrations, in wt%.
    array_Z : np.ndarray
        Atomic numbers.

    Returns
    -------
    float
        Weighted mean atomic number of the backscattered electrons.

    Notes
    -----
    Defined in Appendix 1 of the PAP-paper.
    """
    sum_Zb = 0
    for i in range(len(array_C)):
        sum_Zb += array_C[i] * array_Z[i] ** 0.5
    return sum_Zb**2


def mean_backscattering_coefficient_eta(*, zb: float) -> float:
    """
    Calculate the mean backscattering coefficient, \bar{\eta}.
    "The slight variation of the backscatter coefficient with energy has been neglected."

    Parameters
    ----------
    zb : float
        Mean atomic number.

    Returns
    -------
    float
        Mean backscattering coefficient.

    Notes
    -----
    Defined in Appendix 1 of the PAP-paper.
    """
    return 1.75e-3 * zb + 0.37 * (1 - np.exp(-0.015 * zb**1.3))


def backscattering_factor_R(
    *, array_C: np.ndarray, array_Z: np.ndarray, u0: float
) -> float:
    """
    Calculate the backscattering loss factor, R.
    "The term G(U0) is extracted from the theoretical model of Coulon and Zeller (28)."

    Parameters
    ----------
    array_C : np.ndarray
        Concentrations, in wt%.
    array_Z : np.ndarray
        Atomic numbers.
    u0 : float
        Overvoltage, keV.

    Returns
    -------
    float
        R - the backscattering loss factor.

    Notes
    -----
    Defined in Appendix 1 of the PAP-paper.

    .. math:: R = 1 - \eta \bar{W} (1 - G(U_0)))
    """
    zb = mean_atomic_number_Zb(array_C=array_C, array_Z=array_Z)
    eta = mean_backscattering_coefficient_eta(zb=zb)

    # Calculate \bar{W} (which is work? I am not sure, but it is a constant only used here)
    w = 0.595 + eta / 3.7 + eta**4.55

    # Calculate G(U0), with the help of two intermediate functions
    j_gu = 1 + u0 * (np.log(u0) - 1)
    q_gu = (2 * w - 1) / (1 - w)
    g_of_u = (u0 - 1 - (1 - 1 / u0 ** (1 + q_gu)) / (1 + q_gu)) / ((2 + q_gu) * j_gu)

    return 1 - eta * w * (1 - g_of_u)


## Putting 1/S and R together


def area_F(
    *,
    array_C: np.ndarray,
    array_Z: np.ndarray,
    array_A: np.ndarray,
    e0: float,
    line: str,
    use_q: str = "multiply",
) -> float:
    """
    Calculate the area F.
    The PAP-paper is not clear on how to use Q(U), see the notes below.

    Parameters
    ----------
    array_C : np.ndarray
        Array of concentrations, in wt%.
    array_Z : np.ndarray
        Array of atomic numbers.
    array_A : np.ndarray
        Array of atomic weights, in Da.
    e0 : float
        Beam energy, in keV.
    line : str
        Line of interest, e.g. 'Ga_La'.
    use_q : str, optional
        How to use Q(U). Options are 'multiply', 'divide', or 'ignore'.

    Returns
    -------
    float
        The area F, of the $phi(rho z)$-curve.

    Notes
    -----
    There is a slight contradictive equation.
    In equation (2) it is stated that $n_A = C_A (N^0/A) Q(U) F$, and in equation (3) it is stated that $n_A = C_A (N^0/A) (R/S)$.
    This implies that $(R/S) = Q(U) F$.
    However, equation (13) states that $F = (R/S) \cdot Q(U)$, which is a contradiction to the previous equations.

    It is assumed that Eq. (13) is correct, but the function allows for the user to choose how to use $Q(U)$.
    """
    e_c = theoretical_energy(line=line)
    u = e0 / e_c
    q = ionization_cross_section_Q(line=line, e0=e0)

    # Calculate 1/S
    small_m = set_m_small(line=line)
    big_m = mean_atomic_mass_M(array_C=array_C, array_A=array_A, array_Z=array_Z)
    j = mean_ionzation_potential_J(array_C=array_C, array_A=array_A, array_Z=array_Z)
    one_over_S = deceleration_factor_one_over_S(
        u0=u, e_c=e_c, j=j, m_big=big_m, m_small=small_m
    )

    # Calculate R
    r = backscattering_factor_R(array_C=array_C, array_Z=array_Z, u0=u)

    # Return F, depending on the use_q argument
    if use_q == "multiply":
        return r * one_over_S * q
    elif use_q == "divide":
        return r * one_over_S / q
    elif use_q == "ignore":
        return r * one_over_S
    else:
        raise ValueError(
            f'Invalid use_q argument: {use_q}, must be "multiply", "divide", or "ignore"'
        )
