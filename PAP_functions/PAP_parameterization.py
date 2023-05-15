# Author: Brynjar Morka Mæhlum, @brynjarmorka
# Date created: 7th of May, 2023

# This is a Python implementation of the XPP algorithm for bulk corrections, as described in:
# Quantitative analysis of homogeneous or stratified microvolumes applying the model "PAP"
# by Pouchou, Jean-Louis and Pichoir, Françoise, 1991

# All equations are referenced by their number in the paper in the Notes section of the docstring.

# Made as a part of my master thesis in nanotechnology at NTNU, spring 2023

###################################################################################################

# This file is for:
# The parameterization of $phi(rho z)$

import numpy as np
import hyperspy.api as hs

from PAP_functions.PAP_area_F import (
    area_F,
    mean_backscattering_coefficient_eta,
    mean_atomic_number_Zb,
)
from PAP_functions.PAP_helper_functions import theoretical_energy, get_C_A_Z_arrays


def surface_ionization_phi_zero(*, u: float, eta: float) -> float:
    """
    Calculate phi(0), the surface ionization.

    Parameters
    ----------
    u : float
        Overvoltage
    eta : float
        Mean ionization potential.

    Returns
    -------
    float
        phi(0)

    Notes
    -----
    See Appendix 2 in the PAP-paper.
    """
    return 1 + 3.3 * (1 - 1 / (u ** (2 - 2.3 * eta))) * eta**1.2


def average_depth_of_ionization_R_bar(*, big_f: float, u: float, zb: float) -> float:
    """
    Calculate R_bar, the average depth of ionization.
    I do not know the units of R_bar.

    Parameters
    ----------
    big_f : float
        F, area of $phi(rho z)$
    u : float
        Overvoltage
    zb : float
        Weighted mean atomic number of the specimen.

    Returns
    -------
    float
        R_bar

    Notes
    -----
    Equation 28.

    "If $ F/\\bar{R} < \phi(0) $, impose the condition $ \\bar{R} = F/\phi(0) $"
    I guess the above statement is for numerical stability, but I'm not sure.
    """
    x = 1 + 1.3 * np.log(zb)
    y = 0.2 + zb / 200
    return big_f / (1 + (x * np.log(1 + y * (1 - 1 / u**0.42))) / np.log(1 + y))


def initial_slope_P(*, big_f: float, r_bar: float, zb: float, u: float) -> float:
    """
    P, the initial slope of $phi(rho z)$.

    Parameters
    ----------
    big_f : float
        F, area of $phi(rho z)$
    r_bar : float
        Average depth of ionization.
    zb : float
        Weighted mean atomic number of the specimen.
    u : float
        Overvoltage

    Returns
    -------
    float
        P, the initial slope of $phi(rho z)$.

    Notes
    -----
    Equation 29.

    "If necessary, limit the value $ g \cdot h^4 $ to the value $ 0.9 \cdot b\cdot  \\bar{R}^2 \cdot [b - 2 \phi(0)/F] $"
    I do not think that is necessary.
    """
    g = 0.22 * np.log(4 * zb) * (1 - 2 * np.exp(-zb * (u - 1) / 15))
    h = 1 - 10 * (1 - 1 / (1 + u / 10)) / zb**2
    return g * h**4 * big_f / r_bar**2


def factor_small_b(*, r_bar: float, big_f: float, phi_zero: float) -> float:
    """
    Factor b, used in the calculation of $phi(rho z)$ and $F(\chi)$.

    Parameters
    ----------
    r_bar : float
        \\bar{R}, average depth of ionization.
    big_f : float
        F, area of $phi(rho z)$
    phi_zero : float
        phi(0), surface ionization.

    Returns
    -------
    float
        b

    Notes
    -----
    See Appendix 4 in the PAP-paper.
    """
    return np.sqrt(2) * (1 + np.sqrt(1 - r_bar * phi_zero / big_f)) / r_bar


def factor_small_a(
    *, p: float, b: float, big_f: float, phi_zero: float, r_bar: float
) -> float:
    """
    Factor a, used in the calculation of $phi(rho z)$ and $F(\chi)$.

    Parameters
    ----------
    p : float
        P, initial slope of $phi(rho z)$.
    b : float
        Factor b
    big_f : float
        F, area of $phi(rho z)$
    phi_zero : float
        phi(0), surface ionization.
    r_bar : float
        \\bar{R}, average depth of ionization.

    Returns
    -------
    float
        a

    Notes
    -----
    See Appendix 4 in the PAP-paper.
    """
    return (p + b * (2 * phi_zero - b * big_f)) / (
        b * big_f * (2 - b * r_bar) - phi_zero
    )


def factor_epsilon(*, a: float, b: float) -> float:
    """
    Epsilon is used to calculate A and B, and parameterizes the shape of $phi(rho z)$ and $F(\chi)$.

    Parameters
    ----------
    a : float
        Factor a
    b : float
        Factor b

    Returns
    -------
    float
        epsilon

    Notes
    -----
    See Appendix 4 in the PAP-paper.
    """
    return (a - b) / b


def factor_big_b(
    *, b: float, big_f: float, epsilon: float, p: float, phi_zero: float
) -> float:
    """
    Factor B, used in the calculation of $phi(rho z)$ and $F(\chi)$.

    Parameters
    ----------
    b : float
        Factor b
    big_f : float
        F, area of $phi(rho z)$
    epsilon : float
        Factor epsilon
    p : float
        P, initial slope of $phi(rho z)$.
    phi_zero : float
        phi(0), surface ionization.

    Returns
    -------
    float
        B

    Notes
    -----
    See Appendix 4 in the PAP-paper.
    """
    return (b**2 * big_f * (1 + epsilon) - p - phi_zero * b * (2 + epsilon)) / epsilon


def factor_big_a(
    *, b: float, big_f: float, epsilon: float, phi_zero: float, big_b: float
) -> float:
    """
    Factor A, used in the calculation of $phi(rho z)$ and $F(\chi)$.

    Parameters
    ----------
    b : float
        Factor b
    big_f : float
        F, area of $phi(rho z)$
    epsilon : float
        Factor epsilon
    phi_zero : float
        phi(0), surface ionization.
    big_b : float
        Factor B

    Returns
    -------
    float
        A

    Notes
    -----
    See Appendix 4 in the PAP-paper.
    """
    return (big_b / b + phi_zero - b * big_f) * (1 + epsilon) / epsilon


def emergent_intensity_F_of_chi(
    *, chi: float, phi_zero: float, big_b: float, b: float, big_a: float, epsilon: float
) -> float:
    """
    The emergent intensity $F(\chi)$.

    Parameters
    ----------
    chi : float
        cosecant of the TOA.
    phi_zero : float
        phi(0), surface ionization.
    big_b : float
        Factor B
    b : float
        Factor b
    big_a : float
        Factor A
    epsilon : float
        Factor epsilon

    Returns
    -------
    float
        $F(\chi)$
    """
    return (
        phi_zero + big_b / (b + chi) - big_a * b * epsilon / (b * (1 + epsilon) + chi)
    ) / (b + chi)


def absorption_correction(
    *,
    elements: np.ndarray,
    wt_concentrations: np.ndarray,
    e0: float,
    line: str,
    TOA: float = 35.0,
    use_q: str = "divide",
) -> float:
    """
    Calculate the absorption correction, f(chi) = F(chi) / F

    Parameters
    ----------
    elements : np.ndarray
        Elements in the sample.
    wt_concentrations : np.ndarray
        Weight concentrations, wt%.
    e0 : float
        Beam energy, keV.
    line : str
        X-ray line, e.g. "Mg_Ka".
    TOA : float, optional
        Take-off angle of the EDS detector, by default 35.0
    use_q : str, optional
        See area F in PAP_area_F.py, by default "divide"

    Returns
    -------
    float
        The absorption correction, f(chi) = F(chi) / F

    Notes
    -----
    Page 40 in the PAP-paper.
    """
    e_c = theoretical_energy(line=line)
    u = e0 / e_c
    array_C, array_A, array_Z = get_C_A_Z_arrays(
        elements=elements, concentrations=wt_concentrations
    )
    mu_rho = hs.material.mass_absorption_mixture(
        elements=elements,
        weight_percent=np.array(wt_concentrations) * 100,
        energies=theoretical_energy(line),
    )
    chi = mu_rho / np.sin(np.deg2rad(TOA))
    big_F = area_F(
        array_C=array_C, array_A=array_A, array_Z=array_Z, e0=e0, line=line, use_q=use_q
    )
    zb = mean_atomic_number_Zb(array_C=array_C, array_Z=array_Z)
    eta = mean_backscattering_coefficient_eta(zb=zb)

    phi_zero = surface_ionization_phi_zero(eta=eta, u=u)
    r_bar = average_depth_of_ionization_R_bar(big_f=big_F, u=u, zb=zb)
    p = initial_slope_P(big_f=big_F, zb=zb, u=u, r_bar=r_bar)

    b = factor_small_b(r_bar=r_bar, big_f=big_F, phi_zero=phi_zero)
    a = factor_small_a(p=p, b=b, big_f=big_F, phi_zero=phi_zero, r_bar=r_bar)
    epsilon = factor_epsilon(a=a, b=b)
    big_b = factor_big_b(b=b, big_f=big_F, epsilon=epsilon, p=p, phi_zero=phi_zero)
    big_a = factor_big_a(
        b=b, big_f=big_F, epsilon=epsilon, phi_zero=phi_zero, big_b=big_b
    )

    f_of_chi = emergent_intensity_F_of_chi(
        chi=chi, phi_zero=phi_zero, big_b=big_b, b=b, big_a=big_a, epsilon=epsilon
    )

    return f_of_chi / big_F


def parameterization_phi_of_rhoz(
    *,
    rhoz: np.ndarray,
    elements: np.ndarray,
    wt_concentrations: np.ndarray,
    e0: float,
    line: str,
    TOA: float = 35.0,
    use_q: str = "divide",
) -> np.ndarray:
    """
    Gives the parameterization of phi(rho z) for a range of (rho z).
    Used for e.g. plotting.

    Parameters
    ----------
    rhoz : np.ndarray
        Mass thickness, rho z, in g/cm^2. The X-axis.
    elements : np.ndarray
        Elements in the sample.
    wt_concentrations : np.ndarray
        Weight concentrations, wt%.
    e0 : float
        Beam energy, keV.
    line : str
        X-ray line, e.g. "Mg_Ka".
    TOA : float, optional
        Take-off angle of the EDS detector, by default 35.0
    use_q : str, optional
        See area F in PAP_area_F.py, by default "divide"

    Returns
    -------
    np.ndarray
        phi(rho z)

    Notes
    -----
    Equation 22 in the PAP-paper.
    """
    e_c = theoretical_energy(line=line)
    u = e0 / e_c
    array_C, array_A, array_Z = get_C_A_Z_arrays(
        elements=elements, concentrations=wt_concentrations
    )
    mu_rho = hs.material.mass_absorption_mixture(
        elements=elements,
        weight_percent=np.array(wt_concentrations) * 100,
        energies=theoretical_energy(line),
    )
    chi = mu_rho / np.sin(np.deg2rad(TOA))
    big_F = area_F(
        array_C=array_C, array_A=array_A, array_Z=array_Z, e0=e0, line=line, use_q=use_q
    )
    zb = mean_atomic_number_Zb(array_C=array_C, array_Z=array_Z)
    eta = mean_backscattering_coefficient_eta(zb=zb)

    phi_zero = surface_ionization_phi_zero(eta=eta, u=u)
    r_bar = average_depth_of_ionization_R_bar(big_f=big_F, u=u, zb=zb)
    p = initial_slope_P(big_f=big_F, zb=zb, u=u, r_bar=r_bar)

    b = factor_small_b(r_bar=r_bar, big_f=big_F, phi_zero=phi_zero)
    a = factor_small_a(p=p, b=b, big_f=big_F, phi_zero=phi_zero, r_bar=r_bar)
    epsilon = factor_epsilon(a=a, b=b)
    big_b = factor_big_b(b=b, big_f=big_F, epsilon=epsilon, p=p, phi_zero=phi_zero)
    big_a = factor_big_a(
        b=b, big_f=big_F, epsilon=epsilon, phi_zero=phi_zero, big_b=big_b
    )

    # checking the values of the parameters
    # print(f"big_a = {big_a}")
    # print(f"big_b = {big_b}")
    # print(f"a = {a}")
    # print(f"b = {b}")

    # print(f"big_F = {big_F}")
    # print(f"phi_zero = {phi_zero}")
    # print(f"r_bar = {r_bar}")
    # print(f"slope p = {p}")

    # # calculating F, phi(0), R, and P agian
    # # F = A/a + (phi(0)-A)/b + B/b**2
    # print(f"F = {big_a/a + (phi_zero-big_a)/b + big_b/b**2}")
    # # R = (A/a**2 + (phi(0) -A)/b**2 + 2B/b**3) / F  # (1/F) is not included in the paper, Eq. 24
    # print(f"r_bar = {(big_a/a**2 + (phi_zero-big_a)/b**2 + 2*big_b/b**3)/big_F}")
    # # P = B - a A - b (phi(0) - A)
    # print(f"P = {big_b - a*big_a - b*(phi_zero-big_a)}")
    # # A * np.exp(- a * rhoz) + (B * rhoz + phi_zero - A) * np.exp(- b * rhoz)
    return big_a * np.exp(-a * rhoz) + (big_b * rhoz + phi_zero - big_a) * np.exp(
        -b * rhoz
    )
