import numba
import numpy as np
import math
from scipy.linalg import expm

    
### function that returns lattice spacing in fm, given beta
### * from Necco and Sommer, NPB 622, 328-346 (2002), hep-lat/0108008
### * valid for 5.7 <= beta <= 6.92
### * valid for Nf = 0 (quenched, ie. no quarks)
def fn_a(beta):
    return 0.5 * np.exp(-1.6804 - 1.7331 * (beta - 6.) + 0.7849 * (beta - 6.)**2 - 0.4428 * (beta - 6.)**3)


"""
wilson_flow.py

Standalone Wilson (gradient) flow for SU(3) lattice gauge configurations,

    U.shape == (Nt, Ns, Ns, Ns, 4, 3, 3), dtype=complex

i.e. U[t, x, y, z, mu, :, :] is the SU(3) link matrix at site (t,x,y,z)
pointing in direction mu (mu=0 is time, mu=1,2,3 are space), with periodic
boundary conditions in all four directions.

Physics
-------
The Wilson flow evolves the gauge field according to

    dV_mu(x,t)/dt = Z_mu(V(t)) V_mu(x,t),
    V(0) = U (the input config)

where Z_mu(x) is the Wilson-action force, projected onto the su(3)
Lie algebra (traceless, anti-Hermitian):

    Z_mu(x) = P_ta[ U_mu(x) Sigma_mu(x) ]

    Sigma_mu(x) = sum_{nu != mu} [
          U_nu(x+mu)   U_mu(x+nu)^dagger  U_nu(x)^dagger
        + U_nu(x+mu-nu)^dagger U_mu(x-nu)^dagger U_nu(x-nu)
    ]

    P_ta(M) = 0.5*(M - M^dagger) - (1/(2*Nc)) * Tr(M - M^dagger) * I

Integration uses Luscher's standard 3rd-order Runge-Kutta scheme
(see M. Luscher, "Properties and uses of the Wilson flow in lattice QCD",
JHEP 1008:071, 2010):

    W0 = V(t)
    W1 = exp( (1/4) eps Z0 ) W0,                    Z0 = Z(W0)
    W2 = exp( (8/9) eps Z1 - (17/36) eps Z0 ) W1,   Z1 = Z(W1)
    W3 = exp( (3/4) eps Z2 - (8/9) eps Z1 + (17/36) eps Z0 ) W2,  Z2 = Z(W2)
    V(t+eps) = W3

Usage
-----
    from wilson_flow import wilson_flow

    U = np.load(cfgfile)              # shape (Nt, Ns, Ns, Ns, 4, 3, 3)
    U_flowed = wilson_flow(U, flow_time=1.0, epsilon=0.02)
"""

def shift_site(site, mu, step, Nt, Ns):
    """Shift a site tuple by `step` lattice units in direction mu,
    applying periodic boundary conditions. Same convention as topo.py."""
    coords = list(site)
    if mu == 0:
        coords[0] = (coords[0] + step) % Nt
    else:
        coords[mu] = (coords[mu] + step) % Ns
    return tuple(coords)


def _project_su3_algebra(M):
    """
    Project a general 3x3 complex matrix M onto the su(3) Lie algebra:
    traceless and anti-Hermitian.

        P_ta(M) = 0.5*(M - M^dagger) - (1/6)*Tr(M - M^dagger)*I
    """
    A = 0.5 * (M - M.conj().T)
    A = A - (np.trace(A) / 3.) * np.eye(3, dtype=complex)
    return A


def _staple_sum(U, site, mu, Nt, Ns):
    """
    Sum of the six staples attached to link U_mu(site), for the standard
    Wilson (plaquette) gauge action.
    """
    Sigma = np.zeros((3, 3), dtype=complex)
    x_plus_mu = shift_site(site, mu, +1, Nt, Ns)

    for nu in range(4):
        if nu == mu:
            continue

        # "forward" staple: U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag
        x_plus_nu = shift_site(site, nu, +1, Nt, Ns)
        U_nu_xplusmu = U[x_plus_mu][nu]
        U_mu_xplusnu = U[x_plus_nu][mu]
        U_nu_x = U[site][nu]
        Sigma += U_nu_xplusmu @ U_mu_xplusnu.conj().T @ U_nu_x.conj().T

        # "backward" staple: U_nu(x+mu-nu)^dag U_mu(x-nu)^dag U_nu(x-nu)
        x_minus_nu = shift_site(site, nu, -1, Nt, Ns)
        x_plus_mu_minus_nu = shift_site(x_plus_mu, nu, -1, Nt, Ns)
        U_nu_xplusmuminusnu = U[x_plus_mu_minus_nu][nu]
        U_mu_xminusnu = U[x_minus_nu][mu]
        U_nu_xminusnu = U[x_minus_nu][nu]
        Sigma += U_nu_xplusmuminusnu.conj().T @ U_mu_xminusnu.conj().T @ U_nu_xminusnu

    return Sigma


def _compute_Z(U, Nt, Ns):
    """Compute the su(3)-valued flow force Z_mu(x) for every link."""
    Z = np.zeros_like(U)
    for t in range(Nt):
        for x in range(Ns):
            for y in range(Ns):
                for z in range(Ns):
                    site = (t, x, y, z)
                    for mu in range(4):
                        Umu = U[site][mu]
                        Sigma = _staple_sum(U, site, mu, Nt, Ns)
                        M = Umu @ Sigma
                        Z[site + (mu,)] = -_project_su3_algebra(M)
    return Z


def _evolve_links(U, dW, Nt, Ns):
    """Return U' with U'_mu(x) = expm(dW_mu(x)) @ U_mu(x), link by link."""
    Uprime = np.empty_like(U)
    for t in range(Nt):
        for x in range(Ns):
            for y in range(Ns):
                for z in range(Ns):
                    for mu in range(4):
                        idx = (t, x, y, z, mu)
                        Uprime[idx] = expm(dW[idx]) @ U[idx]
    return Uprime


def _reunitarize(U, Nt, Ns):
    """
    Project every link back onto SU(3) via the closest-unitary-matrix
    (polar decomposition) trick, then fix the determinant phase.
    Numerical drift from repeated expm/matrix-multiply steps is tiny per
    step, but can accumulate over long flow trajectories, so this is
    applied periodically as a safeguard.
    """
    Uout = np.empty_like(U)
    for t in range(Nt):
        for x in range(Ns):
            for y in range(Ns):
                for z in range(Ns):
                    for mu in range(4):
                        idx = (t, x, y, z, mu)
                        M = U[idx]
                        Umat, _, Vh = np.linalg.svd(M)
                        Uunitary = Umat @ Vh
                        det = np.linalg.det(Uunitary)
                        # fix overall phase so det = 1 (SU(3), not just U(3))
                        Uunitary = Uunitary * (det.conjugate() / abs(det)) ** (1.0 / 3.0)
                        Uout[idx] = Uunitary
    return Uout


def wilson_flow(U_in, flow_time, epsilon=0.05, reunitarize_every=10):
    """
    Evolve a single SU(3) lattice gauge configuration under the Wilson
    (gradient) flow.

    Parameters
    ----------
    U_in : ndarray, shape (Nt, Ns, Ns, Ns, 4, 3, 3), complex
        Input configuration, same layout as gauge_latticeqcd.py's self.U.
    flow_time : float
        Total flow time t to integrate to. The flow radius in physical
        units is sqrt(8t), with t in units of a^2.
    epsilon : float, optional
        Integration step size in flow time (default 0.01). Standard safe
        step sizes are around 0.01-0.02 with the 3rd-order integrator
        below; monitor <E(t)> t^2 vs t if you want to check step-size
        independence for your own action/coupling.
    reunitarize_every : int, optional
        Re-project all links back onto exact SU(3) after this many
        integration steps, to control numerical drift over long
        trajectories. Set to 0 to disable.

    Returns
    -------
    U_flowed : ndarray, same shape as U_in
        The configuration flowed to flow time `flow_time`.
    """

    Nt, Ns = U_in.shape[0], U_in.shape[1]

    U = np.array(U_in, copy=True)

    n_steps = int(round(flow_time / epsilon))
    remainder = flow_time - n_steps * epsilon
    steps = [epsilon] * n_steps
    if remainder > 1e-12:
        steps.append(remainder)

    for i, eps in enumerate(steps):
        Z0 = _compute_Z(U, Nt, Ns)
        W1 = _evolve_links(U, (eps / 4.0) * Z0, Nt, Ns)

        Z1 = _compute_Z(W1, Nt, Ns)
        dW2 = eps * (8.0 / 9.0) * Z1 - eps * (17.0 / 36.0) * Z0
        W2 = _evolve_links(W1, dW2, Nt, Ns)

        Z2 = _compute_Z(W2, Nt, Ns)
        dW3 = eps * (3.0 / 4.0) * Z2 - eps * (8.0 / 9.0) * Z1 + eps * (17.0 / 36.0) * Z0
        W3 = _evolve_links(W2, dW3, Nt, Ns)

        U = W3

        if reunitarize_every and (i + 1) % reunitarize_every == 0:
            U = _reunitarize(U, Nt, Ns)

    if reunitarize_every:
        U = _reunitarize(U, Nt, Ns)  # final cleanup

    return U

def average_plaquette_deviation(U, Nt, Ns):
    """
    Return 1 - <Re Tr P> / 3, averaged over all mu<nu plaquettes and all
    sites. This is the (unnormalized) Wilson action density: it is 0 for
    the ordered/identity configuration and increases as the field gets
    rougher. A correctly-signed Wilson flow must *decrease* this quantity
    monotonically with flow time, at least initially.
    """
    total = 0.0
    count = 0
    for t in range(Nt):
        for x in range(Ns):
            for y in range(Ns):
                for z in range(Ns):
                    site = (t, x, y, z)
                    for mu in range(4):
                        for nu in range(mu):
                            x_plus_mu = shift_site(site, mu, +1, Nt, Ns)
                            x_plus_nu = shift_site(site, nu, +1, Nt, Ns)
                            P = (U[site][mu] @ U[x_plus_mu][nu]
                                 @ U[x_plus_nu][mu].conj().T
                                 @ U[site][nu].conj().T)
                            total += 1.0 - np.trace(P).real / 3.0
                            count += 1
    return total / count
