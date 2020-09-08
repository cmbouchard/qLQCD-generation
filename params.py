import numpy as np

### input parameters from PDG:
### The Review of Particle Physics (2020), P.A. Zyla et al. 
### (Particle Data Group), Prog. Theor. Exp. Phys. 2020, 083C01 (2020)
### retrieved 04 Sep 2020 from pdg.lbl.gov
c = 299792458 # m / s
h = 6.62607015 * 1e-34 # J s
hbar = h / (2.0 * np.pi) # J s
e = 1.602176634 * 1e-19 # J / eV


### derived parameters
hbarc_GeVfm = hbar * c / e * 1e15 * 1e-9 # ~0.2 GeV fm
GN_fm2 = 6.70883 * 1e-39 * hbarc_GeVfm**2 # ~2.6e-40 fm^2
lP_fm = GN_fm2**0.5 # ~1.6e-20 fm
