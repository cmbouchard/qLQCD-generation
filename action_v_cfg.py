import os, sys, string
import numba
import numpy as np
import tools_v1 as tool
import params
import gauge_latticeqcd as gl
import lattice_collection as lc

### Script to calculate the evolution of the action as a function of Monte Carlo time
Nstart = 0
Nend = 2000

Nt, Nx, Ny, Nz = 6, 6, 6, 6
action = 'WR_T'
beta = 5.7

### ---- add to action_v_cfg.py, below the imports ------------------------------------------
### Loop starting at start_txyz, following steps = [(direction, +1/-1), ...].
### +1: forward link U_d(x);  -1: backward link U_d(x - d)^dagger  (same helpers as fn_plaquette)
def fn_loop(U, start_txyz, steps):
    line = np.identity(3, dtype='complex128')
    txyz = list(start_txyz)
    for direction, sign in steps:
        if sign > 0:
            line, txyz = gl.fn_line_move_forward(U, line, txyz, direction)
        else:
            line, txyz = gl.fn_line_move_backward(U, line, txyz, direction)
    return line

### Tadpole-improved Luscher-Weisz (Symanzik, tree-level, on-shell) action at a site, Lepage Eq. (103):
###   S(x) = beta * sum_{mu>nu} [ 5/3 (1 - P_{mu nu}/u0^4)
###                               - 1/12 ((1 - R_{mu nu}/u0^6) + (1 - R_{nu mu}/u0^6)) ]
### with P = 1/3 Re Tr(plaquette) and R_{mu nu} = 1/3 Re Tr(2x1 rectangle, long side along mu).
### The constants (1 - ...) make S = 0 on a unit configuration when u0 = 1, and are irrelevant for
### the dynamics. u0 = 1 gives the untadpole-improved WR action.
def fn_eval_point_S_WR(U, t, x, y, z, beta, u0 = 1.):
    start = [t, x, y, z]
    tmp = 0.
    for mu in range(1, 4):        # sum over mu > nu spacetime dimensions
        for nu in range(mu):
            P  = np.real(np.trace(gl.fn_plaquette(U, t, x, y, z, mu, nu))) / 3.
            R1 = np.real(np.trace(fn_loop(U, start, [(mu, 1), (mu, 1), (nu, 1), (mu, -1), (mu, -1), (nu, -1)]))) / 3.
            R2 = np.real(np.trace(fn_loop(U, start, [(nu, 1), (nu, 1), (mu, 1), (nu, -1), (nu, -1), (mu, -1)]))) / 3.
            tmp += (5. / 3.) * (1. - P / u0**4) - (1. / 12.) * ((1. - R1 / u0**6) + (1. - R2 / u0**6))
    return beta * tmp


### if the ensemble was generated with tadpole improvement (action ending in "_T"),
### load the per-configuration u0 log written by gauge_latticeqcd.py during generation
### so that the action is evaluated with the same u0 that was actually used to
### generate each configuration. Non-tadpole ensembles have no u0 log and use u0 = 1.
if action[-2:] == '_T':
    u0_of_cfg = lc.fn_load_u0_log(action, Nt, Nx, Ny, Nz, beta, "./")
else:
    u0_of_cfg = None



def calc_S_QCD(U, u0=1.):
    Nt = len(U)
    Nx = len(U[0])
    Ny = len(U[0, 0])
    Nz = len(U[0, 0, 0])
    S_QCD = np.float64(0)
    for t in range( Nt ):
        for x in range( Nx ):
            for y in range( Ny ):
                for z in range( Nz ):
                
                  if action in ('WR', 'WR_T'):
                    S_QCD += fn_eval_point_S_WR(U, t, x, y, z, beta, u0)
                  else:
                    S_QCD += gl.fn_eval_point_S(U, t, x, y, z, beta, u0)
                
                #end z
            #end y
        #end x
    #end t
    return S_QCD


### output data vs cfg
### * allow plots of evolution of action with configuration number
### * divide action into contributions from:
###   - flat spacetime QCD (standard LQCD action), leading order

dir = './' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '/'
U_infile = dir + 'link_' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '_'

### prepare output file
outfile = './S_v_cfg_' + str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '_' + str(Nstart) + '-' + str(Nend) + '.dat'

fout = open(outfile, 'w')
fout.write('#1:cfg  2:S(QCD)  \n')

for Ncfg in range(Nstart, Nend + 1):

    ### load lattice data
    U = np.load(U_infile + str(Ncfg))

    ### use the u0 that was actually used to generate this configuration
    u0 = u0_of_cfg[Ncfg] if u0_of_cfg is not None else 1.

    ### calculate action
    S_QCD = calc_S_QCD(U, u0)
    fout.write(str(Ncfg) + ' ' + str(S_QCD) + '\n' )

#end Ncfg
fout.close()
