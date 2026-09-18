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
