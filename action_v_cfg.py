import os, sys, string
import numba
import numpy as np
import tools_v1 as tool
import params
import gauge_latticeqcd as gl

### Script to calculate the evolution of the action as a function of Monte Carlo time
Nstart = 2000
Nend = 2500
Nt = 20
Nx = 10
Ny = 10
Nz = 10
action = 'W'
beta = 5.70

#u0file = 'u0_W_T_5x5x5x5_b600'
u0file = None
if u0file != None:
    import u0_W_T_5x5x5x5_b600 as ti
    u0LIST = ti.u0

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

    ### collect tadpole improvement values
    if u0file != None:
        u0 = u0LIST[int(Ncfg/10)]
    else:
        u0 = 1.

    ### calculate action
    S_QCD = calc_S_QCD(U, u0)
    fout.write(str(Ncfg) + ' ' + str(S_QCD) + '\n' )

#end Ncfg
fout.close()
