import os, sys, string
import numba
import numpy as np
import tools_v1 as tool
import gauge_latticeqcd as gl
import params


### Script to calculate the evolution of the action as a function of Monte Carlo time
Nx = 14
Ny = 14
Nz = 14
Nt = 14
action = 'W'
beta = 5.7
u0 = 1.0
Nstart = 919
Nend = 980

#@numba.njit
def calc_S(U):
    Nt, Nx, Ny, Nz = len(U), len(U[0]), len(U[0, 0]), len(U[0, 0, 0])
    S = 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):

                    S += gl.fn_eval_point_S(U, t, x, y, z, beta, u0=1.)

                #end z
            #end y
        #end x
    #end t
    return S


dir = './' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '/'
U_infile = dir + 'link_' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '_'


### prepare output file
outfile = './S_v_cfg_' + str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '.dat'
fout = open(outfile, 'a')

fout.write('#1:cfg  2:S\n')
for Ncfg in range(Nstart, Nend + 1):
    U = np.load(U_infile + str(Ncfg))
    S = calc_S(U)
    fout.write(str(Ncfg) + ' ' + str(S) + '\n' )
#end Ncfg
fout.close()
