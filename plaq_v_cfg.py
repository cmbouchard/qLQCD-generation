import os, sys, string
import numba
import numpy as np
import tools_v1 as tool
import gauge_latticeqcd as gl
import params


### Script to calculate the evolution of the 1x1 Wilson loop as a function of Monte Carlo time
Nx = 14
Ny = 14
Nz = 14
Nt = 14
action = 'W'
beta = 5.7
u0 = 1.0
Nstart = 871
Nend = 997


dir = './' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '/'
U_infile = dir + 'link_' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '_'

### prepare output file
outfile = './plaquette_v_cfg_' + str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '.dat'
fout = open(outfile, 'a')

fout.write('#1:cfg  2:plaquette\n')
for Ncfg in range(Nstart, Nend + 1):
    U = np.load(U_infile + str(Ncfg))
    pl = gl.fn_average_plaquette(U)
    fout.write(str(Ncfg) + ' ' + str(pl) + '\n' )
#end Ncfg
fout.close()
