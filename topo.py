import numpy as np
import gauge_latticeqcd as gl

###########################

Nt, Ns = 6, 6#4, 4#8, 8
beta = 5.7
action = "W"
Ncfg_start = 500#1000#2500
Ncfg_end = 2000#10000#4000

name = action+"_"+str(Nt)+"x"+str(Ns)+"x"+str(Ns)+"x"+str(Ns)+"_b"+str(int(beta * 100))
homepath = "/Users/cmb/Documents/gauge_fields/qLQCD-generation/"
cfgdir = homepath + name + "/"
cfgfile = cfgdir + "link_" + name + "_"
fout = open(homepath + "Q_v_cfg_" + name + ".dat", "w")
fout.write("#Q is the clover-based calculation.\n")
fout.write("#1:cfg 2:Re(Q) 3:Im(Q)\n")
for cfg in range(Ncfg_start, Ncfg_end+1):
  U = np.load(cfgfile + str(cfg))
  Q = 0. + 0.J
  for t in range(Nt):
    for x in range(Ns):
      for y in range(Ns):
        for z in range(Ns):
          Q += gl.fn_topological_charge(U, t, x, y, z)
  fout.write(str(cfg) + " " + str(Q.real) + " " + str(Q.imag) + "\n")

fout.close()
