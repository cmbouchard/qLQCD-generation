import numpy as np
import gauge_latticeqcd as gl
import tools_v1 as tl
import sys

###########################
#Nt, Ns = 4, 4
#Ncfg_start = 10000#8500
#Ncfg_end = 10000

#Nt, Ns = 6, 6
#Ncfg_start = 500
#Ncfg_end = 2000

#Nt, Ns = 8, 8
#Ncfg_start = 3500#2500
#Ncfg_end = 4000

# 500-2000 are thermalized and suitable for analysis
Nt, Ns = 10, 10
Ncfg_start = 500
Ncfg_end = 1500

beta = 5.7
action = "W"

flow = True
flow_t = 1.0

name = action+"_"+str(Nt)+"x"+str(Ns)+"x"+str(Ns)+"x"+str(Ns)+"_b"+str(int(beta * 100))
homepath = "/Users/cmb/Documents/gauge_fields/qLQCD-generation/"
cfgdir = homepath + name + "/"

if flow:
  fout = open(homepath + "Qc_tf_" + str(flow_t) + "_v_cfg_" + name + "_"+str(Ncfg_start)+"-"+str(Ncfg_end)+".dat", "w")
else:
  fout = open(homepath + "Qc_v_cfg_" + name + "_"+str(Ncfg_start)+"-"+str(Ncfg_end)+".dat", "w")
  
fout.write("#Qc is the O(a^2) clover calculation.\n")
fout.write("#Qcr is the O(a^4) clover+rectangle calculation.\n")
if flow:
  fout.write("#lattices Wilson flowed to t_flow = " + str(flow_t) + "\n")
#fout.write("#1:cfg 2:Re(Qc) 3:Im(Qc) 4:Re(Qcr) 5:Im(Qcr)\n")# 6:S\n")
fout.write("#1:cfg 2:Re(Qcr) 3:Im(Qcr) \n")# 6:S\n")

if flow:
  cfgfile = cfgdir + "link_" + name + "_tf_" + str(flow_t) + "_"
else:
  cfgfile = cfgdir + "link_" + name + "_"
    
for cfg in range(Ncfg_start, Ncfg_end+1):
  U = np.load(cfgfile + str(cfg))
  #Qc = 0. + 0.J
  Qcr = 0. + 0.J
  #S = 0.
  for t in range(Nt):
    for x in range(Ns):
      for y in range(Ns):
        for z in range(Ns):
          #Qc += gl.fn_topological_charge(U, t, x, y, z, 'c')
          Qcr += gl.fn_topological_charge(U, t, x, y, z, 'cr')
          #S += gl.fn_eval_point_S(U, t, x, y, z, beta)
  #S /= (Nt * Ns**3)
  #fout.write(str(cfg) + " " + str(Qc.real) + " " + str(Qc.imag) + " " + str(Qcr.real) + " " + str(Qcr.imag) + "\n")# + str(S) + "\n")
  fout.write(str(cfg) + " " + str(Qcr.real) + " " + str(Qcr.imag) + "\n")# + str(S) + "\n")
fout.close()
