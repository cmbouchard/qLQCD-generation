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
Ncfg_start = 1000
Ncfg_end = 1500

beta = 5.7
action = "W"

flow_start = 0
flow_step = 0.5
Nflow = 5

name = action+"_"+str(Nt)+"x"+str(Ns)+"x"+str(Ns)+"x"+str(Ns)+"_b"+str(int(beta * 100))
homepath = "/Users/cmb/Documents/gauge_fields/qLQCD-generation/"
cfgdir = homepath + name + "/"

#if flow:
#  fout = open(homepath + "Qc_wf_" + str(flow_t) + "_v_cfg_" + name + ".dat", "w")
#  #fout = open(homepath + "Qp_wf_" + str(flow_t) + "_v_cfg_" + name + ".dat", "w")
#else:
#  fout = open(homepath + "Qc_v_cfg_" + name + ".dat", "w")
#  #fout = open(homepath + "Qp_v_cfg_" + name + ".dat", "w")
  
#fout.write("#Q is the clover-based calculation.\n")
#fout.write("#Q is the plaquette-based calculation.\n")
#if flow:
#  fout.write("#lattices Wilson flowed to t = " + str(flow_t) + "\n")
#fout.write("#1:cfg 2:Re(Q) 3:Im(Q) 4:S\n")
for _n in range(1, Nflow+1):
  total_flow = _n * flow_step
  
  if _n == 1 and flow_start == 0:
    cfgfile = cfgdir + "link_" + name + "_"
  else:
    cfgfile = cfgdir + "link_" + name + "_tf_" + str(flow_start + (_n-1) * flow_step) + "_"
    
  for cfg in range(Ncfg_start, Ncfg_end+1):
    U_in = np.load(cfgfile + str(cfg))
    U_out = tl.wilson_flow(U_in, flow_step)
    file_out = open(cfgdir + "link_" + name + "_tf_" + str(flow_start + _n * flow_step) + "_" + str(cfg), 'wb')
    np.save(file_out, U_out)  #NOTE: np.save without opening first appends .npy to name
    file_out.close()
    sys.stdout.flush()
    
    #else:
    #  U = U_in
    #Q = 0. + 0.J
    #S = 0.
    #for t in range(Nt):
    #  for x in range(Ns):
    #    for y in range(Ns):
    #      for z in range(Ns):
    #        Q += gl.fn_topological_charge(U, t, x, y, z)
    #        S += gl.fn_eval_point_S(U, t, x, y, z, beta)
    #S /= (Nt * Ns**3)
    #fout.write(str(cfg) + " " + str(Q.real) + " " + str(Q.imag) + " " + str(S) + "\n")
    #print("Re Q("+str(flow_t)+", Im Q("+str(flow_t)+"), S = "+str(Q.real)+" "+str(Q.imag)+" "+str(S))
  #fout.close()
