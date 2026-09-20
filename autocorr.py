import argparse
import numpy as np

### Autocorrelation function of an observable versus configuration number, read from the output of
### action_v_cfg.py (column 2 of S_v_cfg_*.dat or W11_v_cfg_*.dat).
###   e.g.  python autocorr.py --action WR_T --beta 4.4 --Nt 6 --Nx 6 --Ny 6 --Nz 6 --Nstart 0 --Nend 1000 \
###                            --Nstart_analysis 500 --Nend_analysis 1000
def parse_args():
    parser = argparse.ArgumentParser(
        description="Autocorrelation function and autocorrelation length of the observable written by "
                    "action_v_cfg.py.")

    parser.add_argument("--homepath", default='/Users/cmb/Documents/gauge_fields/qLQCD-generation/',
                        help="directory containing the action_v_cfg.py output file; the ACF file is "
                             "written here too (default: /Users/cmb/Documents/gauge_fields/qLQCD-generation/)")
    parser.add_argument("--obs", choices=["action", "W11"], default="action",
                        help="observable to analyse, as chosen with action_v_cfg.py --obs: reads "
                             "S_v_cfg_*.dat for 'action' or W11_v_cfg_*.dat for 'W11' (default: action)")
    parser.add_argument("--action", choices=["W", "WR", "W_T", "WR_T"], default="WR_T",
                        help="action the ensemble was generated with (default: WR_T)")
    parser.add_argument("--beta", type=float, default=4.4, help="beta = 6/g^2 (default: 4.4)")
    parser.add_argument("--Nt", type=int, default=6, help="temporal extent (default: 6)")
    parser.add_argument("--Nx", type=int, default=6, help="spatial extent, x (default: 6)")
    parser.add_argument("--Ny", type=int, default=6, help="spatial extent, y (default: 6)")
    parser.add_argument("--Nz", type=int, default=6, help="spatial extent, z (default: 6)")
    parser.add_argument("--Nstart", type=int, default=500,
                        help="first configuration in the data file, i.e. the Nstart used with "
                             "action_v_cfg.py (default: 500)")
    parser.add_argument("--Nend", type=int, default=6000,
                        help="last configuration in the data file, i.e. the Nend used with "
                             "action_v_cfg.py (default: 6000)")
    parser.add_argument("--Nstart_analysis", type=int, default=None,
                        help="first configuration to include in the analysis (default: --Nstart)")
    parser.add_argument("--Nend_analysis", type=int, default=None,
                        help="last configuration to include in the analysis, inclusive (default: --Nend)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="ACF threshold defining the autocorrelation length (default: 0.1)")

    args = parser.parse_args()
    if args.Nstart_analysis is None:
        args.Nstart_analysis = args.Nstart
    if args.Nend_analysis is None:
        args.Nend_analysis = args.Nend
    if args.Nend < args.Nstart:
        parser.error("--Nend (" + str(args.Nend) + ") must be >= --Nstart (" + str(args.Nstart) + ")")
    if not (args.Nstart <= args.Nstart_analysis <= args.Nend_analysis <= args.Nend):
        parser.error("need Nstart <= Nstart_analysis <= Nend_analysis <= Nend, got " +
                     str(args.Nstart) + ", " + str(args.Nstart_analysis) + ", " +
                     str(args.Nend_analysis) + ", " + str(args.Nend))
    return args

args = parse_args()

homepath = args.homepath

### specify data
Nstart = args.Nstart
Nend = args.Nend
Nt, Nx, Ny, Nz = args.Nt, args.Nx, args.Ny, args.Nz
action = args.action
beta = args.beta
obs = args.obs

### configurations to analyze
Nstart_analysis = args.Nstart_analysis
Nend_analysis = args.Nend_analysis

### set threshold for autocorrelation function
threshold = args.threshold

### point to data file, e.g., S_v_cfg_570_4x4x4x4_W_10000-19998.dat  (or W11_v_cfg_... for --obs W11)
dprefix = 'S_v_cfg_' if obs == 'action' else 'W11_v_cfg_'
dfile = homepath + dprefix + str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '_' + str(Nstart) + '-' + str(Nend) + '.dat'

data = np.loadtxt(dfile, usecols=1, skiprows=1)

### the file holds configurations Nstart..Nend; keep only Nstart_analysis..Nend_analysis
if len(data) != Nend - Nstart + 1:
    raise SystemExit('error: ' + dfile + ' has ' + str(len(data)) + ' rows but --Nstart/--Nend imply ' + str(Nend - Nstart + 1))
data = data[Nstart_analysis - Nstart : Nend_analysis - Nstart + 1]

### Name output file
#Nend_analysis = len(data) * Nskip
fstring_out = str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '_' + str(Nstart_analysis) + '-' + str(Nend_analysis) + '.dat'
### (the W11 ACF is written to A_W11_v_lag_*.dat so that it does not overwrite the action ACF)
aprefix = 'A_v_lag_' if obs == 'action' else 'A_W11_v_lag_'

### Calculate autocorrelation function (ACF)
acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')

### Normalize ACF
acf /= acf[len(data)-1]

### Find autocorrelation length (lag where ACF drops below threshold)
lag = np.arange(len(data))
lag_positive = lag[lag >= 0]  # Only consider non-negative lags
acf_positive = acf[len(data)-1:]  # Only consider non-negative ACF values

### Find the first lag where ACF drops below the threshold
autocorr_length = np.argmax(acf_positive < threshold)

lag_positive = lag_positive[:len(data)]
acf_positive = acf_positive[:len(data)]

### output data to file
fout = open(homepath + aprefix + fstring_out, 'w')
fout.write('#1:lag  2:A\n')
for jj in range(len(data)):
    fout.write(str(lag_positive[jj]) + ' ' + str(acf_positive[jj])+'\n')
fout.close()

"""
# Plot the ACF for non-negative lags
plt.plot(lag_positive, acf_positive)
plt.grid(True)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')
plt.yscale('log')
plt.show()
"""

print("Autocorrelation Length:", autocorr_length)
