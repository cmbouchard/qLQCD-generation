import numpy as np

homepath = '/Users/cmb/Documents/gauge_fields/qLQCD-generation/'

### specify data
Nstart = 10000
Nend = 19998
Nt = 4
Nx = 4
Ny = 4
Nz = 4
action = 'W'
beta = 5.70

### configurations to analyze
Nstart_analysis = 10000
Nend_analysis = 19998


### set threshold for autocorrelation function
threshold = 0.01

### point to data file, e.g., S_v_cfg_570_4x4x4x4_W_10000-19998.dat
dfile = homepath + 'S_v_cfg_' + str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '_' + str(Nstart) + '-' + str(Nend) + '.dat'

data = np.loadtxt(dfile, usecols=1, skiprows=1)

### Name output file
#Nend_analysis = len(data) * Nskip
fstring_out = str(beta * 100) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '_' + str(Nstart_analysis) + '-' + str(Nend_analysis) + '.dat'


#print(len(data))
# prune data from Nstart_analysis/Nskip
#data = data[int(Nstart_analysis/Nskip):Nend_analysis]
#print(len(data))

### Calculate autocorrelation function (ACF)
acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')

### Normalize ACF
acf /= acf[len(data)-1]

### Find autocorrelation length (lag where ACF drops and stays below threshold)
lag = np.arange(len(data))
lag_positive = lag[lag >= 0]  # Only consider non-negative lags
acf_positive = acf[len(data)-1:]  # Only consider non-negative ACF values

### Find the first lag where ACF drops below the threshold
autocorr_length = np.argmax(acf_positive < threshold)

lag_positive = lag_positive[:len(data)]
acf_positive = acf_positive[:len(data)]

### output data to file
fout = open(homepath + 'A_v_lag_' + '_' + fstring_out, 'w')
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
