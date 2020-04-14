from __future__ import print_function
import os, sys, string
import numpy as np
import gauge_latticeqcd as gt
from multiprocessing import Pool

### Functions for better handling lattice collection
### also works with collections of measurements no matter if it is densities or simple numbers
### Functions for:
###   Loading configurations
###   calculating a specific operator passed -> with multiprocessing
###   calculate mean, variance for observables
###   calculate correlations
###   shift collections (ie add lag) 
###   BIN or SKIP observables

### Defines collection of lattice link configurations, takes links from folder
### ALSO WORKS as collection of anything --> useful for observables
def fn_lattice_collection(action, Nt, Nx, Ny, Nz, beta, start, end, step = 1, path = ""):
    collection = []
    for cfg in range(start, end + 1, step):
        U = fn_load_configuration(action, Nt, Nx, Ny, Nz, beta, cfg, path=path)
        collection.append(U)
    print("Configurations for beta ", beta, " collectetd.\n")
    return collection

### Loads configuration given the action, dimensions, beta and number of configurations.
### Configuration must be in folder in same directory.
def fn_load_configuration(action, Nt, Nx, Ny, Nz, beta, cfg, path = ""):
    name = action +'_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100))
    tmp = np.load(path + name + '/link_' + name + '_' + str(cfg))
    U = [[[[[0 for mu in range(4)] for z in range(Nz)] for y in range(Ny)] for x in range(Nx)] for t in range(Nt)]
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for mu in range(4):
                        U[t][x][y][z][mu] = np.matrix(tmp[t][x][y][z][mu])
    sys.stdout.flush()
    return U

### HELPER FUNCTION - ONLY TO ALLOW MULTIPROCESSING. 
### collects calls function and sends arguments properly
def helper(args):
    f = args[0]
    s = args[1:]
    return f(*s)

### Applies given function to EACH lattice configuration and returns a collection of the result for each
### Multiprocessing added -- need helper function to achieve
### A list is formed with all the arguments and the function and is called through helper
def fn_apply_function(collection, function, *args):
    job_args = [(function, item_a, *args) for i, item_a in enumerate(collection)]
    p = Pool(5)
    res = p.map(helper, job_args) #need helper function and to collect arguments as above
    p.terminate()                 #terminate multiprocessing -updated last if issues use p.close() 
    return res

### loads configuration using filename and dimensions (alternative to above)
### NOTE: (CHECK IF POSSIBLE TO OVERLOAD FUNCTION)
def fn_load_file_configuration(Nt, Nx, Ny, Nz, filename):
    tmp = np.load(filename)
    U = [[[[[0 for mu in range(4)] for z in range(Nz)] for y in range(Ny)] for x in range(Nx)] for t in range(Nt)]
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for mu in range(4):
                        U[t][x][y][z][mu] = np.matrix(tmp[t][x][y][z][mu])
    sys.stdout.flush()
    return U
   
### counts number of configurations in a path
def fn_config_counter(path):
    count = 0
    for file in os.listdir(path):
        if file.startswith('link_'+path):
            count += 1 
    return count


### Mean of collection - collection can be array of density matrices etc
def fn_average(collection):
    return np.average(collection, axis = 0)

def fn_mean(collection):
    return np.mean(collection, axis = 0)

### variance of collection
def fn_var(collection):
    return np.var(collection, axis = 0)


### bin cfgs in collection
def fn_bin(coll, Nbin):
    return np.array([fn_mean(coll[i*Nbin:(i+1)*Nbin]) for i in range((len(coll) + Nbin - 1) // Nbin) ])

### skip cfgs in collection
def fn_skip(coll, Nskip):
    return np.array([x for i, x in enumerate(coll) if i % Nskip == 0])



### Shifts array by some lag in specified direction t, x, y, z
### Useful for correlations
def fn_shift(collection, lag, direc):
    axis_direction = direc
    shifted_collection = np.roll(collection, lag, axis = axis_direction)
    return shifted_collection


def fn_correlation_averaging_with_error(collection, lag):
    mean = fn_mean(collection)
    var  = fn_var(collection)
    stdev= np.sqrt(var)
    
    shifted_collection = fn_shift(collection, lag, 0)

    for c in range(len(shifted_collection)):
        for direc in range(1, 4):
            shifted_collection[c] += fn_shift(collection, lag, direc)[c]
        shifted_collection[c] = shifted_collection[c] / 4.

    shifted_mean  = fn_mean(shifted_collection)
    shifted_var   = fn_var(shifted_collection)
    shifted_stdev = np.sqrt(shifted_var)

    tmp = np.zeros(np.shape(collection[0]))
    tmp_err = np.zeros(np.shape(collection[0]))

    for c in range(len(collection)):
        #treat as x_i whatever is in sum of definition of correlation i.e \frac{1}{N_c}\sum{ x_i  }
        #note that mean is an array. 
        x_i = np.multiply((collection[c] - mean), (shifted_collection[c] - shifted_mean))         
        tmp += x_i #for mean, simple addition
        tmp_err += np.multiply(x_i, x_i) #(elementwise) x_i^2 needed for variance
    

    tmp = np.divide(tmp, shifted_stdev)         #x_i includes division by standard deviations
    tmp = np.divide(tmp, stdev)
    tmp_err = np.divide(tmp_err, shifted_var)   #x_i^2 will include division by the square of standard deviations
    tmp_err = np.divide(tmp_err, var)

    tmp = tmp / len(collection) 
    tmp_err = tmp_err / len(collection)         #note that this division by Nc is not in definition of correlation
                                                #this is done only to find averaged x_i^2. so divide by Nc not Nc^2
    tmp_err = np.subtract(tmp_err, np.multiply(tmp,tmp)) #all elementwise

    return [tmp, tmp_err]


### Correlation without averaging in given direction for some lag
def fn_correlation(collection, lag, direc):
    mean = fn_mean(collection)
    var  = fn_var(collection)
    stdev= np.sqrt(var)
    
    shifted_collection = fn_shift(collection, lag, direc)
    shifted_mean  = fn_mean(shifted_collection)
    shifted_var   = fn_var(shifted_collection)
    shifted_stdev = np.sqrt(shifted_var)
    
    tmp = np.zeros(np.shape(collection[0]))
    for c in range(len(collection)):
        tmp += np.multiply((collection[c] - mean), (shifted_collection[c] - shifted_mean))

    tmp = np.divide(tmp, shifted_stdev)
    tmp = np.divide(tmp, stdev)
    return tmp / len(collection)

### alternative version
def fn_correlation_v2(collection, lag):
    
    shifted_collection = fn_shift(collection, lag, 0)
    
    #shift all position by lag. need index for loop because of the way fn_shift is implemented
    for c in range(len(shifted_collection)):
        for direc in range(1,4):
            shifted_collection[c] += fn_shift(collection, lag, direc)[c]
        shifted_collection[c] = shifted_collection[c] / 4.

    tmp = np.zeros(np.shape(collection[0]))
    tmp_er = np.zeros(np.shape(collection[0]))
    #treat O(x) \times  O(x+lag) as x_i element and determine variance and mean -> mean is correlation
    for c, shifted_c in zip(collection, shifted_collection):
        x_i = np.multiply(c, shifted_c)
        tmp += x_i
        tmp_er += np.multiply(x_i, x_i)

    tmp = tmp / len(collection) #correlation array 8x8x8x8
    tmp_er = tmp_er / len(collection) #second moment expectation <X^2> 
    tmp_er = tmp_er - np.multiply(tmp, tmp) #variance on each element of correlation array 8x8x8x8 <X^2> - <X>^2
    tmp_er = np.sqrt(tmp_er)                 #st. deviation
    tmp_er = tmp_er / (len(collection))      #error on mean

    return [tmp, tmp_er]

### normalised alternative version
def fn_correlation_v2_norm(collection, lag):
    shifted_collection = fn_shift(collection, lag, 0)
    
    #shift all position by lag. need index for loop because of the way fn_shift is implemented
    for c in range(len(shifted_collection)):
        for direc in range(1,4):
            shifted_collection[c] += fn_shift(collection, lag, direc)[c]
        shifted_collection[c] = shifted_collection[c] / 4.

    tmp = np.zeros(np.shape(collection[0]))
    tmp_er = np.zeros(np.shape(collection[0]))
    #treat O(x) \times  O(x+lag) as x_i element and determine variance and mean -> mean is correlation
    for c, shifted_c in zip(collection, shifted_collection):
        x_i = np.multiply(c, shifted_c)
        tmp += x_i
        tmp_er += np.multiply(x_i, x_i)

    tmp = tmp / len(collection) #correlation array 8x8x8x8
    tmp_er = tmp_er / len(collection) #second moment expectation <X^2> 
    tmp_er = tmp_er - np.multiply(tmp, tmp) #variance on each element of correlation array 8x8x8x8 <X^2> - <X>^2
    tmp_er = np.sqrt(tmp_er)                 #st. deviation
    tmp_er = tmp_er / (len(collection))      #error on mean

    return [tmp, tmp_er]

    

### Direction with averaging each direction -> SHOULD BE THE CORRECT FORM TO USE -> how can I check?
### NEED TO IMPLEMENT ERRORS THOUGH
def fn_correlation_averaging(collection, lag):
    mean = fn_mean(collection)
    var  = fn_var(collection)
    stdev= np.sqrt(var)
    
    shifted_collection = fn_shift(collection, lag, 0)
    
    for c in range(len(shifted_collection)):
        for direc in range(1,4):
            shifted_collection[c] += fn_shift(collection, lag, direc)[c]
        shifted_collection[c] = shifted_collection[c] / 4.

    shifted_mean  = fn_mean(shifted_collection)
    shifted_var   = fn_var(shifted_collection)
    shifted_stdev = np.sqrt(shifted_var)
    
    tmp = np.zeros(np.shape(collection[0]))
    for c in range(len(collection)):
        tmp += np.multiply((collection[c] - mean), (shifted_collection[c] - shifted_mean))

    tmp = np.divide(tmp, shifted_stdev)
    tmp = np.divide(tmp, stdev)
    return tmp / len(collection)
