import numpy as np
import tifffile
import time
import tomopy


# Read a tomogram by modifying an array in place
def read_tomo(proj_folder, proj_names, angles, tomo, start, stop):
    for indx in range(angles):
        proj_path = proj_folder + proj_names[indx]
        proj = tifffile.imread(proj_path)
        tomo[indx, :, :] = proj[start:stop, :]
      

# Perform reconstructions over a range, and pick the position of best variance
def scan_cor(scan_range, step, guess, tomo, theta, options):
    var = -1
    
    for indx in scan_range:
        tentative_cor = guess + indx * step
        
        recon = tomopy.recon(tomo,
                             theta,
                             center=tentative_cor,
                             algorithm=tomopy.astra,
                             options=options)
        
        temp_var = np.var(recon)
        if temp_var > var:
            var = temp_var
            cor = tentative_cor
            ret = recon
            
    return cor, ret


# Setup a timer
def tic():
    global timer
    timer = time.time()


# Retrieve timer value
def toc():
    return time.time() - timer