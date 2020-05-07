import numpy as np
import tifffile
import time
import tomopy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Read a tomogram by modifying an array in place
def read_tomo(proj_folder, proj_names, angles, tomo, start, stop):
    for indx in range(angles):
        proj_path = proj_folder + '/' + proj_names[indx]
        proj = tifffile.imread(proj_path)
        tomo[indx, :, :] = proj[start:stop, :]
      

# Perform reconstructions over a range, and pick the position of best variance
def coarse_scan_cor(scan_range, step, guess, tomo, theta, options):
    var_array = []
    cor_array = []
    
    for range_indx in scan_range:
        tentative_cor = guess + range_indx * step
        cor_array.append(tentative_cor)
        
        recon = tomopy.recon(tomo,
                             theta,
                             center=tentative_cor,
                             algorithm=tomopy.astra,
                             options=options)
        
        var = np.var(recon)
        var_array.append(var)
                 
    var_range = max(var_array) - min(var_array)
    var_prom = var_range / 10
    cors, _ = find_peaks(var_array, prominence=var_prom)

    num_cors = len(cors)

    plt.figure()
    plt.plot(cor_array, var_array, alpha=0.5)
    plt.scatter(cor_array, var_array, marker='.')
    plt.scatter([cor_array[range_indx] for range_indx in cors],
                [var_array[range_indx] for range_indx in cors],
                marker='X',
                label='Candidate centers of rotation')
    plt.xlabel('Center of rotation [px]')
    plt.ylabel('Reconstruction variance [-]')
    plt.legend()
    plt.grid(True)
    for cor_indx in cors:
        plt.annotate(str(cor_array[cor_indx]),
                     (cor_array[cor_indx], var_array[cor_indx]))
    plt.show()

    if num_cors == 1:
        cor = cor_array[cors[0]]
    else:
        for cor_indx in cors:
            recon = tomopy.recon(tomo,
                                 theta,
                                 center=cor_array[cor_indx],
                                 algorithm=tomopy.astra,
                                 options=options)
            plt.figure()
            plt.title('Center of rotation = ' + str(cor_array[cor_indx]))
            plt.imshow(recon[0, :, :])
            plt.show()
            
        print('Specify which center of rotation to use ...')
        
        input_cor = True
        while input_cor:
            try:
                cor = int(input())
                
                for cor_indx in cors:
                    if cor == cor_array[cor_indx]:
                        input_cor = False
                        print('Using start center: ' + str(cor))
                
                if input_cor:     
                    print('Please input one of the identified '
                          + 'center of rotation ...')
            except:
                print('Please input an integer corresponding to a '
                      + 'candidate center of rotation ...')
            
    return cor


# Perform reconstructions over a range, and pick the position of best variance
def fine_scan_cor(scan_range, step, guess, tomo, theta, options):
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