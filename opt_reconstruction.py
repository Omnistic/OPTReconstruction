import os
import tifffile
import numpy as np
import opt_functions as opt

# Initialization
proj_folder = 'C:\\Users\\david\\Desktop\\opt_data\\'
recon_folder = 'C:\\Users\\david\\Desktop\\reco\\'
prefix = 'recon'

proj_names = os.listdir(proj_folder)
proj_path = proj_folder + proj_names[0]
proj = tifffile.imread(proj_path)

height, width = proj.shape
angles = len(proj_names)
theta = np.linspace(0, 2 * np.pi, angles)

cor = []
tentative_cor = width // 2
coarse_range = 15
coarse_range = range(-coarse_range, coarse_range+1)
coarse_step = 1
init_fine_range = 7
init_fine_range = range(-init_fine_range, init_fine_range+1)
fine_range = 2
fine_range = range(-fine_range, fine_range+1)
fine_step = 0.125 

max_height = 100
tomo_height = max_height
tomo_init = height // 2
tomo_start = tomo_init
tomo_indx = tomo_start
tomo_stop = tomo_start + min(tomo_height, height)

options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}

processing = True
init_cor = True
ascending = True
height_exception = False
test = []

print('Reconstruction started ...')
opt.tic()

# Start tomogram processing
while processing:
    # Special case: trailing portions of the tomogram
    if height_exception:
        tomo_height = tomo_stop - tomo_start
    
    # Read tomogram
    tomo = np.zeros((angles, tomo_height, width))
    opt.read_tomo(proj_folder,
                  proj_names,
                  angles,
                  tomo,
                  tomo_start,
                  tomo_stop)
    
    # Initialize the center of rotation
    if init_cor:
        tentative_cor, _ = opt.scan_cor(coarse_range,
                                        coarse_step,
                                        tentative_cor,
                                        tomo[:, 0, None, :],
                                        theta,
                                        options)
                
        tentative_cor, _ = opt.scan_cor(init_fine_range,
                                        fine_step,
                                        tentative_cor,
                                        tomo[:, 0, None, :],
                                        theta,
                                        options)
        
        center_cor = tentative_cor
        init_cor = False
    
    # Reconstruct tomogram
    if ascending:
        slice_range = range(tomo_height)
    else:
        slice_range = range(tomo_height-1, -1, -1)
        
    for slice_indx in slice_range:
        tentative_cor, recon = opt.scan_cor(fine_range,
                                            fine_step,
                                            tentative_cor,
                                            tomo[:, slice_indx, None, :],
                                            theta,
                                            options)
     
        cor.append(tentative_cor)
        
        # Save individual slices
        save_path = (recon_folder
                     + prefix
                     + '_'            
                     + '{:0>4d}'.format(tomo_indx)
                     + '.tif')
        tifffile.imsave(save_path, recon)
        
        if ascending:
            tomo_indx += 1
        else:
            tomo_indx -= 1
                
    # Adjust tomogram indexes for next iteration
    if height_exception:
        tomo_height = max_height
        height_exception = False

    if ascending:
        if tomo_stop == height:
            ascending = False
            tentative_cor = center_cor
            tomo_indx = tomo_init - 1
            tomo_stop = tomo_init
            tomo_start = tomo_stop - tomo_height
        else:
            tomo_start += tomo_height
            tomo_stop += tomo_height
            
            if tomo_stop > height:
                tomo_stop = height
                height_exception = True
    else:
        if tomo_start == 0:
            processing = False
        else:
            tomo_start -= tomo_height
            tomo_stop -= tomo_height
            
            if tomo_start < 0:
                tomo_start = 0
                height_exception = True

# Complete processing
eta_s = int(opt.toc())
eta_m = eta_s // 60
eta_s -= eta_m * 60

print(('Reconstruction complete (elapsed time: '
       + str(eta_m)
       + ' minutes, and '
       + str(eta_s)
       + ' seconds)'))