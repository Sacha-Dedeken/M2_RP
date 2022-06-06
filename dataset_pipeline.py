# useful fonctions to compute histograms and pre-process the image and the mask

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import os
from pydicom import dcmread
from utils import load_mri, all_mris, all_folders, MRI

patients_IPP = [int(x.name) for x in os.scandir('data/banque') if x.is_dir()]
pat_IPP_exams = [(IPP, [int(e.name) for e in os.scandir('data/banque/'+str(IPP)) if e.is_dir()]) for IPP in patients_IPP]

def compute_histo(frame, **kwargs):
    '''
        compute the histogram of the patch
    '''
    vmax = kwargs.get("vmax", frame.max())
    bins = kwargs.get("bins", vmax)
    density = kwargs.get("density", False)
    counts, values = np.histogram(frame.ravel(), bins=bins, range=(0, vmax), density=density)
    return counts

def std_mri(mri, mask):
    '''
    standardize the full image or the ROI depending of the mask (organ or patch)
    '''
    mri_z = (mri - mri[mask != 0].mean()) / mri[mask != 0].std() # transformation into a Z-score
    mri_std = 32 * ((mri_z - mri_z[mask != 0].min()) / (mri_z[mask != 0].max() - mri_z[mask != 0].min())) # standardization between 1 and 128
    mri_d = np.around(mri_std).astype(int) # discretization
    mri_m = np.where(mask == 0, 0, mri_d+1)
    return mri_m

data = np.zeros((7000, 36))-50

proportion = list()


# the "thresh" part of the code is used to filter the slices that contains too little mask in the image
# correspond to the Oscar Lambret brain dataset but can be useful to adapt on the prostate dataset
# just modify the iteration on the new image paths to compute each histogram
j=0
error = 0
thresh = 0.01 # minimum proportion of the brain mask in % of the entire slice
for IPP, nb_exam in pat_IPP_exams:
    for e in nb_exam:
        meta = load_mri(patient=IPP, exam=e)
        dev = 1*(meta.device == 'Vida')
        mri = np.array(nib.load("HD-BET/HD_BET/dataset_seg/"+str(IPP)+"_"+str(e)+".nii.gz").dataobj)
        mask = np.array(nib.load("HD-BET/HD_BET/dataset_seg/"+str(IPP)+"_"+str(e)+"_mask.nii.gz").dataobj)
        roi = np.where(mask == 0, 0, mri+1)
        roi = std_mri(roi, mask)
        
        for i in range(roi.shape[2]):
            proportion.append(np.sum(mask[:,:,i]) / (mask.shape[0]*mask.shape[1]))
            if np.sum(mask[:,:,i]) > thresh*mask.shape[0]*mask.shape[1]:
                h_freq = compute_histo(roi[:,:,i], vmax=33, density=True)
                h_freq = h_freq[1:] / np.sum(h_freq[1:])
                data[j] = np.concatenate((np.array([dev, IPP, e, i]), h_freq))
                j += 1
            else:
                error+=1

np.save("data_hist1D.npy", data)
np.save("proportion_mask.npy", proportion)







