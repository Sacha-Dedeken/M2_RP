#%%
import os
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from dataclasses import dataclass, field

from pydicom import dcmread

class MRI:
    def __init__(self, patient, exam, shape, **kwargs):
        self.patient = patient
        self.exam = exam
        
        root = kwargs.get("root", "data/banque")
        default_dir = os.path.join(root, str(patient), str(exam), "DICOM")
        self._dir = kwargs.get("dir", default_dir)
        
        self._device = None
        self._slices = np.zeros(shape, dtype=int)
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        for filename in os.listdir(self._dir):
            mri_dicom = dcmread(f"{self._dir}/{filename}")
            k = mri_dicom.InstanceNumber - 1  # index of the slice
            self._slices[k] = mri_dicom.pixel_array
        self._device = mri_dicom.ManufacturerModelName[-4:]
        self._loaded = True

    @property
    def slices(self):
        if not self._loaded:
            self._load()
        return self._slices

    @property
    def device(self):
        if not self._loaded:
            self._load()
        return self._device
    
    @property
    def shape(self):
        return self._slices.shape

    def __repr__(self):
        return f"MRI(patient={self.patient:>6}, exam={self.exam})"


def load_mri(patient, exam, root="data/banque/"):
    """Load a given exam on a given patient and return it as an instance of MRI class"""
    patient, exam = int(patient), int(exam)
    
    # Retrieve shape of the 3D MRI
    exam_dir = os.path.join(root, str(patient), str(exam))
    mri_descr = dcmread(f"{exam_dir}/DICOMDIR").DirectoryRecordSequence
    nx, ny = mri_descr[3].Rows, mri_descr[3].Columns
    nz = len(mri_descr) - 3
    # mri_img = np.zeros((nz, nx, ny), dtype=int)
    # 
    # # Load each slice
    # dir = f"{exam_dir}/DICOM"
    # for filename in os.listdir(dir):
    #     mri_dicom = dcmread(f"{dir}/{filename}")
    #     k = mri_dicom.InstanceNumber - 1  # index of the slice
    #     mri_img[k] = mri_dicom.pixel_array

    # device = mri_dicom.ManufacturerModelName[-4:]

    # return(MRI(patient, exam, device, mri_img))
    return MRI(patient, exam, (nz, nx, ny))


def all_mris(root="data/banque/"):
    """Load and generate each MRI contained in the root folder"""
    for root, dirs, files in os.walk("data/banque"):
        if os.path.basename(root) == "DICOM" and files:
            parent_dir = os.path.dirname(root)
            exam = os.path.basename(parent_dir)
            patient = os.path.basename(os.path.dirname(parent_dir))
            
            yield load_mri(patient, exam)


def all_folders(root="data/banque/", subfolder=""):
    """List all the folders inside a subfolder contained in the root folder"""
    folder = os.path.join(root, subfolder)
    return [
        dir for dir in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, dir))]

# %% –––– Miscellaneous –––––––––––––––––––––––––––––––––––––––––––––––––––––––

def circular_mask(h, w, center=None, radius=None):
    """
    Given the shape of a 2D array, create a circular mask
    centered at 'center' with a radius 'radius'
    """
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def print_loading(i, n, txt=""):
    """Display a loading bar when i out of n jobs are finished."""
    m = int((100 * i / n) // 5)
    print(f"\r{i}/{n}", "[" + "="*m + " "*(20-m) + "]",f"({100 * i / n:.0f}%)  {txt}", end='')
