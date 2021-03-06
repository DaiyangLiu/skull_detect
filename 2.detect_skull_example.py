import numpy as np
import nibabel as nib
import copy
import os
from skull_detector import *

if __name__ == '__main__':
    no_skull_dir = 'F://brain//LGG-1p19Detection//LGG-1p19Detection-Manual-Sorted//T1AT2//'
    no_skull_mask_dir = './/T1AT2_segmentation//'
    no_skull_dir_list = os.listdir(no_skull_dir)
    print(len(no_skull_dir_list))
    count_correct = 0
    for patient_id in no_skull_dir_list:
        patient_t1 = no_skull_dir + patient_id + '//' + patient_id + '_T1.nii.gz'
        patient_mask = no_skull_mask_dir + patient_id + '_T1_mask.nii.gz'
        t1_arr = nib.load(patient_t1).get_data()
        mask_arr = nib.load(patient_mask).get_data()

        skull_de =  skull_detector(t1_arr,mask_arr)
        result = skull_de.skull_detect()

        if result == True:
            count_correct+=1
    print('count_correct:',count_correct)