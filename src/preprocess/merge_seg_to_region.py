import os
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm

TOTAL_LUT = np.zeros(118, dtype=np.uint8)
TOTAL_LUT[1] = 16; TOTAL_LUT[2] = 20; TOTAL_LUT[3] = 20; TOTAL_LUT[4] = 15
TOTAL_LUT[5] = 14; TOTAL_LUT[6] = 18; TOTAL_LUT[7] = 17; TOTAL_LUT[8] = 21
TOTAL_LUT[9] = 21; TOTAL_LUT[10] = 8; TOTAL_LUT[11] = 8; TOTAL_LUT[12] = 8
TOTAL_LUT[13] = 8; TOTAL_LUT[14] = 8; TOTAL_LUT[15] = 11; TOTAL_LUT[16] = 9
TOTAL_LUT[17] = 6; TOTAL_LUT[18] = 19; TOTAL_LUT[19] = 19; TOTAL_LUT[20] = 19
TOTAL_LUT[21] = 22; TOTAL_LUT[22] = 23; TOTAL_LUT[23] = 20; TOTAL_LUT[24] = 20
TOTAL_LUT[25:51] = 27
TOTAL_LUT[51] = 10; TOTAL_LUT[52] = 9; TOTAL_LUT[53] = 9; TOTAL_LUT[54] = 9
TOTAL_LUT[55] = 9; TOTAL_LUT[56] = 9; TOTAL_LUT[57] = 7; TOTAL_LUT[58] = 7
TOTAL_LUT[59] = 9; TOTAL_LUT[60] = 9; TOTAL_LUT[61] = 10; TOTAL_LUT[62] = 9
TOTAL_LUT[63] = 24; TOTAL_LUT[64] = 24; TOTAL_LUT[65] = 24; TOTAL_LUT[66] = 24
TOTAL_LUT[67] = 24; TOTAL_LUT[68] = 24
TOTAL_LUT[69:79] = 27
TOTAL_LUT[79] = 28; TOTAL_LUT[80:90] = 28
TOTAL_LUT[90] = 1; TOTAL_LUT[91] = 1
TOTAL_LUT[92:118] = 27

HEAD_LUT = np.zeros(20, dtype=np.uint8)
HEAD_LUT[1] = 2; HEAD_LUT[2] = 2; HEAD_LUT[3] = 2; HEAD_LUT[4] = 2
HEAD_LUT[5] = 2; HEAD_LUT[6] = 2; HEAD_LUT[7] = 6; HEAD_LUT[8] = 6
HEAD_LUT[9] = 6; HEAD_LUT[10] = 6; HEAD_LUT[11] = 3; HEAD_LUT[12] = 4
HEAD_LUT[13] = 5; HEAD_LUT[14] = 3; HEAD_LUT[15] = 3; HEAD_LUT[16] = 28
HEAD_LUT[17] = 28; HEAD_LUT[18] = 4; HEAD_LUT[19] = 4

OCULO_LUT = np.zeros(20, dtype=np.uint8)
OCULO_LUT[2:20] = 2


def process_case(case_dir):
    seg_path = os.path.join(case_dir, "seg.nii.gz")
    head_path = os.path.join(case_dir, "head_glands_cavities.nii.gz")
    oculo_path = os.path.join(case_dir, "oculomotor_muscles.nii.gz")
    out_path = os.path.join(case_dir, "region.nii.gz")

    # if os.path.exists(out_path):
    #     return case_dir, "skipped"

    seg_nii = nib.load(seg_path)
    seg_data = np.rint(seg_nii.get_fdata(dtype=np.float32)).astype(np.int16)
    result = np.zeros(seg_data.shape, dtype=np.uint8)

    mask = (seg_data > 0) & (seg_data < 118)
    result[mask] = TOTAL_LUT[seg_data[mask]]

    head_data = np.rint(nib.load(head_path).get_fdata(dtype=np.float32)).astype(np.int16)
    mask = (head_data > 0) & (head_data < 20)
    result[mask] = HEAD_LUT[head_data[mask]]

    oculo_data = np.rint(nib.load(oculo_path).get_fdata(dtype=np.float32)).astype(np.int16)
    mask = (oculo_data > 1) & (oculo_data < 20)
    result[mask] = OCULO_LUT[oculo_data[mask]]

    out_hdr = seg_nii.header.copy()
    out_hdr.set_data_dtype(np.uint8)
    out_hdr.set_slope_inter(1, 0)
    out_nii = nib.Nifti1Image(result, seg_nii.affine, out_hdr)
    nib.save(out_nii, out_path)
    return case_dir, "done"
