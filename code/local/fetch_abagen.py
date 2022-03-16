"""
Gets the parcellated AHBA gene expression data using abagen
and saves it as numpy compressed format, to reduce the computational
costs, and also for maximum reproducibility of the analyses it is
important to freeze expression data as it may change in the future
"""
import abagen
import os
import datetime
import numpy as np

abspath = os.path.abspath(__file__)
cwd = os.path.dirname(abspath)
SRC_DIR = os.path.abspath(os.path.join(cwd, '..', '..', 'src'))
ABAGEN_DIR = '/data/group/cng/abagen-data'
if os.path.exists(ABAGEN_DIR):
    data_dir = ABAGEN_DIR
else:
    data_dir = None

for parcellation_name in ['aparc', 'schaefer400', 'schaefer1000', 'sjh']:
    print("Parcellation:", parcellation_name)
    expression_data = abagen.get_expression_data(
        atlas = os.path.join(SRC_DIR, f'tpl-MNI152_desc-{parcellation_name}_parcellation.nii.gz'),
        data_dir = data_dir,
        verbose = 2,
    )
    file_path = os.path.join(
        SRC_DIR, 
        f'ahba_parc-{parcellation_name}_frozen-{datetime.date.today():%Y%m%d}.npz'
    )
    np.savez_compressed(file_path, 
                        data=expression_data.values, 
                        columns=expression_data.columns,
                        index=expression_data.index)
    # Note: use `allow_pickle=True` when loading the data
    
