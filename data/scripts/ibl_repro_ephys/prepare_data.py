"""Load data, processes it, save it."""

import sys
import numpy as np
np.random.seed(42)
import pandas as pd
from datetime import datetime
from one.api import ONE
from kirby.data import (
    Data, 
    IrregularTimeSeries, 
    Interval, 
    DatasetBuilder, 
    ArrayDict
)
from kirby.taxonomy import (
    Task,
    Sex,
    Species,
    SubjectDescription,
)
path_root = '/home/yzhang39/IBL_foundation_model/'
sys.path.append(str(path_root))
from src.utils.ibl_data_utils import (
    prepare_data,
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    load_anytime_behaviors
)
import logging
logging.basicConfig(level=logging.INFO)

SAVE_PATH = "EXAMPLE PATH"
EID = "db4df448-e449-4a6f-a0e7-288711e7a75a"

freeze_file = f'{path_root}/data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

params = {
    'interval_len': 2, 'binsize': 0.02, 'single_region': False, 
    'align_time': 'stimOn_times', 'time_window': (-.5, 1.5), 'fr_thresh': 0.5
}

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', mode='remote'
)


# Load Data




# Extract Spiking Activity





# Extract Trials



# Extract Behavior









