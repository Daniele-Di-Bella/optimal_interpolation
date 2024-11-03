# -*- coding: utf-8 -*-
import sys
import warnings
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from glob import glob
from file_io import list_files, load_dataset, output_saving, plotter
from processor import processor

warnings.filterwarnings('ignore')

__author__ = 'Daniele Di Bella and Leonardo Alvarado'
__email__ = 'leonardo.alvarado@awi.de'
__version__ = '0.0.0'
__release__ = '26.08.2023'


def main(base_dir_ds1=None,
         base_dir_ds2=None,
         extension='nc',
         date_start='yyyy-mm-dd',
         date_end='yyyy-mm-dd',
         do_files_from_dates=False,
         file_pattern_ds1='/path/to/**/fi*.nc',
         file_pattern_ds2='/path/to/**/fi*.nc',
         file_format_in_ds1='s5p_l3',
         file_format_in_ds2='s3a_l3',
         interpolation_method='Optimal Interpolation',  # other methods could be implemented
         out_dir=None,
         varname=None):
    _FILE_FORMAT_IN_SUPPORTED = ["s5p_l3", "s3a_l3",
                                 "s3b_l3"]  # In future versions additional products can be included

    current_time = datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')

    print(f'Synergy processing starts at {formatted_time}')

    # Get the list of files to process
    files_ds1 = None
    files_ds2 = None

    if do_files_from_dates:
        date_index = pd.date_range(date_start, date_end, freq='D')

        if file_format_in_ds1.lower() == 's5p_l3':
            files_ds1 = list_files(base_dir_ds1, date_index, extension)
            if not files_ds1:
                raise FileNotFoundError(f'No files match: {base_dir_ds1}, {date_start}--{date_end} and {extension}')

        else:
            if file_format_in_ds2.lower() == 's3a_l3' or file_format_in_ds2.lower() == 's3b_l3':
                files_ds2 = list_files(base_dir_ds2, date_index, extension)
                if not files_ds2:
                    raise FileNotFoundError(f'No files match: {base_dir_ds2}, {date_start}--{date_end} and {extension}')

            else:
                raise NotImplementedError(
                    f'file_format_in must be in {_FILE_FORMAT_IN_SUPPORTED}. You provided {file_format_in_ds1} and {file_format_in_ds2}')

    else:
        files_ds1 = glob(file_pattern_ds1)  # list of TROPOMI files
        files_ds2 = glob(file_pattern_ds2)  # list of OLCI files

        if not files_ds1:
            raise FileNotFoundError(f'No files match: {file_pattern_ds1}')

        if not files_ds2:
            raise FileNotFoundError(f'No files match: {file_pattern_ds2}')

    # Data processing
    for fileDs1, fileDs2 in tqdm(files_ds1, file=sys.stdout, unit='orbit'), tqdm(files_ds2, file=sys.stdout,
                                                                                 unit='granule'):
        arr_ds1, lat_ds1, lon_ds1 = load_dataset(fileDs1, varname)
        arr_ds2, lat_ds2, lon_ds2 = load_dataset(fileDs2, varname)

        print(f'DOAS dataset loaded: {fileDs1}')
        print(f'OC-PFT dataset loaded: {fileDs2}')

        output = processor(arr_ds1, arr_ds2, lat_ds1, lon_ds1, lat_ds2, lon_ds2, varname, interpolation_method)

        # Saving results
        output_saving(output, varname, fileDs2, out_dir,
                      interpolation_method)  # the parameter "date" is assumed to be "yes"
        ds_path = output_saving(output, varname, fileDs2, out_dir, interpolation_method)

        # Plotting
        plotter(ds_path, varname)


if __name__ == "__main__":
    main(base_dir_ds1='/home/alvarado/projects/tmp/s5poc/cya/',
         base_dir_ds2='/home/alvarado/projects/tmp/s5poc/cya/',
         extension='nc',
         date_start='2018-06-01',
         date_end='2018-06-09',
         do_files_from_dates=True,
         file_pattern_ds1='/path/to/**/fi*.nc',
         file_pattern_ds2='/path/to/**/fi*.nc',
         file_format_in_ds1='s5p_l3',
         file_format_in_ds2='s3a_l3',
         interpolation_method='Optimal Interpolation',
         out_dir=None,
         varname='CYA')
