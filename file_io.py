# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime as dt
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from netCDF4 import Dataset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def list_files(base_dir: str, date_index: pd.DatetimeIndex, extension: str) -> list:
    """
    Taking a directory of your choice, the function returns a list of files that meet the conditions
    specified by the parameters.
    Args:
        base_dir (str): the directory to inquire.
        date_index (pd.DatetimeIndex): the dates to which the files should refer to.
        extension (str): the extension of the files.

    Returns:
        lst (list): list of files that meet the parameters' conditions.
    """

    base_dir = Path(base_dir)

    lst = []

    pattern = f'*.{extension}'  # pattern of file to read

    for d in date_index:
        path = Path(base_dir) / f'{d.year:04d}' / f'{d.month:02d}' / f'{d.day:02d}'
        files = glob(str(path / pattern))
        lst.extend(files)

    return lst


def load_dataset(file, varname: str):  # this function will be expanded for future versions
    """
    Taking a file which can be opened as a xarray dataset, the function returns the array with the values
    associated to each pixel and the arrays containing the latitudes and longitudes of the center point
    of each pixel.
    Args:
        file: the xarray dataset we are referring to
        varname: the variable we are referring to

    Returns:
        lat: the array with the latitude coordinates of the center point of each pixel
        lon: the array with the longitude coordinates of the center point of each pixel
        arr: the array with the values associated to each pixel
    """
    file = Path(file)

    ds = xr.open_dataset(file)

    lat = ds.latitude.values
    lon = ds.longitude.values
    arr = ds[varname].values

    return arr, lat, lon


def output_saving(out_dataset, varname, reference, out_dir, type_of_interpolation, date="yes"):
    """
    This function saves each assimilated dataset in a specific location, associating it with some useful
    information.
    Args:
        out_dataset: the xarray element to be saved
        varname: the variable represented by each interpolation dataset
        reference: the reference data
        out_dir: the directory in which out_dataset is saved
        type_of_interpolation: with a view to including different interpolation methods, this variable
        takes into account the one that you are using
        date: if this variable is yes, the function will create a series of directories in out_dir
        to store the out_datasets in a chronologically ordered manner

    Returns:
        out_file_path: the path to the saved output dataset

    """
    pattern = re.compile(r"(\d{8}T\d{6})")
    match = pattern.search(reference)
    if not match:
        raise ValueError(f"{reference} does not contain a date")

    date_time_str = match.group(1)
    year = date_time_str.strftime("%Y")
    month = date_time_str.strftime("%m")
    day = date_time_str.strftime("%d")

    # Loading out_file into the destination repository
    destination_dir = None

    if not date:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            destination_dir = out_dir

    else:
        if not os.path.exists(os.path.join(out_dir, year, month, day)):
            os.makedirs(os.path.join(out_dir, year, month, day))
            destination_dir = os.path.join(out_dir, year, month, day)

    # Conversion of out_file in NetCDF format
    out_file_name = f'synergistic_product_{date_time_str}.nc'  # manage output name
    out_file_path = os.path.join(destination_dir, out_file_name)

    with Dataset(out_file_path, 'w') as out:
        # General information on code and data
        out.title = (f'{type_of_interpolation} of TROPOMI and OLCI Level-2 PFT products based on '
                     f'chlorophyll-a satellite data')
        out.source = 'remote sensing'
        out.creation_date = str(dt.now())[:19]

        out.info = (f'{type_of_interpolation} is a method for merging data of different kind. In this case '
                    f'it was used to merge the data from OLCI and TROPOMI')
        # out.version = version

        out.originators = 'Daniele Di Bella and Leonardo Alvarado'
        out.originators_contributors = 'Mariana A. Soppa, Svetlana N. Losa, Astrid Bracher et al.'
        out.originators_institution = 'Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, Germany'
        out.contact = 'daniele.dibella@awi.de, leonardo.alvarado@awi.de'
        out.project_number = '?'

        out.Data_conventions = 'CF-1.6'
        out.crs = 'EPSG:4326'

        # Save output information
        out.createDimension('x', out_dataset.varname.values.shape[0])
        out.createDimension('y', out_dataset.varname.values.shape[1])

        lat = out.createVariable('latitude', 'f4', ('x', 'y'), zlib=True, fill_value=np.nan)
        lat[:, :] = out_dataset.latitude[:, :]

        lat.units = 'degrees_north'
        lat.standard_name = 'latitude'
        lat.long_name = 'Latitude in degrees'
        lat.coordinates = 'x y'

        lon = out.createVariable('longitude', 'f4', ('x', 'y'), zlib=True, fill_value=np.nan)
        lon[:, :] = out_dataset.longitude[:, :]

        lon.units = 'degrees_east'
        lon.standard_name = 'longitude'
        lon.long_name = 'Longitude in degrees'
        lon.coordinates = 'x y'

        test = out.createVariable(varname, 'f4', ('x', 'y'), zlib=True, fill_value=np.nan)
        test[:, :] = out_dataset.varname[:, :]

        test.units = 'molecules cm^{-3}'
        test.standard_name = f'{varname}'
        test.long_name = f'Concentration of {varname}'
        test.coordinates = 'lat lon'

        out.close()

    return out_file_path


def plotter(dataset_path, varname):
    """
    With the help of cartopy, this function is devoted to the graphical representation of the output
    xarray datasets
    Args:
        dataset_path: the path leading to the dataset which should be represented
        varname: the variable to represent

    Returns:

    """
    # Fetching data
    ds = nc.Dataset(dataset_path, 'r')
    latitudes = ds.latitude[:, :]
    longitudes = ds.longitude[:, :]
    var = ds[varname][:, :]

    # Figure creation
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    cax = ax.imshow(var, extent=(longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()),
                    transform=ccrs.PlateCarree(), cmap='viridis', origin='lower')

    cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.03, pad=0.1, shrink=1)
    cbar.set_label('Chlorophyll-a [molecules cm^{-3}]')

    ax.gridlines(draw_labels=True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Chlorophyll-a concentration')
    ax.set_global()
    ax.coastlines()

    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    plt.show()
