# -*- coding: utf-8 -*-
import random
import numpy as np
import xarray as xr
from xarray import Dataset


def optimal_interpolation(syn_prod: np.ndarray, y_meas: float, back_sel: list, map_func: str) -> np.ndarray:
    """
    Performs optimal interpolation between one measurement, y_meas, and an array of background data, back_sel.

    Parameters:
        syn_prod (ndarray): a matrix full of zeros in which the results of the operation will be stored.
        y_meas (float): the measurement value.
        back_sel (list): the background values.
        map_func (str): the type of mapping function that we want to use.

    Returns:
        syn_prod (ndarray): the same syn_prod than before, but modified by the operation. Now it bears
        the results of OI.
    """

    xb = np.array([itm[2] for itm in back_sel])
    xb = xb.reshape((len(xb), 1))

    # Creation of R matrix
    R = (0.5 * y_meas) ** 2

    # Defining operator H(xb)
    supported_functions = ["mean"]
    H = None
    Hdot = None

    if map_func not in supported_functions:
        raise NotImplementedError(f"H(xb) must be chosen between {supported_functions}, "
                                  f"you provided {map_func}")
    else:
        if map_func == "mean":
            H = sum(xb) / len(xb)
            Hdot = np.ones((1, len(xb)))
            Hdot = Hdot * 1 / len(xb)

    # Creation of Pb matrix
    '''Ask Leonardo for the data to form err'''
    err = np.array([random.uniform(0, 0.5) for _ in range(0, len(xb))])
    err = err.reshape((len(err), 1))
    eb = np.multiply(xb, err)

    Pb = np.outer(eb, eb.transpose())
    try:
        if not Pb.shape == (len(xb), len(xb)):
            raise Exception(f"Pb shape: {Pb.shape}, should be {(len(xb), len(xb))}")
    except Exception as e:
        print(e)

    # Building K matrix
    Num = np.dot(Pb, np.transpose(Hdot))
    Den = np.dot(Hdot, Num) + R
    # assert Num.shape == (25, 1) and Den.shape == (1, 1)

    K = Num / Den
    # assert K.shape == (25, 1)

    # Optimal interpolation
    xa = xb + (K * (y_meas - H))

    # Loading the data in the syn_prod matrix
    for item in range(0, len(back_sel)):
        m = back_sel[item][0]
        n = back_sel[item][1]
        syn_prod[m][n] += xa[item]

    return syn_prod


def processor(coarse: np.ndarray, smooth: np.ndarray, c_lat: np.ndarray, c_lon: np.ndarray,
              s_lat: np.ndarray, s_lon: np.ndarray, varname: str, interpolation, mapping: str = "mean") -> Dataset:
    # coarse = TROPOMI, smooth = OLCI
    """
    Perform optimal interpolation between 2 datasets: it loops through one of the 2 datasets,
    selects the corresponding part of the other dataset, and executes the optimal interpolation.
    The function was created to operate on one satellite dataset with low spatial resolution (coarse),
    and one with high spatial resolution (smooth).

    Args:
        coarse (ndarray): array of values of the low resolution dataset.
        smooth (ndarray): array of values of the high resolution dataset.
        c_lat (ndarray): array of the latitudes at which the low resolution values were retrieved.
        c_lon (ndarray): array of the longitudes at which the low resolution values were retrieved.
        s_lat (ndarray): array of the latitudes at which the high resolution values were retrieved.
        s_lon (ndarray): array of the longitudes at which the high resolution values were retrieved.
        varname (str): name of the variable that we're measuring.
        interpolation: the mathematical scaffold with which we want to interpolate the data.
        mapping (str): mapping function to pass to optimal_interpolation.

    Returns:
        syne (ndarray): result array bearing optimal interpolation values.
    """

    # Dynamic coarse/smooth ratio
    '''I assuming that the pixels are rectangles or squares.'''
    lat_ratio = np.int32(np.round(s_lat / c_lat))
    lon_ratio = np.int32(np.round(s_lon / c_lon))
    print(f"coarse/smooth latitude ratio = {lat_ratio}°")
    print(f"coarse/smooth latitude ratio = {lon_ratio}°")

    # Creating the matrix for storing results
    syne = np.zeros(smooth.shape)  # same resolution of the OLCI matrix

    # Looping through the TROPOMI grid
    indi_coarse = np.ndindex(coarse.shape)
    map_coarse = np.array([[c_lat[a, b], c_lon[a, b], coarse[a, b]] for a, b in indi_coarse])

    indi_smooth = np.ndindex(smooth.shape)
    map_smooth = np.array([[s_lat[m, n], s_lon[m, n], smooth[m, n]] for m, n in indi_smooth])

    try:
        if not len(map_smooth) == smooth.size:
            raise Exception("Something was wrong in the formation of the mapping array: "
                            "map_smooth.size != smooth.size")
        if not len(map_coarse) == coarse.size:
            raise Exception("Something was wrong in the formation of the mapping array: "
                            "map_coarse.size != coarse.size")
    except Exception as e:
        print(e)
        print(f"len(map_coarse), coarse.size: {len(map_coarse), coarse.size}")
        print(f"len(map_smooth), smooth.size: {len(map_smooth), smooth.size}")

    for i in range(0, len(map_coarse)):
        # Selection of the OLCI pixels

        indices = np.where(
            (map_smooth[:, 0] <= (map_coarse[i, 0] + (lat_ratio / 2))) &
            (map_smooth[:, 0] >= (map_coarse[i, 0] - (lat_ratio / 2))) &
            (map_smooth[:, 1] <= (map_coarse[i, 1] + (lon_ratio / 2))) &
            (map_smooth[:, 1] >= (map_coarse[i, 1] - (lon_ratio / 2)))
        )

        selected = [map_smooth[index, 2] for index in indices]

        assert len(selected) == (lat_ratio * lon_ratio), (f"Selected OLCI pixels: {len(selected)}, "
                                                          f"right_len: {lat_ratio * lon_ratio}")

        if interpolation == "Optimal Interpolation":
            optimal_interpolation(syne, map_coarse[i, 2], selected, mapping)

    # Loading of results into a xr.Dataset
    synergistic = xr.DataArray(syne, dims=('x', 'y'), coords={'x': range(0, syne.shape[0]),
                                                              'y': range(0, syne.shape[1])})
    latitude = xr.DataArray(s_lat, dims=('x', 'y'), coords={'x': range(0, s_lat.shape[0]),
                                                            'y': range(0, s_lat.shape[1])})
    longitude = xr.DataArray(s_lon, dims=('x', 'y'), coords={'x': range(0, s_lon.shape[0]),
                                                             'y': range(0, s_lon.shape[1])})
    results = xr.Dataset({varname: synergistic, 'latitude': latitude, 'longitude': longitude})

    return results
