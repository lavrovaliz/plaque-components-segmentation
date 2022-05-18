# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 18:32:34 2021

@author: E.Lavrova
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os
import pydicom
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d(image, ps, ss):

    xmin = np.min(np.where(image > 0)[0])
    xmax = np.max(np.where(image > 0)[0])
    ymin = np.min(np.where(image > 0)[1])
    ymax = np.max(np.where(image > 0)[1])

    dx = xmax - xmin
    dy = ymax - ymin

    r = round(max(dx, dy) / 2)

    cx = round((xmax + xmin) / 2)
    cy = round((ymax + ymin) / 2)

    bbox_xmin = cx - r - 1
    bbox_ymin = cy - r - 1
    bbox_xmax = cx + r + 1
    bbox_ymax = cy + r + 1

    p = image[bbox_xmin:bbox_xmax, bbox_ymin:bbox_ymax]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    facecolors = ['r', 'b', 'yellow', 'cyan', 'g', 'k']
    alphas = [0.5, 0.15, 1, 1, 1, 1]

    for i in range(1, 7):

        if np.sum(p == i) > 0:
            verts, faces, _, _ = measure.marching_cubes(p == i, spacing=(ps[0],
                                                                         ps[1],
                                                                         ss))
            mesh = Poly3DCollection(verts[faces], alpha=alphas[i - 1])
            mesh.set_facecolor(facecolors[i - 1])
            ax.add_collection3d(mesh)

    ax.set_xlim(0, round(p.shape[0] * ps[0]))
    ax.set_ylim(0, round(p.shape[1] * ps[1]))
    ax.set_zlim(0, round(p.shape[2] * ss))
    ax.set_box_aspect((ps[0], ps[1], ss))
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.set_zlabel('mm', labelpad=50)
    ax.grid(False)

    plt.show()

def get_slice_roi_props(con2d_array, idx_roi, pix_spacing):

        d_hor = 0
        d_ver = 0
        d_min = 0
        d_max = 0
        area = 0

        dim = con2d_array.shape
        con_bin = con2d_array == idx_roi

        if np.sum(con_bin) > 0:

            xmin = np.min(np.where(con_bin == 1)[0])
            xmax = np.max(np.where(con_bin == 1)[0])
            ymin = np.min(np.where(con_bin == 1)[1])
            ymax = np.max(np.where(con_bin == 1)[1])
            d_hor = (xmax - xmin) * pix_spacing[0]
            d_ver = (ymax - ymin) * pix_spacing[1]

            props = measure.regionprops(con_bin.astype(int))
            d_min = (props[0].minor_axis_length) * pix_spacing[0]
            d_max = (props[0].major_axis_length) * pix_spacing[0]

            bound_map = np.zeros(dim, dtype=np.uint8)
            xs = np.where(con_bin > 0)[0]
            ys = np.where(con_bin > 0)[1]
            for x, y in zip(xs, ys):
                if len(np.unique(con_bin[x - 1:x + 2, y - 1:y + 2])) > 1:
                    bound_map[x, y] = 1

            con_inner = con_bin - bound_map

            area = (np.sum(con_inner) + np.sum(bound_map) / 2) * pix_spacing[0] * pix_spacing[1] / 100

        return d_hor, d_ver, d_min, d_max, area


def get_volume_ml(con3d_array, idx_roi, sl_thick, pix_spacing):
    a = 0
    dim = con3d_array.shape
    con_bin = con3d_array == idx_roi

    if np.sum(con_bin) > 0:

        bound_map = np.zeros(dim, dtype=np.uint8)
        for z in range(0, dim[2]):
            img = con_bin[..., z]
            xs = np.where(img > 0)[0]
            ys = np.where(img > 0)[1]
            for x, y in zip(xs, ys):
                if len(np.unique(img[x - 1:x + 2, y - 1:y + 2])) > 1:
                    bound_map[x, y, z] = 1

        con_inner = con_bin - bound_map
        a = ((np.sum(con_inner) + np.sum(bound_map) / 2) * sl_thick * pix_spacing[0] * pix_spacing[1] / 1000)

    return a


def con3d_from_dcmdir(con_dir_name):
    dcms = []

    for item in os.listdir(con_dir_name):
        con_file_name = os.path.join(con_dir_name, item)
        con_file_dcm = pydicom.read_file(con_file_name, force=True)
        con_file_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        dcms.append(con_file_dcm)

    dcms = sorted(dcms, key=lambda s: s.SliceLocation)

    ps = dcms[0].PixelSpacing
    ss = dcms[0].SliceThickness

    con_shape = list(dcms[0].pixel_array.shape)
    con_shape.append(len(dcms))
    con3d = np.zeros(con_shape)

    for i, s in enumerate(dcms):
        con2d = s.pixel_array
        con3d[:, :, i] = con2d

    return con3d, ps, ss


def get_measurements_3d(con3d, ss, ps):

    vol_1 = get_volume_ml(con3d, 1, ss, ps)
    vol_2 = get_volume_ml(con3d, 2, ss, ps)
    vol_3 = get_volume_ml(con3d, 3, ss, ps)
    vol_4 = get_volume_ml(con3d, 4, ss, ps)
    vol_5 = get_volume_ml(con3d, 5, ss, ps)
    vol_6 = get_volume_ml(con3d, 6, ss, ps)

    vol_wall = get_volume_ml((con3d > 1), 1, ss, ps)
    vol_outer = get_volume_ml((con3d > 0), 1, ss, ps)

    measurements_3d = {'vol_lumen': vol_1,
                       'vol_wall': vol_wall,
                       'vol_outer': vol_outer,
                       'vol_lipid': vol_3,
                       'vol_calc': vol_4,
                       'vol_hemor': vol_6}

    return measurements_3d


def get_measurements_2d(con3d, ps):

    measurements_2d = []

    for z in range(0, con3d.shape[2]):
        img = con3d[..., z]

        d_hor_lum, d_ver_lum, d_min_lum, d_max_lum, area_lum = get_slice_roi_props(img, 1, ps)
        measurements_slice = {'slice': z + 1, 'roi': 'lumen',
                              'diam_hor': d_hor_lum, 'diam_vert': d_ver_lum,
                              'diam_min': d_min_lum, 'diam_max': d_max_lum,
                              'area': area_lum}
        measurements_2d.append(measurements_slice)

        d_hor_wall, d_ver_wall, d_min_wall, d_max_wall, area_wall = get_slice_roi_props(img > 1, 1, ps)
        measurements_slice = {'slice': z + 1, 'roi': 'wall',
                              'diam_hor': d_hor_wall, 'diam_vert': d_ver_wall,
                              'diam_min': d_min_wall, 'diam_max': d_max_wall,
                              'area': area_wall}
        measurements_2d.append(measurements_slice)

        d_hor_out, d_ver_out, d_min_out, d_max_out, area_out = get_slice_roi_props(img > 0, 1, ps)
        measurements_slice = {'slice': z + 1, 'roi': 'out',
                              'diam_hor': d_hor_out, 'diam_vert': d_ver_out,
                              'diam_min': d_min_out, 'diam_max': d_max_out,
                              'area': area_out}
        measurements_2d.append(measurements_slice)

        d_hor_3, d_ver_3, d_min_3, d_max_3, area_3 = get_slice_roi_props(img, 3, ps)
        measurements_slice = {'slice': z + 1, 'roi': 'lipid',
                              'diam_hor': d_hor_3, 'diam_vert': d_ver_3,
                              'diam_min': d_min_3, 'diam_max': d_max_3,
                              'area': area_3}
        measurements_2d.append(measurements_slice)

        d_hor_4, d_ver_4, d_min_4, d_max_4, area_4 = get_slice_roi_props(img, 4, ps)
        measurements_slice = {'slice': z + 1, 'roi': 'calc',
                              'diam_hor': d_hor_4, 'diam_vert': d_ver_4,
                              'diam_min': d_min_4, 'diam_max': d_max_4,
                              'area': area_4}
        measurements_2d.append(measurements_slice)

        d_hor_6, d_ver_6, d_min_6, d_max_6, area_6 = get_slice_roi_props(img, 6, ps)
        measurements_slice = {'slice': z + 1, 'roi': 'hemor',
                              'diam_hor': d_hor_6, 'diam_vert': d_ver_6,
                              'diam_min': d_min_6, 'diam_max': d_max_6,
                              'area': area_6}
        measurements_2d.append(measurements_slice)

    measurements_2d_df = pd.DataFrame(measurements_2d)

    return measurements_2d_df