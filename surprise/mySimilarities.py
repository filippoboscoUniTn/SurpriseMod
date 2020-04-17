from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from six.moves import range
from six import iteritems
import h5py

def batched_cosine(n_x, yr, xr, min_support, batch_size, file_path, group_name, dset_name, *args, **kwargs):
    try:
        f = h5py.File(file_path, 'r+')
    except OSError:
        raise ValueError('File {} inesistente'.format(file_path))
    #Apertura file di scrittura della matrice
    dset = f[group_name+'/'+dset_name]
    #Iterate over users/items in batched fashion
    for current_batch in range(0, n_x, batch_size):
        inf = current_batch
        sup = (current_batch + batch_size) if (current_batch + batch_size) < n_x else n_x
        current_batch_size = sup - inf
        prods = np.zeros((current_batch_size, n_x), np.double)
        freq = np.zeros((current_batch_size, n_x), np.int)
        sqi = np.zeros((current_batch_size, n_x), np.double)
        sqj = np.zeros((current_batch_size, n_x), np.double)
        sim = np.empty((current_batch_size, n_x), np.double)
        sim.fill(np.nan)

        for indice, (x, r_xs) in enumerate(list(xr.items())[inf:sup]):
            for y, r_x in r_xs:
                for ox, r_ox in yr[y]:
                    freq[indice, ox] += 1
                    prods[indice, ox] += r_x*r_ox
                    sqi[indice, ox] += r_x**2
                    sqj[indice, ox] += r_ox**2

        for indice, x_i in enumerate(range(inf, sup)):
            for x_j in range(0, n_x):
                if freq[indice, x_j] < min_support:
                    sim[indice, x_j] = 0
                else:
                    denum = np.sqrt(sqi[indice, x_j] * sqj[indice, x_j])
                    num = prods[indice, x_j]
                    if(denum != 0):
                        sim[indice, x_j] = num/denum
                    else:
                        sim[indice, x_j] = 0
            sim[indice, x_i] = 1

        dset[inf:sup] = sim

    f.close()
    return dset_name

def batched_msd(n_x, yr, xr, min_support, batch_size, file_path, group_name, dset_name, *args, **kwargs):
    try:
        f = h5py.File(file_path, 'r+')
    except OSError:
        raise ValueError('File {} esistente'.format(file_path))
    #Apertura file di scrittura della matrice
    dset = f[group_name+'/'+dset_name]
    #Iterate over users/items in batched fashion
    for current_batch in range(0, n_x, batch_size):
        inf = current_batch
        sup = (current_batch + batch_size) if (current_batch + batch_size) < n_x else n_x
        current_batch_size = sup - inf
        sq_diff = np.zeros((current_batch_size, n_x), np.double)
        freq = np.zeros((current_batch_size, n_x), np.int)
        sim = np.empty((current_batch_size, n_x), np.double)
        sim.fill(np.nan)

        for indice, (x, r_xs) in enumerate(list(xr.items())[inf:sup]):
            for y, r_x in r_xs:
                for ox, r_ox in yr[y]:
                    freq[indice, ox] += 1
                    sq_diff[indice, ox] += (r_x - r_ox)**2

        for indice, x_i in enumerate(range(inf, sup)):
            for x_j in range(0, n_x):
                if freq[indice, x_j] < min_support:
                    sim[indice, x_j] = 0
                else:
                    sim[indice, x_j] = 1 / (sq_diff[indice, x_j] / freq[indice, x_j] + 1)
            sim[indice, x_i] = 1

        dset[inf:sup] = sim

    f.close()
    return dset_name

def batched_pearson(n_x, yr, xr, min_support, batch_size, file_path, group_name, dset_name, *args, **kwargs):
    try:
        f = h5py.File(file_path, 'r+')
    except OSError:
        raise ValueError('File {} inesistente'.format(file_path))
    #Apertura file di scrittura della matrice
    dset = f[group_name+'/'+dset_name]
    #Iterate over users/items in batched fashion
    for current_batch in range(0, n_x, batch_size):
        inf = current_batch
        sup = (current_batch + batch_size) if (current_batch + batch_size) < n_x else n_x
        current_batch_size = sup - inf
        prods = np.zeros((current_batch_size, n_x), np.double)
        freq = np.zeros((current_batch_size, n_x), np.int)
        sqi = np.zeros((current_batch_size, n_x), np.double)
        sqj = np.zeros((current_batch_size, n_x), np.double)
        si = np.zeros((current_batch_size, n_x), np.double)
        sj = np.zeros((current_batch_size, n_x), np.double)
        sim = np.empty((current_batch_size, n_x), np.double)
        sim.fill(np.nan)

        for indice, (x, r_xs) in enumerate(list(xr.items())[inf:sup]):
            for y, r_x in r_xs:
                for ox, r_ox in yr[y]:
                    freq[indice, ox] += 1
                    prods[indice, ox] += r_x*r_ox
                    sqi[indice, ox] += r_x**2
                    sqj[indice, ox] += r_ox**2
                    si[indice, ox] += r_x
                    sj[indice, ox] += r_ox

        for indice, x_i in enumerate(range(inf, sup)):
            for x_j in range(x_i + 1, n_x):
                if freq[indice, x_j] < min_support:
                    sim[indice, x_j] = 0
                else:
                    n = freq[indice, x_j]
                    num = n * prods[indice, x_j] - si[indice, x_j] * sj[indice, x_j]
                    denum = np.sqrt((n * sqi[indice, x_j] - si[indice, x_j]**2) *
                                    (n * sqj[indice, x_j] - sj[indice, x_j]**2))
                    if denum == 0:
                        sim[indice, x_j] = 0
                    else:
                        sim[indice, x_j] = num / denum
            sim[indice, x_i] = 1

        dset[inf:sup] = sim

    f.close()
    return dset_name

def batched_pearson_baseline(n_x, yr, xr, min_support, batch_size, file_path, group_name, dset_name, global_mean, x_biases, y_biases, shrinkage=100, *args, **kwargs):
    try:
        f = h5py.File(file_path, 'r+')
    except OSError:
        raise ValueError('File {} già esistente'.format(file_path))
    #Apertura file di scrittura della matrice
    dset = f[group_name+'/'+dset_name]
    #Iterate over users/items in batched fashion
    for current_batch in range(0, n_x, batch_size):
        inf = current_batch
        sup = (current_batch + batch_size) if (current_batch + batch_size) < n_x else n_x
        current_batch_size = sup - inf
        prods = np.zeros((current_batch_size, n_x), np.double)
        freq = np.zeros((current_batch_size, n_x), np.int)
        sq_diff_i = np.zeros((current_batch_size, n_x), np.double)
        sq_diff_j = np.zeros((current_batch_size, n_x), np.double)
        sim = np.empty((current_batch_size, n_x), np.double)
        sim.fill(np.nan)
        for indice, (x, r_xs) in enumerate(list(xr.items())[inf:sup]):
            for y, r_x in r_xs:
                partial_bias = global_mean + y_biases[y]
                for ox, r_ox in yr[y]:
                    freq[indice, ox] += 1
                    diff_i = (r_x - (partial_bias + x_biases[x]))
                    diff_j = (r_ox - (partial_bias + x_biases[ox]))
                    prods[indice, ox] += diff_i * diff_j
                    sq_diff_i[indice, ox] += diff_i**2
                    sq_diff_j[indice, ox] += diff_j**2

        for indice, x_i in enumerate(range(inf, sup)):
            for x_j in range(x_i + 1, n_x):
                if freq[indice, x_j] < min_support:
                    sim[indice, x_j] = 0
                else:
                    sim[indice, x_j] = prods[indice, x_j] / (np.sqrt(sq_diff_i[indice, x_j] *
                                                           sq_diff_j[indice, x_j]))
                    # the shrinkage part
                    sim[indice, x_j] *= (freq[indice, x_j] - 1) / (freq[indice, x_j] - 1 +
                                                         shrinkage)
                    if sim[indice, x_j] == -0:
                        sim[indice, x_j] = 0
            sim[indice, x_i] = 1

        dset[inf:sup] = sim

    f.close()
    return dset_name
