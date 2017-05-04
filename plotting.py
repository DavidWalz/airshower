""" Plotting routines
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from utils import distance2showeraxis


def plot_array(v_stations, values, label='', vmin=None, vmax=None):
    """Plot a map *values* for an detector array specified by *v_stations*.
    """
    xd, yd, zd = v_stations.T / 1000  # in [km]
    circles = [plt.Circle((x, y), 0.650) for x, y in zip(xd.flat, yd.flat)]
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, bottom=0.13)
    coll = PatchCollection(circles, norm=Normalize(vmin=vmin, vmax=vmax))
    coll.set_array(values)
    coll.cmap.set_under('#d3d3d3')
    ax.add_collection(coll)
    ax.set(aspect='equal')
    ax.grid()
    cbar = fig.colorbar(coll)
    cbar.set_label(label)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')


# def plot_traces_of_array_for_one_event(Smu, Sem, v_axis, v_stations, arraysize = 5):
#     '''plot time traces of tanks around the tank with the smallest distance to shower core
#     Sem = EM-SigmalTrace, Smu = Muon-SignalTrace, arraysize = length of array'''
#     t = np.arange(0, 2001, 25)
#     fig, axes = subplots(arraysize, arraysize, sharex=True, figsize=(29, 16), dpi=80, facecolor='w', edgecolor='k')
#     t = np.arange(12.5, 2001, 25)
#     tight_layout()
#     coordVcore = np.argmin(distance2showeraxis(v_stations, v_axis))
#     coordX = int(coordVcore / 11)
#     coordY = coordVcore % 11
#     coords = np.array(0)
#     for j in range(arraysize):
#         for k in range(arraysize):
#             coords = np.append(coords, (coordX-2+j)*11 + (coordY-2+k))
#             coords = coords[1:arraysizearraysize+1]
#     for i, ax in enumerate(axes.flat):
#         try:
#             h1, h2 = Smu[coords[i]], Sem[coords[i]]
#         except TypeError:
#             h2 = np.histogram(0, bins=t)[0].astype(float)
#         ax.step(t, h1 + h2, c='k', where='mid')
#         ax.step(t, h1, label='$\mu$', where='mid')
#         ax.step(t, h2, label='$e\gamma$', where='mid')
#         ax.legend(title='r=%i' % coords[i], fontsize='x-small')
#         ax.set_xlim(0, 1500)
#         ax.grid(True)
#     for k in range(arraysize):
#         for l in range(arraysize):
#             axes[k, l].set_xlabel('$t$ / ns')
#             axes[k, l].set_ylabel('$S$ / VEM')
#     savefig('trace_VEM.png')#, bbox_inches='tight')


if __name__ == '__main__':
    d = np.load('showers.npz')
    logE = d['logE']
    mass = d['mass']
    Xmax = d['Xmax']
    v_core = d['showercore']
    v_axis = d['showeraxis']
    v_max = d['showermax']
    v_stations = d['detector']
    T = d['time']
    S = d['signal']

    # plot example event
    plot_array(v_stations, T[0] * 1E6, label='time [mu s]', vmin=-8, vmax=8)
    plt.savefig('time.png')
    logS = np.log10(S.sum(axis=-1))
    plot_array(v_stations, logS[0], label='log(signal)', vmin=0, vmax=5)
    plt.savefig('signal.png')
