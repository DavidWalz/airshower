""" Plotting routines
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from utils import distance2showeraxis
from pylab import *
import seaborn.apionly as sns


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


def plot_traces_of_array_for_one_event(Smu, Sem, v_axis, v_stations, n=5):
    """ Plot time traces of the n^2 central tanks
        Sem = EM-SigmalTrace, Smu = Muon-SignalTrace
    """
    n0 = int(len(v_stations)**.5)
    i0 = (n0 - n) // 2

    S1 = Smu.reshape(n0, n0, -1)
    S2 = Sem.reshape(n0, n0, -1)

    fig, axes = subplots(n, n, sharex=True, figsize=(29, 16), facecolor='w')
    plt.tight_layout()
    t = np.arange(12.5, 2001, 25)
    v_stations = v_stations.reshape(11,11,3)
    for ix in range(n):
        for iy in range(n):
            ax = axes[ix, iy]
            h1 = S1[ix + i0, iy + i0]
            h2 = S2[ix + i0, iy + i0]
            ax.step(t, h1 + h2, c='k', where='mid')
            ax.step(t, h1, label='$\mu$', where='mid')
            ax.step(t, h2, label='$e\gamma$', where='mid')
            title = "X:"+ str(v_stations[ix + i0,iy + i0][0])+ " / Y:"+ str(v_stations[ix + i0,iy + i0][1])            
            ax.legend(title=title + " / r=" + str(ix + i0) + ", " + str(iy + i0), fontsize='x-small')
            ax.set_xlim(0, 1500)
            ax.grid(True)
            ax.set_xlabel('$t$ / ns', fontsize='x-small')
    ax.set_ylabel('$S$ / VEM', fontsize='x-small')
    savefig('trace_VEM.png')

def plot_time_distribution(t):
    # histogram of time values
    print "time"
    fig, ax = plt.subplots(1)
    ax.hist(t[~np.isnan(t)].flatten(), bins=40, normed=True)
    ax.grid()
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('frequency')
    fig.savefig('./time_distribution.png')

def plot_total_signal_distribution(s):
    print "signal"
    # histogram of signal values
    s = np.sum(s, axis=-1)
    s[np.isnan(s)] = 0
    fig, ax = plt.subplots(1)
    ax.hist(np.sum(s, axis=(-1)), bins=401, normed=True)
    ax.grid()
    plt.yscale('log')
    ax.set_xlabel('signal [a.u.]')
    ax.set_ylabel('frequency')
    fig.savefig('./total_signal_distribution.png')

def plot_energy_distribution(logE):
    print "energy"
    # histogram of Energy values
    fig, ax = plt.subplots(1)
    ax.hist(logE, bins = np.linspace(18.5, 20, 69), normed=False)
    ax.grid()
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel('frequency')
    fig.savefig("./energy_distribution.png")

def plot_energy_vs_tanks_with_hits(logE, s):
    print "energy vc tanks"
    # histogram tons with hits vs. Energy
    nt = np.sum(~np.isnan(s), axis=(1))
    fig = plt.figure()
    ax = sns.regplot(x=logE, y=nt,  x_bins = np.linspace(18.5, 20, 69), fit_reg=False)
    ax.grid()
    ax.set_xlabel('Energy')
    ax.set_ylabel('Number of Stations')
    fig.savefig('./energy_vs_nstations.png')


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
