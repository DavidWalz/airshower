""" Plotting routines
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import seaborn.apionly as sns

import utils


def maybe_save(fig, fname):
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')
        plt.close()


def plot_array(v_stations, values, v_core=None, v_axis=None,
               label='', title='', vmin=None, vmax=None, fname=None):
    """Plot a map *values* for an detector array specified by *v_stations*. """
    print('Plot event')
    xd, yd, zd = v_stations.T / 1000  # in [km]
    circles = [plt.Circle((x, y), 0.650) for x, y in zip(xd.flat, yd.flat)]
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, bottom=0.13)
    coll = PatchCollection(circles, norm=Normalize(vmin=vmin, vmax=vmax))
    coll.set_array(values)
    coll.cmap.set_under('#d3d3d3')
    ax.add_collection(coll)
    cbar = fig.colorbar(coll)
    cbar.set_label(label)
    ax.grid(True)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    # plot shower direction and core
    if (v_core is not None) and (v_axis is not None):
        x, y, z = v_core / 1000 - 3 * v_axis
        dx, dy, dz = 3 * v_axis
        plt.arrow(x, y, dx, dy, lw=2, head_width=0.4, head_length=0.5, fc='r', ec='r')
    maybe_save(fig, fname)


def plot_array_traces(Smu, Sem, v_stations, n=5, fname=None):
    """ Plot time traces of the n^2 central tanks. """
    print('Plot time traces event')
    n0 = int(len(v_stations)**.5)  # number of stations along one axis
    i0 = (n0 - n) // 2  # start index for the sub-array to be plotted

    S1 = Smu.reshape(n0, n0, -1)
    S2 = Sem.reshape(n0, n0, -1)
    v = v_stations.reshape(n0, n0, 3)

    fig, axes = plt.subplots(n, n, sharex=True, figsize=(29, 16), facecolor='w')
    plt.tight_layout()
    t = np.arange(12.5, 2001, 25)
    for ix in range(n):
        for iy in range(n):
            ax = axes[ix, iy]
            h1 = S1[ix + i0, iy + i0]
            h2 = S2[ix + i0, iy + i0]
            ax.step(t, h1 + h2, c='k', where='mid')
            ax.step(t, h1, label='$\mu$', where='mid')
            ax.step(t, h2, label='$e\gamma$', where='mid')
            ax.legend(fontsize='x-small')
            ax.grid(True)
            ax.set_title('%.1f, %.1f km' % tuple(v[ix + i0, iy + i0, 0:2] / 1000))
            ax.set_xlabel('$t$ / ns', fontsize='x-small')
            ax.set_ylabel('$S$ / VEM', fontsize='x-small')
            ax.set_xlim(0, 1500)
    maybe_save(fig, fname)


def plot_time_distribution(T, fname=None):
    """ histogram of time values """
    print('Plot time distribution')
    fig, ax = plt.subplots(1)
    ax.hist(T[~np.isnan(T)].flatten() * 1E6, bins=40, normed=True)
    ax.grid()
    ax.set_xlabel('time [$\mu$ s]')
    ax.set_ylabel('relative frequency')
    maybe_save(fig, fname)


def plot_signal_distribution(S, fname=None):
    """ histogram of signal values """
    print('Plot total signal distribution')
    s = np.sum(S, axis=-1)  # sum over time trace per station
    s[np.isnan(s)] = 0
    s = np.sum(s, axis=-1)  # sum over stations
    fig, ax = plt.subplots(1)
    ax.hist(np.log10(s), bins=31, normed=True)
    ax.grid()
    ax.set_xlabel('$\log_{10}$(total signal) [a.u.]')
    ax.set_ylabel('relative frequency')
    maybe_save(fig, fname)


def plot_energy_distribution(logE, fname=None):
    """ histogram of energy values """
    print('Plot energy distribution')
    fig, ax = plt.subplots(1)
    ax.hist(logE, bins=np.linspace(18.5, 20, 31))
    ax.grid()
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel('frequency')
    maybe_save(fig, fname)


def plot_zenith_distribution(zenith, fname=None):
    """ histogram of zenith values """
    print('Plot zenith distribution')
    fig, ax = plt.subplots(1)
    ax.hist(np.rad2deg(zenith), bins=np.linspace(0, 60, 31))
    ax.grid()
    ax.set_xlabel('zenith [degree]')
    ax.set_ylabel('frequency')
    maybe_save(fig, fname)


def plot_phi_distribution(phi, fname=None):
    """ histogram of phi values """
    print('Plot phi distribution')
    fig, ax = plt.subplots(1)
    ax.hist(np.rad2deg(phi), bins=np.linspace(-180, 180, 31))
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.grid()
    ax.set_xlabel('phi [degree]')
    ax.set_ylabel('frequency')
    maybe_save(fig, fname)


def plot_xmax_distribution(Xmax, fname=None):
    """ histogram of Xmax values """
    print('Plot Xmax distribution')
    fig, ax = plt.subplots(1)
    ax.hist(Xmax, bins=np.linspace(600, 1200, 31))
    ax.grid()
    ax.set_xlabel('Xmax [g/cm$^2$]')
    ax.set_ylabel('frequency')
    maybe_save(fig, fname)


def plot_stations_vs_energy(logE, S, fname=None):
    """ histogram of stations with hits vs energy """
    print('Plot stations vs energy')
    nt = np.sum(~np.isnan(S), axis=1)
    fig = plt.figure()
    bins = np.linspace(18.5, 20, 31)
    ax = sns.regplot(x=logE, y=nt, x_bins=bins, fit_reg=False)
    ax.grid()
    ax.set_ylim(0)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Number of Stations')
    maybe_save(fig, fname)


def plot_stations_vs_zenith(zen, S, fname=None):
    """ histogram of stations with hits vs zenith """
    print('Plot stations vs zenith')
    nt = np.sum(~np.isnan(S), axis=1)
    fig = plt.figure()
    ax = sns.regplot(x=np.rad2deg(zen), y=nt, x_bins=np.linspace(0, 60, 31), fit_reg=False)
    ax.grid()
    ax.set_ylim(0)
    ax.set_xlabel('Zenith')
    ax.set_ylabel('Number of Stations')
    maybe_save(fig, fname)


def plot_stations_vs_phi(phi, S, fname=None):
    """ histogram of stations with hits vs zenith """
    print('Plot stations vs phi')
    nt = np.sum(~np.isnan(S), axis=1)
    fig = plt.figure()
    ax = sns.regplot(x=np.rad2deg(phi), y=nt, x_bins=np.linspace(-180, 180, 31), fit_reg=False)
    ax.grid()
    ax.set_ylim(0)
    ax.set_xlabel('Phi')
    ax.set_ylabel('Number of Stations')
    maybe_save(fig, fname)


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
    S1 = d['signal1']
    S2 = d['signal2']
    phi, zenith = utils.vec2ang(v_axis)

    # ------------------------------------
    # plot example event
    # ------------------------------------
    for i in range(3):
        title = 'logE=%.2f, Xmax=%.2f, zenith=%.2f' % (logE[i], Xmax[i], np.rad2deg(zenith[i]))

        plot_array(
            v_stations, T[i] * 1E6, v_core=v_core[i], v_axis=v_axis[i],
            vmin=-10, vmax=10, label='time [mu s]', title=title,
            fname='plots/example-%i-time.png' % i)

        logStot = np.log10(S.sum(axis=-1))
        plot_array(
            v_stations, logStot[i], v_core=v_core[i], v_axis=v_axis[i],
            vmin=1, vmax=5, label='time [mu s]', title=title,
            fname='plots/example-%i-signal.png' % i)

        plot_array_traces(
            Smu=S1[i], Sem=S2[i], v_stations=v_stations, n=5,
            fname='plots/example-%i-traces.png' % i)

    # ------------------------------------
    # plot distribution of all events
    # ------------------------------------
    plot_time_distribution(T, fname='plots/time_distribution.png')
    plot_signal_distribution(S, fname='plots/signal_distribution.png')
    plot_energy_distribution(logE, fname='plots/energy_distribution.png')
    plot_xmax_distribution(Xmax, fname='plots/xmax_distribution.png')
    plot_zenith_distribution(zenith, fname='plots/zenith_distribution.png')
    plot_phi_distribution(phi, fname='plots/phi_distribution.png')
    plot_stations_vs_energy(logE, S, fname='plots/stations_vs_energy.png')
    plot_stations_vs_zenith(zenith, S, fname='plots/stations_vs_zenith.png')
    plot_stations_vs_zenith(phi, S, fname='plots/stations_vs_phi.png')
