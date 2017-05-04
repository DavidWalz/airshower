from __future__ import division

import numpy as np
from utils import *
import atmosphere
import gumbel
from pylab import *
atm = atmosphere.Atmosphere()
import gumbel
import plotting
def station_response(r, dX, logE=19, A=1):
    """ Simulate time trace f(t) for a given SD station and shower.
    Args:
        r (float): radial distance to shower axis in [m]
        dX (float): distance to Xmax in [g/cm^2]
        logE (float): log10(E/eV)
        A (float): mass number
    Returns:
        time traces f_muon(t) and f_em(t) in [VEM] for t = 0 - 2000 ns in bins of 25 ns
    """
    # strength of muon and em signal
    # scaling with energy
    S0 = 1200 * 10**(logE - 19)
    # scaling with mass number
    a = A**0.15
    b = 2 ## EM scaling
    S1 = S0 * a / (a + 1) * 1/(b+1)
    S2 = S0 * 1 / (a + 1) * b/(b+1)
    # TODO: add scaling with zenith angle (CIC)
    # ...
    # scaling with distance of station to shower axis
    S1 *= np.minimum((r / 1000)**-4.3, 1000)
    S2 *= np.minimum((r / 1000)**-5.5, 1000)
    # scaling with traversed atmosphere to station
    S1 *= np.minimum((dX / 100)**-0.1, 10)
    S2 *= np.minimum((dX / 100)**-0.8, 10)

    # limit total signal, otherwise we get memory problems when drawing that many samples from distribution
    Stot = S1 + S2
    Smax = 1E6
    if Stot > Smax:
        S1 *= Smax / Stot
        S2 *= Smax / Stot

    # draw number of detected muons and em particles
    N1 = np.maximum(int(S1 + S1**.5 * np.random.randn()), 0)  # number of muons
    N2 = np.maximum(int(S2 + S2**.5 * np.random.randn()), 0)  # number of em particles

    # parameters of log-normal distributions
    mean1 = np.log(200 * (r / 750)**1.2 * (1 - 0.2 * dX / 1000.))
    mean2 = np.log(320 * (r / 750)**1.2 * (1 - 0.1 * dX / 1000.))
    sigma1 = 0.7
    sigma2 = 0.7

    # draw samples from distributions and create histograms
    shift = np.exp(mean2) - np.exp(mean1)
    bins = np.arange(0, 2001, 25)  # time bins [ns]
    h1 = np.histogram(np.random.lognormal(mean1, sigma1, size=N1), bins=bins)[0]
    h2 = np.histogram(np.random.lognormal(mean2, sigma2, size=N2) + 0.5*shift, bins=bins)[0]

    # total signal (simplify: 1 particle = 1 VEM)
    return h1, h2


def detector_response(logE, mass, v_axis, v_max, v_stations):
    """ Simulate the detector response for all SD stations and one event
    """
    r = distance2showeraxis(v_stations, v_axis)  # radial distance to shower axis
    phi, zen = vec2ang(v_max - v_stations)  # direction of shower maximum relative to stations
    # distance to shower maximum in [g/cm^2]
    dX = atm.get_atmosphere(
        zen,  # zenith angle of shower maximum seen from each station
        h_low=v_stations[:, 2],  # height of stations
        h_up=v_max[2])  # height of shower maximum

    # time traces for each station
    n = len(v_stations)
    S1 = np.zeros((n, 80))  # muon traces
    S2 = np.zeros((n, 80))  # em traces

    for j in range(n):
        h1, h2 = station_response(r=r[j], dX=dX[j], logE=logE, A=mass)
        S1[j] = h1
        S2[j] = h2

    return S1, S2


def rand_shower_geometry(N, logE, mass):
    """ Generate random shower geometries: Xmax, axis, core, maximum. """
    N = len(logE)

    # shower axis
    phi = 2 * np.pi * (np.random.rand(N) - 0.5)
    zenith = rand_zenith(N)
    v_axis = ang2vec(phi, zenith)
#### TESTIN ######################################################
    phi = 2 *0* np.pi * (np.random.rand(N) - 0.5)
    zenith = 60*np.pi/180*rand_zenith(N)

    # shower core on ground (random offset w.r.t. grid origin)
    v_core = 0*SPACING * (np.random.rand(N, 3) - 0.5)
    v_core[:, 2] = HEIGHT  # core is always at ground level

    # shower maximum, require h > hmin
    Xmax = np.empty(N)
    hmax = np.empty(N)
    i = 0
    while i < N:
        Xmax[i] = gumbel.rand_gumbel(logE[i], mass[i])
        hmax[i] = atm.get_vertical_height(zenith[i], Xmax[i])
        if hmax[i] < HEIGHT + 200:
            continue  # shower maximum too low, repeat
        i += 1

    # position of shower maximum
    dmax = (hmax - HEIGHT) / np.cos(zenith)  # distance to Xmax
    v_max = v_core + v_axis * dmax[:, np.newaxis]

    return Xmax, v_axis, v_core, v_max


if __name__ == '__main__':
    # detector array
    nb_stations = 11**2  # number of stations
    v_stations = triangular_array(nb_stations**.5)  # x,y,z coordinates of SD stations

    # showers
    nb_events = 2
    logE = 18.5 + 1.5 * np.random.rand(nb_events)
    logE = 20 *np.ones(nb_events)
    mass = 1 * np.ones(nb_events)
    Xmax, v_axis, v_core, v_max = rand_shower_geometry(nb_events, logE, mass)

    # detector response for each event
    T = np.zeros((nb_events, nb_stations))
    S1 = np.zeros((nb_events, nb_stations, 80))
    S2 = np.zeros((nb_events, nb_stations, 80))
    for i in range(nb_events):
        print(i)
        s1, s2 = detector_response(logE[i], mass[i], v_axis[i], v_max[i], v_stations)
        S1[i] = s1
        S2[i] = s2
        T[i] = arrival_time_planar(v_stations, v_core[i], v_axis[i])

    # add per ton noise on arrival time
    T += 20E-9 * np.random.randn(*T.shape)

    # add time offset per event (to account for core position)
    T += 100E-9 * (np.random.rand(nb_events, 1) - 0.5)

#    # add relative noise to signal pulse
#    S1 += 0.02 * S1 * np.random.randn(*S1.shape)
#    S2 += 0.02 * S2 * np.random.randn(*S2.shape)
#    # add absolute noise to signal pulse
#    S1 += 0.5 + 0.2 * np.random.randn(*S1.shape)
#    S2 += 0.5 + 0.2 * np.random.randn(*S2.shape)

    plotting.plot_traces_of_array_for_one_event(Smu=S1[0], Sem=S2[0], v_axis=v_axis[0], v_stations=v_stations, arraysize = 5)
    # trigger threshold: use only stations with sufficient signal-to-noise
##    c = S.sum(axis=-1) < 80 * 0.55
##    T[c] = np.NaN
##    S[c] = np.NaN

    # TODO: apply array trigger (3T5)

    # save
#    np.savez_compressed(
#        'showers.npz',
#        logE=logE,
#        mass=mass,
#        Xmax=Xmax,
#        time=T,
#        signal=S,
#        showercore=v_core,
#        showeraxis=v_axis,
#        showermax=v_max,
#        detector=v_stations)
