from __future__ import division

import numpy as np
import utils
import atmosphere
import gumbel
import plotting


HEIGHT = 1400
SPACING = 1500


def station_response(r, dX, logE=19, zenith=0, mass=1):
    """ Simulate time trace f(t) for a given SD station and shower.
    Args:
        r (float): radial distance to shower axis in [m]
        dX (float): distance to Xmax in [g/cm^2]
        logE (float): log10(E/eV)
        zenith (float): zenith angle
        mass (float): mass number
    Returns:
        time traces f_muon(t) and f_em(t) in [VEM] for t = 0 - 2000 ns in bins of 25 ns
    """
    # signal strength, scaling with energy
    S0 = 900 * 10**(logE - 19)
    # # scaling with zenith angle (similar to S1000 --> S38 relation, CIC)
    # x = np.cos(zenith)**2 - np.cos(np.deg2rad(38))**2
    # S0 *= 1 + 0.95 * x - 1.4 * x**2 - 1.1 * x**3

    # relative scaling mu/em and scaling with mass number
    r1 = 0.3 * mass**0.15
    r2 = 0.7
    S1 = S0 * r1 / (r1 + r2)
    S2 = S0 * r2 / (r1 + r2)

    # scaling with distance of station to shower axis
    S1 *= (np.maximum(r, 150) / 1000)**-4.7
    S2 *= (np.maximum(r, 150) / 1000)**-6.1

    # scaling with traversed atmosphere to station
    S1 *= np.minimum((dX / 100)**-0.1, 10)
    S2 *= np.minimum((dX / 100)**-0.4, 10)

    # limit total signal (saturation / memory constraints)
    Stot = S1 + S2
    Smax = 1E7
    if Stot > Smax:
        S1 *= Smax / Stot
        S2 *= Smax / Stot

    # draw number of detected muons and em particles
    N1 = np.maximum(int(S1 + S1**.5 * np.random.randn()), 0)  # number of muons
    N2 = np.maximum(int(S2 + S2**.5 * np.random.randn()), 0)  # number of em particles

    # parameters of log-normal distributions
    mean1 = np.log(50 + 140 * (r / 750)**1.4 * (1 - 0.2 * dX / 1000.))
    mean2 = np.log(80 + 200 * (r / 750)**1.4 * (1 - 0.1 * dX / 1000.))
    sigma1 = 0.7
    sigma2 = 0.7

    # draw samples from distributions and create histograms
    shift = (np.exp(mean2) - np.exp(mean1)) / 1.5
    bins = np.arange(0, 2001, 25)  # time bins [ns]
    h1 = np.histogram(np.random.lognormal(mean1, sigma1, size=N1), bins=bins)[0]
    h2 = np.histogram(np.random.lognormal(mean2, sigma2, size=N2) + shift, bins=bins)[0]

    # total signal (simplify: 1 particle = 1 VEM)
    return h1, h2


def detector_response(logE, mass, v_axis, v_core, v_max, v_stations):
    """ Simulate the detector response for all SD stations and one event. """
    _, zenith = utils.vec2ang(v_axis)

    r = utils.distance2showeraxis(v_stations, v_core, v_axis)  # radial distance to shower axis
    phi, zen = utils.vec2ang(v_max - v_stations)  # direction of shower maximum relative to stations

    # distance from each station to shower maximum in [g/cm^2]
    dX = atmosphere.slant_depth(v_stations[:, 2], zen) - atmosphere.slant_depth(v_max[2], zen)

    # time traces for each station
    n = len(v_stations)
    S1 = np.zeros((n, 80))  # muon traces
    S2 = np.zeros((n, 80))  # em traces

    for j in range(n):
        h1, h2 = station_response(r=r[j], dX=dX[j], logE=logE, zenith=zenith, mass=mass)
        S1[j] = h1
        S2[j] = h2

    return S1, S2


def rand_shower_geometry(logE, mass):
    """ Generate random shower geometries: Xmax, axis, core, maximum, virtual origin """
    nb_showers = len(logE)

    # 1) random shower axis
    phi = 2 * np.pi * (np.random.rand(nb_showers) - 0.5)
    zenith = utils.rand_zenith(nb_showers)
    v_axis = utils.ang2vec(phi, zenith)

    # 2) random shower core on ground (offset w.r.t. grid origin)
    r = SPACING / 2 * np.random.rand(nb_showers)**.5
    p = np.random.rand(nb_showers) * 2 * np.pi
    x = r * np.cos(p)
    y = r * np.sin(p)
    z = HEIGHT * np.ones_like(p)
    v_core = np.c_[x, y, z]

    # 3) random shower maximum, require Xmax 200m above ground
    Xmax = gumbel.rand_gumbel(logE, mass)
    Xmax_max = atmosphere.slant_depth(HEIGHT + 200, zenith)

    while not np.all(Xmax < Xmax_max):
        idx = Xmax > Xmax_max  # resample shower maxima less than 200 above ground
        Xmax[idx] = gumbel.rand_gumbel(logE[idx], mass[idx])

    # 4) point of shower maximum
    h = atmosphere.height_at_slant_depth(Xmax, zenith)
    d = (h - HEIGHT) / np.cos(zenith)
    v_max = v_core + v_axis * d[:, np.newaxis]

    # 5) virtual origin at 50% of Xmax
    h = atmosphere.height_at_slant_depth(0.5 * Xmax, zenith)
    d = (h - HEIGHT) / np.cos(zenith)
    v_origin = v_core + v_axis * d[:, np.newaxis]

    return Xmax, v_axis, v_core, v_max, v_origin


def rand_events(logE, mass, v_stations, fname=None, wavefront='planar'):
    """Simulate events for given energy, mass, and detector positions."""

    nb_events = len(logE)
    nb_stations = len(v_stations)

    # simulation showers
    print('simulating showers')
    Xmax, v_axis, v_core, v_max, v_origin = rand_shower_geometry(logE, mass)

    # detector response for each event
    print('simulating detector response')
    T = np.zeros((nb_events, nb_stations))
    S1 = np.zeros((nb_events, nb_stations, 80))
    S2 = np.zeros((nb_events, nb_stations, 80))
    for i in range(nb_events):
        # print('%4i, logE = %.2f' % (i, logE[i]))
        S1[i], S2[i] = detector_response(logE[i], mass[i], v_axis[i], v_core[i], v_max[i], v_stations)
        if wavefront == 'planar':
            T[i] = utils.arrival_time_planar(v_stations, v_core[i], v_axis[i])
        elif wavefront == 'spherical':
            T[i] = utils.arrival_time_spherical(v_stations, v_origin[i], v_core[i])
        else:
            raise ValueError('wavefront must be "planar" or "spherical"')

    # total signal
    S = S1 + S2
    # add per ton noise on arrival time
    Stot = S.sum(axis=-1)
    sigma = 60E-9 * 8. / (1 + np.log10(Stot + 1))  # varies from ~(1-8)*60 ns
    T += sigma * np.random.randn(*T.shape)
    # add time offset per event (to account for random core position)
    T += 100E-9 * (np.random.rand(nb_events, 1) - 0.5)
    # add relative noise to signal pulse
    S += 0.05 * S * np.random.randn(*S.shape)
    # add absolute noise to signal pulse
    noise_level = 1.2
    S += noise_level * (1 + 0.5 * np.random.randn(*S.shape))
    S = np.clip(S, 0, None)
    # trigger threshold: use only stations with sufficient signal-to-noise
    c = S.sum(axis=-1) < 80 * noise_level * 1.2
    T[c] = np.NaN
    S[c] = np.NaN

    return {
        'logE': logE,
        'mass': mass,
        'Xmax': Xmax,
        'time': T,
        'signal': S,
        'signal1': S1,
        'signal2': S2,
        'showercore': v_core,
        'showeraxis': v_axis,
        'showermax': v_max,
        'detector': v_stations}


if __name__ == '__main__':
    # detector array, vector of (x,y,z) positions
    v_stations = utils.station_coordinates(11, layout='offset')

    # simulate events
    n = 1000
    logE = 18.5 + 1.5 * np.random.rand(n)
    mass = 1
    data = rand_events(logE, mass, v_stations)

    phi, zenith = utils.vec2ang(data['showeraxis'])
    plotting.plot_time_distribution(data['time'], fname='time_distribution.png')
    plotting.plot_signal_distribution(data['signal'], fname='signal_distribution.png')
    plotting.plot_energy_distribution(data['logE'], fname='energy_distribution.png')
    plotting.plot_xmax_distribution(data['Xmax'], fname='xmax_distribution.png')
    plotting.plot_zenith_distribution(zenith, fname='zenith_distribution.png')
    plotting.plot_phi_distribution(phi, fname='phi_distribution.png')
    plotting.plot_stations_vs_energy(data['logE'], data['signal'], fname='stations_vs_energy.png')
    plotting.plot_stations_vs_zenith(zenith, data['signal'], fname='stations_vs_zenith.png')
    plotting.plot_array_traces(Smu=data['signal1'][1], Sem=data['signal2'][1], v_stations=data['detector'], n=5, fname='example-trace.png')
