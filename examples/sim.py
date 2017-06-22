import numpy as np
from airshower import shower, utils, plotting
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--mass", default=1, type=int, help='mass number')
parser.add_argument("--nfiles", default=1, type=int, help='mass number')
parser.add_argument("--nevents", default=1000, type=int, help='mass number')
args = parser.parse_args()

print 'Simulating'
print 'A', args.mass
print 'nfiles', args.nfiles
print 'nevents', args.nevents

v_stations = utils.station_coordinates(9, layout='offset')

for i in range(args.nfiles):
    logE = 18.5 + 1.5 * np.random.rand(args.nevents)
    mass = args.mass * np.ones(args.nevents)
    data = shower.rand_events(logE, mass, v_stations)

    np.savez_compressed(
        'showers-A%i-%i.npz' % (args.mass, i),
        detector=data['detector'],
        logE=data['logE'],
        mass=data['mass'],
        Xmax=data['Xmax'],
        showercore=data['showercore'],
        showeraxis=data['showeraxis'],
        showermax=data['showermax'],
        time=data['time'],
        signal=data['signal'])
