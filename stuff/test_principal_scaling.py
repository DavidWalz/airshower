from pylab import *


# scaling with zenith angle (S1000 --> S38 relation, CIC)
zenith = np.linspace(0, 60, 100)
x = np.cos(np.deg2rad(zenith))**2 - np.cos(np.deg2rad(38))**2
y = 1 + 0.95 * x - 1.63 * x**2 - 1.21 * x**3
figure()
plot(zenith, y)
xlabel('zenith [deg]')
ylabel('signal')
ylim(0)
grid()
savefig('scaling-zenith.png')


# scaling with mass number and relative scaling mu/em
A = np.linspace(1, 60)
a = A**0.15
S1 = a / (a + 1)
S2 = 1 / (a + 1) * 1.5
figure()
plot(A, S1, label='$\mu$')
plot(A, S2, label='$e \gamma$')
legend()
ylim(0)
xlabel('mass number')
ylabel('signal')
grid()
savefig('scaling-A.png')


# scaling with distance of station to shower axis
r = np.logspace(1.8, 3.5, 100)
S1 = (np.maximum(r, 100) / 1000)**-4.3
S2 = (np.maximum(r, 100) / 1000)**-5.5
figure()
plot(r, S1, label='$\mu$')
plot(r, S2, label='$e \gamma$')
legend()
loglog()
xlabel('radial distance [m]')
ylabel('signal')
grid()
savefig('scaling-r.png')


# scaling with traversed atmosphere to station
dX = np.linspace(10, 800, 100)
S1 = (np.maximum(dX, 10) / 100)**-0.05
S2 = (np.maximum(dX, 10) / 100)**-0.2
figure()
plot(dX, S1, label='$\mu$')
plot(dX, S2, label='$e \gamma$')
legend()
ylim(0)
xlabel('distance to Xmax [g/cm2]')
ylabel('signal')
grid()
savefig('scaling-dX.png')