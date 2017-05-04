from pylab import *

close('all')


def mean1(r, dx):
    return 30 + 180 * (r / 750)**1.4 * (1 - 0.2 * dX / 1000.)

def mean2(r, dx):
    return 50 + 300 * (r / 750)**1.4 * (1 - 0.1 * dX / 1000.)


figure()
r = linspace(100, 3000)
dX = 600
plot(r, mean1(r, dX))
plot(r, mean2(r, dX))
plot([750, 850, 1250], [220, 260, 500], 'C0+')
plot([750, 850, 1250], [420, 500, 1000], 'C1+')
xlabel('$r$ [m]')
ylabel('$\Delta t$ [ns]')
grid()


for r in [750, 850, 1250]:
    m1 = np.log(mean1(r, dX))
    m2 = np.log(mean2(r, dX))

    figure()
    bins = np.arange(0, 2001, 25)  # time bins [ns]
    axvline(exp(m1), color='C0')
    axvline(exp(m2), color='C1')
    hist(np.random.lognormal(m1, 0.7, size=10000), bins=bins, alpha=0.7)
    hist(np.random.lognormal(m2, 0.7, size=10000) + exp(m2) - exp(m1), bins=bins, alpha=0.7)
    xlim(0, 1500)
    show()


# r = 750
# for dX in [300, 600, 900]:
#     mean1 = np.log(200 * (r / 750)**1.2 * (1 - 0.2 * dX / 1000.))
#     mean2 = np.log(320 * (r / 750)**1.2 * (1 - 0.1 * dX / 1000.))
#     sigma1 = 0.7  # * (r / 1000.)**0.15 * (1 - 0.2 * dX / 1000.)
#     sigma2 = 0.7  # * (r / 1000.)**0.15

#     figure()
#     bins = np.arange(0, 2001, 25)  # time bins [ns]
#     axvline(exp(mean1), color='C0')
#     axvline(exp(mean2), color='C1')
#     shift = exp(mean2)-exp(mean1)
#     hist(np.random.lognormal(mean1, sigma1, size=10000), bins=bins, alpha=0.7)
#     hist(np.random.lognormal(mean2, sigma2, size=10000) + shift, bins=bins, alpha=0.7)
#     xlim(0, 1500)
#     show()
