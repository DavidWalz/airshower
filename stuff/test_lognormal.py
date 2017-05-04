from pylab import *
from scipy.stats import lognorm


# scipy.stats.lognorm
# ----------------------------------------------
# p(x) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(ln(x)/s)**2)
# lognorm takes s as a shape parameter.
# The probability density above is defined in the “standardized” form.
# To shift and/or scale the distribution use the loc and scale parameters.
# lognorm.pdf(x, s, loc, scale) = lognorm.pdf(y, s) / scale with y = (x - loc) / scale.
# A common parametrization is in terms of mu and sigma, of the unique normally distributed random variable X such that exp(X) = Y.
# This parametrization corresponds to setting s = sigma and scale = exp(mu).

# np.random.lognormal
# ---------------------------------------------
# p(x) = 1 / (sigma*x*sqrt(2*pi) * exp(-1/2*(ln(x)-mu)**2)/sigma**2)


def julie(mu, sigma, bins=linspace(0, 20, 101)):
    figure()
    N = 10000
    hist(np.random.lognormal(mean=mu, sigma=sigma, size=N), bins=bins, alpha=0.6)
    hist(lognorm.rvs(sigma, loc=mu, scale=exp(mu), size=N) - mu, bins=bins, alpha=0.6)
    text(0.05, 0.9, '$\mu$=%.2f, $\sigma$=%.2f' % (mu, sigma), transform=gca().transAxes)


julie(1, 1)
# julie(1, 0.5)
# julie(1.5, 1)
# julie(2, 1)
# julie(2.5, 1)
# julie(2.5, 0.5)
julie(log(250), 0.7, bins=linspace(0, 1000, 101))

show()
