import gpflow
import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt

# from gpflow.test_util import notebook_niter
np.random.seed(123)

# make a dataset with two outputs, correlated, heavy-tail noise. One has more noise than the other.
X1 = np.random.rand(100, 1)
X2 = np.random.rand(50, 1) * 0.5
Y1 = np.sin(6*X1) + np.random.standard_t(3, X1.shape)*0.03
Y2 = np.sin(6*X2+ 0.7) + np.random.standard_t(3, X2.shape)*0.1

plt.plot(X1, Y1, 'x', mew=2)
plt.plot(X2, Y2, 'x', mew=2)

plt.show()

# a Coregionalization kernel. The base kernel is Matern 3/2, and acts on the first ([0]) data dimension.
# the 'Coregion' kernel indexes the outputs, and acts on the second ([1]) data dimension
k1 = gpflow.kernels.Matern32(1, active_dims=[0])
coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[1])
kern = k1 * coreg

# build a variational model. This likelihood switches between Student-T noise with different variances:
lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.StudentT(), gpflow.likelihoods.StudentT()])

# Augment the time data with ones or zeros to indicate the required output dimension
stacked_X1 = np.hstack((X1, np.zeros_like(X1)))
stacked_X2 = np.hstack((X2, np.ones_like(X2)))
X_augmented = np.vstack((stacked_X1, stacked_X2))

# Augment the Y data to indicate which likelihood we should use
stacked_Y1 = np.hstack((Y1, np.zeros_like(X1)))
stacked_Y2 = np.hstack((Y2, np.ones_like(X2)))
Y_augmented = np.vstack((stacked_Y1, stacked_Y2))

# now build the GP model as normal
m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)


# fit the covariance function parameters
# gpflow.train.ScipyOptimizer().minimize(m, maxiter=notebook_niter(1000))
gpflow.train.ScipyOptimizer().minimize(m)

def plot_gp(x, mu, var, color='k'):
    plt.plot(x, mu, color=color, lw=2)
    plt.plot(x, mu + 2*np.sqrt(var), '--', color=color)
    plt.plot(x, mu - 2*np.sqrt(var), '--', color=color)

def plot(m):
    xtest = np.linspace(0, 1, 100)[:,None]
    line, = plt.plot(X1, Y1, 'x', mew=2)
    stacked_xtest1 = np.hstack((xtest, np.zeros_like(xtest)))
    mu, var = m.predict_f(stacked_xtest1)
    plot_gp(xtest, mu, var, line.get_color())

    line, = plt.plot(X2, Y2, 'x', mew=2)
    stacked_xtest2 = np.hstack((xtest, np.ones_like(xtest)))
    mu, var = m.predict_f(stacked_xtest2)
    plot_gp(xtest, mu, var, line.get_color())

plot(m)

plt.show()

m.kern.kern_list[1].W = np.random.randn(2, 1)
gpflow.train.ScipyOptimizer().minimize(m, maxiter=2000)
#
plot(m)
plt.show()

coreg.W.value @ coreg.W.value.T + np.diag(coreg.kappa.value)
