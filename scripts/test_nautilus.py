from nautilus import Prior
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from nautilus import Sampler
import corner
import time
import matplotlib.pyplot as plt

try:
    import schwimmbad
    from mpi4py import MPI
except:
    print("Can not import MPI, use single process")
try:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1


def log_likelihood_ring(param_dict):
    r = np.sqrt(param_dict['a']**2 + param_dict['b']**2)
    return -0.5 * ( (r-1)/0.01 )**2

def log_likelihood_funnel(param):
    #a, b = param_dict['a'], param_dict['b']
    a, b = param[0], param[1]
    return (norm.logpdf(a, loc=0.5, scale=0.1) +
            norm.logpdf(b, loc=0.5, scale=np.exp(20 * (a - 0.5)) / 100))

prior = Prior()
prior.add_parameter('a', dist=(-5, +5))
prior.add_parameter('b', dist=(-5, +5))

if __name__ == "__main__":
    if size>1:
        pool = schwimmbad.choose_pool(mpi=True, processes=1)
        sampler = Sampler(prior, log_likelihood_funnel, pass_dict=False, n_live=3000, pool=pool)
    else:
        sampler = Sampler(prior, log_likelihood_funnel, pass_dict=False, n_live=3000)
    t_start = time.time()
    sampler.run(verbose=True)
    t_end = time.time()
    print('Total time: {:.1f}s'.format(t_end - t_start))
    points, log_w, log_l = sampler.posterior()
    fig = corner.corner(
        points, weights=np.exp(log_w), bins=200, labels=prior.keys, color='purple',
        plot_datapoints=False, 
        #range=np.repeat(2, len(prior.keys)),
        range=[[0,1], [0,1]],
    )

    plt.savefig('nautilus_corner.png')