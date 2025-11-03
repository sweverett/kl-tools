import numpy as np
import ultranest
import matplotlib.pyplot as plt
from scipy.stats import norm
import corner
import time
try:
    import schwimmbad
    import mpi4py
    from mpi4py import MPI
except:
    print("Can not import MPI, use single process")
try:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(f'MPI initialized with {size} processes, rank {rank}')
except:
    rank = 0
    size = 1
#from schwimmbad import MPIPool
#pool = schwimmbad.choose_pool(mpi=True, processes=1)

# Define a simple log-likelihood: Gaussian centered at 0
def log_likelihood_ring(params):
    x, y = params
    r = np.sqrt(x*x + y*y)
    return -0.5 * ((r-1)/0.01)**2

def log_likelihood_funnel(x):
    return (norm.logpdf(x[0], loc=0.5, scale=0.1) +
            norm.logpdf(x[1], loc=0.5, scale=np.exp(20 * (x[0] - 0.5)) / 100))

# Uniform prior in [-5, 5] for both parameters
def transform(cube):
    return 10 * cube - 5

comm = MPI.COMM_WORLD

sampler = ultranest.ReactiveNestedSampler(
    param_names=['x', 'y'],
    loglike=log_likelihood_funnel,
    transform=transform,
)
t_start = time.time()
result = sampler.run(
    min_num_live_points=1000,
    min_ess=10000,
    #callback=sampler.print_progress  # prints progress in real time
)
t_end = time.time()
print('Total time: {:.1f}s'.format(t_end - t_start))

# Print status after run
sampler.print_results()
# fig = sampler.plot_corner()

fig = corner.corner(
    result['weighted_samples']['points'],
    weights=result['weighted_samples']['weights'],
    labels=result['paramnames'], bins=200, color='purple',
    plot_datapoints=False, 
    range=[[0,1], [0,1]],
    )

plt.savefig('ultranest_corner.png')