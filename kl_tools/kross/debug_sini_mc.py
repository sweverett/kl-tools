from argparse import ArgumentParser
import numpy as np
from scipy.special import erf
from scipy.stats import lognorm, norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--n_samples', '-n', type=int, default=200000,
        help='Number of Monte Carlo samples'
        )
    parser.add_argument(
        '--n_sini_samples', '-i', type=int, default=500,
        help='Number of sini values to sample in the posterior, between 0 and 1'
        )
    parser.add_argument(
        '--sini', type=float, default=0.5,
        help='True sin(i) value'
        )
    parser.add_argument(
        '--v_tf', type=float, default=200,
        help='Predicted TF 3D velocity (km/s)'
        )
    parser.add_argument(
        '--sigma_vmaj', type=float, default=10,
        help='Uncertainty in the observed v_maj (km/s)'
        )
    parser.add_argument(
        '--v_maj_offset', type=float, default=0,
        help='Observed v_maj offset from sini*TF-predicted value in units of sigma_vmaj'
        )
    parser.add_argument(
        '--sigma_tf', type=float, default=0.05,
        help='Uncertainty in the Tully-Fisher relation (dex)'
        )
    parser.add_argument(
        '--no_prior', '-p', action='store_true', help='Turn off the i prior'
    )
    parser.add_argument(
        '--plot_both', '-b', action='store_true',
        help='Plot posterior for both i and sin(i)'
    )
    parser.add_argument(
        '--show', '-s', action='store_true', help='Show the plots'
    )
    return parser.parse_args()


def mu_i(v_maj, v_tf, sig_vmaj, sig_vcirc, i):
    if i == 0:
        i = 1e-10
    sini = np.sin(i)

    mu1 = v_maj / sini
    mu2 = v_tf

    var1 = (sig_vmaj / sini)**2
    var2 = sig_vcirc**2

    num = (mu1 * var1) + (mu2 * sig_vcirc)
    den = var1 + var2

    return num / den

def sigma_i(sig_vmaj, sig_vcirc, i):
    if i == 0:
        i = 1e-10
    sini = np.sin(i)

    var1 = (sig_vmaj / sini)**2
    var2 = sig_vcirc**2

    num = var1 * var2
    den = var1 + var2

    return num / den

def gauss_product(mu1, mu2, var1, var2):
    return norm.pdf(mu1, loc=mu2, scale=np.sqrt(var1 + var2))

def lognormal_vcirc(v_circ, v_tf, sigma_tf):
    # need to scale sigma_tf to account for difference in log-space
    return lognorm.pdf(v_circ, s=sigma_tf*np.log(10), scale=v_tf)

# Function for P(v_circ | v_maj, i) - Gaussian in v_circ
def normal_vcirc(v_circ, v_maj, i, sigma_vmaj):
    mean = v_maj / np.sin(i)
    sigma = sigma_vmaj / np.sin(i)
    return norm.pdf(v_circ, loc=mean, scale=sigma)

# Integrand for the v_circ part
def integrand_vcirc(v_circ, v_maj, i, v_TF, sigma_vmaj, sigma_TF):
    P_vcirc_vTF = lognormal_vcirc(v_circ, v_TF, sigma_TF)
    P_vcirc_vmaj = normal_vcirc(v_circ, v_maj, i, sigma_vmaj)
    return P_vcirc_vmaj * P_vcirc_vTF

def estimate_mstar_from_vtf(v_tf, alpha=4.51, log_M100=9.49):
    '''
    Calculate log10(M_star) from the given v_tf using the inverted TF relation.
    '''

    log_mstar = alpha * np.log10(v_tf/100) + log_M100
    return log_mstar

def main():

    #--------------------------------------------------------------------------
    # setup

    args = parse_args()

    n_samples = args.n_samples
    n_sini_samples = args.n_sini_samples
    true_sini = args.sini
    true_i = np.arcsin(true_sini)
    v_tf = args.v_tf # predicted TF 3D velocity (km/s)
    sigma_vmaj = args.sigma_vmaj # km/s
    v_maj_offset = args.v_maj_offset # Number of sigma_vmaj's to offset
    sigma_tf = args.sigma_tf # dex
    no_prior = args.no_prior # turn off the sin(i) prior
    plot_both = args.plot_both
    show = args.show

    # derived quantities
    sigma_vcirc = sigma_tf * v_tf * np.log(10) # km/s
    v_maj = (v_tf * true_sini) + (v_maj_offset * sigma_vmaj) # km/s
    true_mu = mu_i(v_maj, v_tf, sigma_vmaj, sigma_vcirc, true_i)
    true_sigma = sigma_i(sigma_vmaj, sigma_vcirc, true_i)
    true_ratio = true_mu / true_sigma

    log_mstar = estimate_mstar_from_vtf(v_tf)

    vcirc_max = 100 * v_maj
    # vcirc_max = np.inf

    numerical_posterior = []
    analytic_posterior = []

    i_samples = np.linspace(0.025, np.pi/2-0.025, n_sini_samples)
    for i in i_samples:
        sini = np.sin(i)

        # do the marginalization integration numerically
        integrated, _ = quad(
            integrand_vcirc, 0, vcirc_max, args=(
                v_maj,
                i,
                v_tf,
                sigma_vmaj,
                sigma_tf
                )
            )

        # now analytic posterior
        gauss_term = gauss_product(
            (v_maj / sini),
            v_tf,
            (sigma_vmaj / sini)**2,
            sigma_vcirc**2
        )
        mu = mu_i(v_maj, v_tf, sigma_vmaj, sigma_vcirc, i)
        sigma = sigma_i(sigma_vmaj, sigma_vcirc, i)
        ratio = mu / sigma
        analytic = gauss_term * (1. + erf(ratio / np.sqrt(2)))/2.
        # analytic = gauss_term

        # add priors, if desired
        if not no_prior:
            integrated *= sini
            analytic *= sini

        numerical_posterior.append(integrated)
        analytic_posterior.append(analytic)

    # normalize the posteriors
    numerical_posterior = np.array(numerical_posterior)
    analytic_posterior = np.array(analytic_posterior)
    numerical_posterior /= np.trapz(numerical_posterior, i_samples)
    analytic_posterior /= np.trapz(analytic_posterior, i_samples)

    # plot the results
    i_samples_deg = np.degrees(i_samples)
    true_i_deg = np.degrees(true_i)
    plt.figure(figsize=(12, 6))
    plt.plot(i_samples_deg, numerical_posterior, label='Numerical')
    plt.plot(i_samples_deg, analytic_posterior, label='Analytic')
    plt.axvline(true_i_deg, color='k', linestyle='--', label='Truth')
    plt.xlabel('Inclination Angle (degrees)')
    plt.ylabel('Posterior Probability')
    plt.legend()
    plt.title('Posterior Distribution of Inclination Angle (Simple)')

    plt.tight_layout()

    outfile = './i_integrated_simple.png'
    plt.savefig(outfile, dpi=300)

    if show is True:
        plt.show()

    #--------------------------------------------------------------------------
    # Plots taken from the original code

    #--------------------------------------------------------------------------
    # find the sin(i) corresponding to the maximum posterior

    sini_samples = np.sin(i_samples)

    max_posterior_idx = np.argmax(numerical_posterior)
    max_sin_i = sini_samples[max_posterior_idx]

    max_posterior_analytic_idx = np.argmax(analytic_posterior)
    max_sin_i_analytic = sini_samples[max_posterior_analytic_idx]

    plt.figure(figsize=(12, 6))
    sini_samples = np.sin(i_samples)
    plt.plot(sini_samples, numerical_posterior, label='sin(i) Posterior from Integration')
    plt.plot(sini_samples, analytic_posterior, label='sin(i) Posterior from Analytic Approximation')
    plt.axvline(
        true_sini, color='k', linestyle=':', label=f'True sin(i) = {true_sini:.4f}')
    plt.axvline(
        max_sin_i, color='b', linestyle='--', label=f'MAP sin(i) = {max_sin_i:.4f} (Numerical)'
    )
    plt.axvline(
        max_sin_i_analytic, color='orange', linestyle='--', label=f'MAP sin(i) = {max_sin_i_analytic:.4f} (Analytic)'
    )
    # if sin_i_map is not None:
    #     plt.axvline(
    #         sin_i_map, color='r', linestyle='--', label=f'MAP sin(i) = {sin_i_map:.4f} (analytic; last time)'
    #         )
    plt.xlabel('sin(i)')
    plt.ylabel('Posterior Probability Density')
    plt.legend()

    prior = 'with' if no_prior is False else 'without'
    title1 = rf'Posterior Distribution of $\sin(i)$ ({prior} prior)' + '\n'
    title2 = rf'$\log(M_\star)$ = {log_mstar:.2f}; $v_{{TF}}$ = {v_tf:.2f} km/s; $\sigma_{{TF}}$ = {sigma_tf:.2f} dex' + '\n'
    title3 = f'$v_{{maj}}$ = {v_maj:.2f} km/s; $\sigma_{{vmaj}}$ = {sigma_vmaj:.2f} km/s; $v_{{maj}}$ offset = {v_maj_offset:.2f} $\sigma_{{vmaj}}$' + '\n'
    title4 = rf'True $\mu(i)/\sigma(i)$ = {true_ratio:.2f}'
    plt.title(title1+title2+title3+title4)

    plt.tight_layout()

    outfile = './sini_integrated.png'
    plt.savefig(outfile, dpi=300)

    if show is True:
        plt.show()

    if plot_both is True:
        plt.figure(figsize=(12, 6))
        plt.plot(i_samples_deg, numerical_posterior, label='i Posterior from Integration')
        plt.plot(i_samples_deg, analytic_posterior, label='i Posterior from Analytic Approximation')
        true_i = np.arcsin(true_sini)
        max_i = np.arcsin(max_sin_i)
        max_i_analytic = np.arcsin(max_sin_i_analytic)
        plt.axvline(
            np.rad2deg(true_i), color='k', linestyle=':', label=f'True i = {true_i:.4f}')
        plt.axvline(
            np.rad2deg(max_i), color='b', linestyle='--', label=f'MAP i = {max_i:.4f} (Numerical)'
        )
        plt.axvline(
            np.rad2deg(max_i_analytic), color='orange', linestyle='--', label=f'MAP i = {max_i_analytic:.4f} (Analytic)'
        )
        # if sin_i_map is not None:
        #     i_map = np.arcsin(sin_i_map)
        #     plt.axvline(
        #         i_map, color='r', linestyle='--', label=f'MAP i = {i_map:.4f} (analytic; last time)'
        #         )
        plt.xlabel('i')
        plt.ylabel('Posterior Probability Density')
        plt.legend()

        prior = 'with' if no_prior is False else 'without'
        title1 = rf'Posterior Distribution of $i$ ({prior} prior)' + '\n'
        title2 = rf'$\log(M_\star)$ = {log_mstar:.2f}; $v_{{TF}}$ = {v_tf:.2f} km/s; $\sigma_{{TF}}$ = {sigma_tf:.2f} dex' + '\n'
        title3 = f'$v_{{maj}}$ = {v_maj:.2f} km/s; $\sigma_{{vmaj}}$ = {sigma_vmaj:.2f} km/s; $v_{{maj}}$ offset = {v_maj_offset:.2f} $\sigma_{{vmaj}}$' + '\n'
        title4 = rf'True $\mu(i)/\sigma(i)$ = {true_ratio:.2f}'
        plt.title(title1+title2+title3+title4)

        plt.tight_layout()

        outfile = './i_integrated.png'
        plt.savefig(outfile, dpi=300)

        if show is True:
            plt.show()

    return

if __name__ == '__main__':
    main()