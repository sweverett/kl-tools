import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from scipy.special import erf
from scipy.optimize import root_scalar
from argparse import ArgumentParser

from kl_tools.kross.tfr import estimate_vtf
import kl_tools.kross.analytic_estimators as estimator

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--n_samples', '-n', type=int, default=200000,
        help='Number of Monte Carlo samples'
        )
    parser.add_argument(
        '--n_gplus_samples', '-g', type=int, default=500,
        help='Number of gplus values to sample in the posterior, between '
        ' -0.2 and 0.2'
        )
    parser.add_argument(
        '--n_sini_samples', '-i', type=int, default=500,
        help='Number of sini values to sample in the posterior, between 0 and 1'
        )
    parser.add_argument(
        '--gplus', type=float, default=0.1,
        help='True g_+ value'
        )
    parser.add_argument(
        '--sini', type=float, default=0.5,
        help='True sin(i) value'
        )
    parser.add_argument(
        '--eint', type=float, default=0.1,
        help='Intrinsic ellipticity'
    )
    parser.add_argument(
        '--v_tf', type=float, default=200,
        help='Predicted TF 3D velocity (km/s)'
        )
    parser.add_argument(
        '--sig_vmaj', type=float, default=10,
        help='Uncertainty in the observed v_maj (km/s)'
        )
    parser.add_argument(
        '--v_maj_offset', type=float, default=0,
        help='Observed v_maj offset from sini*TF-predicted value in units of sigma_vmaj'
        )
    parser.add_argument(
        '--sig_tf', type=float, default=0.05,
        help='Uncertainty in the Tully-Fisher relation (dex)'
        )
    parser.add_argument(
        '--eobs', type=float, default=0.1,
        help='Observed ellipticity'
    )
    parser.add_argument(
        '--sig_eobs', type=float, default=0.3,
        help='Uncertainty in the observed ellipticity'
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

#-------------------------------------------------------------------------------
# helper methods

def estimate_mstar_from_vtf(v_tf, alpha=4.51, log_M100=9.49):
    '''
    Calculate log10(M_star) from the given v_tf using the inverted TF relation.
    '''

    log_mstar = alpha * np.log10(v_tf/100) + log_M100
    return log_mstar

def root_equation(sin_i, v_maj, v_tf, sigma_vmaj, sigma_vcirc, no_prior):
    '''
    Find the sin(i) that satisfies the extremized equation (requires prior)
    '''

    # the estimate was derived assuming a prior on i of the form sin(i)
    if no_prior is True:
        return None

    mu = estimator.mu_sini(v_maj, sin_i, v_tf, sigma_vmaj, sigma_vcirc)
    var = estimator.var_sini(sigma_vmaj, sigma_vcirc, sin_i)

    bracket_expression = 1 + (v_maj * sin_i * mu / sigma_vmaj**2) -\
          (sin_i**2 / sigma_vmaj**2) * (mu**2 + var)

    return bracket_expression

def main():

    #--------------------------------------------------------------------------
    # setup

    args = parse_args()

    n_samples = args.n_samples
    n_sini_samples = args.n_sini_samples
    true_gplus = args.gplus
    true_sini = args.sini
    true_eint = args.eint
    v_tf = args.v_tf # predicted TF 3D velocity (km/s)
    sig_vmaj = args.sig_vmaj # km/s
    v_maj_offset = args.v_maj_offset # Number of sig_vmaj's to offset
    sig_tf = args.sig_tf # dex
    eobs = args.eobs # observed ellipticity
    sig_eobs = args.sig_eobs # uncertainty in observed ellipticity
    no_prior = args.no_prior # turn off the sin(i) prior
    plot_both = args.plot_both
    show = args.show

    # derived quantities
    sig_vcirc = sig_tf * v_tf * np.log(10) # km/s
    v_maj = (v_tf * true_sini) + (v_maj_offset * sig_vmaj) # km/s
    true_mu = estimator.mu_sini(v_maj, true_sini, v_tf, sig_vmaj, sig_vcirc)
    true_sigma = estimator.sigma_sini(sig_vmaj, sig_vcirc, true_sini)
    true_ratio = true_mu / true_sigma

    #--------------------------------------------------------------------------
    # find the log(M_star) corresponding to the passed v_tf

    log_mstar = estimate_mstar_from_vtf(v_tf)

    #--------------------------------------------------------------------------
    # perform root finding to find the sin(i) where the expression is zero

    # the analytic estimate assumes a prior on i of the form sin(i)
    if no_prior is False:
        sol = root_scalar(
            root_equation,
            args=(v_maj, v_tf, sig_vmaj, sig_vcirc, no_prior),
            bracket=[0.0, 1.0]
            )

        if sol.converged:
            sin_i_map = sol.root
        else:
            sin_i_map = None
    else:
        sin_i_map = None

    #--------------------------------------------------------------------------
    # now run the Monte Carlo simulations

    gplus_i_vals = np.linspace(-0.2, 0.2, args.n_gplus_samples)
    posterior = np.zeros_like(gplus_i_vals)
    posterior_analytic = np.zeros_like(gplus_i_vals)

    for j, gplus in enumerate(gplus_i_vals):
        sin_i_vals = np.linspace(0.0, 1.0, n_sini_samples)
        for k, sin_i in enumerate(sin_i_vals):
            i = np.arcsin(sin_i)

            # sample lots of v_circ's from log-normal distribution
            v_circ_samples = estimator.lognormal_base10(
                np.log10(v_tf), sig_tf, n_samples
                )

            # use sampled v_circ's to compute *expected* v_maj's
            v_maj_model = v_circ_samples * sin_i
            likelihood = norm.pdf(v_maj_model, loc=v_maj, scale=sig_vmaj)

            # now average these likelihoods to get the mean posterior (if no prior)
            posterior[k] = np.mean(likelihood)

            # now compute the analytic posterior given our log-normal->normal approx
            mu = estimator.mu_sini(v_maj, sin_i, v_tf, sig_vmaj, sig_vcirc)
            sig = estimator.sigma_sini(sig_vmaj, sig_vcirc, sin_i)

            gauss_product = estimator.gaussian_product_sini(
                v_maj, v_tf, sig_vmaj, sig_vcirc, sin_i
                )

            posterior_analytic[k] = gauss_product * 0.5 * (
                1.+erf(mu/(np.sqrt(2)*sig))
                )

            # only add the sini term if we're using the prior
            if no_prior is False:
                prior = sin_i
                posterior[k] *= prior
                posterior_analytic[k] *= prior

    # normalize the posterior
    posterior /= np.trapz(posterior, sin_i_vals)
    posterior_analytic /= np.trapz(posterior_analytic, sin_i_vals)

    # plt.plot(sin_i_vals, posterior, label='MC')
    # plt.plot(sin_i_vals, posterior_analytic, label='Analytic')
    # plt.axvline(true_sini, color='k', linestyle=':')
    # plt.legend()
    # plt.show()

    # TODO: keep investigating...
    # plt.plot(sin_i_vals, posterior_analytic)
    # plt.show()

    #--------------------------------------------------------------------------
    # find the sin(i) corresponding to the maximum posterior

    max_posterior_idx = np.argmax(posterior)
    max_sin_i = sin_i_vals[max_posterior_idx]

    max_posterior_analytic_idx = np.argmax(posterior_analytic)
    max_sin_i_analytic = sin_i_vals[max_posterior_analytic_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(sin_i_vals, posterior, label='sin(i) Posterior from Monte Carlo')
    plt.plot(sin_i_vals, posterior_analytic, label='sin(i) Posterior from Analytic Approximation')
    plt.axvline(
        true_sini, color='k', linestyle=':', label=f'True sin(i) = {true_sini:.4f}')
    plt.axvline(
        max_sin_i, color='b', linestyle='--', label=f'MAP sin(i) = {max_sin_i:.4f} (Monte Carlo)'
    )
    plt.axvline(
        max_sin_i_analytic, color='orange', linestyle='--', label=f'MAP sin(i) = {max_sin_i_analytic:.4f} (analytic posterior)'
    )
    if sin_i_map is not None:
        plt.axvline(
            sin_i_map, color='r', linestyle='--', label=f'MAP sin(i) = {sin_i_map:.4f} (analytic; last time)'
            )
    plt.xlabel('sin(i)')
    plt.ylabel('Posterior Probability Density')
    plt.legend()

    prior = 'with' if no_prior is False else 'without'
    title1 = rf'Posterior Distribution of $\sin(i)$ ({prior} prior)' + '\n'
    title2 = rf'$\log(M_\star)$ = {log_mstar:.2f}; $v_{{TF}}$ = {v_tf:.2f} km/s; $\sigma_{{TF}}$ = {sig_tf:.2f} dex' + '\n'
    title3 = f'$v_{{maj}}$ = {v_maj:.2f} km/s; $\sigma_{{vmaj}}$ = {sig_vmaj:.2f} km/s; $v_{{maj}}$ offset = {v_maj_offset:.2f} $\sigma_{{vmaj}}$' + '\n'
    title4 = rf'True $\mu(i)/\sigma(i)$ = {true_ratio:.2f}'
    plt.title(title1+title2+title3+title4)

    plt.tight_layout()

    outfile = './sini_mc.png'
    plt.savefig(outfile, dpi=300)

    if show is True:
        plt.show()

    if plot_both is True:
        plt.figure(figsize=(12, 6))
        plt.plot(np.arcsin(sin_i_vals), posterior, label='i Posterior from Monte Carlo')
        plt.plot(np.arcsin(sin_i_vals), posterior_analytic, label='i Posterior from Analytic Approximation')
        true_i = np.arcsin(true_sini)
        max_i = np.arcsin(max_sin_i)
        max_i_analytic = np.arcsin(max_sin_i_analytic)
        plt.axvline(
            true_i, color='k', linestyle=':', label=f'True i = {true_i:.4f}')
        plt.axvline(
            max_i, color='b', linestyle='--', label=f'MAP i = {max_i:.4f} (Monte Carlo)'
        )
        plt.axvline(
            max_i_analytic, color='orange', linestyle='--', label=f'MAP i = {np.arcsin(max_i_analytic):.4f} (analytic posterior)'
        )
        if sin_i_map is not None:
            i_map = np.arcsin(sin_i_map)
            plt.axvline(
                i_map, color='r', linestyle='--', label=f'MAP i = {i_map:.4f} (analytic; last time)'
                )
        plt.xlabel('i')
        plt.ylabel('Posterior Probability Density')
        plt.legend()

        prior = 'with' if no_prior is False else 'without'
        title1 = rf'Posterior Distribution of $i$ ({prior} prior)' + '\n'
        title2 = rf'$\log(M_\star)$ = {log_mstar:.2f}; $v_{{TF}}$ = {v_tf:.2f} km/s; $\sigma_{{TF}}$ = {sig_tf:.2f} dex' + '\n'
        title3 = f'$v_{{maj}}$ = {v_maj:.2f} km/s; $\sigma_{{vmaj}}$ = {sig_vmaj:.2f} km/s; $v_{{maj}}$ offset = {v_maj_offset:.2f} $\sigma_{{vmaj}}$' + '\n'
        title4 = rf'True $\mu(i)/\sigma(i)$ = {true_ratio:.2f}'
        plt.title(title1+title2+title3+title4)

        plt.tight_layout()

        outfile = './i_mc.png'
        plt.savefig(outfile, dpi=300)

        if show is True:
            plt.show()

    return

if __name__ == '__main__':
    main()