import time, sys

import numpy as np
from sedpy.observate import load_filters
import astropy.io.fits as fits
from astropy.table import Table

from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer


# --------------
# Model Definition
# --------------
def delogify(logMass=0, **extras):
    """Change the stellar mass parameter to log10-based
    """
    return 10**logMass

def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
                add_neb=False, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a delay-tau SFH parameter set. ---
    # This has 6 free parameters:
    #   "mass", "logzsol", "dust2", "dust_index", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # Change the stellar mass parameter to the log10 based "logMass"
    model_params["mass"]["isfree"] = False
    model_params["mass"]["depends_on"] = delogify
    model_params["logMass"] = dict(N=1, isfree=True)
    # add dust extinction slope index into sampling
    # dust_type = 4: Kriek & Conroy 2013
    model_params["dust_type"] = {"N": 1, "isfree": False, "init": 4, 
                                "units": "FSPS index"}
    model_params["dust_index"] = {"N": 1, "isfree": True, 
                              "units": "power-law multiplication of Calzetti"}
    # update IMF type 
    # (0-Salpeter, 1-Chabrier, 2-Kroupa, 3-van Dokkum, 4-Dave, 5-tabulated)
    model_params["imf_type"]["init"] = 1  


    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 13.
    model_params["logMass"]["init"] = 8.0
    model_params["dust_index"]["init"] = 0.0

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["logMass"]["init_disp"] = 0.04
    model_params["tau"]["init_disp"] = 3.0
    model_params["tage"]["init_disp"] = 5.0
    model_params["tage"]["disp_floor"] = 2.0
    model_params["dust2"]["disp_floor"] = 0.1
    model_params["dust_index"]["dist"] = 0.1

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_params["logMass"]["prior"] = priors.TopHat(mini=6, maxi=12)
    model_params["dust_index"]["prior"] = priors.TopHat(mini=-3.0, maxi=1.0)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.0, maxi=0.40)

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SpecModel(model_params)

    return model

# --------------
# Observational Data
# --------------

# Here we are going to put together some filter names
jwst_nircam = ['F090W', 'F115W', 'F150W', 'F182M', 'F200W', 'F210M', 'F277W',
        'F335M', 'F356W', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M']
hst_acs = ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP', ]
hst_wfc3 = ['F105W', 'F125W', 'F140W', 'F160W']
wise = ["W3", "W4"]
filters_used = jwst_nircam+hst_acs+hst_wfc3+wise

# The first filter set is Johnson/Cousins, the second is SDSS. We will use a
# flag in the photometry table to tell us which set to use for each object
# (some were not in the SDSS footprint, and therefore have Johnson/Cousins
# photometry)
#
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
filtersets = ["jwst_%s"%bp.lower() for bp in jwst_nircam]
filtersets += ["acs_wfc_%s"%bp.lower() for bp in hst_acs]
filtersets += ["wfc3_ir_%s"%bp.lower() for bp in hst_wfc3]
filtersets += ["wise_%s"%bp.lower() for bp in wise]
filters = load_filters(filtersets)


def build_obs(ind=0, phottable='demo_photometry.fits', **kwargs):
    """Load photometry from an ascii/fits file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)
    from prospect.utils.obsutils import fix_obs

    # Here we will read in an fits catalog
    catalog = Table().read(phottable)
    Nrows = len(catalog)
    assert ind<Nrows, f'ind = {ind} is larger than the size of table {Nrows}!'

    # Find the object right now
    obj_gal = catalog[ind]
    objid = obj_gal['ID']
    zgal = obj_gal['z_spec']
    print(f'Building observation of object {objid} at redshift {zgal:.2f}')
    # convert the flux from nJy  to maggy & set negative fluxes to zero
    # NOTE: the factor of 1.25 is aperture correction
    nJy2maggy = 1.25*1.0e-9/3631.0
    maggies = np.array([obj_gal["%s_CIRC0"%bp]*nJy2maggy for bp in filters_used]).flatten()
    maggies[maggies<0] = 0.0
    magerr = np.array([obj_gal["%s_CIRC0_e"%bp]*nJy2maggy for bp in filters_used]).flatten()
    print(f'Object flux:')
    for i,bp in enumerate(filters_used):
        flux = maggies[i]
        flux_err = magerr[i]
        print(f'--- {bp:10s}: {flux:.2e} pm {flux_err:.2e} nJy')

    # Build output dictionary.
    obs = dict(redshift=zgal, wavelength=None, spectrum=None,unc=None,
           maggies=maggies, maggies_unc=magerr, filters=filters,
           mask=(np.isfinite(maggies) & np.isfinite(magerr)) )
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(observations, **extras):
    # use the defaults
    return observations

# -----------
# Everything
# ------------

def build_all(**kwargs):
    observations = build_obs(**kwargs)
    observations = build_noise(observations, **kwargs)
    model = build_model(**kwargs)
    sps = build_sps(**kwargs)

    return (observations, model, sps)


if __name__ == '__main__':

    # --- Parser with default arguments ---
    parser = prospect_args.get_parser()
    # --- Add custom arguments ---
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--ind', type=int, default=1,
                        help="ONE-index row number in the table to fit.")

    # --- Configure ---
    args = parser.parse_args()
    config = vars(args)
    config["param_file"] = __file__
    catalog = Table().read(config['phottable'])
    # SLURM array jobs start from ind=1, we have to convert it to zero-indexed
    config['ind'] = config['ind'] - 1
    assert config['ind'] < len(catalog), f'{config["ind"]} beyond catalog size {len(catalog)}!'
    config['objid'] = catalog[config['ind']]['ID']
    config['object_redshift'] = catalog[config['ind']]['z_spec']
    config['nested_method']="rwalk"
    print(f'\nRunning SED fitting for object {config["objid"]}...')
    print("\n\nRunning with the following configuration:")
    for k,v in config.items():
        print(f'>>> {k} = {v}')

    # --- Get fitting ingredients ---
    obs, model, sps = build_all(**config)
    config["sps_libraries"] = sps.ssp.libraries
    print(model)

    if args.debug:
        sys.exit()

    # --- Set up output ---
    target_dir = "/xdisk/timeifler/jiachuanxu/jwst/fengwu_catalog/sed_fitting"
    hfile = f'{target_dir}/ID{config["objid"]}_DelayedTau_result.h5'

    #  --- Run the actual fit ---
    print(f'\nStart SED fitting...')
    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, 
        noise=(None, None), **config)

    print("\nWriting to {}".format(hfile))
    writer.write_hdf5(hfile,
                      config,
                      model,
                      obs,
                      output["sampling"][0],
                      None,
                      sps=sps,
                      tsample=output["sampling"][1],
                      toptimize=0.0
                      )

    try:
        hfile.close()
    except(AttributeError):
        pass
