# Quickstart

Details later. Here is a quick overview on how to replciate the COSMOS KL measurement w/ KROSS data.

## Data Aquisition

1. Download the KROSS data, all inside `REPO_DIR/data/kross` (relative dirs will be given)
- KROSS catalog: https://astro.dur.ac.uk/KROSS/data/kross_release_v2.fits (`data/kross/`)
- KROSS datacubes: https://astro.dur.ac.uk/KROSS/data/KROSS_cubes.tar.gz (`data/kross/cubes/`)
- KROSS H-alpha intensity maps: https://astro.dur.ac.uk/KROSS/data/KROSS_Halpha_maps.tar.gz (`data/kross/halpha`/)
- KROSS velocity maps: https://astro.dur.ac.uk/KROSS/data/KROSS_velocity_maps.tar.gz (`data/kross/vmaps`)
- - NOTE: You may have to move the vmaps from the unpacked `velocity/out` dir to the `vmaps` dir
2. Download the COSMOS data
- For now, just ask Spencer for a tarball. Will streamline later (put in `kl-tools/data/cosmos`)

# Running the scripts

To run a typical KROSS-COSMOS measurement, do the following (detailed options at the top of each script):

1. Go into the analysis dir `kl_tools/kross`
2. Make our own velocity map measurements: `python velocity_map_fitter.py`
- `-c` to overwrite on repeated runs, `-v` for verbose printing
3. Generate the initial KL sample w/ sini & shear estimates: `python generate_kl_sample.py -v`
- `-c` and `-v` do the usual
- The sample is saved to a fits file in `sample/`
4. Add the estimated COSMOS shears to the sample file: `python add_cosmos_shear.py sample/kl_sample.fits`
- `-c` to overwrite on repeated runs
- `-p` to see a plot of the COSMOS shear map
5. Run the analysis w/ the pre-defined selections: `python kl_analysis.py sample/kl_sample_with_cosmos.fits`

To make the diagnostics:
1. Fit the rotation curves: `python rotation_curve_fitter.py`
- `-c` and `-v` do the usual
2. Make the diagnostics: `python kl_sample_diagnostics.py`
- Lots of options at the top of the cript
- Outputs are saved to: `plots/diagnostics`
- Collated file: `kross-cosmos-diagnostics.pdf`