# kl-tools: Kinematics Lensing Toolkit

## Installation

1. You will need an installation of `conda`. I recommend [`miniforge3`](https://www.google.com/search?client=firefox-b-1-d&q=miniforge3)
2. Run `make install`
3. Check your installation by running `make test`

You should be good to go! Make sure to `conda activate kl-tools` whenever you want to run or develop the code.

## Getting Started

Until more comprehensive documentation is written, please checkout `notebooks/getting_started.py` in this repo.

## Updating the `kl-tools` environment

If you need a package that does not currently exist in the `kl-tools`, do the following:
1. Make sure you're in the `kl-tools` environment
2. `conda install {your-package}`
3. `conda export --no-builds --from-history -f environment.yaml`
4. `git add environment.yaml` on your branch
5. Merge in PR after it's been approved
6. Inform other users that they likely need to rerun `make install` during their next pull, likely on slack

**NOTE:** This will be streamlined once we have our scripts setup for `conda-lock`.

## Collaborating

We are just getting started using `kl-tools` as a larger group. Let's stick to the following guidelines:
1. Low threshold for posting issues - let's help one another!
2. No pushing to `main` without a PR - please work on a branch (ideally in the format of `{user}/{branch-name}`) and submit a pull request with a reviewer tagged. Default is @sweverett.
3. Keep most branches focused on a single issue and with a short lifetime. Let's merge in features quickly and not leave branches (or PRs) dangling.
4. Until we have CI (continuous integration) setup, please make sure the unit tests run on your branch before sumitting a PR. Do this by running `make test`.
