#!/bin/bash

# exit as soon as a command fails:
set -e

CWD=$(pwd)
SYS_DIR="$(realpath "$(dirname "$0")")"
REPO_DIR="$(realpath "$SYS_DIR"/../)"

ENV_FILE="$REPO_DIR/environment.yaml"
ENV_NAME="$(grep '^name:' "$ENVFILE" | cut -d' ' -f2)"
echo "Environment name: $ENV_NAME"

# always execute this script with bash, so that conda shell.hook works.
# relevant conda bug: https://github.com/conda/conda/issues/7980
if test "$BASH_VERSION" = ""
then
    exec bash "$0" "$@"
fi

# check if the environment already exists
if conda info --envs | grep -q "$ENVNAME"; then
    echo "Environment '$ENVNAME' already exists. Removing it first..."
    conda deactivate || true
    conda env remove -n "$ENVNAME" --yes
fi

# install environment fresh
echo "Installing '$ENV_NAME' from reproducible conda-lock.yml..."
conda-lock install --name "$ENV_NAME" "$REPO_DIR/conda-lock.yml"

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENVNAME"

# Install additional packages that are not in the conda-lock.yml, due to 
# known issues with pip dependencies in conda-lock.
echo "Pip instaling kl-tools..."
pip install --no-build-isolation --no-deps --editable "REPO_DIR/."

echo "Pip installing pocomc..."
pip install --no-build-isolation --no-deps pocomc

echo "Pip installing ultranest..."
pip install --no-build-isolation --no-deps ultranest

echo "kl-tools installed successfully"