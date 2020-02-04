#!/bin/bash

# Path to the parent directory of miniconda installation directory
APP_DIR=~/.local/apps

ana () {
        shell="$(basename $SHELL)"
        __conda_setup="$(${APP_DIR}/miniconda/bin/conda shell.$shell hook 2> /dev/null)"
        if [ $? -eq 0 ]
        then
                eval "$__conda_setup"
        else
                if [ -f "${APP_DIR}/miniconda/etc/profile.d/conda.sh" ]
                then
                        . "${APP_DIR}/miniconda/etc/profile.d/conda.sh"
                else
                        export PATH="${APP_DIR}/miniconda/bin:$PATH"
                fi
        fi
        unset __conda_setup
}

ana

conda create -y -n hydrodata python pip
conda activate hydro
pip install -U -r requirements.txt

conda deactivate

conda create -y -n nhdplus -c conda-forge r-rgdal r-sf r-curl r-httr r-prettymapr r-rosm r-optparse r-geojsonio
conda activate nhdplus

# The CRAN server can be changed. This is the Texas server.
Rscript -e \
"options(repos = 'https://cran.revolutionanalytics.com'); \
install.packages('nhdplusTools');"
