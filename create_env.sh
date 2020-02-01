#!/bin/bash

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

conda create -y -n hydrodata python=3.7
conda create -y -n nhdplus -c conda-forge r-rgdal r-sf r-curl r-httr r-prettymapr r-rosm r-optparse r-geojsonio

conda activate nhdplus
Rscript -e \
"options(repos = 'https://cran.revolutionanalytics.com'); \
install.packages('nhdplusTools');"

conda deactivate

conda activate hydro
pip install -U -r requirements.txt

