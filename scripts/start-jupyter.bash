#!bin/bash
FILE="/root/.jupyter/jupyter_notebook_config.json"
if [ ! -f "$FILE" ]; then
    jupyter notebook password
    echo "Run script again after you've set up the password."
else
    jupyter lab --ip 0.0.0.0 --allow-root --no-browser
fi
