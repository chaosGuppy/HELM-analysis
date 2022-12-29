# HELM Analysis

An app for sharing insights on HELM data.

## Install

In a clean virtual environment, do

```
pip install -r requirements.txt
```

Then, run the following to create your environment file

```
echo "export HELM_DATA_DIR=<directory-where-you-want-the-data>" > .env
```

Replace `<directory-where-you-want-the-data>` with your chosen path. It need not exist.

Finally, run

```
sh download-data.sh
```

to download the data.

## Run

```
sh run-dashboard.sh
```

The dashboard should open in a browser window.

## Contributing

Please create visualizations for any interesting analysis you do, and add them to the app.
Make a PR with your changes, and tag Lawrence (chaosGuppy).
The dashboard is built using [streamlit](https://docs.streamlit.io/).
Formatting should be [black](https://github.com/psf/black).

## Overview of code

load.py contains functions for reading in data.

accuracy.py contains functions for calculating correctness of model responses
(for example, parsing MATH responses to grab the answer) and for computing accuracy across trials and models.

difficulty.py contains functions for calculating trial difficulty, and for transforming arrays of difficulties (e.g. into quantile form)

agent_characteristic.py contains functions for creating agent characteristic curves

dashboard.py is the dashboard.
