# ml_playground

## Installation

This repo uses Poetry for dependency management. To set up this project, first install
[Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10
installed on your system.

Then, configure poetry to set up a virtual environment that uses Python 3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the
following command:
```bash
poetry install --no-root
```

Activate the environment:
```bash
poetry shell
```
