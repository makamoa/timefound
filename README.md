
# Alpha Series

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.




## Installation
### Prerequisites

- Python 3
- pip
- git

### Geting repository
```bash
git clone https://github.com/makamoa/foundation.git
cd foundation
```

### Setting Up the Environment

Create a virtual environment:
```bash
python3 -m venv /Users/antonvoskresenskii/Documents/venvs/alpha
```

Activate the virtual environment:
```bash
activat moment
```

Upgrade pip:
```bash
pip install --upgrade pip
```

Navigate to the root folder of the project via terminal and install the required packages:

```bash
pip install -r requirements.txt --no-cache-dir
```

## Testing

This project uses `pytest` for testing.

Then, you can run the tests with the following command:

```bash
pytest --log-cli-level=INFO -s
```

The `--log-cli-level=INFO` option sets the logging level to INFO, meaning that all INFO, WARNING, ERROR, and CRITICAL messages will be displayed in the console. The `-s` option disables output capturing, allowing you to see print statements output for passing tests.

This will run all the tests in the project. If you want to run a specific test, you can specify the file name:

```bash
pytest tests/test_notebooks.py --log-cli-level=INFO -s
pytest tests/test_modules.py --log-cli-level=INFO -s
pytest tests/test_formatting.py --log-cli-level=INFO -s
```

## Test coverage

```bash
pytest --cov=./
pytest --cov=./ --cov-report html
```

## Using the Notebook Processing Script

The `utils.py` script is used to process Jupyter notebooks in a specified directory. It performs two main tasks:
1. The script updates the format of each notebook to at least version 4.5
2. The script assigns a unique ID to each cell in the notebooks

### How to Use

To use the script, you need to run it from the root directory:

```python
python3 src/utils.py
```

## Using the linter

black notebooks/

## Project Structure

- notebooks/: Contains Jupyter notebooks for exploratory data analysis and model training
- src/: Contains the source code for the project
- tests/: Contains test scripts
- data/: Contains data files
- config/: Contains configuration files
