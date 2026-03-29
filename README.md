# KAN DoS Detection Thesis

## Overview

This repository contains the code and supporting material for a thesis project on Denial of Service (DoS) attack detection in IoT environments using Kolmogorov-Arnold Networks (KANs).

The project focuses on binary DoS detection from the CICIDS2017 dataset and includes training, evaluation, feature analysis, and hyperparameter study scripts.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|   |-- train.py
|   |-- analyze.py
|   |-- feature_analysis.py
|   `-- hyperparameter_study.py
|-- experiment_data/
|-- figures/
`-- model/
```

## Main Files

- `src/train.py`: prepares the dataset, trains the KAN model, and saves experiment artifacts.
- `src/analyze.py`: evaluates a trained model and produces performance metrics and plots.
- `src/feature_analysis.py`: analyzes feature importance, feature statistics, and correlations.
- `src/hyperparameter_study.py`: runs comparative hyperparameter experiments on different KAN configurations.
- `requirements.txt`: lists the Python dependencies required to run the project.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ComfortablyEnum/kan-dos-detection-thesis.git
cd kan-dos-detection-thesis
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts from the project root.

Train the model:

```bash
python src/train.py
```

Analyze the trained model:

```bash
python src/analyze.py
```

Run feature analysis:

```bash
python src/feature_analysis.py
```

Run the hyperparameter study:

```bash
python src/hyperparameter_study.py
```

## Dataset

The project is based on the CICIDS2017 dataset, with a focus on Wednesday traffic containing benign samples and DoS attack classes.

## Notes

- Some scripts generate output folders such as `experiment_data/`, `figures/`, and model artifacts during execution.
- The `pykan` dependency is installed directly from GitHub through `requirements.txt`.

## Acknowledgments

- CICIDS2017 dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- pykan project: https://github.com/KindXiaoming/pykan
