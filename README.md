# TAED2_LuminIA

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Traffic signs detection using YOLOv8

## Traffic Signs Detection and Classification 

This repository implements a comprehensive traffic sign detection and classification system using the YOLO deep learning model. The system classifies traffic signs into four main categories: prohibitory, danger, mandatory, and other. It features a well-structured dataset containing a diverse set of annotated images, which capture various traffic sign types and conditions, ensuring robust model performance across different environments. Data integrity is maintained through rigorous validation techniques. The project is designed to support scalable, reproducible experiments with a modular pipeline that facilitates continuous model evaluation and improvement. Additionally, it includes tools for monitoring model performance and metrics, making it easier to fine-tune parameters and optimize results over time.

- <a target='_blank' href = 'https://github.com/taed2-2425q1-gced-upc/TAED2_LuminIA/blob/main/docs/model_card.md'> Model Card </a>
- <a target = '_blank' href = 'https://github.com/taed2-2425q1-gced-upc/TAED2_LuminIA/blob/main/docs/dataset_card.md' > Data Card </a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         taed2_luminia and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── taed2_luminia   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes taed2_luminia a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Instructions

To set up and run the project code, follow these steps:

**Cloning the repository**

With the aim of running it locally, the first step consists of cloning the repository to your local machine as follows:

```bash
git clone git clone https://github.com/taed2-2425q1-gced-upc/TAED2_LuminIA.git
```

**Installing Dependencies**

Once the repository is cloned in your local machine, the required dependencies and libraries need to be installed through Poetry as follows:

```bash
poetry install
```

**Getting the data and model**

Both the data and the model can be retrieved from the DVC remote repository executing the following command:

```bash
dvc pull
```

**Running the pipeline**

Once all is set up, the entire pipeline can be executed to preprocess the images, train the model, and evaluate the results through the following command:

```bash
dvc repro
```

**Contact**

For any questions or issues during execution, please contact the project development team members listed below:

- **Laia Álvarez** - laia.alvarez.capell@estudiantat.upc.edu
- **Adrián Cerezuela** - adrian.cerezuela@estudiantat.upc.edu
- **Eva Jiménez** - eva.jimenez.vargas@estudiantat.upc.edu
- **Roger Martínez** - roger.martinez.gilibets@estudiantat.upc.edu
- **Ramon Ventura** - ramon.ventura@estudiantat.upc.edu
