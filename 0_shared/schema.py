"""The schema of the dataset in the run directory: kion.py, or yambda.py when DATASET=yambda."""
import importlib
import os

dataset = importlib.import_module(os.environ.get("DATASET", "kion"))
