import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(ROOT_DIR, "input_data")
TEST_DATA = os.path.join(DATA_PATH, "test.csv")
TRAIN_DATA = os.path.join(DATA_PATH, "train.csv")
VAL_DATA = os.path.join(DATA_PATH, "val.csv")

MODEL_NAME = 'SkolkovoInstitute/russian_toxicity_classifier'

MODEL_PATH = os.path.join(ROOT_DIR, "result")


