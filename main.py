from preprocessing import Preprocessing
from constants import *

FILENAME = "lichess_games.csv"

# Initialize Preprocessing class with directory paths and column mapping
preprocessor = Preprocessing(
    filename=FILENAME,
    raw_dir=RAW_DIR,
    preprocessed_dir=PREPROCESSED_DIR,
    column_mapping=COLUMN_MAPPING[FILENAME],
)

# Run the preprocessing steps in one call
preprocessor.process_data()