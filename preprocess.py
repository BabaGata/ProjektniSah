from preprocessing.preprocess_lichess import PreprocessLichess
from constants import *
import pickle
import os

FILENAME = "lichess_games.csv"

# Initialize Preprocessing class with directory paths and column mapping
preprocessor = PreprocessLichess(
    filename=FILENAME,
    raw_dir=RAW_DIR,
    preprocessed_dir=PREPROCESSED_DIR,
    column_mapping=COLUMN_MAPPING[FILENAME],
)

# Run the preprocessing steps in one call
processed_df = preprocessor.process_data()
# Save with pickle
with open(os.path.join(PREPROCESSED_DIR, FILENAME.replace(".csv", ".pkl")), 'wb') as f:
    pickle.dump(processed_df, f)

print("Preprocessed data saved successfully!")