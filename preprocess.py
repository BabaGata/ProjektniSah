import os
import pickle
import pandas as pd
from preprocessing.preprocess_lichess import PreprocessLichess
from constants import *

# Define the input CSV filename
FILENAME = "lichess_games_6M.csv"

def main():
    """Main function to run the preprocessing pipeline."""
    print("Initializing preprocessing...")

    # Initialize PreprocessLichess
    preprocessor = PreprocessLichess(
        filename=FILENAME,
        raw_dir=RAW_DIR,
        preprocessed_dir=PREPROCESSED_DIR,
        column_mapping=COLUMN_MAPPING[FILENAME],
        eco_mapping=ECO_MAPPING
    )

    # Process data in chunks and save batches
    preprocessor.process_data()

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()