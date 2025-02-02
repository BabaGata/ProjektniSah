from preprocessing.preprocess_lichess import PreprocessLichess
from constants import *
import pickle
import os
import pandas as pd

FILENAME = "lichess_games_6M.csv"
eco_mapping_df = pd.read_csv("data/ECO_codes_mapping.csv")
eco_mapping = {
    row["eco"]: {
        "index": idx,
        "opening_ply": len(row["pgn"].split())  # Count number of moves in PGN
    }
    for idx, row in eco_mapping_df.iterrows()
}

# Initialize Preprocessing class with directory paths and column mapping
preprocessor = PreprocessLichess(
    filename=FILENAME,
    raw_dir=RAW_DIR,
    preprocessed_dir=PREPROCESSED_DIR,
    column_mapping=COLUMN_MAPPING[FILENAME],
    eco_mapping=eco_mapping
)

# Run the preprocessing steps in one call
processed_df = preprocessor.process_data()

# Save with pickle
with open(os.path.join(PREPROCESSED_DIR, FILENAME.replace(".csv", ".pkl")), 'wb') as f:
    pickle.dump(processed_df, f)

print("Preprocessed data saved successfully!")