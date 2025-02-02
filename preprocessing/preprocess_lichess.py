import pandas as pd
import chess
import chess.pgn
import numpy as np
import os
import io
from constants import *
from tqdm import tqdm

class PreprocessLichess:    
    def __init__(self, filename, raw_dir, preprocessed_dir, column_mapping, eco_mapping, chunk_size=100000):
        self.filename = filename
        self.raw_dir = raw_dir
        self.preprocessed_dir = preprocessed_dir
        self.column_mapping = column_mapping
        self.eco_mapping = eco_mapping
        self.chunk_size = chunk_size  # Process in smaller chunks to reduce memory usage

    def format_moves(self, moves_str):
        """Format move sequence correctly for PGN by adding move numbers."""
        moves = moves_str.split()
        return " ".join([f"{(i//2)+1}. {moves[i]} {moves[i+1]}" if i+1 < len(moves) else f"{(i//2)+1}. {moves[i]}" for i in range(0, len(moves), 2)])

    def convert_to_pgn(self, row):
        """Convert a CSV row into a PGN string."""
        result = "1-0" if row[self.column_mapping["result"]] == "white" else "0-1" if row[self.column_mapping["result"]] == "black" else row[self.column_mapping["result"]]
        
        pgn = f"""[Event "?"]
[Site "Lichess.org"]
[Date "?"]
[Round "?"]
[White "{row[self.column_mapping['white_id']]}"]
[Black "{row[self.column_mapping['black_id']]}"]
[WhiteElo "{row[self.column_mapping['white_rating']]}"]
[BlackElo "{row[self.column_mapping['black_rating']]}"]
[Result "{result}"]
[Opening "{row[self.column_mapping['opening_name']]}"]
[ECO "{row[self.column_mapping['opening_eco']]}"]\n"""
        
        return pgn + f"{self.format_moves(row[self.column_mapping['moves']])} {result}\n"

    def board_to_matrix(self, board):
        """Convert a chess.Board into an 8x8 numpy matrix."""
        board_matrix = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - chess.square_rank(square), chess.square_file(square)
                board_matrix[row, col] = PIECE_TO_INT[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)
        return board_matrix

    def get_game_matrices(self, pgn_string):
        """Convert PGN into a list of board matrices."""
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        board = game.board()
        return [self.board_to_matrix(board) for move in game.mainline_moves() if not board.push(move)]

    def process_data(self):
        """Load raw data in chunks, preprocess, and save in smaller batches."""
        raw_file_path = os.path.join(self.raw_dir, self.filename)
        
        # Ensure preprocessed directory exists
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        valid_columns = [col for col in self.column_mapping.values() if col is not None]

        chunk_number = 0
        for chunk in pd.read_csv(raw_file_path, chunksize=self.chunk_size):
            print("Processing Chunk: ", chunk_number)
            chunk = chunk[valid_columns]

            data = []
            for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc="Processing Data"):
                pgn = self.convert_to_pgn(row)
                matrices = self.get_game_matrices(pgn)

                eco_code = self.eco_mapping.get(row[self.column_mapping["opening_eco"]])
                if eco_code is None:
                    continue

                opening_ply = row[self.column_mapping["opening_ply"]] if self.column_mapping["opening_ply"] else OPENING_MOVES

                data.append({
                    "matrices": np.array(matrices, dtype=np.int32),
                    "encoded_eco": eco_code,
                    "opening_ply": opening_ply
                })

            # Save each chunk as a separate file
            output_file = os.path.join(self.preprocessed_dir, f"{self.filename}_{chunk_number}.pkl")
            batch_df = pd.DataFrame(data)
            batch_df.to_pickle(output_file)
            print(f"Saved chunk {chunk_number} to {output_file}")
        
        chunk_number += 1
        print("All chunks processed successfully!")