import pandas as pd
import chess
import chess.pgn
import numpy as np
import os
import io
from constants import *

class PreprocessLichess:    
    def __init__(self, filename, raw_dir, preprocessed_dir, column_mapping, eco_mapping):
        self.filename = filename
        self.raw_dir = raw_dir
        self.preprocessed_dir = preprocessed_dir
        self.column_mapping = column_mapping
        self.eco_mapping = eco_mapping
    
    def format_moves(self, moves_str):
        """Format move sequence correctly for PGN by adding move numbers."""
        moves = moves_str.split()
        formatted_moves = []
        
        for i in range(0, len(moves), 2):
            move_number = (i // 2) + 1
            move_pair = f"{move_number}. {moves[i]}"
            if i + 1 < len(moves):
                move_pair += f" {moves[i + 1]}"
            formatted_moves.append(move_pair)
        
        return " ".join(formatted_moves)

    def convert_to_pgn(self, row):
        """Convert a CSV row into a properly formatted PGN string."""
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
        
        # Format moves using the helper method
        pgn += f"{self.format_moves(row[self.column_mapping['moves']])} {result}\n"
        
        return pgn

    def board_to_matrix(self, board):
        """Convert a chess.Board into an 8x8 numpy matrix."""
        board_matrix = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                row = 7 - chess.square_rank(square)  # Flip row to match board visualization
                col = chess.square_file(square)
                value = PIECE_TO_INT[piece.piece_type]
                board_matrix[row, col] = value if piece.color == chess.WHITE else -value
        return board_matrix

    def get_game_matrices(self, pgn_string):
        """Convert a PGN game into a list of board matrices for each move."""
        pgn = io.StringIO(pgn_string)
        game = chess.pgn.read_game(pgn)
        board = game.board()
        matrices = []
        
        for move in game.mainline_moves():
            board.push(move)
            matrices.append(self.board_to_matrix(board))
        
        return matrices

    def process_data(self):
        """Load raw data, preprocess, and save as pickle file."""
        df = pd.read_csv(os.path.join(self.raw_dir, self.filename))

        data = []
        for _, row in df.iterrows():
            pgn = self.convert_to_pgn(row)
            matrices = self.get_game_matrices(pgn)

            eco_info = self.eco_mapping.get(row[self.column_mapping["opening_eco"]])

            if eco_info is None:
                continue

            opening_ply = row[self.column_mapping["opening_ply"]] if self.column_mapping["opening_ply"] else eco_info["opening_ply"]


            data.append({
                "matrices": np.array(matrices, dtype=np.int32),
                "encoded_eco": eco_info["index"],
                "opening_ply": opening_ply
            })

        return pd.DataFrame(data)

        
        