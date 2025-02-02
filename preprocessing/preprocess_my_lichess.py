import os
import pandas as pd
import chess
import chess.pgn
import numpy as np
import io
from constants import *

class PreprocessMyLichess:
    def __init__(self, pgn_filename, username, my_dir=MY_DIR, eco_mapping=ECO_MAPPING):
        self.pgn_filename = pgn_filename
        self.username = username  # Username to check player color
        self.my_dir = my_dir
        self.eco_mapping = eco_mapping

    def board_to_matrix(self, board):
        """Convert a chess.Board into an 8x8 numpy matrix."""
        board_matrix = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - chess.square_rank(square), chess.square_file(square)
                board_matrix[row, col] = PIECE_TO_INT[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)
        return board_matrix

    def get_game_matrices(self, game):
        """Convert PGN into a list of board matrices."""
        board = game.board()
        return [self.board_to_matrix(board) for move in game.mainline_moves() if not board.push(move)]

    def process_data(self):
        """Load raw .pgn, preprocess, and save it to a pickle file."""
        print(self.my_dir)
        pgn_file_path = os.path.join(self.my_dir, 'lichess', self.pgn_filename)

        # Read the PGN file
        with open(pgn_file_path, "r") as f:
            pgn_data = f.read()

        # Split the file into individual games based on triple newline (ensuring complete games)
        games = pgn_data.split("\n\n\n")

        data = []

        # Process each game from PGN
        for game_str in games:
            try:
                # Convert PGN string to a chess game object
                game = chess.pgn.read_game(io.StringIO(game_str))

                # Extract White and Black player names
                white_player = game.headers.get("White", "").lower()
                black_player = game.headers.get("Black", "").lower()

                # Determine player's color
                if self.username.lower() == white_player:
                    player_color = "white"
                elif self.username.lower() == black_player:
                    player_color = "black"
                else:
                    continue  # Skip games where the username is not present

                # Extract ECO code and map it to encoded ECO
                eco_code = game.headers.get("ECO")
                result = game.headers.get("Result")
                encoded_eco = self.eco_mapping.get(eco_code)
                if encoded_eco is None:
                    continue

                board = game.board()
                # Get the game matrices (board state for each move)
                matrices = [self.board_to_matrix(board) for move in game.mainline_moves() if not board.push(move)]

                # Append the processed data to the list
                data.append({
                    "matrices": np.array(matrices, dtype=np.int32),
                    "encoded_eco": encoded_eco,
                    "opening_ply": OPENING_MOVES,
                    "result": result,
                    "player_color": player_color
                })

            except Exception as e:
                continue  # Skip invalid games

        # Save the processed data as a pickle file
        output_file = os.path.join(self.my_dir, "preprocessed", self.pgn_filename.replace("pgn", "pkl"))
        batch_df = pd.DataFrame(data)
        batch_df.to_pickle(output_file)

        print(f"Processed data saved to: {output_file}")