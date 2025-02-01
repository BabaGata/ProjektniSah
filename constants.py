import chess

PREPROCESSED_DIR = "data/preprocessed/"
RAW_DIR = "data/raw/"

PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

LICHESS_GAMES_CSV = {
    "winner": "winner",
    "white_id": "white_id",
    "black_id": "black_id",
    "white_rating": "white_rating",
    "black_rating": "black_rating",
    "opening_name": "opening_name",
    "opening_eco": "opening_eco",
    "opening_ply": "opening_ply",
    "moves": "moves"
}

COLUMN_MAPPING = {
    "lichess_games.csv": LICHESS_GAMES_CSV
}
