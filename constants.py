import chess

PREPROCESSED_DIR = "data/preprocessed/"
RAW_DIR = "data/raw/"
MAX_MOVES = 40
OPENING_MOVES = 10

PIECE_TO_INT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

LICHESS_GAMES_20K = {
    "result": "winner",
    "white_id": "white_id",
    "black_id": "black_id",
    "white_rating": "white_rating",
    "black_rating": "black_rating",
    "opening_name": "opening_name",
    "opening_eco": "opening_eco",
    "opening_ply": "opening_ply",
    "moves": "moves"
}

LICHESS_GAMES_6M = {
    "result": "Result",
    "white_id": "White",
    "black_id": "Black",
    "white_rating": "WhiteElo",
    "black_rating": "BlackElo",
    "opening_name": "Opening",
    "opening_eco": "ECO",
    "opening_ply": None,
    "moves": "AN"
}

COLUMN_MAPPING = {
    "lichess_games_20K.csv": LICHESS_GAMES_20K,
    "lichess_games_6M.csv": LICHESS_GAMES_6M
}