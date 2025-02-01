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

ECO_MAPPING = {
    "A00":{
        "name":"Polish (Sokolsky) opening",
        "moves":"1. b4"
    },
    "A01":{
        "name":"Nimzovich-Larsen attack",
        "moves":"1. b3"
    },
    "A02":{
        "name":"Bird's opening",
        "moves":"1. f4"
    },
    "A04":{
        "name":"Reti opening",
        "moves":"1. Nf3"
    },
    "A10":{
        "name":"English opening",
        "moves":"1. c4"
    },
    "A40":{
        "name":"Queen's pawn",
        "moves":"1. d4"
    },
    "A42":{
        "name":"Modern defence, Averbakh system",
        "moves":"1. d4 d6 2. c4 g6 3. Nc3 Bg7 4. e4"
    },
    "A43":{
        "name":"Old Benoni defence",
        "moves":"1. d4 c5"
    },
    "A45":{
        "name":"Queen's pawn game",
        "moves":"1. d4 Nf6"
    },
    "A47":{
        "name":"Queen's Indian defence",
        "moves":"1. d4 Nf6 2. Nf3 b6"
    },
    "A48":{
        "name":"King's Indian, East Indian defence",
        "moves":"1. d4 Nf6 2. Nf3 g6"
    },
    "A50":{
        "name":"Queen's pawn game",
        "moves":"1. d4 Nf6 2. c4"
    },
    "A51":{
        "name":"Budapest defence",
        "moves":"1. d4 Nf6 2. c4 e5"
    },
    "A53":{
        "name":"Old Indian defence",
        "moves":"1. d4 Nf6 2. c4 d6"
    },
    "A56":{
        "name":"Benoni defence",
        "moves":"1. d4 Nf6 2. c4 c5"
    },
    "A57":{
        "name":"Benko gambit",
        "moves":"1. d4 Nf6 2. c4 c5 3. d5 b5"
    },
    "A60":{
        "name":"Benoni defence",
        "moves":"1. d4 Nf6 2. c4 c5 3. d5 e6"
    },
    "A80":{
        "name":"Dutch",
        "moves":"1. d4 f5"
    },
    "B00":{
        "name":"King's pawn opening",
        "moves":"1. e4"
    },
    "B01":{
        "name":"Scandinavian (centre counter) defence",
        "moves":"1. e4 d5"
    },
    "B02":{
        "name":"Alekhine's defence",
        "moves":"1. e4 Nf6"
    },
    "B06":{
        "name":"Robatsch (modern) defence",
        "moves":"1. e4 g6"
    },
    "B07":{
        "name":"Pirc defence",
        "moves":"1. e4 d6 2. d4 Nf6 3. Nc3"
    },
    "B10":{
        "name":"Caro-Kann defence",
        "moves":"1. e4 c6"
    },
    "B20":{
        "name":"Sicilian defence",
        "moves":"1. e4 c5"
    },
    "C00":{
        "name":"French defence",
        "moves":"1. e4 e6"
    },
    "C20":{
        "name":"King's pawn game",
        "moves":"1. e4 e5"
    },
    "C21":{
        "name":"Centre game",
        "moves":"1. e4 e5 2. d4 exd4"
    },
    "C23":{
        "name":"Bishop's opening",
        "moves":"1. e4 e5 2. Bc4"
    },
    "C25":{
        "name":"Vienna game",
        "moves":"1. e4 e5 2. Nc3"
    },
    "C30":{
        "name":"King's gambit",
        "moves":"1. e4 e5 2. f4"
    },
    "C40":{
        "name":"King's knight opening",
        "moves":"1. e4 e5 2. Nf3"
    },
    "C41":{
        "name":"Philidor's defence",
        "moves":"1. e4 e5 2. Nf3 d6"
    },
    "C42":{
        "name":"Petrov's defence",
        "moves":"1. e4 e5 2. Nf3 Nf6"
    },
    "C44":{
        "name":"King's pawn game",
        "moves":"1. e4 e5 2. Nf3 Nc6"
    },
    "C45":{
        "name":"Scotch game",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4"
    },
    "C46":{
        "name":"Three knights game",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Nc3"
    },
    "C47":{
        "name":"Four knights, Scotch variation",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. d4"
    },
    "C50":{
        "name":"Italian Game",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Bc4"
    },
    "C51":{
        "name":"Evans gambit",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4"
    },
    "C53":{
        "name":"Giuoco Piano",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3"
    },
    "C55":{
        "name":"Two knights defence",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6"
    },
    "C60":{
        "name":"Ruy Lopez (Spanish opening)",
        "moves":"1. e4 e5 2. Nf3 Nc6 3. Bb5"
    },
    "D00":{
        "name":"Queen's pawn game",
        "moves":"1. d4 d5"
    },
    "D01":{
        "name":"Richter-Veresov attack",
        "moves":"1. d4 d5 2. Nc3 Nf6 3. Bg5"
    },
    "D02":{
        "name":"Queen's pawn game",
        "moves":"1. d4 d5 2. Nf3"
    },
    "D03":{
        "name":"Torre attack (Tartakower variation)",
        "moves":"1. d4 d5 2. Nf3 Nf6 3. Bg5"
    },
    "D04":{
        "name":"Queen's pawn game",
        "moves":"1. d4 d5 2. Nf3 Nf6 3. e3"
    },
    "D06":{
        "name":"Queen's Gambit",
        "moves":"1. d4 d5 2. c4"
    },
    "D07":{
        "name":"Queen's Gambit Declined, Chigorin defence",
        "moves":"1. d4 d5 2. c4 Nc6"
    },
    "D10":{
        "name":"Queen's Gambit Declined Slav defence",
        "moves":"1. d4 d5 2. c4 c6"
    },
    "D16":{
        "name":"Queen's Gambit Declined Slav accepted, Alapin variation",
        "moves":"1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4"
    },
    "D17":{
        "name":"Queen's Gambit Declined Slav, Czech defence",
        "moves":"1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 Bf5"
    },
    "D20":{
        "name":"Queen's gambit accepted",
        "moves":"1. d4 d5 2. c4 dxc4"
    },
    "D30":{
        "name":"Queen's gambit declined",
        "moves":"1. d4 d5 2. c4 e6"
    },
    "D43":{
        "name":"Queen's Gambit Declined semi-Slav",
        "moves":"1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Nf3 c6"
    },
    "D50":{
        "name":"Queen's Gambit Declined, 4.Bg5",
        "moves":"1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5"
    },
    "D70":{
        "name":"Neo-Gruenfeld defence",
        "moves":"1. d4 Nf6 2. c4 g6 3. f3 d5"
    },
    "D80":{
        "name":"Gruenfeld defence",
        "moves":"1. d4 Nf6 2. c4 g6 3. Nc3 d5"
    },
    "E00":{
        "name":"Queen's pawn game",
        "moves":"1. d4 Nf6 2. c4 e6"
    },
    "E01":{
        "name":"Catalan, closed",
        "moves":"1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2"
    },
    "E10":{
        "name":"Queen's pawn game",
        "moves":"1. d4 Nf6 2. c4 e6 3. Nf3"
    },
    "E11":{
        "name":"Bogo-Indian defence",
        "moves":"1. d4 Nf6 2. c4 e6 3. Nf3 Bb4+"
    },
    "E12":{
        "name":"Queen's Indian defence",
        "moves":"1. d4 Nf6 2. c4 e6 3. Nf3 b6"
    },
    "E20":{
        "name":"Nimzo-Indian defence",
        "moves":"1. d4 Nf6 2. c4 e6 3. Nc3 Bb4"
    },
    "E60":{
        "name":"King's Indian defence",
        "moves":"1. d4 Nf6 2. c4 g6"
    },
}