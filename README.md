# Kaggle dataset links
1. preuzeto (lichess_games.csv) - [Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess): Cijele partije sa informacijama o otvaranju, preko 20K partija
2. [Chess Openings High ELO](https://www.kaggle.com/datasets/alexandrelemercier/all-chess-openings): Statistika o najcescim otvaranjima i prvih par poteza u partiji, nema cijelih partija koliko sam vidjela
3. [High-Elo Games Chess Opening Dataset](https://www.kaggle.com/datasets/arashnic/chess-opening-dataset): Slicni, ako ne i isti dataset ko i ovaj prije (2.)
4. [Online Chess Games](https://www.kaggle.com/datasets/ulrikthygepedersen/online-chess-games): Isti ko lichess (1.), jedino je naziv otvaranja i varijante izdvojen u zasebne stupce
4. [Chess Games](https://www.kaggle.com/datasets/arevel/chess-games): Lichess, trebalo bi bit preko 6M partija
4. [60,000+ Chess Game Dataset (Chess.com)](https://www.kaggle.com/datasets/adityajha1504/chesscom-user-games-60000-games): Chess.com, nema naziva otvanranja al se lako iz prvih par poteza izvuce
4. [Chess Games of Woman Grandmasters (2009 - 2021)](https://www.kaggle.com/datasets/rohanrao/chess-games-of-woman-grandmasters): Chess.com, nema naziva otvanranja al se lako iz prvih par poteza izvuce
4. [Chess position evaluation](https://www.kaggle.com/datasets/petert1/chess-position-evaluation?select=train_houdini_normal_977000.csv): There is a dataset of about 1 million chess positions in FEN string format with Houdini evaluatio. Python skripte za Houdini.
4. [Chess Games Analysis Corpus](https://www.kaggle.com/datasets/adityajha1504/corpuschessstratergybook): This is the corpus of the book, Chess Strategy by Edward Lasker, which is divided into two parts. Knjiga, analiza partija, korisno za nlp.
4. [python-chess library data](https://github.com/niklasf/python-chess/tree/master/data): Git repo sa hrpu podataka u formatima koje nemam pojma kako procesirat al ako bude trebalo das e nade.


# Korisni library, modeli, algoritmi...
1. [python-chess library](https://python-chess.readthedocs.io/en/latest/): Pretvorba FEN formata partije u chess board

# Ideje za projektni
- Iz dataseta od partija trenirati NN iz sredisnjica partija i predvidati otvaranja.
- Treniranje modela kao AlphaZero i potom mijenjanje izlaznog sloja i fine tuneanje modela da se predvidaju otvaranja iz sredisnjica.