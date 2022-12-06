# Antimaia Games

This directory accompanies my blog post [When Suboptimal Minimax is Better](https://lukesalamone.github.io/posts/suboptimal-minimax/) which outlines opponent modeling in chess. Using the algorithm outlined in the post, I have simulated 400 games:

| White | Black | Games | White win % | Average half moves per game |
| ----- | ----- | ----- | ----------- | --------------------------- |
| Antimaia 1900 | Maia 1900 | 100 | 100% | 37 |
| Maia 1900 | Antimaia 1900 | 100 | 0% | 43 |
| Maia 1900 | Stockfish @10 | 100 | 0% | 67 |
| Stockfish @10 | Maia 1900 | 100 | 100% | 70 |

## Note
 - Maia 1900 uses Maia Chess weights trained on games from players rated near 1900.  
 - Antimaia 1900 uses Maia 1900 weights and Stockfish @10.  
 - Stockfish @10 is Stockfish 14.1 evaluated at depth 10.  

## Analysis

Antimaia finishes its games much more quickly (12-16 moves faster!). This demonstrates that Antimaia is exploiting weaknesses in Maia's policy as determined by Stockfish, and playing high risk/high reward moves which usually pay off.
