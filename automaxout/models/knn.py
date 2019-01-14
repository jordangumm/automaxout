"""K-Nearest Neighbors (KNN)

The primary purpose of this module is to provide automated KNN generation for
the creation of `local knowledge` features.  Basically, map a tournament game
to a cluster of highly similar games that were already played.

Diversity of `local knowledge` KNN models should provide a better sense for
how the game may turn out.

Local Knowledge Contexts:
    * Previous Tournament KNN
    * Previous Regular Season KNN
    * Current Regular Season KNN


TODO:
    probably start with already-proven set of statistics e.g. four factors

"""

from sklearn.neighbors import NearestNeighbors






def get_quasi_probability(team1: int, team2: int, season: int):
    """Calculate quasi-probability of team1 beating team2 using local contexts.

    Args:
        team1:  kaggle identifier
        team2:  kaggle identifier
        season: tournament year

    """
    
