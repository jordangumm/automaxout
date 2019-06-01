"""Automated Maxout model selection."""
import random
import sys

from typing import Any, Callable

import click
import pandas as pd

from deap  import base, creator
from deap  import tools
from deap  import algorithms
from numpy import mean


def random_node_count(self, low: int, high: int) -> int:
    """Select random node count divisible by pool size.

    Number of nodes must be divisible by pool size.
    Half of number of nodes must also be divisible by pool size.

    Args:
        low:  lowest node count to consider
        high: highest node count to consider

    TODO:
        Look into using pool size as hyperparameter.
        Maxout paper stated pool size of 2 is good.

    """
    num = random.randint(low, high)
    while (num/2) % 2 or num % 2:
        num = random.randint(low, high)
    return num


class GeneticSelector():
    """Evolutionary Model Selection.

    Attributes:
        maxout:      class of Maxout architecture to train
        X:           example features
        y:           example classes
        node_min:    minimum nodes
        node_max:    maximum nodes
        layer_min:   minimum layers
        layer_max:   maximum layers
        dropout_min: minimum dropout
        dropout_max: maximum dropout
        ngen:        number of hyperparameter generations to search

    """
    def __init__(
            self,
            maxout:      Callable[Any, Any],
            X:           pd.DataFrame,
            y:           pd.DataFrame,
            node_min:    int,
            node_max:    int,
            layer_min:   int,
            layer_max:   int,
            dropout_min: float,
            dropout_max: float,
            ngen:        int,
    ) -> None:
        """Initialize EvolveSelector.

        Args:
            maxout:      class of Maxout architecture to train
            X:           example features
            y:           example classes
            node_min:    minimum nodes
            node_max:    maximum nodes
            layer_min:   minimum layers
            layer_max:   maximum layers
            dropout_min: minimum dropout
            dropout_max: maximum dropout
            ngen:        number of hyperparameter generations to search

        """
        self.maxout      = maxout
        self.X           = X
        self.y           = y
        *,
        self.node_min    = node_min
        self.node_max    = node_max
        self.layer_min   = layer_min
        self.layer_max   = layer_max
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.ngen        = ngen

    def evaluate(self, indi):
        """Evaluate individual.

        Args:
            indi: hyperparameter set to evaluate

        Returns:
            Log loss score.

        TODO:
            Integrate boosting-based evaluation.
        
        """
        probs = self.maxout(
            num_layers=indi[0],
            num_nodes=indi[1],
            dropout_p=indi[2],
        ).fit(self.X, self.y, self.X, self.y)  # TODO set validation data sets with boosting

        score = log_loss(self.y, probs)  # TODO base score on validation

        print(indi, '\tscore: ', score)
        return score

    def random_mutation(self, indi):
        """Mutate an individual by replacing attributes.
        
        Mutations (replacements) occur at 50% probability.
        
        Args:
            indi: individual to mutate
            
        """
        # randomly reset number of nodes on coin flip
        if random.randint(0, 1):
            indi[0] = random_node_count(self.node_min, self.node_max)

        # randomly reset number of layers on coin flip
        if random.randint(0, 1):
            indi[1] = random.randint(self.layer_min, self.layer_max)

        # randomly reset dropout on coin flip
        if random.randint(0, 1):
            indi[2] = random.random(self.dropout_min, self.dropout_max)

        # randomly reset early stopping on coin flip
        if random.randint(0, 1):
            indi[3] = random.randint(1, 3)  # TODO add as arguments

        return indi

    def select_best(self, inds, k):
        """Select k best non-duplicate individuals.

        Args:
            inds: set of individuals to select from
            k:    number of individuals to select

        Returns:
            Set of best scoring individuals.

        """
        best = []
        best_added = 0
        while True:
            scores = [mean(ind.fitness.values) for ind in inds]
            index = scores.index(min(scores))
            if inds[index] not in best:
                best.append(individuals[index])
                best_added += 1
            print '{}: {}'.format(inds[index], scores[index])
            del inds[index]
            if best_added == k:
                break
        return best

    def get_toolbox(self):
        num_layer_min, num_layer_max = 1, 2
        num_nodes_min, num_nodes_max = 2, 100
        dropout_min, dropout_max = 0.0, 0.5

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("num_layers", random.randint, num_layer_min, num_layer_max)
        toolbox.register("num_nodes", self.get_random_numnodes_for_poolsize, num_nodes_min, num_nodes_max)
        toolbox.register("dropout_p", random.uniform, dropout_min, dropout_max)
        toolbox.register("early_stop", random.randint, 1, 3)  # TODO add as argument
        toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.num_layers, toolbox.num_nodes, toolbox.dropout_p), n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxUniform)
        toolbox.register("mutate", self.random_mutation)
        toolbox.register("select", selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)

        return toolbox

    def select_best_model(self):
        """ """
        toolbox = self.get_toolbox()
        pop = toolbox.population(n=10)

        # evaluate the entire population
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self.ngen):
            # select the next generation of individuals
            offspring = toolbox.select(pop, len(pop))
            # clone the selected individuals
            offspring = map(toolbox.clone, offspring)

            # apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.randint(0, 1):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # the population is entirely replaced by the offspring
            pop[:] = offspring
        best = toolbox.select(pop, 1)[0]
        print 'best: {}'.format(mean(best.fitness.values))


@click.command()
@click.argument('ngen', type=click.INT)
def run(ngen):
    for i, s in enumerate((2010,2011,2012,2013,2014,2015,2016)):
        if i == 0:
            df = pd.read_csv('data/final/{}_tourney_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('data/final/{}_tourney_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    features = df.keys().tolist()
    features.remove('won')
    features.remove('season')

    selector = ModelSelector(df=df, features=features,
                        eval_type='bayes_loss', ngen=ngen)
    selector.select_best_model()


if __name__ == "__main__":
    run()
