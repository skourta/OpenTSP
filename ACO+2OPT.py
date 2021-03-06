# Hybrid scheme Implementation provided by Omar Tahmi, refactored into a command line program by Smail KOURTA
import numpy as np
import Parser
import argparse
import time
import random


# Permuter les aretes entre 2 noeuds.
def swap_2opt(tour, i, j):
    tour[i:j+1] = tour[j:i-1:-1]
    return tour

# Calculer Le cout d'une tournée dans un graphe donné.


def coast_of_tour(graphe, tour):
    return graphe[np.roll(tour, 1), tour].sum()


def solve_tsp_2opt(graphe, initial_tour=None):
    # Si le param
    if initial_tour is None:
        initial_tour = generate_initial_tour(graphe)
    tour = initial_tour.copy()
    dimension = len(graphe)
    final_coast = initial_coast = coast_of_tour(graphe, tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, dimension - 2):
            for j in range(i+1, dimension):
                current_coast = coast_of_tour(graphe, tour)
                new_coast = coast_of_tour(graphe, swap_2opt(tour[:], i, j))
                if current_coast > new_coast:
                    improved = True
                    swap_2opt(tour, i, j)
                    final_coast = new_coast
    return initial_tour, tour, initial_coast, final_coast


class AC:
    class Ant:
        def __init__(self, alpha, beta, weights, pherom):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = weights.shape[0]
            self.weights = weights
            self.pheroms = pherom
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            unvisited_nodes = [node for node in range(
                self.num_nodes) if node not in self.tour]
            rowWeights = self.weights[self.tour[-1]][unvisited_nodes]
            rowPheroms = self.pheroms[self.tour[-1]][unvisited_nodes]
            tauIetaI = [(rowPheroms[i] ** self.alpha) * ((1 / rowWeights[i]) ** self.beta) for i in
                        range(len(unvisited_nodes))]
            roulette_wheel = np.sum(tauIetaI)
            random_value = random.random()
            for i, unvisited_node in enumerate(unvisited_nodes):
                random_value -= tauIetaI[i] / roulette_wheel
                if random_value <= 0:
                    return unvisited_node

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.weights[self.tour[i]
                                              ][self.tour[(i + 1) % self.num_nodes]]
            return self.distance

    def __init__(self, weights, mode='ACS', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0,
                 beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = weights.shape[0]
        self.nodes = nodes
        self.weights = weights
        self.pheroms = np.full(
            (weights.shape[0], weights.shape[1]), initial_pheromone)
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.ants = [self.Ant(alpha, beta, self.weights, self.pheroms)
                     for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.pheroms[tour[i]][tour[(
                i + 1) % self.num_nodes]] += weight * pheromone_to_add

    def _acs(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.pheroms[i][j] *= (1.0 - self.rho)

    def _elitist(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance

            self._add_pheromone(
                self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.pheroms[i][j] *= (1.0 - self.rho)

    def _max_min(self):
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = float("inf")
            for ant in self.ants:
                ant.find_tour()
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance

            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour,
                                    iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self._add_pheromone(self.global_best_tour,
                                    self.global_best_distance)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_distance
            min_pheromone = max_pheromone * self.min_scaling_factor
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.pheroms[i][j] *= (1.0 - self.rho)
                    if self.pheroms[i][j] > max_pheromone:
                        self.pheroms[i][j] = max_pheromone
                    elif self.pheroms[i][j] < min_pheromone:
                        self.pheroms[i][j] = min_pheromone

    def run(self):
        # print('Started : {0}'.format(self.mode))
        start_time = time.time()
        if self.mode == 'ACS':
            self._acs()
        elif self.mode == 'Elitist':
            self._elitist()
        else:
            self._max_min()

        # 2OPT
        initial_tour, self.global_best_tour, init_coast, self.global_best_distance = solve_tsp_2opt(
            self.weights,                                                                                            self.global_best_tour)
        #########

        end_time = time.time()
        print(self.global_best_tour)
        print(self.global_best_distance)
        print(end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance",
                        help="Path to the instance file",
                        required=True,)
    parser.add_argument("--mode",
                        help="AC Mode",
                        default="ACS")
    parser.add_argument("--colony_size",
                        help="AC Colony Size",
                        default=10)
    parser.add_argument("--elitist_weight",
                        help="AC Elitist Weight",
                        default=1.0)
    parser.add_argument("--min_scaling_factor",
                        help="AC Min Scaling Factor",
                        default=0.001)
    parser.add_argument("--alpha",
                        help="AC Alpha Parameter",
                        default=1.0)
    parser.add_argument("--beta",
                        help="AC Beta Parameter",
                        default=3.0)
    parser.add_argument("--rho",
                        help="AC Rho Parameter",
                        default=0.1)
    parser.add_argument("--pheromone_deposit_weight",
                        help="AC Pheromone Deposit Weight",
                        default=1.0)
    parser.add_argument("--initial_pheromone",
                        help="AC Initial Pheromone",
                        default=1.0)
    parser.add_argument("--steps",
                        help="AC Steps",
                        default=100)
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    ac = AC(np.array(instance.data), mode=args.mode, colony_size=int(args.colony_size), elitist_weight=float(args.elitist_weight), min_scaling_factor=float(args.min_scaling_factor), alpha=float(args.alpha),
            beta=float(args.beta), rho=float(args.rho), pheromone_deposit_weight=float(args.pheromone_deposit_weight), initial_pheromone=float(args.initial_pheromone), steps=int(args.steps), nodes=None, labels=None)
    ac.run()
