import numpy as np
import Parser
import argparse
import time
from random import randint

def generate_initial_tour(graphe):
  dim = len(graphe)
  initial_vector = np.arange(dim).tolist()
  tour = []
  for i in range(dim):
      index = randint(0,dim-i-1)
      tour.append(initial_vector[index])
      del initial_vector[index]
  return tour

#Permuter les aretes entre 2 noeuds.
def swap_2opt(tour, i, j):
  tour[i:j+1] = tour[j:i-1:-1]
  return tour

#Calculer Le cout d'une tournée dans un graphe donné.
def coast_of_tour(graphe, tour):
  return graphe[np.roll(tour, 1), tour].sum()

def solve_tsp_2opt(graphe, initial_tour=None):
  #Si le param
  if initial_tour is None : initial_tour = generate_initial_tour(np.array(graphe))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    # parser.add_argument("--initial_tour",
    #                     help="Initial Tour",)
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    start_time = time.time()
    # if args.start is not None:
    #     initial_tour, tour, init_coast, cost = solve_tsp_2opt(graphe=np.array(instance.data),)
    # else:
    initial_tour, tour, init_coast, cost = solve_tsp_2opt(graphe=np.array(instance.data))
    end_time = time.time()
    print(tour)
    print(cost)
    print(end_time - start_time)
