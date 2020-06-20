import numpy as np
import random
import argparse
import Parser
import time
import copy

def generate_first_solution(graphe, v_depart=None):
    # Faire une copie du graphe vu qu'il va subir a des modification
    _graphe = graphe.copy()
    # La liste chemin gardera trace de notre parcour
    chemin = []
    # Selection d'un point de depart
    if v_depart is None: depart = v_depart = np.random.randint(0, len(graphe))
    depart = v_depart
    chemin.append(v_depart)
    # Creation de l'ensemble des noeuds non visités
    noeudsNonVisite = set(np.delete(np.arange(0, len(graphe)), v_depart).flatten())
    # Mise a zero du coup de la solution
    cout = 0
    while (len(noeudsNonVisite) != 0):
        # Retourner le plus proche voisin
        v_suivante = np.argmin(_graphe[v_depart, :])
        # Màj du chemin
        chemin.append(v_suivante)
        # Màj du cout
        cout += _graphe[v_depart, v_suivante]
        # Visiter le prochain neoud
        noeudsNonVisite.remove(v_suivante)
        v_depart = v_suivante
        # Mettre vers les noeuds deja visité a l'infini
        _graphe[v_depart, chemin] = _graphe[chemin, v_depart] = float("inf")

    # Ajouter le cout de retour
    cout += graphe[v_suivante, depart]

    return chemin, cout


def find_neighborhood(solution, matrice):
    neighborhood_of_solution = []

    for n in solution[1:-1]:
        idx1 = solution.index(n)
        for kn in solution[1:-1]:
            idx2 = solution.index(kn)
            if n == kn:
                continue

            _tmp = copy.deepcopy(solution)
            _tmp[idx1] = kn
            _tmp[idx2] = n

            distance = 0
            for i in range(len(matrice)):
                distance += matrice[_tmp[i - 1]][_tmp[i]]

            _tmp.append(distance)

            if _tmp not in neighborhood_of_solution:
                neighborhood_of_solution.append(_tmp)

    indexOfLastItemInTheList = len(neighborhood_of_solution[0]) - 1

    neighborhood_of_solution.sort(key=lambda x: x[indexOfLastItemInTheList])
    return neighborhood_of_solution


def tabu_search(matrice, iters, size, start_node=None):
    # Generation d'une solution initial
    solution, best_cost = generate_first_solution(matrice, start_node)
    # Initialisation de la liste tabou
    tabu_list = list()

    best_solution_ever = solution

    for count in range(iters):
        # Generation des voisins de la solution
        neighborhood = find_neighborhood(solution, matrice)
        index_of_best_solution = 0
        best_solution = neighborhood[index_of_best_solution]
        best_cost_index = len(best_solution) - 1

        found = False
        while found is False:

            for i in range(len(best_solution)):
                if best_solution[i] != solution[i]:
                    first_exchange_node = best_solution[i]
                    second_exchange_node = solution[i]
                    break

            if [first_exchange_node, second_exchange_node] not in tabu_list and [second_exchange_node,
                                                                                 first_exchange_node] not in tabu_list:
                tabu_list.append([first_exchange_node, second_exchange_node])
                found = True
                solution = best_solution[:-1]
                cost = neighborhood[index_of_best_solution][best_cost_index]
                if cost < best_cost:
                    best_cost = cost
                    best_solution_ever = solution
            else:
                index_of_best_solution = index_of_best_solution + 1
                best_solution = neighborhood[index_of_best_solution]

        if len(tabu_list) >= size:
            _ = tabu_list.pop(0)

    return best_solution_ever, best_cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--iterations",
                        help="Number of Iterations", )
    parser.add_argument("--size",
                        help="Size of Tabu List", )
    parser.add_argument("--start",
                        help="Starting Node", default=0)
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    start_time = time.time()
    tour, cost = tabu_search(np.array(instance.data), iters=int(args.iterations), size=int(args.size), start_node=int(args.start))
    end_time = time.time()
    print(tour)
    print(cost)
    print(end_time - start_time)
