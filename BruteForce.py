# B&B Implementation provided by IFERROUDJENE Mouloud, refactored into a command line program by Smail KOURTA

import numpy as np
import Parser
import argparse
import time


def prochaine_permutation(L):
    n = len(L)
    i = n - 2
    while i >= 0 and L[i] >= L[i + 1]:
        i -= 1
    if i == -1:
        return False
    j = i + 1
    while j < n and L[j] > L[i]:
        j += 1
    j -= 1
    L[i], L[j] = L[j], L[i]
    left = i + 1
    right = n - 1
    while left < right:
        L[left], L[right] = L[right], L[left]
        left += 1
        right -= 1
    return True


# calcule du cout d'un chemin
def coutChemin(villes, v_depart, chemin):
    cout = 0
    v_actuelle = v_depart
    for v_suiv in chemin:
        cout += villes[v_actuelle][v_suiv]
        v_actuelle = v_suiv
    # Ajouter la distance vers la ville de depart
    cout += villes[v_actuelle][v_depart]
    return cout, chemin.copy()


def tspBruteForce(graphe, v_depart):
    # Creation du premier chemin
    chemin = []
    for ville in range(len(graphe)):
        if ville != v_depart:
            chemin.append(ville)

    # Initialisaion du minimum chemin
    cout_min_chemin, min_chemin = coutChemin(graphe, v_depart, chemin)

    # Pour afficher les chemins et leurs cout decommenter cette instruction
    # print(cout_min_chemin,min_chemin )

    # Tester et comparer toutes les permutations
    while prochaine_permutation(chemin):
        # Cacule du cout du chemin
        cout_courant_chemin, courant_chemin = coutChemin(
            graphe, v_depart, chemin)

        # Pour afficher les chemins et leurs cout decommenter cette instruction
        # print(cout_courant_chemin,courant_chemin )

        # La mise a jour du chemin minimum et de son cout
        if (cout_min_chemin > cout_courant_chemin):
            min_chemin = courant_chemin
            cout_min_chemin = cout_courant_chemin

    return [v_depart] + min_chemin, cout_min_chemin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--start",
                        help="Starting Node", )
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    start_time = time.time()
    if args.start is not None:
        tour, cost = tspBruteForce(np.array(instance.data), int(args.start))
    else:
        tour, cost = tspBruteForce(np.array(instance.data), 0)
    end_time = time.time()
    print(tour)
    print(cost)
    print(end_time - start_time)
