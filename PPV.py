# Nearest Neighbor Implementation provided by Oussama BENDJABALLAH, refactored into a command line program by Smail KOURTA
import numpy as np
import Parser
import argparse
import time


def PPV(graphe, v_depart=None):
    # Faire une copie du graphe vu qu'il va subir a des modification
    _graphe = graphe.copy()
    # La liste chemin gardera trace de notre parcour
    chemin = []
    # Selection d'un point de depart
    if v_depart is None:
        depart = v_depart = np.random.randint(0, len(graphe))
    depart = v_depart
    chemin.append(v_depart)
    # Creation de l'ensemble des noeuds non visités
    noeudsNonVisite = set(
        np.delete(np.arange(0, len(graphe)), v_depart).flatten())
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
        _graphe[v_depart, chemin] = float("inf")
        _graphe[chemin, v_depart] = float("inf")

    # Ajouter le cout de retour
    cout += graphe[v_suivante, depart]

    return chemin, cout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--start",
                        help="Starting Node",)
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    start_time = time.time()
    if args.start is not None:
        tour, cost = PPV(np.array(instance.data), int(args.start))
    else:
        tour, cost = PPV(np.array(instance.data))
    end_time = time.time()
    print(tour)
    print(cost)
    print(end_time - start_time)
