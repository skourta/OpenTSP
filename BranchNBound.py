# B&B Implementation provided by Smail KOURTA, refactored into a command line program by Smail KOURTA
import numpy as np
import Parser
import argparse
import time


def reduce_matrix(data):
    # Faire une copie de la matrice pour ne pas faire des changement sur la matrice originale
    reduced = np.copy(data)
    # Recuperer les minimus de toutes les lignes
    rowMin = np.array([np.min(data[i, :]) for i in range(data.shape[0])])
    # Transformer le vecteur colonne en vecteur ligne
    rowMin = rowMin[:, np.newaxis]
    # Metter ∞ a 0 pour pouvoir faire la soustraction
    rowMin[rowMin == np.inf] = 0
    # Soustraction des minimums des lignes chaqun de sa ligne correspondante
    reduced = data - rowMin
    # Recuperer les minimums des columns
    colMin = np.array([np.min(reduced[:, i]) for i in range(reduced.shape[0])])
    # Metter ∞ a 0 pour pouvoir faire la soustraction
    colMin[colMin == np.inf] = 0
    # Soustraction des minimums des colonnes chaqun de sa colonne correspondante
    reduced = reduced - colMin
    # Retourner la matrice reduite et la sommes de tous les minimums
    return reduced, np.sum(rowMin) + np.sum(colMin)


def getNextNode(fullNode, visited, active, start, nodeSuiv, size):
    # recuperer les donnes a utiliser
    data = fullNode[1]['matrix']
    node = fullNode[1]['node']
    cost = fullNode[1]['cost']
    parentLevel = fullNode[1]['level']
    for i in range(0, data.shape[0]):
        nodeVisited = list.copy(fullNode[1]['visited'])
        # Verifier que l'on a pas deja visite ce noeud
        if not (i in nodeVisited):
            # Faire une copie de la matrice d'evalution
            temp = np.copy(data)
            # Mettre la ligne du noeud parent a ∞
            temp[node, :] = np.inf
            # Mettre la colonne du noeud fils ∞
            temp[:, i] = np.inf
            # Mettre le poid du noeud parent au fils a ∞ pour ne pas revenir au parent
            temp[i, start] = np.inf
            # Faire la reduction de la matrice en sauvgaredant la matrice reduite et la sommes des poids reduits
            childReduced, childCost = reduce_matrix(temp)
            # Calculer l'evaluation du noeud fils
            childCost += data[node, i] + cost
            # Ajouter le noeud fils a la liste des noeuds visites
            nodeVisited.append(i)
            # Ajouter le noeud fils a la liste des noeuds actifs
            active.append(
                (nodeSuiv, {'node': i, 'cost': childCost, 'level': parentLevel + 1, 'visited': nodeVisited, 'matrix': np.copy(childReduced)}))
            nodeSuiv += 1
    # Recuperer les couts des neouds actifs pour calculer le cout minimum
    seq = [x[1]['cost'] for x in active]
    # Recuperer le minimum des couts
    minim = min(seq)
    index = -1
    # Recuperer l'index du noeud avec la borne inf la plus petite
    index = seq.index(minim)
    # Supprimer le neoud choisi de la liste des noeuds actifs et retourner ce noeud avec son cout et ca matrice reduite
    nextNode = active.pop(index)
    # Si on trouve une solution condidate on elage tous les neouds ayant une borne inf > a cette solution
    if len(nextNode[1]['visited']) == size:
        for x in range(len(active) - 1, -1, -1):
            if active[x][1]['cost'] > nextNode[1]['cost']:
                active.pop(x)
    return nextNode, nodeSuiv


def branchNbound(start, data):
    # Initialiser les variables a utiliser
    # Racine
    root = start
    # Liste des neouds visites
    visited = []
    # Liste des noeuds actifs
    active = []
    # Creer des identifiants pour les noeuds actifs
    nodeSuiv = 0
    # Calculer la matrice reduite et l'evaluation du noeud racine
    reduced, cost = reduce_matrix(np.copy(data))
    # Ajouter la racine a la liste des noeuds visites
    visited.append(root)
    # Creer un noeud actif
    nextNode = (nodeSuiv, {'node': start, 'cost': cost, 'level': 0, 'visited': [start],
                           'matrix': np.copy(reduced)})
    nodeSuiv += 1
    # Recuperer le prochain noeud a exploiter
    nextNode, nodeSuiv = getNextNode(
        nextNode, visited, active, start, nodeSuiv, data.shape[0])
    # Continuer a recuperer le prochain noeud actif tant que l'on a pas visite tous les noeuds
    while (len(active) > 0):
        # Ajouter le noeud choisi a la liste des noeuds visites
        visited.append(nextNode[1]['node'])
        # Recuperer le prochain noeud a exploiter
        nextNode, nodeSuiv = getNextNode(
            nextNode, visited, active, start, nodeSuiv, data.shape[0])
    visited = np.copy(nextNode[1]['visited'])
    # calculer le cout de la solution trouvee
    summ = 0
    for i in range(len(visited) - 1):
        summ += (data[visited[i], visited[i + 1]])
    summ += data[visited[len(visited) - 1], start]
    # print(active)
    return visited, summ


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    # print(pd.DataFrame(data=instance.data))
    start_time = time.time()
    tour, cost = branchNbound(0, np.array(instance.data))
    end_time = time.time() - start_time
    print(tour)
    print(cost)
    print(end_time)
    return tour, cost, end_time


if __name__ == "__main__":
    run()
