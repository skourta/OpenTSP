import numpy as np
import pandas as pd
import Parser
import time
import argparse


def PPV(graphe, v_depart=None):
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
    return np.array(chemin)


# Permuter les aretes entre 2 noeuds.
def swap_2opt(tour, i, j):
    tour[i:j + 1] = tour[j:i - 1:-1]
    return tour


# Calculer Le cout d'une tournée dans un graphe donné.
def coast_of_tour(graphe, tour):
    return graphe[np.roll(tour, 1), tour].sum()


def solve_tsp_2opt(graphe, tours):
    resultats = []
    dimension = len(graphe)
    # Si le param
    for tr in tours:
        tour = tr[0].copy()
        final_coast = initial_coast = tr[1]
        improved = True
        while improved:
            improved = False
            for i in range(1, dimension - 2):
                for j in range(i + 1, dimension):
                    current_coast = coast_of_tour(graphe, tour)
                    new_coast = coast_of_tour(graphe, swap_2opt(tour.copy(), i, j))
                    if current_coast > new_coast:
                        improved = True
                        swap_2opt(tour, i, j)
                        final_coast = new_coast
        resultats.append([tour, final_coast, 1 / final_coast])

    return np.array(resultats)


def generatePopulation(graphe, populationSize, algorithme="Random"):
    population = []
    grapheSize = len(graphe)

    if (algorithme == "Random"):
        for i in range(populationSize):
            # Generation d'un individu
            sol = np.random.permutation(grapheSize)
            # Ajouter l'individu a la population
            population.append(sol)
        return np.array(population)

    if (algorithme == "PPV"):
        for i in range(populationSize):
            # v_depart = np.random.randint(0,grapheSize)
            v_depart = i % grapheSize
            sol = PPV(graphe, v_depart)
            population.append(sol)
        return np.array(population)


def evaluateIndividu(indiv, graphe):
    cout = 0
    for i in range(len(graphe)):
        cout += graphe[indiv[i - 1]][indiv[i]]
    return cout


def evaluatePopulation(graphe, population):
    evaluated_population = []
    for pop in population:
        cout = evaluateIndividu(pop, graphe)
        fitness = 1 / cout
        evaluated_population.append([pop, cout, fitness])
    return np.array(evaluated_population)


def selectParents(evaluated_population, parents_Size, eliteSize=2, Algorithme="RouletteWheel"):
    # Liste ordonnée d'indices
    ind = np.argsort(evaluated_population[:, 1])
    sorted_population = evaluated_population[ind]

    if (Algorithme == "Elitiste"):
        # Prendre uniquement les elites (Meilleurs individus)
        parents = sorted_population[:parents_Size, :]
        return parents
    else:
        # Choisir le elites a inclure la liste des parents
        parents = sorted_population[:eliteSize, :]

    if (Algorithme == "Tournoi"):
        # Un certain nombre d'individus sont sélectionnés au hasard dans la population
        # Et l'elite du groupe est choisi comme premier parent.
        # Cette opération est répétée pour choisir le deuxième parent.
        populationSize = len(evaluated_population)
        for i in range(parents_Size - eliteSize):
            # Selectionner un nombre aleatoire d'individus pour un tournoi
            selectedSize = np.random.randint(2, populationSize)
            selectedIndice = np.unique(np.random.randint(0, populationSize, selectedSize))
            selected_population = evaluated_population[selectedIndice]
            # Prendre le meileure individus de ce trounois
            elite_indice = np.argmin(selected_population[:, 1])
            selected_elite = selected_population[elite_indice][:]
            # Ajouter l'elite a la liste des parents
            parents = np.insert(parents, 0, selected_elite, axis=0)
        return parents

    if (Algorithme == "RouletteWheel"):
        # Nous avons mis en place la roue de la roulette en calculant un poids de forme relatif pour chaque individu.
        df = pd.DataFrame(np.array(sorted_population), columns=["Index", "cout", "Fitness"])
        # Calcules des somme cumulatives de fitness
        df['cum_sum'] = df.Fitness.cumsum()
        # Cacule
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        # Ici nous comparons un nombre tiré au hasard à ces poids pour sélectionner le parents
        for i in range(0, parents_Size - eliteSize):
            # Faire tourner la roulette
            pick = np.random.uniform(0, 100)
            for i in range(0, len(sorted_population)):
                if pick <= df.iat[i, 4]:
                    parents = np.insert(parents, 0, sorted_population[i][:], axis=0)
                    break
        return parents


def inter(lst1, lst2, offset=0):
    # retourne les elements dupliqués de la liste lst2 dans la liste lst1 avec leurs indices
    lst = [(lst1.index(value) + offset, value) for value in lst1 if value in lst2]
    return lst


def croisement(parent1, parent2, nbPointsCroissement):
    fils1 = parent1.tolist()
    fils2 = parent2.tolist()
    # Choisi aleatoirement deux points de découpe
    rng = np.random.default_rng()
    points = np.sort(rng.choice(len(parent1), size=nbPointsCroissement, replace=False), axis=0).tolist()
    co_points = [0]
    for i in range(len(points)):
        if i % 2 == 0:
            co_points.append(points[i])
        else:
            co_points.append(points[i] + 1)
    co_points.append(len(parent1))
    # Les sous chaines s_a1,s_a2=[],[]
    s_a1, s_a2 = [], []
    for i in range(len(points) // 2):
        s_a1.append(fils1[points[2 * i]:points[2 * i + 1] + 1])
        s_a2.append(fils2[points[2 * i]:points[2 * i + 1] + 1])
    # intervertir entre les deux parcours
    for i in range(len(points) // 2):
        fils1[points[2 * i]: points[2 * i + 1] + 1] = s_a2[i]
        fils2[points[2 * i]: points[2 * i + 1] + 1] = s_a1[i]
    # recenser les villes qui n'apparaissent pas dans chacun des deux villes
    index1, index2 = [], []
    for i in range(len(points) // 2 + 1):
        for v in s_a2:
            index1 += inter(fils1[co_points[2 * i]:co_points[2 * i + 1]], v, co_points[2 * i])
        for u in s_a1:
            index2 += inter(fils2[co_points[2 * i]:co_points[2 * i + 1]], u, co_points[2 * i])
    # Remplir les trous dans chaque parcours
    for i in range(len(index1)):
        fils1[index1[i][0]] = index2[i][1]
        fils2[index2[i][0]] = index1[i][1]
    return fils1, fils2


def croisementPopulation(parents, nbPointsCroissement=2):
    fils = []
    random_list = np.random.permutation(len(parents))
    for i in range(0, len(parents) - 1, 2):
        fils1, filsP2 = croisement(parents[random_list[i], 0], parents[random_list[i + 1], 0], nbPointsCroissement)
        fils.extend((fils1, filsP2))
    return np.array(fils)


def swap_mutation(individu, probaMutation):
    for i in range(len(individu)):
        chance = np.random.uniform()
        if (chance < probaMutation):
            j = int(chance * len(individu))
            individu[i], individu[j] = individu[j], individu[i].copy()
    return individu


def mutationPopulation(population, probaMutation=None):
    mutated_population = []
    # C'est une probabilité tres faible [0.01:0.001] ou P=1/len(indiv)
    if probaMutation is None:
        probaMutation = 1 / population.shape[1]
    for individu in population:
        mutated_population.append(swap_mutation(individu, probaMutation))
    return np.array(mutated_population)


def remplacement(init_population, new_population, methode="Generationnel"):
    if methode == "Elitiste":
        size = len(new_population)
        pop = np.append(init_population, new_population, axis=0)
        ind = np.argsort(pop[:, 1])
        sorted_pop = pop[ind]
        return sorted_pop[:size, :]

    if methode == "RouletteWheel":
        size = len(new_population)
        pop = np.append(init_population, new_population, axis=0)
        return selectParents(pop, size, Algorithme="RouletteWheel")

    if methode == "Tournoi":
        size = len(new_population)
        pop = np.append(init_population, new_population, axis=0)
        return selectParents(pop, size, Algorithme="Tournoi")

    if methode == "Generationnel":
        return new_population


def nextGeneration(graphe, population, parents_size, eliteSize, SelectionAlgo, nbPointCroisement, probaMutation,
                   remplacementAlgo):
    init_population = population.copy()

    # Selection
    # SelectionAlgo : [RouletteWheel, Tournoi, Elitiste]
    parents = selectParents(population, parents_size, eliteSize, SelectionAlgo)

    # Croisement
    fils = croisementPopulation(parents, nbPointCroisement)

    # Mutation
    mutated_fils = evaluatePopulation(graphe, mutationPopulation(fils, probaMutation))
    new_population = np.array(np.append(parents, mutated_fils, axis=0))

    # Remplacement
    # Modes: Elitiste , Generationnel
    new_population = remplacement(init_population, new_population, remplacementAlgo)

    return new_population


def HRH_Ag_2opt(graphe, population_size, nbgenerations, parents_size, eliteSize=2, genAlgo="Random",
                SelectionAlgo="RouletteWheel", nbPointCroisement=4, probaMutation=None,
                remplacementAlgo="Generationnel"):
    # Generation de population initial
    # genAlgo: Random,PPV
    start_time = time.time()
    population = generatePopulation(graphe, population_size, genAlgo)
    # Evaluation de population
    population = evaluatePopulation(graphe, population)
    for i in range(0, nbgenerations):
        population = nextGeneration(graphe, population, parents_size, eliteSize, SelectionAlgo, nbPointCroisement,
                                    probaMutation, remplacementAlgo)
    # 2OPT
    # print("AG time:", time.time() - start_time, "sec")
    start_time = time.time()
    population = solve_tsp_2opt(graphe, population)
    # print("AG + 2opt time:", time.time() - start_time, "sec")
    totalTime = time.time() - start_time
    sol = population[np.argsort(population[:, 1]), :2][0].tolist()
    return sol[0].tolist(), sol[1], totalTime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--population_size",
                        help="Size of Population", default=28)
    parser.add_argument("--nbgenerations",
                        help="Number of Generations", default=2)
    parser.add_argument("--parents_size",
                        help="Size of parent", default=14)
    parser.add_argument("--eliteSize",
                        help="Size of Elite", default=2)
    parser.add_argument("--genAlgo",
                        help="Generation Algorithm", default="PPV")
    parser.add_argument("--SelectionAlgo",
                        help="Selection Algorithm", default="Tournoi")
    parser.add_argument("--nbPointCroisement",
                        help="Selection Algorithm", default=2)
    parser.add_argument("--probaMutation",
                        help="Mutation Probability", default=None)
    parser.add_argument("--remplacementAlgo",
                        help="Replacement Algorithm", default="Tournoi")
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    probaMutation = None
    if args.probaMutation != "None" and args.probaMutation is not None:
        probaMutation = float(args.probaMutation)
    # start_time = time.time()
    tour, cost, total_time = HRH_Ag_2opt(np.array(instance.data), population_size=int(args.population_size),
                             nbgenerations=int(args.nbgenerations), parents_size=int(args.parents_size),
                             eliteSize=int(args.eliteSize), genAlgo=args.genAlgo, SelectionAlgo=args.SelectionAlgo,
                             nbPointCroisement=int(args.nbPointCroisement),
                             probaMutation=probaMutation, remplacementAlgo=args.remplacementAlgo)
    # end_time = time.time()
    print(tour)
    print(cost)
    print(total_time)
