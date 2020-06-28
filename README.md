# OpenTSP

**OpenTSP** is a Python Library to solve the Symmetric Traveling Salesman Problem(TSP), it provides different methods written in Python to solve TSP varying from exact ones to hybrid meta-heuristics.

## 1.1. Requirements
This project is based on Python.

## 1.2. Build

1. First thing you need to do is clone this repository
```
git clone https://github.com/skourta/OpenTSP
```
2. Positing yourself in the root of the project:
```
cd   OpenTSP
```

## 1.3. Usage
You can each program using the terminal py providing the parameters of the algorithm through the command line arguments.
Example:
```
python AC.py --instance=datasets/bays29.tsp --colony_size=100
```
## Credits
| Algorithm        | Provided By           |
| ---------------- | --------------------- |
| 2OPT             | BENABED Youcef        |
| AC               | KOURTA Smail          |
| ACO[2OPT]        | TAHMI Omar            |
| ACO+2OPT         | TAHMI Omar            |
| AG               | IFERROUDJENE Mouloud  |
| AG[2OPT]         | IFERROUDJENE Mouloud  |
| AG+2OPT          | IFERROUDJENE Mouloud  |
| Branch and Bound | KOURTA Smail          |
| Brute Force      | IFERROUDJENE Mouloud  |
| Greedy Algorithm | BENDJABALLAH Oussama  |
| Nearest Neighbor | TAHMI Omar            |
| Tabu Search      | BENBELGACEM Rahma Aya |
| Or-Tools Usage   | KOURTA Smail          |
| TSPlib Parser    | KOURTA Smail          |

All programs were refactored by skourta to be used as command line programs.
