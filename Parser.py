import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import math
import pandas as pd


class TSPInstance:
    specsKeywords = ['NAME', 'TYPE', 'COMMENT', 'DIMENSION', 'CAPACITY', 'EDGE_WEIGHT_TYPE', 'DISPLAY_DATA_TYPE',
                     'EDGE_WEIGHT_FORMAT', 'EDGE_DATA_FORMAT', 'NODE_COORD_TYPE', 'DISPLAY_DATA_TYPE', 'EOF']
    dataKeywords = ['NODE_COORD_SECTION', 'DEPOT_SECTION', 'DEMAND_SECTION', 'EDGE_DATA_SECTION',
                    'FIXED_EDGES_SECTION', 'DISPLAY_DATA_SECTION', 'TOUR_SECTION', 'EDGE_WEIGHT_SECTION']
    curLine = ""

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, path):
        self.rawData = open(path)
        self.readSpecs()

    def readSpecs(self):
        specsSection = []
        self.curLine = self.rawData.readline().rstrip()
        while (not (self.curLine in self.dataKeywords)):
            specsSection.append(self.curLine)
            self.curLine = self.rawData.readline().rstrip()
        strippedSpecSection = [lin.split(': ') for lin in specsSection]
        for i in range(len(strippedSpecSection)):
            setattr(self, strippedSpecSection[i][0], strippedSpecSection[i][1])

    def readData(self):
        if (self.curLine == "EDGE_WEIGHT_SECTION"):
            if (self['EDGE_WEIGHT_FORMAT'] == "FULL_MATRIX"):
                temp = []
                for i in range(int(self['DIMENSION'])):
                    lin = [int(i) for i in self.rawData.readline().split()]
                    lin = [i if i > 0 else float('inf') for i in lin]
                    temp.append(lin)
                setattr(self, 'data', temp)
            self.curLine = self.rawData.readline().rstrip()
            if self.curLine == "DISPLAY_DATA_SECTION":
                if self['DISPLAY_DATA_TYPE'] == "TWOD_DISPLAY":
                    temp = []
                    for i in range(int(self['DIMENSION'])):
                        lin = [float(i) for i in self.rawData.readline().split()]
                        lin.pop(0)
                        temp.append(lin)
                    setattr(self, 'diplay_data', temp)
        if self.curLine == "NODE_COORD_SECTION":
            if self['EDGE_WEIGHT_TYPE'] == "GEO":
                coords = []
                for i in range(int(self['DIMENSION'])):
                    lin = [float(i) for i in self.rawData.readline().split()]
                    coord = (lin[1], lin[2])
                    coords.append(coord)
                setattr(self, 'diplay_data', coords)
                data = []
                for i in range(len(coords)):
                    row = []
                    for j in range(len(coords)):
                        if i == j:
                            row.append(float("inf"))
                        else:
                            row.append(self.geo_dist(coords[i], coords[j]))
                    data.append(row)
                setattr(self, 'data', data)

    def geo_dist(self, i, j):
        RRR = 6378.388
        lat1 = math.radians(i[0])
        lon1 = math.radians(i[1])
        lat2 = math.radians(j[0])
        lon2 = math.radians(j[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return RRR * c

