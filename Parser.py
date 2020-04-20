import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
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
        while(not (self.curLine in self.dataKeywords)):
            specsSection.append(self.curLine)
            self.curLine = self.rawData.readline().rstrip()
        strippedSpecSection = [lin.split(': ') for lin in specsSection]
        for i in range(len(strippedSpecSection)):
            setattr(self, strippedSpecSection[i][0], strippedSpecSection[i][1])

    def readData(self):
        if(self.curLine == "EDGE_WEIGHT_SECTION"):
            if(self['EDGE_WEIGHT_FORMAT'] == "FULL_MATRIX"):
                temp = []
                for i in range(int(self['DIMENSION'])):
                    lin = [int(i) for i in self.rawData.readline().split()]
                    lin = [i if i > 0 else float('inf') for i in lin]
                    temp.append(lin)
                setattr(self, 'data', temp)
        self.curLine = self.rawData.readline().rstrip()
        if(self.curLine == "DISPLAY_DATA_SECTION"):
            if(self['DISPLAY_DATA_TYPE'] == "TWOD_DISPLAY"):
                temp = []
                for i in range(int(self['DIMENSION'])):
                    lin = [float(i) for i in self.rawData.readline().split()]
                    lin.pop(0)
                    temp.append(lin)
                setattr(self, 'diplay_data', temp)
