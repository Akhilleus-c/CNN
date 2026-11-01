import numpy as np
import generalUseFunctions as guf
from convention import *
import json
import pandas as pd
class Nn:
    def __init__(self, path, layer_count):
        self.inpts = self._take_inputs(path)
        self.previousLayer = self.inpts
        self.layers = layer_count
        
    def _take_inputs(self, pathToJpg):
        img = color_changer(r"C:\Users\zeker\Projeler\resm1.jpg")
        fltr = np.random.random(9).reshape(3, 3)
        return filterize(img, fltr)
    def loadparameters(self):
        try:
            self.df = pd.read_json("parameters.json")
        except FileNotFoundError:
            with open("parameters.json", "w") as jf:
                pars = [guf.initialize_randomly(len(self.inpts), 16), guf.initialize_randomly(16, 16), guf.initialize_randomly(16,4)]
                json.dump(pars, jf, indent=2)
            self.df = pd.read_json("parameters.json")
    def startEstimating(self):
        try:
            lyrs = len(self.df)
            weights = self.df["weights"]
            biases = self.df["biases"]
        except AttributeError:
            raise Exception("Firstly you have to load the parameters. :)")
        for i in range(lyrs):
            res = guf.forwardpass(self.previousLayer, np.array(weights[i][:-2]).reshape(weights[i][-1], weights[i][-2]), biases[i])
            self.previousLayer = res
        self.result = self.previousLayer
        self.previousLayer = self.inpts
    def backPropagation(self):
        pass
# Daha bitmedi :(

nn = Nn("a", 12)
nn.loadparameters()

print(nn.startEstimating())
