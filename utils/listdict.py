###
### Define a class for a dict with lists, to facilitate collection of statistics
###

import math

class ListDict:

    def __init__(self):
        self.listdict = {}

    def append(self, d):
        for k, v in d.items():
            if k in self.listdict.keys():
                self.listdict[k].append(v)
            else:
                self.listdict[k] = [v]

    def mean(self):
        means = {}
        for k, v in self.listdict.items():
            if k.find('perplexity') < 0: # Cheat: perplexity values are treated differently
                means[k] = sum(v) / len(v)
            else:
                means[k] = math.exp(sum([math.log(elem) for elem in v]) / len(v))
        return means
