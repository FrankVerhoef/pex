###
### Define a class for a dict with lists, to facilitate collection of statistics
###

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
            means[k] = sum(v) / len(v)
        return means
