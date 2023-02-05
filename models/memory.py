from collections import Counter
import networkx as nx


class TextMemory:
    """
    Simple memory class to hold a list of facts in the form of text sentences.
    The memory has a weight for each fact:
        - for a new fact, the weight is set to the median of all weights plus 1
        - if a fact is stored again, the weight is increased with the median (if current value is below median) or with 1 otherwise
        - if a fact is recalled, the weight is increased by 1
    When a new fact is added while the memory is full, the fact with the lowest count is removed before the new fact is added
    """

    def __init__(self, id, maxsize):
        """
        Initialize empty memory
        """
        self.id = id
        self.mem = Counter()
        self.maxsize = maxsize

    def add(self, fact):
        """
        Adds a fact to the memory, or renew
        """
        newfact = fact not in self.mem
        remaining = self.maxsize - len(self.mem) - newfact
        if remaining < 0:
            self.compress(n=-remaining)
        sorted_weight = sorted(self.mem.values())
        median = sorted_weight[(len(sorted_weight)-1) // 2] if len(sorted_weight) > 0 else 0
        if newfact:
            self.mem[fact] = median + 1
        else:
            self.mem[fact] += median if self.mem[fact] < median else 1 

    def recall(self, keys):
        """
        Returns a list of memories that match with any of the key words
        """
        memories = []
        for fact in self.mem.keys():
            if self.match(fact, keys):
                memories.append(fact)
                self.mem[fact] += 1

        return memories

    def match(self, fact, keys):
        """
        Returns True of the fact contains any of the key words
        """
        words = set(fact.split())
        keys = set(keys)
        return len(words & keys) > 0

    def compress(self, n=1):
        """
        Removes n facts from memory, based on the value of the counter.
        """
        if n >= len(self.mem):
            self.mem = Counter()
        else:
            facts, weights = list(self.mem.keys()), list(self.mem.values())
            weights_sortec_idx = sorted(range(len(weights)), key=weights.__getitem__)
            for i in range(n):
                del self.mem[facts[weights_sortec_idx[i]]]

    def __repr__(self):
        return "<TextMemory id: {}, maxsize: {}>".format(self.id, self.maxsize)

    def __str__(self):
        s = "TextMemory {}\n".format(self.id)
        s += "Weight: Fact\n"
        s += '\n'.join([
            "{:6}: {}".format(weight, fact)
            for fact, weight in self.mem.items()
        ])
        return(s)

class GraphMemory:
    """
    Simple memory class to hold a graph with facts in the form of text triples (subject, object, predicate).
    The graph is a directed graph with possibility of multiple edges (preicates) between any two nodes (subject and object).
    The graph has a weight for each node and edge:
        - for a new node, the weight is set to the median of all node weights plus 1
        - if a node occurs again in another, the weight is increased with the median (if current weight is below median) or with 1 otherwise
        - for a new edge, the weight is set to 1
        - if an edge with the same predicate occurs again in another fact, the weight is increased with 1
        - if a fact is recalled, the weights of the corresponding subject, object and predicate is increased by 1
    When new nodes need to be added while the memory is full, the nodes with the lowest weight x degree are removed (including connected edges)
    before the new nodes are added. The number of edges is unlimited.
    """

    def __init__(self, id, maxsize):
        """
        Initialize memory with a single node that serves as identifier of the 'owner' of the memory.
        """

        self.id = id
        self.mem = nx.MultiDiGraph()
        self.mem.add_node(id, weight=1)
        self.maxsize = maxsize

    def add(self, fact):
        """
        Adds a fact to the knowledge graph
        Fact must be a triple with three strings
        """

        subject, predicate, object = fact

        # First, make space in memory if it is full
        newsubject = subject not in self.mem.nodes
        newobject = object not in self.mem.nodes
        remaining = self.maxsize - len(self.mem.nodes) - newsubject - newobject
        if remaining < 0:
            self.compress(n=-remaining)

        # Add subject, object and relationship to graph if not already present, and set or update weights
        sorted_nodes = sorted(self.mem.nodes, key=lambda x: self.mem.nodes[x]['weight'])
        median = self.mem.nodes[sorted_nodes[(len(self.mem.nodes)-1) // 2]]['weight'] if len(self.mem.nodes) > 0 else 0
        if newsubject:
            self.mem.add_node(subject, weight=median + 1)
        else:
            self.mem.nodes[subject]['weight'] += median if self.mem.nodes[subject]['weight'] < median else 1
        if newobject:
            self.mem.add_node(object, weight=median + 1)
        else:
            self.mem.nodes[object]['weight'] += median if self.mem.nodes[object]['weight'] < median else 1
        if (subject, object) not in self.mem.edges:
            self.mem.add_edge(subject, object, type=predicate, weight=1)
        else:
            for e in self.mem[subject][object].values():
                if e['type'] == predicate:
                    e['weight'] += 1


    def recall(self, keys):
        """
        Returns all paths starting from the central node that are connected with nodes that match with any of the key words.
        The result is a list of triples (subject, object, predicate), where subject and object are text strings and 
        predicate is a dict with predicate['type'] is a string with relatioship type and predicate['weight'] is a weight value.
        """

        memories = []
        paths = {self.id: []}
        targets = set(self.mem.nodes)
        targets.remove(self.id)
        boundary = list(nx.edge_boundary(self.mem, [self.id], targets, data=True))

        # Repeat until there are no new edges in the boundary
        while len(boundary) > 0:

            boundary_tails = set()

            # Loop over all the facts in the boundary
            for fact in boundary:
                head, tail, attributes = fact
                paths[tail] = paths[head] + [fact]
        
                # If the fact matches with the key words, remember the path and update weights of nodes and edges
                if self.match(fact, keys):
                    memories.append(paths[tail])
                    for (h, t, a) in paths[tail]:
                        self.mem.nodes[t]['weight'] += 1
                        for i in range(len(self.mem[h][t])):
                            r = self.mem[h][t][i]
                            if r['type'] == a['type']:
                                r['weight'] += 1

                # Remove tail node from targetset, unless it contains any of the key words
                if len(set(tail.split()) & keys) == 0:
                    boundary_tails.add(tail)
                    targets.remove(tail)

            # Determine new boundary edges
            boundary = list(nx.edge_boundary(self.mem, boundary_tails, targets, data=True))

        return memories

    def match(self, fact, keys):
        head, tail, attributes = fact
        words = set(tail.split()) | set(attributes['type'].split())
        keys = set(keys)
        return len(words & keys) > 0

    def compress(self, n=1):
        """
        Removes 'n' nodes from the memory graph, including connected edges
        The ranking of nodes to be removed is determined by combination of the degree of the node
        and the number of times the object has been part of a fact that was added or recalled.
        """
        if n >= len(self.mem.nodes):
            self.mem = nx.MultiDiGraph()
            self.mem.add_node(id)
        else:
            sorted_nodes = sorted(nx.degree(self.mem), key=lambda x: self.mem.nodes[x[0]]['weight'] * x[1])
            for i in range(n):
                self.mem.remove_node(sorted_nodes[i][0])

    def __repr__(self):
        return "<GraphMemory id: {}, maxsize: {}>".format(self.id, self.maxsize)

    def __str__(self):
        s = "GraphMemory {}\n".format(self.id)
        s += "Weight: Object\n"
        s += '\n'.join([
            "{:4}: {}".format(self.mem.nodes[object]['weight'], object)
            for object in self.mem.nodes
        ]) + '\n'
        s += "Used: Fact\n"
        s += '\n'.join([
            "{:4}: {} - {} - {}".format(
                attributes['weight'], object, attributes['type'], subject, self.mem.nodes[subject]['weight'])
            for object, subject, attributes in self.mem.edges(data=True)
        ])
        return(s)

MEMORY_MODEL = {
    'sentencelist': TextMemory,
    'textgraph': GraphMemory
}
MEMORY_TYPES = list(MEMORY_MODEL.keys())

if __name__ == '__main__':

    # testtype = 'sentencelist'
    testtype = 'textgraph'

    memory = MEMORY_MODEL[testtype](id='<John>', maxsize=10)
    facts = {
        'sentencelist': [
            "I'm a perfectionist.", 
            "If things aren't done right I'll redo them again and again.", 
            "I take forever to get tasks done so I start early and clock out late.", 
            "I work too much.", 
            "I think I need a vacation."
            "I work in the military.", 
            "I've been all over the world.", 
            "I like things that explode.", 
            "I also like kittens.", 
            "I like being a soldier",
            "If you're a soldier than you're part of the military",
            "Brownies are my favorite dessert."
        ],
        'textgraph': [
            ['<John>', 'is a', 'perfectionist'],
            ['in the military', 'is a', 'job'],
            ['<John>', 'have workstyle', 'redo things again and again'],
            ['<John>', 'have workstyle', 'start early and clock out late'],
            ['<John>', 'work', 'too much'],
            ['<John>', 'need', 'a vacation'],
            ['<John>', 'work', 'in the military'],
            ['<John>', 'like', 'things to explode'],
            ['<John>', 'like', 'kittens'],
            ['<John>', 'like', 'being a soldier'],
            ['being a soldier', 'part of', 'in the military'],
            ['<John>', 'have favorite dessert', 'brownies']
        ]
    }[testtype]

    print('--- ADD ---')
    keys = {"job", "like", "military"}
    for f in facts:
        memory.add(f)
        print(memory)  

    print('--- RECALL ---')
    recalled = memory.recall(keys)
    for r in recalled:
        print(r)

    print('--- COMPRESS ---')
    memory.compress(6)
    print(memory)

    print('--- RECALL AGAIN ---')
    recalled = memory.recall(keys)
    for r in recalled:
        print(r)

