"""Node2Vec: Self Implementation (Random Walk)"""

import argparse
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

def parser():
    parser = argparse.ArgumentParser(description="Arguments for Node2Vec (Random Walk)")
    parser.add_argument('--input_graph', nargs='?', default='graph/karate.edgelist', 
                        help='Specifying Input Graph')
    parser.add_argument('--output_emb', nargs='?', default='karate.emb', 
                        help='Output Location')
    parser.add_argument('--walk_length', type=int, default=40,
                        help="Walk Length for Random Walk. Default value is 40")
    parser.add_argument('--p', type=float, default=1,
                        help='Return Parameter (p). High p => High chances of return. Default value is 1')
    parser.add_argument('--q', type=float, default=1,
                        help='In-Out Parameter (q). High q => High chances of outward walk. Default value is 1')
    parser.add_argument('--paths', type=int, default=100,
                        help="Number of times a fresh node is sampled.")
    parser.add_argument('--window_size', type=int, default=2,
                        help='Size of Window for Skip-gram. Default value is 2')
    parser.add_argument('--dim', type=int, default=100, 
                        help="Specifies dimension for generated embedding")
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help="Learning rate of Word2Vec")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Epochs for Word2Vec")
    
    parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Weighted or Unweighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Directed or undirected. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()

#########################################################################################################
class N2V():
    def __init__(self, G, args):
        self.G = G
        self.output = args.output_emb
        self.len = args.walk_length
        self.p = args.p
        self.q = args.q
        self.paths = args.paths
        self.size = args.window_size
        self.dim = args.dim
        self.epochs = args.epochs
        self.weighted = args.weighted
        self.directed = args.directed

    def get_first_probs(self):
        node_considered = {}
        for node in self.G.nodes(): 
            temp = sorted(self.G.neighbors(node)) 
            probs = [self.G[node][x]['weight'] for x in temp]
            normalised = [float(y)/sum(probs) for y in probs]
            node_considered[node] = self.get_node(normalised) # obtained index of highest probability neighbour
        
        edges = {}
        if self.directed:
            for edge in self.G.edges():
                edges[edge] = self.get_edge(edge) # obtained index based on Random Walk parameters
        else:
            for edge in self.G.edges():
                edges[edge] = self.get_edge(edge)
                edges[(edge[1], edge[0])] = self.get_edge(edge, True)

        self.nodes = node_considered
        self.edges = edges

    def get_node(self, normalised):
        max = 0
        index = -1
        for ind, val in enumerate (normalised):
            a = val*np.random.rand()
            if a>max:
                index = ind
        return index

    def get_edge(self, edge, invert = False):
        if invert:
            start = edge[1]
            end = edge[0]
        else:
            start = edge[0]
            end = edge[1]
        G = self.G
        p = self.p
        q = self.q
        probs = []
        for nbr in sorted(G.neighbors(end)):
            if nbr == start:
                probs.append(G[end][nbr]['weight']*p)
            elif G.has_edge(nbr, start):
                probs.append(G[end][nbr]['weight'])
            else:
                probs.append(G[end][nbr]['weight']*q)
        normalised = [float(prob)/sum(probs) for prob in probs]
        return self.get_node(normalised)

    def make_walk(self):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Node2Vec Running..')
        for iter in range(self.paths):
            print("Iteration ", str(iter+1))
            for node in nodes:
                walks.append(self.walk(node))
        return walks
    
    def walk(self, start):
        G = self.G
        nodes = self.nodes
        edges = self.edges
        walk = [start]
        while len(walk) < self.len:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[nodes[cur]])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[edges(prev, cur)]
                    walk.append(next)
            else:
                print("Lone Node")
                break
            return walk

    def embedding(self, walks):
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=self.dim, window=self.size, sg=1, iter=self.epochs)
        model.wv.save_word2vec_format(self.output)

#########################################################################################################
def make_graph(args):
    if args.weighted:
        G = nx.read_edgelist(path = args.input_graph, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(path = args.input_graph, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()
    return G

def main(args):
    """print(args.weighted)
    print(args.directed)"""
    G = make_graph(args)
    model = N2V(G, args)
    model.get_first_probs()
    walks = model.make_walk()
    model.embedding(walks)

if __name__ == '__main__':
    args = parser()
    main(args)

#########################################################################################################