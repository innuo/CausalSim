import cdt
import networkx as nx




class CausalStructure:
    def __init__(self, variable_names, dag=None):
        self.variable_names = variable_names
        self.dag = nx.empty_graph(len(variable_names), create_using=nx.DiGraph())
        self.dag = nx.relabel_nodes(self.dag, dict(zip(range(len(variable_names)), variable_names)))

        if dag is not None:
            self.dag = nx.compose(dag, self.dag)

    # TODO: improve structure learning
    def learn_structure(self, dataset):
        gs = cdt.causality.graph.bnlearn.GS()
        dataset_dag = gs.create_graph_from_data(dataset.df)
        self.update_structure(dataset_dag, 'replace', 'self')

    def update_structure(self, dag, merge_type='replace', priority='self'):
        self.merge(dag, merge_type, priority)
        self.topo_sorted = list(nx.topological_sort(self.dag))
        self.parents = dict(zip(self.variable_names, [[]] * len(self.variable_names)))
        for v in self.topo_sorted:
            self.parents[v] = list(nx.DiGraph.predecessors(self.dag, v))

    def merge(self, dag, merge_type="union", priority="self"):
        """ merge_type = "union" is a simple compose of the two graphs with possible cycles
            merge_type = "replace" adds edges from dag to self.dag,
                          any conflicts in the edges are resolved by the priority arg
        """
        if merge_type == 'union':
            self.dag = nx.compose(dag, self.dag)
            if not nx.is_directed_acyclic_graph(self.dag):
                print('Error: After merge no longer a DAG')
        elif merge_type == 'replace':
            g1 = self.dag if priority == 'self' else dag
            g2 = dag if priority == 'self' else self.dag
            for e in g2.edges:
                if not g1.has_edge(e[1], e[0]):
                    g1.add_edge(e[0], e[1])
            self.dag = g1
        else:
            print('Error: merge_type' + merge_type + ' not supported')

    def plot(self):
        nx.draw_networkx(self.dag)
    pass

if __name__ == '__main__':
    #import pandas as pd
    #df =  pd.read_csv("~/work/tinkering/data/5d.csv")
    #gs = cdt.causality.graph.bnlearn.GS()
    #k = gs.create_graph_from_data(df)
    #nx.draw_networkx(k)

    from datahandler import dataset as ds
    r = ds.DataUtils("~/work/tinkering/data/5d.csv")
    cs = CausalStructure(r.variable_names)
    cs.learn_structure(r)
    cs.plot()