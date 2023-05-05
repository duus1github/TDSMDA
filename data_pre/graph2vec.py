"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--input-path",
                        nargs="?",
                        default="../dataSet/input_data/mir_dag/",
                        # default="E:/WorkSpace/Research/dataSets/output/",
	                help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="../dataSet/out_data/mir_dag/",
	                help="Embeddings path.")

    parser.add_argument("--dimensions", # shape defind 128
                        type=int,
                        default=128,
	                help="Number of dimensions. Default is 128.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
	                help="Number of workers. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
	                help="Number of epochs. Default is 10.")

    parser.add_argument("--min-count", # Minimum count of structural features
                        type=int,
                        default=5,
	                help="Minimal structural feature count. Default is 5.")

    parser.add_argument("--wl-iterations", # The number of iterations of Weisfeiler-Lehman
                        type=int,
                        default=2,
	                help="Number of Weisfeiler-Lehman iterations. Default is 2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
	                help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--down-sampling",
                        type=float,
                        default=0.0001,
	                help="Down sampling rate of features. Default is 0.0001.")

    return parser.parse_args()
args = parameter_parser()
class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
        # features = {k: v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        # features = {int(k): v for k, v in features}
        features = {k: v for k, v in features}
        # for k,v in features:
        #     print("k:",k)
        #     print("v:",v)
    return graph, features, name  # Get the image, feature, and corresponding name


def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc


def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        # out.append([identifier] + list(model.docvecs["g_" + identifier]))
        out.append([identifier] + list(model.dv["g_" + identifier]))
    column_names = ["type"] + ["x_" + str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)


def graph2vec_main():
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    # path = os.path.join(args.input_path,'*.json')
    # print(path)
    for step in range(1,4):
        input_path = args.input_path+str(step)+'/'
        output_path = args.output_path+str(step)+'.csv'
        graphs = glob.glob(os.path.join(input_path,"*.json"))
        # graphs = sorted(graphs,key = lambda info:int(info.split('\\').split('.')[0]))
        # print(args.input_path)
        # print(os.path.join(args.input_path,"*.json"))
        print("\nFeature extraction started.\n")
        document_collections = Parallel(n_jobs=args.workers)(
            delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))

        print("\nOptimization started.\n")

        model = Doc2Vec(document_collections,
                        vector_size=args.dimensions,
                        window=0,
                        min_count=args.min_count,
                        dm=0,
                        sample=args.down_sampling,
                        workers=args.workers,
                        epochs=args.epochs,
                        alpha=args.learning_rate)

        save_embedding(output_path, model, graphs, args.dimensions)


if __name__ == "__main__":

    graph2vec_main()
