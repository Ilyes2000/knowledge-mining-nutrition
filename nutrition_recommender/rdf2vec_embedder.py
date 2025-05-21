import rdflib
import random
from gensim.models import Word2Vec

class RDF2VecEmbedder:
    """
    Génération d'embeddings RDF2Vec à partir d'un KG Turtle.
    """
    def __init__(self, ttl_path):
        self.graph = rdflib.Graph()
        self.graph.parse(ttl_path, format='ttl')

    def _walk(self, start, length):
        path = [start]
        for _ in range(length-1):
            neighbors = (
                list(self.graph.objects(path[-1], None))
                + list(self.graph.subjects(None, path[-1]))
            )
            if not neighbors: break
            path.append(random.choice(neighbors))
        return [str(n) for n in path]

    def generate_walks(self, num_walks=100, walk_length=8):
        nodes = list(set(self.graph.subjects()))
        walks = []
        for n in nodes:
            for _ in range(num_walks):
                walks.append(self._walk(n, walk_length))
        return walks

    def fit(self, num_walks=100, walk_length=8, embed_size=64, window=5, epochs=10):
        walks = self.generate_walks(num_walks, walk_length)
        self.w2v = Word2Vec(walks, vector_size=embed_size,
                            window=window, min_count=1, sg=1, epochs=epochs)

    def get_embedding(self, uri):
        return self.w2v.wv[str(uri)]
