import numpy as np
import skfuzzy as fuzz        
import os

class FuzzyClustering:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def load_embeddings(self, path):
        embeddings_path = path
        if not os.path.exists(embeddings_path):
            raise ValueError("Embeddings file not found")
        embeddings = np.load(embeddings_path)
        return embeddings

    def run_clusterings(self, embeddings):
        print("Running Fuzzy C-Means Clustering…")
        data = embeddings.T
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data,
            c=self.n_clusters,
            m=2,
            error=0.005,
            maxiter=1000,
            init=None,
        )
        return cntr, u

    def save_clusters(self, cntr, membership):
        np.save("data/cluster_centers.npy", cntr)
        np.save("data/document_membership.npy", membership)
        print("Cluster centers and document membership saved successfully.")
    
if __name__ == "__main__":
  
  clustering = FuzzyClustering(n_clusters=15)
  
  embeddings = clustering.load_embeddings("data/embeddings.npy")
  
  print("Ebeddings loaded successfully." , embeddings.shape)
  
  centers , membership = clustering.run_clusterings(embeddings)
  
  print("Clustering centers" , centers.shape)
  print("Document membership shape" , membership.shape)
  
  clustering.save_clusters(centers , membership)
    
