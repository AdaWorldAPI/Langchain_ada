import faiss
import numpy as np

d = 64                            # dimension
nb = 1000                         # database size
nq = 5                            # number of queries

np.random.seed(123)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

index = faiss.IndexFlatL2(d)      # L2 distance index
index.add(xb)                     # add vectors to index
D, I = index.search(xq, 5)        # search

print("Nearest indices:\n", I)
