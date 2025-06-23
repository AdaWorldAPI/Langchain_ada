import numpy as np
import faiss
import time

# --------- Config ---------
d = 64             # Dimensionality
nlist = 64        # # of coarse centroids
m = 8               # # of subquantizers
nbits = 8           # bits per codebook



train_size = 256000
query_size = 64

faiss.omp_set_num_threads(16)

# --------- Generate Data ---------
train_vectors = np.random.randn(train_size, d).astype("float32")
query_vectors = np.random.randn(query_size, d).astype("float32")

# --------- Build Ground Truth Flat Index ---------
true_index = faiss.IndexFlatL2(d)
true_index.add(train_vectors)

# --------- Build IVFPQ Index ---------
quantizer = faiss.IndexFlatL2(d)
pq_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# --------- Train ---------
start = time.time()
pq_index.train(train_vectors)
print(f"⏱️ Trained in {time.time() - start:.3f} seconds")

# --------- Add ---------
start = time.time()
pq_index.add(train_vectors)
print(f"➕ Added vectors in {time.time() - start:.3f} seconds")

# --------- Search ---------
pq_index.nprobe = 16
start = time.time()
D_pq, I_pq = pq_index.search(query_vectors, k=10)
search_time = time.time() - start
print(f"🔍 Searched {query_size} queries in {search_time:.3f} seconds")
print(f"⚡ QPS ≈ {query_size / search_time:.1f} queries/sec")

# --------- Ground Truth Search ---------
D_true, I_true = true_index.search(query_vectors, k=10)

# --------- Recall@10 ---------
recall_at_10 = np.mean([
    len(set(I_true[i]) & set(I_pq[i])) > 0
    for i in range(len(query_vectors))
])
print(f"🎯 Recall@10: {recall_at_10:.2%}")
for nprobe in [8, 16, 32, 48, 64]:
    pq_index.nprobe = nprobe
    D, I = pq_index.search(query_vectors, k=10)
    recall = np.mean([
        len(set(I_true[i]) & set(I[i])) > 0
        for i in range(len(query_vectors))
    ])
    print(f"nprobe={nprobe}, Recall@10={recall:.2%}")
