import time
import numpy as np

class EvpBits:
    """
    Optimized version of EvpBits using Python's arbitrary-precision integers
    for memory efficiency and fast scalar operations, while supporting fast
    conversion back to NumPy for batched operations.
    """
    __slots__ = ['ones', 'negative_ones', 'dim']

    def __init__(self, ones, negative_ones, dim):
        # ones and negative_ones are now Python integers representing bit vectors
        self.ones = ones
        self.negative_ones = negative_ones
        self.dim = dim

    @classmethod
    def from_embedding(cls, embedding, non_zeros):
        dim = len(embedding)
        if non_zeros >= dim:
            raise ValueError(f"non_zeros ({non_zeros}) must be strictly less than embedding dimension dim ({dim})")
            
        abs_emb = np.abs(embedding)
        
        # using argsort to mimic Rust's arg_sort exactly
        top_indices = np.argsort(abs_emb)[-non_zeros:]
        
        ones = np.zeros(dim, dtype=bool)
        negative_ones = np.zeros(dim, dtype=bool)
        
        for idx in top_indices:
            val = embedding[idx]
            if val > 0:
                ones[idx] = True
            elif val < 0:
                negative_ones[idx] = True
                
        # Pack to bytes, then integer
        ones_int = int.from_bytes(np.packbits(ones).tobytes(), 'little')
        negative_ones_int = int.from_bytes(np.packbits(negative_ones).tobytes(), 'little')
        
        return cls(ones_int, negative_ones_int, dim)

    @classmethod
    def from_embeddings(cls, dataset, non_zeros, chunk_size=8192):
        num_rows, dim = dataset.shape
        if non_zeros >= dim:
            raise ValueError(f"non_zeros ({non_zeros}) must be strictly less than embedding dimension dim ({dim})")
            
        data_evp = []
        
        for i in range(0, num_rows, chunk_size):
            chunk = dataset[i:i+chunk_size]
            abs_chunk = np.abs(chunk)
            
            top_indices = np.argpartition(abs_chunk, -non_zeros, axis=1)[:, -non_zeros:]
                
            rows = np.arange(chunk.shape[0])[:, np.newaxis]
            top_vals = chunk[rows, top_indices]
            
            ones_mat = np.zeros(chunk.shape, dtype=bool)
            negative_ones_mat = np.zeros(chunk.shape, dtype=bool)
            
            ones_mat[rows, top_indices] = top_vals > 0
            negative_ones_mat[rows, top_indices] = top_vals < 0
            
            # Pack bit arrays and convert to integers
            ones_packed = np.packbits(ones_mat, axis=1)
            neg_packed = np.packbits(negative_ones_mat, axis=1)
            
            for j in range(chunk.shape[0]):
                o_int = int.from_bytes(ones_packed[j].tobytes(), 'little')
                n_int = int.from_bytes(neg_packed[j].tobytes(), 'little')
                data_evp.append(cls(o_int, n_int, dim))
                
        return data_evp

        
def get_max_similarity(dim, non_zeros):
    """
    Returns the maximum possible EvpBits similarity for a given dimension dim
    and number of non-zero elements. This corresponds to the similarity of
    a vector with itself.
    """
    return float(non_zeros + 2 * dim)


def evp_similarity(a, b):
    """
    Computes the EvpBits similarity between two EvpBits objects.
    Highly optimized using Python's built-in fast integer bit counting.
    """
    aa = (a.ones & b.ones).bit_count()
    bb = (a.negative_ones & b.negative_ones).bit_count()
    cc = (a.ones & b.negative_ones).bit_count()
    dd = (b.ones & a.negative_ones).bit_count()
    
    return float((aa + bb + a.dim * 2) - (cc + dd))

def evp_similarity_batch(queries, targets):
    """
    Computes similarities between two lists/arrays of EvpBits objects using 
    optimized matrix multiplication.
    Returns a numpy array of shape (len(queries), len(targets)).
    """
    if not queries or not targets:
        return np.array([[]])
        
    dim = queries[0].dim
    byte_len = (dim + 7) // 8
    
    Q_len = len(queries)
    T_len = len(targets)
    
    # 1. Convert integers to bytes, load into uint8 numpy arrays
    Qo_bytes = b''.join(q.ones.to_bytes(byte_len, 'little') for q in queries)
    Qn_bytes = b''.join(q.negative_ones.to_bytes(byte_len, 'little') for q in queries)
    
    To_bytes = b''.join(t.ones.to_bytes(byte_len, 'little') for t in targets)
    Tn_bytes = b''.join(t.negative_ones.to_bytes(byte_len, 'little') for t in targets)
    
    Qo_pack = np.frombuffer(Qo_bytes, dtype=np.uint8).reshape(Q_len, byte_len)
    Qn_pack = np.frombuffer(Qn_bytes, dtype=np.uint8).reshape(Q_len, byte_len)
    To_pack = np.frombuffer(To_bytes, dtype=np.uint8).reshape(T_len, byte_len)
    Tn_pack = np.frombuffer(Tn_bytes, dtype=np.uint8).reshape(T_len, byte_len)
    
    # 2. Unpack to boolean/uint8 arrays of correct dim
    Qo_unpacked = np.unpackbits(Qo_pack, axis=1)[:, :dim]
    Qn_unpacked = np.unpackbits(Qn_pack, axis=1)[:, :dim]
    To_unpacked = np.unpackbits(To_pack, axis=1)[:, :dim]
    Tn_unpacked = np.unpackbits(Tn_pack, axis=1)[:, :dim]
    
    # 3. Create ternary matrices
    Q = (Qo_unpacked.astype(np.int8) - Qn_unpacked.astype(np.int8)).astype(np.float32)
    T_T = (To_unpacked.astype(np.int8) - Tn_unpacked.astype(np.int8)).astype(np.float32).T
    
    return np.dot(Q, T_T) + 2 * dim

def compute_all_similarities_batch(evp_list, k_top=100, batch_size=1000):
    """
    Computes all-pairs EvpBits similarities directly from the underlying 
    `ones` and `negative_ones` boolean arrays of the `EvpBits` objects.
    Returns an array of shape (N, K) containing the indices of the Top-K elements for each vector.
    """
    N = len(evp_list)
    if N == 0:
        return np.array([])
    dim = evp_list[0].dim      
    byte_len = (dim + 7) // 8
    
    print("Re-assembling matrices from EvpBits objects for fast batched computation...", flush=True)
    convert_start = time.time()
    
    To_bytes = b''.join(e.ones.to_bytes(byte_len, 'little') for e in evp_list)
    Tn_bytes = b''.join(e.negative_ones.to_bytes(byte_len, 'little') for e in evp_list)
    
    To_pack = np.frombuffer(To_bytes, dtype=np.uint8).reshape(N, byte_len)
    Tn_pack = np.frombuffer(Tn_bytes, dtype=np.uint8).reshape(N, byte_len)
    
    To_unpacked = np.unpackbits(To_pack, axis=1)[:, :dim]
    Tn_unpacked = np.unpackbits(Tn_pack, axis=1)[:, :dim]
    
    T = (To_unpacked.astype(np.int8) - Tn_unpacked.astype(np.int8)).astype(np.float32)
    T_T = T.T
    print(f"Matrix re-assembly took {time.time() - convert_start:.2f} s")
    
    # Computing all-pairs similarity
    sim_start = time.time()    
    top_100_indices = np.zeros((N, k_top), dtype=np.int32)    
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        B_T = T[i:end]
        
        # Calculate dot product
        sim = np.dot(B_T, T_T)
        sim += 2 * dim
        
        # Extract top 100 indices
        top_k = np.argpartition(sim, -k_top, axis=1)[:, -k_top:]
        
        # Sort the top K elements properly by similarity (descending)
        top_k_vals = np.take_along_axis(sim, top_k, axis=1)
        sort_order = np.argsort(-top_k_vals, axis=1)
        top_100_indices[i:end] = np.take_along_axis(top_k, sort_order, axis=1)
        
        # Progress reporting
        if (i // 20000) > (max(0, i - batch_size) // 20000):
            print(f"\rProcessed {i}/{N} vectors... ({time.time() - sim_start:.2f} s elapsed)", end="", flush=True)
            
    print("\n", end="")
    print(f"Similarity computation and top {k_top} extraction took {time.time() - sim_start:.2f} s", flush=True)
    
    return top_100_indices