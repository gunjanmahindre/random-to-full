# random-to-full
Here we try to recover the complete NxN distance matrix from a few randomly selected entries. We use Low-rank Matrix Completion and informed upper and lower bounds calculated from only the observed distances.

## Steps:
1. Take a distance matrix (D) for a network of size [N,N]
2. Select a certain percentage of distance entries from the complete D matrix.
3. The D matrix is symmetric so ij entry and ji entry both are deleted.
4. Calculate upper and lower bounds for missing entries
5. Recover the complete [N,N] D matrix using Low-rank Matrix Completion while satisfying the bounds

## Performance evaluation metrics:
1. Mean error (percentage error)
2. Absolute hop distance error
3. Clustering coefficient 
4. Average node degree 
5. KL divergence for degree distribution 
6. KL divergence for distance distribution 

Metrics (1) and (2) are measured to understand the prediction error.
Metrics (3,4,5) and (6) are measured so that we can comapre wheather we can restore original network character
