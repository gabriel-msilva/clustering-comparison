# A comparison of four Clustering algorithms

*Access the [`clustering_comparison.md`](https://github.com/gabriel-msilva/clustering-comparison/blob/master/clustering_comparison.md) file to read the full project.*

This kernel is a comparison between four different unsupervised learning clustering algorithms over simulated data. Four algorithms are tested:

* k-Means;
* Hierarchical Agglomerative Clustering (HAC);
* Expectation–Maximization clustering using Gaussian Mixture Models (GMM); and
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

I generated five datasets from normal and uniform distribuitions to simulate practical cases. Each dataset presents a different pattern and challenge for the algorithm.

It is not my intent to cover each algorithm in details, but I tried to tweak some of the parameters to avoid underperformance due to bad usage. Clustering is highly dependent on the particular dataset and the analysis objective. However, it is interesting to see how each algorithm performs on each case to give an idea of when to use a particular algorithm.
