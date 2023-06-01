from sklearn.cluster import AgglomerativeClustering

def agglomaritive(vectorised_corpus):
    model = AgglomerativeClustering(metric='euclidean', linkage='ward', compute_distances=True)
    clustering = model.fit(vectorised_corpus)
    return clustering

