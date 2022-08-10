import Visualisation
from scipy.stats import entropy
from matplotlib.pyplot import bar
import matplotlib.pyplot as plt

def get_topic_entropy(cluster_topic_matrix):
    entropies = []
    for i in range(len(cluster_topic_matrix)):
        topic = cluster_topic_matrix.iloc[:, i]
        entropies.append(entropy(topic))
    return entropies

cid = 9
tid = 7
clustering = 'ClusterResults/' + str(cid) + '/model'
topic_model = 'LDAResults/' + str(tid) + '/model'
corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
directory = 'figures/'
clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,c_order=20, t_order=20, mode='distribution')

entropies = get_topic_entropy(cluster_topic_matrix)
entropy_rank = sorted(zip(range(len(entropies)), entropies), key=lambda x:x[1])
print(sum(entropies)/len(entropies))

x = range(0,40,2)
y = [x[1] for x in entropy_rank]
x_ticks = [x[0] for x in entropy_rank]
fig, axes = plt.subplots(2,1,figsize=(10,9))
axes[0].bar(x, y, width=1)
axes[0].set_xticks(x, x_ticks)
axes[0].set_xlabel("topic")
axes[0].set_ylabel('entropy')
axes[0].set_title("Topic Model Run 7")

cid = 9
tid = 1
clustering = 'ClusterResults/' + str(cid) + '/model'
topic_model = 'LDAResults/' + str(tid) + '/model'
corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
_, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,c_order=20, t_order=20, mode='distribution')

entropies = get_topic_entropy(cluster_topic_matrix)
entropy_rank = sorted(zip(range(len(entropies)), entropies), key=lambda x:x[1])
print(sum(entropies)/len(entropies))

x = range(0,40,2)
y = [x[1] for x in entropy_rank]
x_ticks = [x[0] for x in entropy_rank]
axes[1].bar(x, y, width=1)
axes[1].set_xticks(x, x_ticks)
axes[1].set_xlabel("topic")
axes[1].set_ylabel('entropy')
axes[1].set_title("Topic Model Run 1")

plt.savefig("Topic_Entropy_by_Clusters.pdf")
plt.show()

