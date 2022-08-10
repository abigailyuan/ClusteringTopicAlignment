import rbo
import matplotlib.pyplot as plt
import numpy as np

# load ranked lists
wordnet_rank = 'topic_rank_wordnet.txt'
entropy_rank = 'topic_rank_entropy.txt'
KL_rank = 'topic_rank_KL.txt'
pairwise_rank = 'topic_rank_pairwise.txt'

# wordnet = [39,18,34,30,38,36,4,25,22,5,14,9,24,28,23,10,33,2,37,26,15,31,27,16,29,32,1,17,0,13,7,19,12,35,21,6,11,20,3,8]
# entropy = [3,21,5,7,14,35,23,34,33,15,25,38,36,24,19,29,17,18,20,37,13,10,26,0,22,6,12,1,2,32,30,4,31,8,27,9,11,28,16,39]
#
# KL = [28,30,20,36,24,31,37,14,17,0,33,8,26,11,32,13,34,16,3,1,7,35,5,21,9,39,6,4,27,19,22,12,10,23,2,18,38,29,25,15]
# pairwise = [25,33,1,30,12,32,19,18,4,2,15,39,31,5,35,13,9,27,17,8,29,21,10,37,24,6,38,11,7,23,36,16,26,20,3,22,14,28,0,34]
#
# print('KL similarity:')
# print(rbo.RankingSimilarity(wordnet, KL).rbo())
#
# print('entropy similarity:')
# print(rbo.RankingSimilarity(wordnet, entropy).rbo())
#
# print('pairwise similarity:')
# print(rbo.RankingSimilarity(wordnet, pairwise).rbo())

def read_file(filename):
    '''return a list of tuple representing the rank and score'''
    fp = open(filename,'r')
    rows = fp.readlines()
    fp.close()
    scores = []
    for row in rows:
        row = row.split()
        topic = row[0]
        score = float(row[1])
        scores.append((topic, score))
    return sorted(scores, key=lambda x:x[1], reverse=True)


scores = read_file(pairwise_rank)

topics = [x[0] for x in scores]
print(topics)
scores = [x[1] for x in scores]


# plt.plot(scores)
#
# plt.ylabel('Pairwise Word Coherence')
# # plt.ylim(0,1)
# plt.xlabel('Topic ID')
# plt.savefig('pairwise_topic_dist.pdf')
# plt.show()

# make data:
x = 1+np.arange(40)
y = scores

# plot
fig, ax = plt.subplots(figsize=(12,8))


ax.bar(x,y, width=0.8, edgecolor="white", linewidth=0.6)

ax.set(xlim=(0.5, 41))
ax.set_xticks(x)
ax.set_xticklabels(topics)
ax.set_xlabel('Topic ID')
ax.set_ylabel('Pairwise Word Coherence Score per Topic')
plt.savefig('pairwise_topic_rank.pdf')
# plt.show()