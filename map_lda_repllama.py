from mappings.GreedyMapping import feature_mapping
import pickle
import torch
import numpy as np


projected_features = pickle.load(open('ProcessedWSJ/projected_features.pkl','rb'))
topics = pickle.load(open('Results/LDA/wsj_topic_doc_matrix.pkl','rb'))

print(projected_features.shape)
print(topics.shape)


# 1. move to CPU & convert
if torch.is_tensor(projected_features):
    projected_np = projected_features.cpu().numpy()
else:
    projected_np = np.asarray(projected_features)

if torch.is_tensor(topics):
    topics_np = topics.cpu().numpy()
else:
    topics_np = np.asarray(topics)
    
mapping = feature_mapping(projected_np, topics_np)
pickle.dump(mapping, open('Results/wsj_repllama_lda_mapping.pkl','wb'))
print(type(mapping))
print(mapping)