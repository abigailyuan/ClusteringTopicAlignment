from scipy import spatial
import numpy as np

def greedy_mapping(random_projections, topics):
    projection_pool = random_projections
    mappings = []
    projection_id = 0
    while projection_pool:
        projection = projection_pool.pop(0)
        max_sim = 0
        curr_map_topic = None
        # calculate cosine similarity
        for idx in range(len(topics)):
            topic = topics[idx]
            sim = abs(1 - spatial.distance.cosine(projection, topic))
            if sim > max_sim:
                max_sim = sim
                curr_map_topic = idx
        topics.pop(idx)
        mappings.append((projection_id, idx))
        projection_id += 1
    return mappings

def global_mapping(random_projections, topics):
    mappings = []
    sim_lst = []
    for rpx in range(len(random_projections)):
        rp = random_projections[rpx]
        for topicx in range(len(topics)):
            topic = topics[topicx]
            sim = abs(1 - spatial.distance.cosine(rp, topic))
            sim_lst.append((sim, rpx, topicx))

    # get global mappings
    sim_lst = sorted(sim_lst, reverse=True)
    topic_pool = [i for i in range(len(topics))]
    projection_pool = [i for i in range(len(random_projections))]
    for sim, rpx, topicx in sim_lst:
        if (rpx in projection_pool) and (topicx in topic_pool):
            mappings.append((rpx, topicx))
            projection_pool.remove(rpx)
            topic_pool.remove(topicx)
        if topic_pool == [] or projection_pool == []:
            break
    return mappings

def feature_mapping(random_projections, topics):
    mappings = []
    for rid in range(len(random_projections)):
        rp = random_projections[rid]

        max_sim = -1
        curr_topic = None
        for tid in range(len(topics)):
            topic = topics[tid]
            sim = 1 - spatial.distance.cosine(rp, topic)
            if sim >= max_sim:
                curr_topic = tid
                max_sim = sim
        mappings.append((rid, curr_topic))
    return mappings



