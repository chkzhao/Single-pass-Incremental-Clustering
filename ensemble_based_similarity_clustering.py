import numpy as np

def _ifInSameCluster(list, i, j):
    if list[i] == list[j]:
        return 1
    return 0

def _similarityMatrixCal(list, weight_list):

    similarity_matrix = np.zeros((5000, 5000))

    for i in range(5000):
        for j in range(5000):
            if i < j:
                continue
            else:
                similarity_matrix[i][j] = np.dot(weight_list, [_ifInSameCluster(list[k], i, j) for k in range(5)])

    return similarity_matrix

def _getMaxSimilarity(sim_matrix, clusters, curr_index):
    max_cluster = 0
    max_similarity = 0

    for key, value in clusters.items():
        similarity = []
        for item in value:
            if curr_index < item:
                similarity.append(sim_matrix[item][curr_index])
            else:
                similarity.append(sim_matrix[curr_index][item])
        if np.mean(similarity) > max_similarity:
            max_similarity = np.mean(similarity)
            max_cluster = key

    return max_cluster, max_similarity

def ensembledClustering_partition(titles_list, disc_list, tags_list, time_list, geo_list, weight_list, theta):

    list = [titles_list, disc_list, tags_list, time_list, geo_list]
    sim_matrix = _similarityMatrixCal(list, weight_list)

    clusters = {}
    num_clusters = 0

    for index in range(5000):
        if num_clusters == 0:
            clusters[num_clusters] = []
            clusters[num_clusters].append(index)
            num_clusters += 1
        else:
            max_cluster, max_value = _getMaxSimilarity(sim_matrix, clusters, index)

            if max_value > theta:
                clusters[max_cluster].append(index)
            else:
                clusters[num_clusters] = []
                clusters[num_clusters].append(index)
                num_clusters += 1

    print("The number of total cluster is {}.".format(num_clusters))

    cluster_list = []
    for _ in range(5000):
        cluster_list.append(0)
    for key, value in clusters.items():
        for item in value:
            cluster_list[item] = key
    return cluster_list

def ensembleWithSimilarityMatrix(similarity_matrix, theta):
    clusters = {}
    num_clusters = 0

    for index in range(len(similarity_matrix)):
        if num_clusters == 0:
            clusters[num_clusters] = []
            clusters[num_clusters].append(index)
            num_clusters += 1
        else:
            max_cluster, max_value = _getMaxSimilarity(similarity_matrix, clusters, index)

            if max_value > theta:
                clusters[max_cluster].append(index)
            else:
                clusters[num_clusters] = []
                clusters[num_clusters].append(index)
                num_clusters += 1

    print("The number of total cluster is {}.".format(num_clusters))

    cluster_list = []
    for _ in range(len(similarity_matrix)):
        cluster_list.append(0)
    for key, value in clusters.items():
        for item in value:
            cluster_list[item] = key
    return cluster_list


def _getCentroid_time(index, df):
    sum = 0
    for i in index:
        sum += df[i]
    return sum / len(index)


def _getCentroid_location(indexs, df):
    centroid = [0, 0]
    lon_list = []
    lat_list = []
    for index in indexs:
        lon_list.append(df[index][0])
        lat_list.append(df[index][1])
    centroid[0] = np.mean(lon_list)
    centroid[1] = np.mean(lat_list)
    return centroid


def single_pass_cluster(feature, threshold, similiarty_func):
    num_doc, num_fea = feature.shape
    all_idx = [i for i in range(num_doc)]
    zeros_idx = [idx for idx, doc in enumerate(feature) if sum(doc) == 0]
    nozeros_idx = [idx for idx in all_idx if idx not in zeros_idx]
    first = nozeros_idx[0]

    cluster = {0: [first]}
    centroid = {0: feature[first].tolist()}

    for idx_doc in nozeros_idx[1:]:
        max_similairty = 0
        max_cluster = 0
        for idx_cluster in range(len(cluster)):
            similarity = similiarty_func(feature[idx_doc], centroid[idx_cluster])
            print(similarity)
            if similarity > max_similairty:
                max_similairty = similarity
                max_cluster = idx_cluster

        if max_similairty > threshold:
            cluster[max_cluster].append(idx_doc)
        else:
            cluster[len(cluster)] = [idx_doc]
        centroid = {i: np.mean(feature[cluster[i]], axis=0).tolist() for i in cluster.keys()}
    cluster[len(cluster)] = zeros_idx
    return cluster


def consine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.sum(np.square(vec2))) * np.sqrt(np.sum(np.square(vec1))))


def single_pass_cluster_similarity(doc_similiary, threshold):
    cluster = {0: [0]}
    n, m = doc_similiary.shape
    for i in range(n):
        max_similarity = 0
        max_cluster = 0
        for j in range(len(cluster)):
            similarity = np.mean(doc_similiary[i, cluster[j]])
            if similarity > max_similarity:
                max_similarity = similarity
                max_cluster = j

        if max_similarity > threshold:
            cluster[max_cluster].append(i)
        else:
            cluster[len(cluster)] = [i]
    return cluster


def angleToRadian(angle):
    return angle * np.pi / 180


def time_similarity(time1, time2):
    if (abs(time1 - time2) / (60 * 24)) > 365:
        return 0
    return 1 - (abs(time1 - time2) / 525600)


def location_similarity(location1, location2):
    if (location1[0] == 0 and location1[1] == 0) or (location2[0] == 0 and location2[1] == 0):
        return 0
    dlon = location2[0] - location1[0]
    dlat = location2[1] - location1[1]
    a = np.sin(dlat / 2) ** 2 + np.cos(angleToRadian(location1[1])) * np.cos(angleToRadian(location2[1])) * np.sin(
        dlon / 2) ** 2
    c = 2 * np.arcsin(a ** (1 / 2))
    return 1 - c


def single_pass_cluster_centorid_ensemble(doc_num, Tag, Title, Description, Time, Geo, threshold,
                                          threshold_each_feature, weight):
    for i in range(len(Geo)):
        if Geo[i] == "n/a":
            Geo[i] = [0, 0]
        else:
            part1, part2 = Geo[i].split(" ")[0], Geo[i].split(" ")[1]
            longitude, latitude = part1.split("=")[1], part2.split("=")[1]
            Geo[i] = [float(longitude), float(latitude)]

    cluster = {0: [0]}
    for i in range(int(doc_num))[1:]:
        vote = {"tag": [], "Title": [], "Description": [], "Time": [], "Geo": []}

        cluster_centroid_Tag = {i: np.mean(Tag[cluster[i]], axis=0) for i, val in cluster.items()}
        cluster_centroid_Title = {i: np.mean(Title[cluster[i]], axis=0) for i, val in cluster.items()}
        # cluster_centroid_AllText = {i:np.mean(All_Text[cluster[i]],axis=0) for i,val in cluster.items()}
        cluster_centroid_Description = {i: np.mean(Description[cluster[i]], axis=0) for i, val in cluster.items()}
        cluster_centroid_Time = {i: _getCentroid_time(val, Time) for i, val in cluster.items()}
        cluster_centroid_Geo = {i: _getCentroid_location(val, Geo) for i, val in cluster.items()}

        for j in range(len(cluster)):

            if consine_similarity(Tag[i], cluster_centroid_Tag[j]) > threshold_each_feature[0]:
                vote["tag"].append(1)
            else:
                vote["tag"].append(0)

            if consine_similarity(Title[i], cluster_centroid_Title[j]) > threshold_each_feature[1]:
                vote["Title"].append(1)
            else:
                vote["Title"].append(0)

            # if consine_similarity(All_Text[i],cluster_centroid_AllText[j]) > threshold_each_feature[2]:
            #    vote["AllText"].append(1)
            # else:
            #    vote["AllText"].append(0)

            if consine_similarity(Description[i], cluster_centroid_Description[j]) > threshold_each_feature[2]:
                vote["Description"].append(1)
            else:
                vote["Description"].append(0)
            # print(Time[i],cluster_centroid_Time[j])
            if time_similarity(Time[i], cluster_centroid_Time[j]) > threshold_each_feature[3]:
                vote["Time"].append(1)
            else:
                vote["Time"].append(0)

            if location_similarity(Geo[i], cluster_centroid_Geo[j]) > threshold_each_feature[4]:
                vote["Geo"].append(1)
            else:
                vote["Geo"].append(0)

        vote_result = [0 for i in range(len(cluster))]
        item = 0
        for fea, score in vote.items():
            for idx, grade in enumerate(score):
                vote_result[idx] += weight[item] * grade
            item += 1
        if max(vote_result) > threshold:
            idx = np.argmax(vote_result)
            cluster[idx].append(i)
        else:
            cluster[len(cluster)] = [i]

    print("The number of total cluster is {}.".format(len(cluster)))
    cluster_list = []
    for _ in range(int(doc_num)):
        cluster_list.append(0)
    for key, value in cluster.items():
        for item in value:
            cluster_list[item] = key
    return cluster_list







