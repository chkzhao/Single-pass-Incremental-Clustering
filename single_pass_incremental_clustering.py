import numpy as np
#import bcubed

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

def _angleToRadian(angle):
  return angle * np.pi / 180

def _time_similarity(time1, time2):
    if (abs(time1 - time2) / (60 * 24)) > 365:
        return 0
    return 1 - (abs(time1 - time2) / 525600)

def _location_similarity(location1, location2):
  if (location1[0]==0 and location1[1]==0) or (location2[0]==0 and location2[1]==0):
    return 0
  dlon = location2[0] - location1[0]
  dlat = location2[1] - location1[1]
  a = np.sin(dlat/2)**2 + np.cos(_angleToRadian(location1[1]))*np.cos(_angleToRadian(location2[1]))*np.sin(dlon/2)**2
  c = 2*np.arcsin(a**(1/2))
  return 1-c

def _consine_similarity(vec1, vec2):
  return np.dot(vec1,vec2)/(np.sqrt(np.sum(np.square(vec2)))*np.sqrt(np.sum(np.square(vec1))))

def _getMaxSimilarity_time(clusters, curr_index, df):
  max_cluster = 0
  max_similarity = 0
  similarity = 0

  for key,value in clusters.items():
    sum = 0
    similarity = 0

    centroid = np.mean([df[val] for val in value])
    #for num in value:
    #  sum += df[num]
    #centroid = sum / len(value)
    similarity = _time_similarity(centroid, df[curr_index])

    if similarity > max_similarity:
      max_similarity = similarity
      max_cluster = key

  return max_cluster, max_similarity

def _getMaxSimilarity_location(centroids, curr_loc):
  max_cluster = 0
  max_similarity = 0
  #similarity = 0

  for key, centroid in centroids.items():
    #sum = 0
    #similarity = 0

    #for num in value:
    #  sum += df[num]
    #centroid = sum / len(value)
    similarity = _location_similarity(centroid, curr_loc)

    if similarity > max_similarity:
      max_similarity = similarity
      max_cluster = key

  return max_cluster, max_similarity

def _getMaxSimilarity_text(centroids, curr_doc):
  max_cluster = 0
  max_similarity = 0

  for key, centroid in centroids.items():
    similarity = _consine_similarity(centroid, curr_doc)

    if similarity > max_similarity:
      max_similarity = similarity
      max_cluster = key

  return max_cluster, max_similarity

def _singlePass_time(df, theta):
  treshold = theta
  clusters = {}
  num_clusters = 0

  for index in range(len(df)):
    if num_clusters == 0:
      clusters[num_clusters] = []
      clusters[num_clusters].append(index)
      num_clusters += 1

    else:
      max_cluster, max_value = _getMaxSimilarity_time(clusters, index, df)

      if max_value > treshold:
        clusters[max_cluster].append(index)

      else:
        clusters[num_clusters] = []
        clusters[num_clusters].append(index)
        num_clusters += 1


  print("The number of total cluster is {}.".format(num_clusters))
  return clusters

def _singlePass_location(df, theta):
  treshold = theta
  clusters = {}
  centroids = {}
  num_clusters = 0

  for index in range(len(df)):
    if num_clusters == 0:
      clusters[num_clusters] = []
      clusters[num_clusters].append(index)
      centroids[num_clusters] = df[index]
      num_clusters += 1
    else:
      max_cluster, max_value = _getMaxSimilarity_location(centroids, df[index])

      if max_value > treshold:
        clusters[max_cluster].append(index)
        centroids[max_cluster] = _getCentroid_location(clusters[max_cluster], df)
      else:
        clusters[num_clusters] = []
        clusters[num_clusters].append(index)
        centroids[num_clusters] = df[index]
        num_clusters += 1

  print("The number of total cluster is {}.".format(num_clusters))
  return clusters

def _singlePass_text(df, theta):

  clusters = {}
  centroids = {}
  num_clusters = 0

  for index in range(len(df)):
    #if sum(data[i]) == 0:

    if num_clusters == 0 or sum(df[index]) == 0:
      clusters[num_clusters] = []
      clusters[num_clusters].append(index)
      centroids[num_clusters] = df[index]
      num_clusters += 1
    else:
      max_cluster, max_value = _getMaxSimilarity_text(centroids, df[index])

      if max_value > theta:
        clusters[max_cluster].append(index)
        centroids[max_cluster] = np.mean(df[clusters[max_cluster]], axis=0).tolist()
      else:
        clusters[num_clusters] = []
        clusters[num_clusters].append(index)
        centroids[num_clusters] = df[index]
        num_clusters += 1

  print("The number of total cluster is {}.".format(num_clusters))
  return clusters





def fit_data_time(data, theta = 0.95):
  clusters = _singlePass_time(data, theta)
  #print(type(clusters))
  cluster_list = []
  for _ in range(len(data)):
    cluster_list.append(0)
  for key,value in clusters.items():
    for item in value:
      cluster_list[item] = key
  return cluster_list

def fit_data_location(data, theta = 0.65):
  for i in range(len(data)):
    if data[i] == "n/a":
      data[i] = [0,0]
    else:
      part1, part2 = data[i].split(" ")[0], data[i].split(" ")[1]
      longitude, latitude = part1.split("=")[1], part2.split("=")[1]
      data[i] = [float(longitude), float(latitude)]
  #return data
  clusters = _singlePass_location(data, theta)

  cluster_list = []
  for _ in range(len(data)):
    cluster_list.append(0)
  for key, value in clusters.items():
    for item in value:
      cluster_list[item] = key
  return cluster_list

def fit_data_text(data, theta):

  clusters = _singlePass_text(data, theta)

  cluster_list = []
  for _ in range(len(data)):
    cluster_list.append(0)
  for key, value in clusters.items():
    for item in value:
      cluster_list[item] = key
  return cluster_list

#def

def test_func(data):
  print(type(data))
  print(len(data))
  print(len(data[0]))




#a = ["n/a", "GeoData[longitude=-77.58403 latitude=43.15915 accuracy=15]", "GeoData[longitude=-77.58403 latitude=43.15915 accuracy=15]"]
#print(fit_data_location(a))

'''
def bcube_cal_test(clusters, classi):
  clusters_dic = {}
  classi_dic = {}

  for i in range(len(clusters)):
    if "item{}".format(clusters[i]) not in clusters_dic.keys():
      clusters_dic["item{}".format(clusters[i])] = []
      clusters_dic["item{}".format(clusters[i])].append(i)
    else:
      clusters_dic["item{}".format(clusters[i])].append(i)
  #for j in range(len(classi)):

def calPrec(clusters, classi):
  sum_precision = 0
  for i in range(len(clusters)):
    same = 0
    length_temp_cluster = clusters.count(clusters[i])
    for j in range(len(clusters)):
      if (clusters[j] == clusters[i]) & (classi[j] == classi[i]):
        same += 1
    indi_precision = same/length_temp_cluster
    sum_precision += indi_precision
  return sum_precision/len(clusters)

def calRec(clusters, classi):
  sum_recall = 0
  for i in range(len(classi)):
    same = 0
    length_temp_classi = classi.count(classi[i])
    for j in range(len(classi)):
      if (classi[j] == classi[i]) & (clusters[j] == clusters[i]):
        same += 1
    indi_recall = same/length_temp_classi
    sum_recall += indi_recall
  return sum_recall/len(classi)

def bcubed_calcualte(clusters, classi):
  f_score = (2*calPrec(clusters, classi)*calRec(clusters, classi)/(calPrec(clusters, classi)+calRec(clusters, classi)))
  return f_score

#a = [1,1,1,1]
#b = [2,2,2,2]

#dic = np.array([[1,1,1,1], [2,2,2,2], [3,3,3,3]])
#d = {0:[0,2], 1:[1,2]}

#c = {i:np.mean(dic[d[i]], axis=0).tolist() for i in d.keys()}
#print(c)
#clusters = [1,2,2,4,4,1,3]
#classi = [135961,135961,123862,123862,123862,148825,40459]
#bcubed_num = bcubed_cal(clusters,classi)
#print(bcubed_num)

#clusters = {"item1":set([0,5]), "item2":set([1,2]), "item3":set([6]), "item4":set([3,4])}
#classi = {"item1":set([0,1]), "item2":set([2,3,4]), "item3":set([5]), "item4":set([6])}



#bcubed_compute(clusters, classi)

#precision = bcubed.precision(clusters, classi)
#recall = bcubed.recall(clusters, classi)
#f_score = bcubed.fscore(precision, recall)
#print(f_score)

#list = [175483,172344,172345, 175646, 174367,173467,107822,40493]
#cluster = fit_data(list)

#print(cluster)
'''







