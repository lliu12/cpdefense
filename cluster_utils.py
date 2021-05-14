from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch

def device_to_weights(devices):
    arr = []
    for i in range(len(devices)):
        new_torch = torch.empty([1]).cuda()
        for k in devices[i]['weights']:
            if 'weight'  in k:
                new_torch = torch.cat((new_torch, devices[i]['weights'][k].flatten()), 0)
        arr.append(new_torch.detach().cpu().numpy())
    return arr

def compute_max_update_norm(cluster):
        return np.max([torch.norm(flatten(client['weights'])).item() for client in cluster])

    
def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([flatten(client['weights']) for client in cluster]), 
                                  dim=0)).item()

def flatten(source): 
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

    return angles.detach().numpy()

def cosine_sim_decomp(devices):
    similarities = pairwise_angles([client['weights'] for client in devices])
    return similarities

def pca_decomp(devices, dim):
    arr = device_to_weights(devices)
    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(arr)
    return X_pca

def tsne_decomp(devices, dim):
    arr = device_to_weights(devices)
    tsne = TSNE(n_components=dim)
    X_tsne = tsne.fit_transform(arr)
    return X_tsne

def agg_cluster(data):
  clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
  c1 = np.argwhere(clustering.labels_ == 0).flatten() 
  c2 = np.argwhere(clustering.labels_ == 1).flatten() 
  return c1,c2

def bipartion_cluster(devices, data, cluster_indices, EPS_1, EPS_2):
    cluster_indices_new = []
    for idc in cluster_indices:
        max_norm = compute_max_update_norm([devices[i] for i in idc])
        mean_norm = compute_mean_update_norm([devices[i] for i in idc])
            
        if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and c_round>20:
                        
            c1, c2 = cluster_clients(similarities[idc][:,idc]) 
            cluster_indices_new += [c1, c2]

        else:
            cluster_indices_new += [idc]
    cluster_indices = cluster_indices_new
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    return cluster_indices, client_clusters


def kmeans_cluster(data, num_clusters):
    knn = KMeans(n_clusters=num_clusters).fit(data) 
    preds = knn.predict(data)
    return preds

def gmm_cluster(data, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(data) 
    preds = gmm.predict(data)
    return preds
    
