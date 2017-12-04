import numpy as np
from sklearn.cluster import KMeans
from getUsers import loadSingleData


def compute_k_means(data, K=5):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def get_data_clusters(cluster_assignments, cluster_centers, data):
    k = cluster_centers.shape[0]
    m = data.shape[0]
    return [np.asarray([data[j] for j in range(m) if cluster_assignments[j] == i]) for i in range(k)]



def generate_multivariate_gaussian_attack(cluster_assignments, cluster_centers, data, counts,
                                          count_threshold=5, tile_fn=None, sample_guassian=True, sample_num=1):
    '''
    Generate one attack per mean using multivariate gaussian
    '''
    k = cluster_centers.shape[0]
    clusters = get_data_clusters(cluster_assignments, cluster_centers, data)
    attacks = []
    for i in range(k):
        if counts[i] >= count_threshold:
            mean = cluster_centers[i]

            attack = [mean]
            if sample_guassian:
                cov = np.cov(clusters[i].T)

                # Add small epsilon to cov is cov is too small
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig < 0:
                    cov -= 10 * min_eig * np.eye(*cov.shape)

                attack = [np.random.multivariate_normal(mean, cov) for _ in range(sample_num)]

            if tile_fn:
                attack = tile_fn(clusters[i], attack)

            attacks.extend(attack)
    return np.asarray(attacks)



def tile_attacks(data, attacks):
    n = data.shape[1]

    tiled_attacks = []
    for d in data:
        for a in attacks:
            example = np.concatenate((d, a))
            tiled_attacks.append(example)

    return np.asarray(tiled_attacks)


def generate_tiled_k_means_attack(K, sample_guassian=False, sample_num=1, data_file='keystroke.csv'):
    labels, data = loadSingleData(data_file)
    cluster_assignments, cluster_centers = compute_k_means(data, K=K)
    counts = [np.sum(cluster_assignments == i) for i in range(cluster_centers.shape[0])]
    attacks = generate_multivariate_gaussian_attack(cluster_assignments, cluster_centers, data, counts,
                                                    tile_fn=tile_attacks, sample_guassian=sample_guassian,
                                                    sample_num=sample_num)
    print(counts)
    return attacks


def main():
    filename = 'keystroke.csv'
    labels, data = loadSingleData(filename)
    cluster_assignments, cluster_centers = compute_k_means(data, K=5)
    counts = [np.sum(cluster_assignments == i) for i in range(cluster_centers.shape[0])]
    attacks = generate_multivariate_gaussian_attack(cluster_assignments, cluster_centers, data, counts, tile_fn=tile_attacks)

    print(attacks.shape)



if __name__ == '__main__':
    main()
