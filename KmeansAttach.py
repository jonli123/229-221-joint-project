import numpy as np
from sklearn.cluster import KMeans
from getUsers import loadSingleData


def compute_k_means(data, K=5):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def compute_covariance(cluster_assignments, cluster_centers, data):
    k = cluster_centers.shape[0]
    m = data.shape[0]
    covs = []
    for i in range(k):
        cluster_points = np.asarray([data[j] for j in range(m) if cluster_assignments[j] == i])
        covariance = np.cov(cluster_points.T)
        covs.append(covariance)
    return covs


def generate_multivariate_gaussian_attack(means, covs, counts, count_threshold=5):
    '''
    Generate one attack per mean using multivariate gaussian
    '''
    attacks = []
    for i in range(means.shape[0]):
        if counts[i] >= count_threshold:
            mean = means[i]
            cov = covs[i]

            # Add small epsilon to cov is cov is too small
            min_eig = np.min(np.real(np.linalg.eigvals(cov)))
            if min_eig < 0:
                cov -= 10 * min_eig * np.eye(*cov.shape)

            attack = np.random.multivariate_normal(mean, cov)
            attacks.append(attack)
    return attacks


# def tile_attacks(data, attacks):
#     #tile each mean and data


def main():
    filename = 'keystroke.csv'
    labels, data = loadSingleData(filename)
    cluster_assignments, cluster_centers = compute_k_means(data, K=5)
    covs = compute_covariance(cluster_assignments, cluster_centers, data)
    counts = [np.sum(cluster_assignments == i) for i in range(cluster_centers.shape[0])]
    attacks = generate_multivariate_gaussian_attack(cluster_centers, covs, counts)
    print(attacks)



if __name__ == '__main__':
    main()
