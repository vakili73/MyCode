import glob
import numpy as np

from sklearn import metrics
from sklearn.metrics import pairwise_distances

from Reporter import Report
from Schema.Utils import load_features


def intra_cluster_distance(distances_row, labels, i):
    mask = labels == labels[i]
    mask[i] = False
    if not np.any(mask):
        # cluster of size 1
        return 0
    a = np.sum(distances_row[mask])
    return a


def inter_cluster_distance(distances_row, labels, i):
    mask = labels != labels[i]
    b = np.sum(distances_row[mask])
    return b


def member_count(labels, i):
    mask = labels == i
    return len(labels[mask])


def betacv(data, labels, metric='euclidean'):
    distances = pairwise_distances(data, metric=metric)
    n = labels.shape[0]
    A = np.array([intra_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    B = np.array([inter_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    a = np.sum(A)
    b = np.sum(B)
    labels_unq = np.unique(labels)
    members = np.array([member_count(labels, i) for i in labels_unq])
    N_in = np.array([i*(i-1) for i in members])
    n_in = np.sum(N_in)
    N_out = np.array([i*(n-i) for i in members])
    n_out = np.sum(N_out)
    betacv = (a/n_in)/(b/n_out)
    return betacv


if __name__ == "__main__":

    rpt = Report(file_dir='./report_incluster.log')

    for f_path in glob.glob("./logs/features/*.txt"):
        if f_path.endswith('_train.txt'):
            continue

        f_test = f_path.split('/')[-1]

        X_test, y_test = load_features(f_test)

        rpt.write_text(f_test[:-4]).flush()

        silscore = metrics.silhouette_score(X_test, y_test, metric='euclidean')
        rpt.write_text('euclidean_silscore,'+str(silscore)).flush()

        # silscore = metrics.silhouette_score(X_test, y_test, metric='jensenshannon')
        # rpt.write_text('jensenshannon_silscore,'+str(silscore)).flush()

        betacv_score = betacv(X_test, y_test, metric='euclidean')
        rpt.write_text('euclidean_betacv,'+str(betacv_score)).flush()

        # betacv_score = betacv(X_test, y_test, metric='jensenshannon')
        # rpt.write_text('jensenshannon_betacv,'+str(betacv_score)).flush()

        rpt.end_line()
