import glob

from jqmcvi import dunn_fast

from sklearn.metrics import davies_bouldin_score

from Reporter import Report
from Schema.Utils import load_features


if __name__ == "__main__":

    rpt = Report(file_dir='./cluster_scores.log')

    for f_path in glob.glob("./logs/features/*.txt"):
        if f_path.endswith('_train.txt'):
            continue

        f_test = f_path.split('/')[-1]

        X_test, y_test = load_features(f_test)

        rpt.write_text(f_test[:-4]).flush()

        print(f_test)

        dbscore = davies_bouldin_score(X_test, y_test)
        rpt.write_text('davies_bouldin_score,'+str(dbscore)).flush()

        dunnscore = dunn_fast(X_test, y_test)
        rpt.write_text('dunn_index_score,'+str(dunnscore)).flush()

        rpt.end_line()
