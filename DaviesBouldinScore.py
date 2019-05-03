import glob

from sklearn.metrics import davies_bouldin_score

from Reporter import Report
from Schema.Utils import load_features


if __name__ == "__main__":

    rpt = Report(file_dir='./davies_bouldin_score.log')

    for f_path in glob.glob("./logs/features/*.txt"):
        if f_path.endswith('_train.txt'):
            continue

        f_test = f_path.split('/')[-1]

        X_test, y_test = load_features(f_test)

        rpt.write_text(f_test[:-4]).flush()

        dbscore = davies_bouldin_score(X_test, y_test)
        rpt.write_text('davies_bouldin_score,'+str(dbscore)).flush()
        print(f_test+' '+'euclidean_silscore,'+str(dbscore))

        rpt.end_line()
