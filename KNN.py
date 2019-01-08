import time
import glob

from Config import __KNNS
from Reporter import Report
from Database import INFORM
from Runner import clf_report
from Schema.Utils import load_features
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":

    METRICS = ['minkowski', 'jensenshannon']

    rpt = Report(file_dir='./report_knn.log')

    for f_path in glob.glob("./logs/features/*_train.txt"):
        f_train = f_path.split('/')[-1]
        f_test = f_train[:-10] + '.txt'

        X_train, y_train = load_features(f_train)
        X_test, y_test = load_features(f_test)

        db = f_test.split('_')[0]
        n_cls = INFORM[db]['n_cls']

        rpt.write_text(f_test[:-4]).flush()

        for knn in __KNNS:
            rpt.write_knn(knn['weights'], knn['n_neighbors']).flush()

            for metric in METRICS:
                rpt.write_text('metric,'+metric).flush()
                title = 'knn_'+f_test[:-4]+'_'+metric + \
                    '_weights_{}_neighbors_{}'.format(
                        knn['weights'], knn['n_neighbors'])
                print(title)

                clf = KNeighborsClassifier(**knn, metric=metric)
                clf.fit(X_train, y_train)

                start_time = time.time()
                y_score = clf.predict_proba(X_test)
                end_time = time.time() - start_time
                rpt.write_text('time,'+str(end_time)).flush()

                clf_report(rpt, y_test, y_score, n_cls, title)
                print("--- %s seconds ---" % (time.time() - start_time))

        rpt.end_line()
