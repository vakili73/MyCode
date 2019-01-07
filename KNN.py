import time
import glob
import numpy as np
import tensorflow as tf

from Config import __KNNS
from Config import METHODS
from Config import DATASETS

from Reporter import Report
from Runner import clf_report
from Runner import getFeatures

from Database import INFORM
from Database import load_data
from Database.Utils import reshape

from Schema import load_schema
from Schema.Utils import load_weights

from tensorflow.keras import backend as K
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":

    METRICS = ['minkowski', 'jensenshannon', 'kullbackleibler']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    rpt = Report(file_dir='./report_knn.log')

    for f_path in glob.glob("./logs/models/*.h5"):
        f_name = f_path.split('/')[-1]
        rpt.write_text(f_name[:-11]).flush()
        f_split = f_name[:-11].split('_')

        db = f_split[0]
        db_opt = DATASETS[db]

        n_cls = INFORM[db]['n_cls']
        shape = INFORM[db]['shape']
        X_train, X_test, y_train, y_test = load_data(db)
        X_train = reshape(X_train/255.0, shape)
        X_test = reshape(X_test/255.0, shape)

        bld = f_split[1]
        bld_opt = METHODS[bld]

        schm = db_opt['schema']
        schema = load_schema(schm)
        getattr(schema, 'build'+bld)(shape, n_cls)

        load_weights(schema.model, f_name)

        embed_feature = getFeatures(schema.getModel(), X_test)
        embed_feature_train = getFeatures(schema.getModel(), X_train)

        for knn in __KNNS:
            for metric in METRICS:
                start_time = time.time()
                rpt.write_knn(knn['weights'], knn['n_neighbors']).flush()
                rpt.write_text('metric,'+metric).flush()
                print(f_name[:-11]+'_'+metric)
                clf = KNeighborsClassifier(**knn, metric=metric)
                clf.fit(embed_feature_train, y_train)
                y_score = clf.predict_proba(embed_feature)
                clf_report(rpt, y_test, y_score, n_cls, f_name[:-11])
                print("--- %s seconds ---" % (time.time() - start_time))

        rpt.end_line()

    pass
