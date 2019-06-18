import os
import glob

import numpy as np

from Reporter import Report
from Database import INFORM
from Schema.Utils import load_features


def mean_entropy(X, y=None, axis=-1, n_cls=None):
    def __mean_entropy(_X):
        _X = np.clip(_X, np.finfo(np.float).eps, 1.0)
        return np.mean(-np.sum(_X * np.log(_X), axis=axis))

    if n_cls == None:
        return __mean_entropy(X)

    else:
        result = []
        for i in range(n_cls):
            X_temp = X[y == i]
            result.append(__mean_entropy(X_temp))
        return np.mean(result)


if __name__ == "__main__":

    rpt = Report(file_dir='./report_entropy.log')

    for f_path in glob.glob("./logs/features/*_None.txt"):
        print(f_path)
        X_test, y_test = load_features(f_path[2:], os.getcwd())

        db = f_path.split('/')[-1].split('_')[0]
        n_cls = INFORM[db]['n_cls']

        rpt.write_text('_'.join(f_path.split('_')[-5:-1])).flush()

        # mean_entropy_all_samples = mean_entropy(X_test)
        # mean_entropy_all_feature = mean_entropy(X_test, axis=0)
        mean_entropy_cls_samples = mean_entropy(X_test, y_test, n_cls=n_cls)
        mean_entropy_cls_feature = mean_entropy(X_test, y_test, 0, n_cls=n_cls)
        
        # rpt.write_text('mean_entropy_all_samples').write_text(mean_entropy_all_samples).flush()
        # rpt.write_text('mean_entropy_all_feature').write_text(mean_entropy_all_feature).flush()
        rpt.write_text('mean_entropy_cls_samples').write_text(mean_entropy_cls_samples).flush()
        rpt.write_text('mean_entropy_cls_feature').write_text(mean_entropy_cls_feature).flush()

        rpt.end_line()

    pass
