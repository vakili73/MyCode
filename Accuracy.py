import glob

from Utils import load_history
from Utils import plot_comparison_accu


if __name__ == "__main__":

    for f_path in glob.glob("./logs/histories/*ConventionalV1*.cpkl"):

        _f_path = f_path.split('_')
        _f_path[1] = 'MyModelV1'

        history = load_history(f_path)
        _history = load_history('_'.join(_f_path))

        db = _f_path[0].split('/')[-1]
        title = db + '_' + '_'.join(_f_path[2:4])
        title += '_' + _f_path[4][:-5]

        print(title)
        plot_comparison_accu(history, _history, title)