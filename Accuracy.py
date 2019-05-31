import glob

from Utils import load_history
from Utils import plot_comparison_accu


if __name__ == "__main__":

    y_limit = 500

    cmp_list = {
        'True_None': 'With Augmentation',
        'False_None': 'Without Augmentation',
    }

    for cmpls, title in cmp_list.items():

        orders = {
            'MLV0': 'PROPOSED',
            'MSE': '',
            'MAE': '',
            'MAPE': '',
            'MSLE': '',
            'HNG': '',
            'SHNG': '',
            'LCH': '',
            'CRE': '',
            'KLD': '',
            'POS': '',
            'COS': '',
            }

        files = sorted(glob.glob("./logs/histories/*"+cmpls+".cpkl"))

        hists = []
        lables = []
        for _ord, alias in orders.items():
            for f in files:
                if '_'+_ord+'_' in f:
                    hists.append(load_history(f))
                    lable = f.split('_')[-3]
                    lables.append(lable if alias == '' else alias)

        plot_comparison_accu(hists, lables, title, y_limit=y_limit)
