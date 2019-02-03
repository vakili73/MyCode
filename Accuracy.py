import glob

from Utils import load_history
from Utils import plot_comparison_accu


if __name__ == "__main__":

    cmp_list = [
        'True_None',
        'False_None',
    ]

    for cmpls in cmp_list:

        files = sorted(glob.glob("./logs/histories/*"+cmpls+".cpkl"))

        hists = []
        lables = []
        for f in files:
            hists.append(load_history(f))
            lable = f.split('_')[-3]
            lables.append(lable)

        plot_comparison_accu(hists, lables, cmpls)