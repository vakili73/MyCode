import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class MySiamese(Sequence):

    def __init__(self, x_set, y_set, n_cls, batch_size=128):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = [np.where(self.y == i)[0] for i in range(n_cls)]
        self.min_len = [self.indices[i].size for i in range(n_cls)]
        self.n_cls = n_cls

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = []
        for _ in range(self.batch_size):
            anchor = np.random.permutation(range(self.n_cls))[0]
            index = np.random.randint(self.min_len[anchor], size=2)
            batch.append((self.x[self.indices[anchor][index[0]]],
                          self.x[self.indices[anchor][index[1]]],
                          to_categorical(anchor, self.n_cls),
                          to_categorical(anchor, self.n_cls)))
        in_1, in_2, out_1, out_2 = zip(*batch)
        return [np.stack(in_1), np.stack(in_2)], np.concatenate([
            np.stack(out_1), np.stack(out_2)], axis=-1)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
