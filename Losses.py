import Metrics
import tensorflow as tf

from tensorflow.keras import backend as K


# %% Losses function

def cross_entropy(**kwargs):
    """
    van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D.
    (2015). Siamese Neural Networks for One-Shot Image Recognition.
    ICML - Deep Learning Workshop, 7(11), 956–963. 
    https://doi.org/10.1017/CBO9781107415324.004
    """
    def _loss(y_true, y_pred):
        loss = -(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
        return loss

    return _loss


def contrastive(margin=1.25, **kwargs):
    """
    Hadsell, R., Chopra, S., & LeCun, Y. 
    (2006). Dimensionality reduction by learning an invariant mapping. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 1735–1742). 
    https://doi.org/10.1109/CVPR.2006.100
    """
    def _loss(y_true, y_pred):
        loss = (1 - y_true) * K.square(y_pred) + y_true * \
            K.square(K.maximum(0.0, margin - y_pred))
        return loss

    return _loss


def triplet(alpha=0.2, **kwargs):
    """
    Schroff, F., Kalenichenko, D., & Philbin, J. 
    (2015). FaceNet: A unified embedding for face recognition and clustering. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 07–12–June, pp. 815–823). 
    https://doi.org/10.1109/CVPR.2015.7298682
    """
    def _loss(y_true, y_pred):
        pos_dist = y_pred[:, 0]
        neg_dist = y_pred[:, 1]
        loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
        return loss

    return _loss


def my_loss_v1(**kwargs):
    n_cls = kwargs['n_cls']
    e_len = kwargs['e_len']

    def _loss(y_true, y_pred):
        embeds_apn = []
        _len = 0
        for i in range(len(e_len)):
            embed_a = y_pred[:, _len:(_len+e_len[i])]
            embed_p = y_pred[:, (_len+e_len[i]):(_len+(e_len[i]*2))]
            embed_n = y_pred[:, (_len+(e_len[i]*2)):(_len+(e_len[i]*3))]
            embeds_apn.append((embed_a, embed_p, embed_n))
            _len += e_len[i]*3

        output_a = y_pred[:, _len:(_len+n_cls)]
        output_p = y_pred[:, (_len+n_cls):(_len+(n_cls*2))]
        output_n = y_pred[:, (_len+(n_cls*2)):(_len+(n_cls*3))]

        true_a = y_true[:, :n_cls]
        true_p = y_true[:, n_cls:(n_cls*2)]
        true_n = y_true[:, (n_cls*2):(n_cls*3)]

        def __loss(anc, pos, neg):

            pos_dist_l2 = Metrics.squared_l2_distance(anc, pos)
            neg_dist_l2 = Metrics.squared_l2_distance(anc, neg)

            """
            Symmetrised Kullback and Leibler
            Kullback, S.; Leibler, R.A. (1951).
            "On information and sufficiency".
            Annals of Mathematical Statistics. 22 (1): 79–86.
            doi:10.1214/aoms/1177729694. MR 0039968.
            """
            pos_dist_kl = Metrics.kullback_leibler(anc, pos) +\
                Metrics.kullback_leibler(pos, anc)
            neg_dist_kl = Metrics.kullback_leibler(anc, neg) +\
                Metrics.kullback_leibler(neg, anc)

            """
            Squared Jensen-Shannon distance
            Endres, D. M.; J. E. Schindelin (2003).
            "A new metric for probability distributions".
            IEEE Trans. Inf. Theory. 49 (7): 1858–1860.
            doi:10.1109/TIT.2003.813506.
            """
            # pos_dist_js = Metrics.jensen_shannon(anc, pos)
            # neg_dist_js = Metrics.jensen_shannon(anc, neg)

            """
            Squared Hellinger distance
            Nikulin, M.S.
            (2001) [1994], "Hellinger distance"
            in Hazewinkel, Michiel, Encyclopedia of Mathematics, Springer Science+Business Media B.V.
            Kluwer Academic Publishers, ISBN 978-1-55608-010-4
            """
            # pos_dist_hl = Metrics.squared_hellinger(anc, pos)
            # neg_dist_hl = Metrics.squared_hellinger(anc, neg)

            # pos_dist = K.tanh(pos_dist_kl)
            # neg_dist = K.tanh(neg_dist_kl)

            _loss = \
                - ((K.tanh(pos_dist_l2))*K.log(K.maximum(K.tanh(pos_dist_l2), K.epsilon())) +
                   (K.tanh(neg_dist_l2))*K.log(K.maximum(K.tanh(neg_dist_l2), K.epsilon())))\
                - ((K.tanh(pos_dist_kl))*K.log(K.maximum(K.tanh(pos_dist_kl), K.epsilon())) +
                   (K.tanh(neg_dist_kl))*K.log(K.maximum(K.tanh(neg_dist_kl), K.epsilon())))
            return _loss

        loss = 0
        for i in range(len(e_len)):
            loss += __loss(*embeds_apn[i])
        loss += \
            Metrics.cross_entropy(true_a, output_a) +\
            Metrics.cross_entropy(true_p, output_p) +\
            Metrics.cross_entropy(true_n, output_n)
        return loss

    return _loss


def my_loss_v2(**kwargs):
    e_len = kwargs['e_len']

    def _loss(y_true, y_pred):
        embeds_apn = []
        _len = 0
        for i in range(len(e_len)):
            embed_a = y_pred[:, _len:(_len+e_len[i])]
            embed_p = y_pred[:, (_len+e_len[i]):(_len+(e_len[i]*2))]
            embed_n = y_pred[:, (_len+(e_len[i]*2)):(_len+(e_len[i]*3))]
            embeds_apn.append((embed_a, embed_p, embed_n))
            _len += e_len[i]*3

        def __loss(anc, pos, neg):

            pos_dist_l2 = Metrics.squared_l2_distance(anc, pos)
            neg_dist_l2 = Metrics.squared_l2_distance(anc, neg)

            """
            Symmetrised Kullback and Leibler
            Kullback, S.; Leibler, R.A. (1951).
            "On information and sufficiency".
            Annals of Mathematical Statistics. 22 (1): 79–86.
            doi:10.1214/aoms/1177729694. MR 0039968.
            """
            pos_dist_kl = Metrics.kullback_leibler(anc, pos) +\
                Metrics.kullback_leibler(pos, anc)
            neg_dist_kl = Metrics.kullback_leibler(anc, neg) +\
                Metrics.kullback_leibler(neg, anc)

            """
            Squared Jensen-Shannon distance
            Endres, D. M.; J. E. Schindelin (2003).
            "A new metric for probability distributions".
            IEEE Trans. Inf. Theory. 49 (7): 1858–1860.
            doi:10.1109/TIT.2003.813506.
            """
            # pos_dist_js = Metrics.jensen_shannon(anc, pos)
            # neg_dist_js = Metrics.jensen_shannon(anc, neg)

            """
            Squared Hellinger distance
            Nikulin, M.S.
            (2001) [1994], "Hellinger distance"
            in Hazewinkel, Michiel, Encyclopedia of Mathematics, Springer Science+Business Media B.V.
            Kluwer Academic Publishers, ISBN 978-1-55608-010-4
            """
            # pos_dist_hl = Metrics.squared_hellinger(anc, pos)
            # neg_dist_hl = Metrics.squared_hellinger(anc, neg)

            # pos_dist = K.tanh(pos_dist_kl)
            # neg_dist = K.tanh(neg_dist_kl)

            _loss = \
                - ((K.tanh(pos_dist_l2))*K.log(K.maximum(K.tanh(pos_dist_l2), K.epsilon())) +
                   (K.tanh(neg_dist_l2))*K.log(K.maximum(K.tanh(neg_dist_l2), K.epsilon())))\
                - ((K.tanh(pos_dist_kl))*K.log(K.maximum(K.tanh(pos_dist_kl), K.epsilon())) +
                   (K.tanh(neg_dist_kl))*K.log(K.maximum(K.tanh(neg_dist_kl), K.epsilon())))
            return _loss

        loss = 0
        for i in range(len(e_len)):
            loss += __loss(*embeds_apn[i])
        return loss

    return _loss


def my_loss_v3(**kwargs):

    def _loss(y_true, y_pred):
        m = 0.5 * (y_true + y_pred)
        return Metrics.kullback_leibler(y_pred, m) +\
            Metrics.kullback_leibler(m, y_pred)

    return _loss


def my_loss_v4(**kwargs):
    e_len = kwargs['e_len']

    def _loss(y_true, y_pred):
        output_a = y_pred[:, :(e_len)]
        output_p = y_pred[:, (e_len):(e_len*2)]
        output_n = y_pred[:, (e_len*2):(e_len*3)]

        loss = \
            K.sqrt(Metrics.jensen_shannon(output_a, output_p)) +\
            -K.log(K.tanh(K.sqrt(Metrics.jensen_shannon(output_a, output_n))))
        return loss

    return _loss
