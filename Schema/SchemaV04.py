
from .BaseSchema import BaseSchema

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import epsilon
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Sequential


class SchemaV04(BaseSchema):
    def __init__(self):
        super().__init__()
        pass

    def buildConventionalV1(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer = layers.Dense(128, activation='sigmoid')
        #  kernel_regularizer=l2(0.01))
        model.add(layer)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildConventionalV2(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='relu'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer = layers.Dense(128, activation='relu')
        #  kernel_regularizer=l2(0.01))
        model.add(layer)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildConventionalV3(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer = layers.Dense(128, activation='sigmoid')
        #  kernel_regularizer=l2(0.01))
        model.add(layer)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='sigmoid'))

        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildSiameseV1(self, shape, n_cls, distance='l1'):
        """
        The model used in [1]. Which uses the function of cross-entropy. It is assumed that 1 for the same and 0 for different images.

        [1] van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D. 
            (2015). Siamese Neural Networks for One-Shot Image Recognition.
            ICML - Deep Learning Workshop, 7(11), 956–963. 
            https://doi.org/10.1017/CBO9781107415324.004
        """
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.input = model.input
        self.output = model.output

        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)

        embedded_1 = model(input_1)
        embedded_2 = model(input_2)

        def output_shape(input_shape):
            return input_shape

        if distance == 'l1':
            distance_layer = layers.Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]),
                output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])
        elif distance == 'l2':
            distance_layer = layers.Lambda(
                lambda tensors: K.square(tensors[0] - tensors[1]),
                output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])

        prediction = layers.Dense(1, activation='sigmoid')(distance)
        #   kernel_regularizer=l2(0.01))(distance)

        self.model = Model(inputs=[input_1, input_2], outputs=prediction)
        return self

    def buildSiameseV2(self, shape, n_cls, distance='l2'):
        """
        Which uses the function of contrastive. It is assumed that 0 for the same and 1 for different images.

        [1] Hadsell R, Chopra S, LeCun Y. 
            Dimensionality reduction by learning an invariant mapping. 
            Innull 2006 Jun 17 (pp. 1735-1742). IEEE.
        """
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.input = model.input
        self.output = model.output

        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)

        embedded_1 = model(input_1)
        embedded_2 = model(input_2)

        def output_shape(input_shape):
            return input_shape[0], 1

        if distance == 'l1':
            distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])
        elif distance == 'l2':
            distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True) + epsilon()), output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])

        self.model = Model(inputs=[input_1, input_2], outputs=distance)
        return self

    def buildTripletV1(self, shape, n_cls, distance='l2'):
        """
        Hoffer, E., & Ailon, N. 
        (2015). Deep metric learning using triplet network. 
        Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9370(2010), 84–92. 
        https://doi.org/10.1007/978-3-319-24261-3_7
        """
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        embedded_a = model(input_a)
        embedded_p = model(input_p)
        embedded_n = model(input_n)

        def output_shape(input_shape):
            return input_shape[0], 1

        if distance == 'l1':
            pos_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])
        elif distance == 'l2':
            pos_distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True) + epsilon()), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True) + epsilon()), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])

        concat = layers.Concatenate(axis=-1)([pos_distance, neg_distance])
        softmax = layers.Activation('softmax')(concat)

        self.model = Model(inputs=[input_a, input_p, input_n], outputs=softmax)
        return self

    def buildTripletV2(self, shape, n_cls, distance='l2'):
        """
        Schroff, F., Kalenichenko, D., & Philbin, J. 
        (2015). FaceNet: A unified embedding for face recognition and clustering. 
        In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 07–12–June, pp. 815–823). 
        https://doi.org/10.1109/CVPR.2015.7298682
        """
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        embedded_a = model(input_a)
        embedded_p = model(input_p)
        embedded_n = model(input_n)

        def output_shape(input_shape):
            return input_shape[0], 1

        if distance == 'l1':
            pos_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])
        elif distance == 'l2':
            pos_distance_layer = layers.Lambda(
                lambda tensors:
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors:
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])

        concat = layers.Concatenate(axis=-1)([pos_distance, neg_distance])

        self.model = Model(inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def getClfModel(self):
        return Model(self.input, self.clf_out)

    def buildMyModel(self, shape, n_cls):
        return self.buildMyModelV0(shape, n_cls)

    def buildMyModelS(self, shape, n_cls):
        return self.buildMyModelV0(shape, n_cls)

    def buildMyModelV0(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer01 = model.output
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.e_len = [128]
        self.output = layer01
        self.input = model.input
        self.clf_out = model.output

        input_a = layers.Input(shape=shape)
        input_o = layers.Input(shape=shape)

        embed_model = Model(inputs=model.input, outputs=layer01)
        embed_a = embed_model(input_a)
        embed_o = embed_model(input_o)

        output_a = model(input_a)
        output_o = model(input_o)

        concat = layers.Concatenate()(
            [embed_a, embed_o,
             output_a, output_o])

        self.model = Model(
            inputs=[input_a, input_o], outputs=concat)
        return self

    def buildMyModelV1(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer02 = model.output
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer01 = model.output
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.e_len = [512, 128]
        self.output = layer01
        self.input = model.input
        self.clf_out = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        layer02_model = Model(inputs=model.input, outputs=layer02)
        layer02_a = layer02_model(input_a)
        layer02_p = layer02_model(input_p)
        layer02_n = layer02_model(input_n)

        embed_model = Model(inputs=model.input, outputs=layer01)
        embed_a = embed_model(input_a)
        embed_p = embed_model(input_p)
        embed_n = embed_model(input_n)

        output_a = model(input_a)
        output_p = model(input_p)
        output_n = model(input_n)

        concat = layers.Concatenate()(
            [layer02_a, layer02_p, layer02_n,
             embed_a, embed_p, embed_n,
             output_a, output_p, output_n])

        self.model = Model(
            inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def buildMyModelV2(self, shape, n_cls):
        model = Sequential()
        model.add(layers.Conv2D(32, (5, 5), padding='same', input_shape=shape))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        layer05 = layers.Dense(128, activation='sigmoid')(
            layers.Flatten()(model.output))
        model.add(layers.Conv2D(64, (5, 5), padding='same'))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        layer04 = layers.Dense(128, activation='sigmoid')(
            layers.Flatten()(model.output))
        model.add(layers.Conv2D(32, (5, 5), padding='same'))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        layer03 = layers.Dense(128, activation='sigmoid')(model.output)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer02 = model.output
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer01 = model.output
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.e_len = [128, 128, 128, 512, 128]
        self.output = layer01
        self.input = model.input
        self.clf_out = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        layer05_model = Model(inputs=model.input, outputs=layer05)
        layer05_a = layer05_model(input_a)
        layer05_p = layer05_model(input_p)
        layer05_n = layer05_model(input_n)

        layer04_model = Model(inputs=model.input, outputs=layer04)
        layer04_a = layer04_model(input_a)
        layer04_p = layer04_model(input_p)
        layer04_n = layer04_model(input_n)

        layer03_model = Model(inputs=model.input, outputs=layer03)
        layer03_a = layer03_model(input_a)
        layer03_p = layer03_model(input_p)
        layer03_n = layer03_model(input_n)

        layer02_model = Model(inputs=model.input, outputs=layer02)
        layer02_a = layer02_model(input_a)
        layer02_p = layer02_model(input_p)
        layer02_n = layer02_model(input_n)

        embed_model = Model(inputs=model.input, outputs=layer01)
        embed_a = embed_model(input_a)
        embed_p = embed_model(input_p)
        embed_n = embed_model(input_n)

        output_a = model(input_a)
        output_p = model(input_p)
        output_n = model(input_n)

        concat = layers.Concatenate()(
            [layer05_a, layer05_p, layer05_n,
             layer04_a, layer04_p, layer04_n,
             layer03_a, layer03_p, layer03_n,
             layer02_a, layer02_p, layer02_n,
             embed_a, embed_p, embed_n,
             output_a, output_p, output_n])

        self.model = Model(
            inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def buildMyModelV3(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer02 = model.output
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.e_len = [512, 128]
        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        layer02_model = Model(inputs=model.input, outputs=layer02)
        layer02_a = layer02_model(input_a)
        layer02_p = layer02_model(input_p)
        layer02_n = layer02_model(input_n)

        embed_model = Model(inputs=model.input, outputs=model.output)
        embed_a = embed_model(input_a)
        embed_p = embed_model(input_p)
        embed_n = embed_model(input_n)

        concat = layers.Concatenate()(
            [layer02_a, layer02_p, layer02_n,
             embed_a, embed_p, embed_n])

        self.model = Model(
            inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def buildMyModelV4(self, shape, n_cls):
        model = Sequential()
        model.add(layers.Conv2D(32, (5, 5), padding='same', input_shape=shape))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        layer05 = layers.Dense(128, activation='sigmoid')(
            layers.Flatten()(model.output))
        model.add(layers.Conv2D(64, (5, 5), padding='same'))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        layer04 = layers.Dense(128, activation='sigmoid')(
            layers.Flatten()(model.output))
        model.add(layers.Conv2D(32, (5, 5), padding='same'))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        layer03 = layers.Dense(128, activation='sigmoid')(model.output)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer02 = model.output
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.e_len = [128, 128, 128, 512, 128]
        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        layer05_model = Model(inputs=model.input, outputs=layer05)
        layer05_a = layer05_model(input_a)
        layer05_p = layer05_model(input_p)
        layer05_n = layer05_model(input_n)

        layer04_model = Model(inputs=model.input, outputs=layer04)
        layer04_a = layer04_model(input_a)
        layer04_p = layer04_model(input_p)
        layer04_n = layer04_model(input_n)

        layer03_model = Model(inputs=model.input, outputs=layer03)
        layer03_a = layer03_model(input_a)
        layer03_p = layer03_model(input_p)
        layer03_n = layer03_model(input_n)

        layer02_model = Model(inputs=model.input, outputs=layer02)
        layer02_a = layer02_model(input_a)
        layer02_p = layer02_model(input_p)
        layer02_n = layer02_model(input_n)

        embed_model = Model(inputs=model.input, outputs=model.output)
        embed_a = embed_model(input_a)
        embed_p = embed_model(input_p)
        embed_n = embed_model(input_n)

        concat = layers.Concatenate()(
            [layer05_a, layer05_p, layer05_n,
             layer04_a, layer04_p, layer04_n,
             layer03_a, layer03_p, layer03_n,
             layer02_a, layer02_p, layer02_n,
             embed_a, embed_p, embed_n])

        self.model = Model(
            inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def buildMyModelV5(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer = layers.Dense(128, activation='sigmoid')
        #  kernel_regularizer=l2(0.01))
        model.add(layer)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildMyModelV6(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer = layers.Dense(128, activation='sigmoid')
        #  kernel_regularizer=l2(0.01))
        model.add(layer)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='sigmoid'))

        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildMyModelV7(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))

        self.e_len = 128
        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        embed_model = Model(inputs=model.input, outputs=model.output)
        embed_a = embed_model(input_a)
        embed_p = embed_model(input_p)
        embed_n = embed_model(input_n)

        concat = layers.Concatenate()(
            [embed_a, embed_p, embed_n])

        self.model = Model(
            inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def buildMyModelT(self, shape, n_cls):
        return self.buildMyModelV8(shape, n_cls)

    def buildMyModelV8(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer01 = model.output
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.e_len = [128]
        self.output = layer01
        self.input = model.input
        self.clf_out = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        embed_model = Model(inputs=model.input, outputs=layer01)
        embed_a = embed_model(input_a)
        embed_p = embed_model(input_p)
        embed_n = embed_model(input_n)

        output_a = model(input_a)
        output_p = model(input_p)
        output_n = model(input_n)

        concat = layers.Concatenate()(
            [embed_a, embed_p, embed_n,
             output_a, output_p, output_n])

        self.model = Model(
            inputs=[input_a, input_p, input_n], outputs=concat)
        return self

    def buildMyModelV9(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(512, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer02 = model.output
        model.add(layers.Dense(128, activation='sigmoid'))
        #    kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))
        layer01 = model.output
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.e_len = [512, 128]
        self.output = layer01
        self.input = model.input
        self.clf_out = model.output

        input_a = layers.Input(shape=shape)
        input_o = layers.Input(shape=shape)

        layer02_model = Model(inputs=model.input, outputs=layer02)
        layer02_a = layer02_model(input_a)
        layer02_o = layer02_model(input_o)

        embed_model = Model(inputs=model.input, outputs=layer01)
        embed_a = embed_model(input_a)
        embed_o = embed_model(input_o)

        output_a = model(input_a)
        output_o = model(input_o)

        concat = layers.Concatenate()(
            [layer02_a, layer02_o,
             embed_a, embed_o,
             output_a, output_o])

        self.model = Model(
            inputs=[input_a, input_o], outputs=concat)
        return self

    def build(self, shape):
        """
        [1] Designed by the experimental result and LeNet-5[3] inspiration

        [2] https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

        [3] Cun, Y. L., Bottou, L., Bengio, Y., & Haffiner, P. 
            (1998). Gradient based learning applied to document recognition. 
            Proceedings of IEEE, 86(11), 86(11):2278-2324.
        """
        model = Sequential()
        model.add(layers.Conv2D(32, (5, 5), padding='same', input_shape=shape))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (5, 5), padding='same'))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(32, (5, 5), padding='same'))
        # kernel_regularizer=l2(0.01)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        return model

    pass
