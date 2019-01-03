from tensorflow.keras import optimizers

TOP_K_ACCU = [1, 5]

__VERBOSE = 2
__EPOCHS = 9999

PATIENCE = 20
BATCHSIZE = 128
OPTIMIZER = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
FITGENOPTS = {
    'workers': 8,
    'epochs': __EPOCHS,
    'verbose': __VERBOSE,
    'max_queue_size': 80,
}
STEPS_PER = {
    'steps_per_epoch': 200,
    'validation_steps': 50,
}
FITOPTS = {
    'epochs': __EPOCHS,
    'verbose': __VERBOSE,
    'batch_size': BATCHSIZE,
}

__SHOTS = [50, None]

__DATAGEN_OPT_COLORED_IMAGE = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'channel_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
}

__DATAGEN_OPT_B_AND_W_IMAGE = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
}

DATASETS = {
    "cifar10": {
        "schema": 'V03',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "cifar100": {
        "schema": 'V03',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "fashion": {
        "schema": 'V01',
        "shots": __SHOTS,
        "dgen_opt": {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
        },
    },
    "homus": {
        "schema": 'V02',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    # "mingnet": {
    #     "schema": 'V04',
    #     "shots": __SHOTS,
    #     "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    # },
    "mnist": {
        "schema": 'V01',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    "nist": {
        "schema": 'V02',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    # "omniglot": {
    #     "schema": 'V01',
    #     "shots": [None],
    #     "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    # },
    "stl10": {
        "schema": 'V04',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "svhn": {
        "schema": 'V03',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
}

__N_JOBS = 8
__KNNS = [
    {'n_neighbors': 1,
     'weights': 'uniform',
     'n_jobs': __N_JOBS, },
    {'n_neighbors': 5,
     'weights': 'uniform',
     'n_jobs': __N_JOBS, },
    {'n_neighbors': 10,
     'weights': 'distance',
     'n_jobs': __N_JOBS, },
    {'n_neighbors': 15,
     'weights': 'distance',
     'n_jobs': __N_JOBS, },
]

__SVMS = [
    {'kernel': 'linear',
     'gamma': 'scale', },
    {'kernel': 'rbf',
     'gamma': 'scale', },
    # {'kernel': 'poly',
    #  'gamma': 'scale', },
    # {'kernel': 'sigmoid',
    #  'gamma': 'scale', },
]

METHODS = {
    'ConventionalV1': {
        'loss': 'K-categorical_crossentropy',
        'metrics': ['K-acc'],
        'datagen': 'Original',
        'classification': '',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'ConventionalV2': {
    #     'loss': 'K-categorical_crossentropy',
    #     'metrics': ['K-acc'],
    #     'datagen': 'Original',
    #     'classification': '',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'MyModelV1': {
        'loss': 'L-my_loss_v1',
        'metrics': ['L-my_accu'],
        'datagen': 'MyTriplet',
        'classification': 'getClfModel',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'MyModelV2': {
    #     'loss': 'L-my_loss_v1',
    #     'metrics': ['L-my_accu'],
    #     'datagen': 'MyTriplet',
    #     'classification': 'getClfModel',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV3': {
    #     'loss': 'L-my_loss_v2',
    #     'metrics': [],
    #     'datagen': 'MyTriplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV4': {
    #     'loss': 'L-my_loss_v2',
    #     'metrics': [],
    #     'datagen': 'MyTriplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'SiameseV1': {
        'loss': 'K-binary_crossentropy',
        'metrics': [],
        'datagen': 'SiameseV1',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    'SiameseV2': {
        'loss': 'L-contrastive',
        'metrics': [],
        'datagen': 'SiameseV2',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'TripletV1': {
    #     'loss': 'K-mean_squared_error',
    #     'metrics': [],
    #     'datagen': 'Triplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'TripletV2': {
        'loss': 'L-triplet',
        'metrics': [],
        'datagen': 'Triplet',
        'knn': __KNNS,
        'svm': __SVMS,
    },
}
