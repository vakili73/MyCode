
TOP_K_ACCU = [1]  # , 5

__VERBOSE = 2
__EPOCHS = 9999

PATIENCE = 20
BATCHSIZE = 128
OPTIMIZER = 'adadelta'
FITGENOPTS = {
    'epochs': __EPOCHS,
    'verbose': __VERBOSE,
    'use_multiprocessing': True,
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

__SHOTS = [None]  # 50,

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
        "schema": 'V05',
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
    "mingnet": {
        "schema": 'V04',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
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
    "omniglot": {
        "schema": 'V01',
        "shots": [None],
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
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
        'loss': {
            'MSE': 'K-mean_squared_error',
            'MAE': 'K-mean_absolute_error',
            'MAPE': 'K-mean_absolute_percentage_error',
            'MSLE': 'K-mean_squared_logarithmic_error',
            'SHNG': 'K-squared_hinge',
            'HNG': 'K-hinge',
            # 'CHNG': 'K-categorical_hinge',
            'LCH': 'K-logcosh',
            'CRE': 'K-categorical_crossentropy',
            'KLD': 'K-kullback_leibler_divergence',
            'POS': 'K-poisson',
            'COS': 'K-cosine_proximity',
        },
        'metrics': ['K-acc'],
        'datagen': 'Original',
        'classification': '',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'ConventionalV2': {
    #     'loss': {'CRE': 'K-categorical_crossentropy'},
    #     'metrics': ['K-acc'],
    #     'datagen': 'Original',
    #     'classification': '',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'ConventionalV3': {
    #     'loss': {
    #         'MSE': 'K-mean_squared_error',
    #         'MAE': 'K-mean_absolute_error',
    #         'SHNG': 'K-squared_hinge',
    #         'HNG': 'K-hinge',
    #         'CHNG': 'K-categorical_hinge',
    #         'LCH': 'K-logcosh',
    #         'CRE': 'K-categorical_crossentropy',
    #         'KLD': 'K-kullback_leibler_divergence',
    #         'POS': 'K-poisson',
    #         'COS': 'K-cosine_proximity',
    #     },
    #     'metrics': ['K-acc'],
    #     'datagen': 'Original',
    #     'classification': '',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'MyModel': {
        'loss': {
            'MLCZ': 'L-my_loss_CZ',
            'MLCO': 'L-my_loss_CO',
            },
        'metrics': ['L-my_sacc'],
        'datagen': 'MySiameseCombined',
        'classification': 'getClfModel',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    'MyModelS': {
        'loss': {
            'MLSZ': 'L-my_loss_SZ',
            'MLSO': 'L-my_loss_SO',
            },
        'metrics': ['L-my_sacc'],
        'datagen': 'MySiamese',
        'classification': 'getClfModel',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    'MyModelT': {
        'loss': {
            'MLTZ': 'L-my_loss_TZ',
            'MLTO': 'L-my_loss_TO',
            },
        'metrics': ['L-my_accu'],
        'datagen': 'MyTriplet',
        'classification': 'getClfModel',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'MyModelV0': {
    #     'loss': {
    #         'MLV0': 'L-my_loss_v0',
    #         },
    #     'metrics': ['L-my_sacc'],
    #     'datagen': 'MySiamese',
    #     'classification': 'getClfModel',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV9': {
    #     'loss': {
    #         'MLV9': 'L-my_loss_v0',
    #         'MLV10': 'L-my_loss_v0_1',
    #         },
    #     'metrics': ['L-my_sacc'],
    #     'datagen': 'MySiamese',
    #     'classification': 'getClfModel',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV1': {
    #     'loss': {
    #         'MLV1': 'L-my_loss_v1',
    #         'MLV11': 'L-my_loss_v1_1',
    #         },
    #     'metrics': ['L-my_accu'],
    #     'datagen': 'MyTriplet',
    #     'classification': 'getClfModel',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV2': {
    # 'loss': {'MLV1': 'L-my_loss_v1'},
    #     'metrics': ['L-my_accu'],
    #     'datagen': 'MyTriplet',
    #     'classification': 'getClfModel',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV3': {
    # 'loss': {'MLV2': 'L-my_loss_v2'},
    #     'metrics': [],
    #     'datagen': 'MyTriplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV4': {
    # 'loss': {'MLV2': 'L-my_loss_v2'},
    #     'metrics': [],
    #     'datagen': 'MyTriplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV5': {
    #     'loss': {'MLV3': 'L-my_loss_v3'},
    #     'metrics': ['K-acc'],
    #     'datagen': 'Original',
    #     'classification': '',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV6': {
    #     'loss': {'MLV3': 'L-my_loss_v3'},
    #     'metrics': ['K-acc'],
    #     'datagen': 'Original',
    #     'classification': '',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV7': {
    #     'loss': {'MLV4': 'L-my_loss_v4'},
    #     'metrics': [],
    #     'datagen': 'MyTriplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'MyModelV8': {
    #     'loss': {
    #         'SMLV1': 'L-my_loss_v1',
    #         'SMLV11': 'L-my_loss_v1_1',
    #         },
    #     'metrics': ['L-my_accu'],
    #     'datagen': 'MyTriplet',
    #     'classification': 'getClfModel',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    # 'SiameseV1': {
    #     'loss': {'BCRE': 'K-binary_crossentropy'},
    #     'metrics': [],
    #     'datagen': 'SiameseV1',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'SiameseV2': {
        'loss': {'CONT': 'L-contrastive'},
        'metrics': [],
        'datagen': 'SiameseV2',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'TripletV1': {
    #     'loss': {'MSE': 'K-mean_squared_error'},
    #     'metrics': [],
    #     'datagen': 'Triplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'TripletV2': {
        'loss': {'TRIP': 'L-triplet'},
        'metrics': [],
        'datagen': 'Triplet',
        'knn': __KNNS,
        'svm': __SVMS,
    },
}
