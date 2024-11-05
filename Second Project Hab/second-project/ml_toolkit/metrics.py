import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, \
                            roc_auc_score, \
                            confusion_matrix, \
                            accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score


def calc_metrics(y_true, y_pred):
    metrics = {
        'report': classification_report(y_true, y_pred),
        'roc-auc': roc_auc_score(y_true, y_pred),
        'confusion': confusion_matrix(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': recall_score(np.logical_not(y_true), np.logical_not(y_pred))
    }

    print("Classification Report:")
    print(metrics['report'])
    print()
    print("ROC-AUC Score:", metrics['roc-auc'])
    print()
    print("Confusion Matrix:")
    print(metrics['confusion'])

    return metrics


def model_experiment(features, target, Model, model_args={}, oversample=0, undersample=0, scaling=None, validation=None):
    
    print("Beginning Experiment for Model:", Model)

    # TODO: Add validation (K-folds, hold-one out, etc.)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    if oversample:
        smote = SMOTE(sampling_strategy=oversample, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if undersample:
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_train, y_train = undersample.fit_resample(X_train, y_train)
    
    if scaling:
        scaler = scaling()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = Model(**model_args)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = calc_metrics(y_test, y_pred)

    print("-"*50)
    print()

    return model, metrics


def find_best_models(X, y, models, param_grids, metrics, refit="f1"):

    best_models = []

    if len(models) != len(param_grids):
        raise ValueError("Number of models must match number of param_grids. These must have the same length.")
    
    for model, params in zip(models, param_grids):

        grid_solver = GridSearchCV( estimator = model, 
                                    param_grid = params, 
                                    scoring = metrics,
                                    cv = 5,
                                    refit = refit,
                                    n_jobs = -1,
                                )

        model_result = grid_solver.fit(X, y)

        best_models.append(model_result)

    return best_models

def model_experiment_avg_metrics(features, target, Model, model_args={}, n_splits=5, oversample=0, undersample=0, scaling=None, validation=None):
    
    print("Beginning Experiment for Model:", Model)

    skfold = StratifiedKFold(n_splits = n_splits)
    y_test_real = []
    y_pred = []

    for train_index, test_index in skfold.split(features, target): 
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]

        if oversample:
            smote = SMOTE(sampling_strategy=oversample, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if undersample:
            undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            X_train, y_train = undersample.fit_resample(X_train, y_train)
        
        if scaling:
            scaler = scaling()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = Model(**model_args)

        model.fit(X_train, y_train)
        
        # Prediccion
        yhat = model.predict(X_test)
        y_pred.extend(yhat)
        
        # Valores reales
        y_test_real.extend(y_test)


    metrics = calc_metrics(y_test_real, y_pred)

    print("-"*50)
    print()

    return metrics


def find_best_models(X, y, models, param_grids, metrics, refit="f1"):

    best_models = []

    if len(models) != len(param_grids):
        raise ValueError("Number of models must match number of param_grids. These must have the same length.")
    
    for model, params in zip(models, param_grids):

        grid_solver = GridSearchCV( estimator = model, 
                                    param_grid = params, 
                                    scoring = metrics,
                                    cv = 5,
                                    refit = refit,
                                    n_jobs = -1,
                                )

        model_result = grid_solver.fit(X, y)

        best_models.append(model_result)

    return best_models