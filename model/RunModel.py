import utils.utils as ut
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def is_balance_needed(y_train):
    target_distribution = y_train.value_counts(normalize=True) * 100
    min_class_percentage = target_distribution.min()
    return min_class_percentage < 39

def balance_dataset(x_train, y_train, config):
    print(f'{ut.get_time_now()} :: Running model: Verifying if dataset balancing is necessary')

    if is_balance_needed(y_train):
        if config['dataset_balance'].get('undersampling', False):
            print(f'{ut.get_time_now()} :: Running model: Applying under sampling')
            x_train, y_train = random_under_sampling(x_train, y_train)
        elif config['dataset_balance'].get('smote', False):
            print(f'{ut.get_time_now()} :: Running model: Applying SMOTE')
            x_train, y_train = smote_sampling(x_train, y_train)
        else:
            print(f'{ut.get_time_now()} :: Running model: No balancing method has been specified')

    return x_train, y_train

def random_under_sampling(x_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def smote_sampling(x_train, y_train):
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def split_dataset(df):
    x = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

def needs_scaling(x_train, config):
    print(f'{ut.get_time_now()} :: Running model: Checking if scaling is needed')

    if config['applied_model'].get('name') in config['models_need_scaling']:
        # 🔹 Filtrar solo columnas numéricas para evitar errores con Timedelta
        numeric_columns = x_train.select_dtypes(include=[np.number])

        # 🔹 Verificar si quedan columnas después del filtrado
        if numeric_columns.empty:
            print(f'{ut.get_time_now()} :: Running model: No numeric columns in X_train,scaling will not be applied')
            return False

        # 🔹 Calcular estadísticas solo sobre datos numéricos
        ranges = numeric_columns.max() - numeric_columns.min()
        max_range = ranges.max()
        min_range = ranges.min()
        range_ratio = max_range / (min_range + 1e-6)  # Evitar división por cero

        # 🔹 Calcular desviación estándar y detectar outliers
        std_devs = numeric_columns.std()
        high_std_count = (std_devs > 10).sum()
        q1 = numeric_columns.quantile(0.25)
        q3 = numeric_columns.quantile(0.75)
        iqr = q3 - q1
        outliers_count = ((numeric_columns < (q1 - 1.5 * iqr)) | (numeric_columns > (q3 + 1.5 * iqr))).sum().sum()

        # 🔹 Decisión sobre escalado
        needs_scaling = (range_ratio > 100) or (high_std_count > 0) or (outliers_count > 0)

        print(f'{ut.get_time_now()} :: Running model: Is scaling needed analysis:')
        print(f'   - Max range: {max_range}, Min range: {min_range}, Ratio: {range_ratio:.2f}')
        print(f'   - Features with high STD (>10): {high_std_count}')
        print(f'   - Detected outliers: {outliers_count}')
        print(f"   - Used model: {config['applied_model'].get('name')}")
        print(f"   - Is scaling needed?: {'Yes' if needs_scaling else  'No'}")

        return needs_scaling

def apply_scaling(x_train, x_test, config):
    """
    Apply scaling to training and testing datasets based on the specified model
    in the configuration.

    This function determines whether scaling is required for the training (`x_train`)
    and testing (`x_test`) datasets based on the provided configuration (`config`).
    Depending on the model to be applied, it selects an appropriate scaler such as
    `StandardScaler` or `MinMaxScaler` to transform the datasets. If scaling is not
    required or the model does not dictate scaling, the datasets are returned without
    modification.

    Parameters:
    x_train (array-like): The training dataset to be scaled.
    x_test (array-like): The testing dataset to be scaled.
    config (dict): A dictionary containing configuration details, including the
                   model name under the key `applied_model`.

    Returns:
    Tuple[array-like, array-like]: A tuple containing the transformed training and
                                   testing datasets if scaling was applied, or the
                                   original datasets if scaling was not required.
    """
    print(f'{ut.get_time_now()} :: Running model: Scaling started')
    model_name = config['applied_model'].get('name')

    # Choosing the method
    if model_name in ["LogisticRegression", "svm", "pca", "neural_network"]:
        print(f'{ut.get_time_now()} :: Running model: Applying StandardScaler')
        scaler = StandardScaler()
    elif model_name in ["knn"]:
        print(f'{ut.get_time_now()} :: Running model: Applying MinMaxScaler')
        scaler = MinMaxScaler()
    else:
        print(f'{ut.get_time_now()} :: Running model: No scaler required')
        return x_train, x_test  # Si el modelo no requiere escalado, retornar sin cambios

    # 🔹 Verificar si 'datetime' existe y establecerlo como índice
    datetime_column = "datetime" if "datetime" in x_train.columns else None
    if datetime_column:
        x_train = x_train.set_index(datetime_column)
        x_test = x_test.set_index(datetime_column)

    # 🔹 Seleccionar solo columnas numéricas para escalar
    numeric_columns = x_train.select_dtypes(include=[np.number]).columns
    x_train_numeric = x_train[numeric_columns]
    x_test_numeric = x_test[numeric_columns]

    # 🔹 Asegurar que no haya NaNs antes del escalado
    x_train_numeric.fillna(x_train_numeric.median(), inplace=True)
    x_test_numeric.fillna(x_test_numeric.median(), inplace=True)

    # 🔹 Aplicar escalado solo a las columnas numéricas
    x_train_scaled = scaler.fit_transform(x_train_numeric)
    x_test_scaled = scaler.transform(x_test_numeric)

    # 🔹 Convertir de nuevo en DataFrame con los mismos índices y nombres de columnas
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=numeric_columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=numeric_columns, index=x_test.index)

    # 🔹 Restaurar las columnas no numéricas
    x_train_final = x_train_scaled.join(x_train.drop(columns=numeric_columns, errors='ignore'))
    x_test_final = x_test_scaled.join(x_test.drop(columns=numeric_columns, errors='ignore'))

    # 🔹 Restaurar 'datetime' como columna y ORDENARLO para evitar mezclas
    if datetime_column:
        x_train_final = x_train_final.reset_index().sort_values(by="datetime").reset_index(drop=True)
        x_test_final = x_test_final.reset_index().sort_values(by="datetime").reset_index(drop=True)

    # 🔹 Comprobar si hay NaNs y eliminarlos
    nan_count_train = x_train_final.isna().sum().sum()
    nan_count_test = x_test_final.isna().sum().sum()
    if nan_count_train > 0 or nan_count_test > 0:
        print(f"⚠️ Warning: Found {nan_count_train} NaNs in X_train and {nan_count_test} in X_test. Filling with median values.")
        x_train_final.fillna(x_train_final.median(), inplace=True)
        x_test_final.fillna(x_test_final.median(), inplace=True)

    print(f'{ut.get_time_now()} :: Running model: Scaling finished')

    return x_train_final, x_test_final

def train_and_evaluate(x_train, x_test, y_train, y_test, config):
    """
    Provides a function to train and evaluate a machine learning model based on a provided
    configuration. This function selects the model dynamically, trains it on the provided
    training dataset, makes predictions using the test dataset, and evaluates its performance.

    Arguments:
        x_train (Any): The feature set for training the model.
        x_test (Any): The feature set for testing the model.
        y_train (Any): The target values for training the model.
        y_test (Any): The target values for evaluating the model.
        config (dict): A dictionary containing configuration settings, including the model
                       name and specific hyperparameters for some classifier types.

    Returns:
        Any: The trained model instance.

    Raises:
        ValueError: If the model name provided in the configuration is not recognized.
    """
    model_name = config["applied_model"]["name"]
    print(f'{ut.get_time_now()} :: Running model: Training {model_name}')

    # 🔹 Filtrar solo columnas numéricas para evitar errores con timestamps
    x_train = x_train.select_dtypes(include=[np.number])
    x_test = x_test.select_dtypes(include=[np.number])

    # 🔹 Selección del modelo basado en `model_config.json`
    if model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=config["RandomForestClassifier"]["n_estimators"],
            max_depth=config["RandomForestClassifier"]["max_depth"],
            random_state=config["RandomForestClassifier"]["random_state"]
        )
    elif model_name == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError(f"❌ Model '{model_name}' not recognized in configuration.")

    # 🔹 Entrenar modelo
    model.fit(x_train, y_train)

    # 🔹 Hacer predicciones en el test set
    y_pred = model.predict(x_test)

    # 🔹 Evaluar el modelo
    print(f'{ut.get_time_now()} :: Running model: Evaluation Results:')
    print(f'🔹 Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'🔹 Classification Report:\n{classification_report(y_test, y_pred)}')
    print(f'🔹 Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

    return model


def run_model(df):
    print(f'{ut.get_time_now()} :: Running model: Starting running model')

    # Loading config
    config = ut.load_config('model_config')

    # Splitting train and test
    x_train, x_test, y_train, y_test = split_dataset(df)

    # Applying balance if it is necessary
    x_train, y_train = balance_dataset(x_train, y_train, config)

    if needs_scaling(x_train, config):
        x_train, x_test = apply_scaling(x_train, x_test, config)

    train_and_evaluate(x_train, x_test, y_train, y_test, config)

    return x_train, x_test, y_train, y_test





