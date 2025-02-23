import pandas as pd
import numpy as np
import utils.utils as ut
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from model.model_utils import is_balance_needed

def get_df_prediction(df):
    """
    Extracts a copy of the last row from the provided DataFrame, excluding the
    'target' column, which is dropped.

    This function is useful for preparing the last row of a DataFrame for
    predictive tasks by isolating it while neglecting the specified 'target'
    column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which the last row will be extracted and processed.

    Returns
    -------
    pandas.DataFrame
        A copy of the last row of the provided DataFrame, with the 'target'
        column removed.
    """
    df_prediction = df.iloc[[-1]].copy()  # Keep the last column
    df_prediction = df_prediction.drop(columns=['target'])  # set target to nun
    df_prediction.head()
    return df_prediction

def split_dataset(df):
    """
    Splits a dataset into training and testing subsets. The function takes a DataFrame,
    drops its last row, separates features (x) and labels (y) based on a specified
    'target' column, and then splits them into training and testing datasets.
    The training set contains 80% of the data, while the testing set contains
    20%. The split is stratified based on the target variable, and a fixed random
    state is used for reproducibility.

    Arguments:
        df (DataFrame): The input dataset that contains features and a 'target'
        column for labels.

    Returns:
        Tuple[DataFrame, DataFrame, Series, Series]: A tuple containing the training
        feature set, testing feature set, training labels, and testing labels.

    Raises:
        None
    """
    df = df.drop(df.index[-1]).copy()
    x = df.drop(columns=['target'])
    y = df['target']
    x_train, x_test, y_train, y_test=  train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    x_val, x_test, y_val, y_test=  train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    return x_train, x_test, x_val, y_train, y_test, y_val

def balance_dataset(x_train, y_train, config):
    """
    Balances the dataset based on the configuration settings provided.

    This function assesses whether balancing is necessary for the dataset using
    the `is_balance_needed` function. If balancing is required, it checks the
    configuration settings to determine the method of balancing. The available
    methods include under-sampling and SMOTE. If neither method is specified,
    the dataset is left unaltered. After balancing, the modified training data
    is returned.

    Parameters:
    x_train : Any
        The training features dataset to be used for the model.
    y_train : Any
        The training labels corresponding to the features.
    config : dict
        A configuration dictionary specifying dataset balancing preferences. It
        should include a 'dataset_balance' key with options such as 'under_sampling'
        and 'smote'.

    Returns:
    Tuple[Any, Any]
        A tuple containing the potentially modified `x_train` and `y_train` after
        balancing the dataset if applicable.

    Raises:
    None
    """
    print(f'{ut.get_time_now()} :: Running model: Verifying if dataset balancing is necessary')

    if is_balance_needed(y_train):
        print(f'{ut.get_time_now()} :: Running model: balancing is necessary')
        if config['dataset_balance'].get('under_sampling', False):
            print(f'{ut.get_time_now()} :: Running model: Applying under sampling')
            x_train, y_train = random_under_sampling(x_train, y_train)
        elif config['dataset_balance'].get('smote', False):
            print(f'{ut.get_time_now()} :: Running model: Applying SMOTE')
            x_train, y_train = smote_sampling(x_train, y_train)
        else:
            print(f'{ut.get_time_now()} :: Running model: No balancing method has been specified')
    else:
        print(f'{ut.get_time_now()} :: Running model: No balancing method has been applied')

    return x_train, y_train

def random_under_sampling(x_train, y_train):
    """
    Performs random under-sampling to balance the classes in the given training data.

    This function utilizes RandomUnderSampler from the imbalanced-learn library
    to reduce the number of samples in the majority class randomly. It ensures that
    each class in the training dataset has a balanced representation, which might
    improve the performance of classifiers sensitive to imbalanced data.

    Args:
        x_train: The feature set of the training data where under-sampling will
            be performed.
        y_train: The target labels corresponding to the training data feature set.

    Returns:
        A tuple containing the resampled feature set and the corresponding
        resampled target labels.

    Raises:
        This function does not explicitly raise any exceptions but may propagate
        exceptions raised by RandomUnderSampler.
    """
    rus = RandomUnderSampler(random_state=42)
    x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def smote_sampling(x_train, y_train):
    """
    Resamples the input training dataset to address class imbalance issues using the Synthetic Minority
    Oversampling Technique (SMOTE).

    Args:
        x_train: Training feature set.
        y_train: Training target set.

    Returns:
        A tuple containing the resampled feature and target sets:
            - x_train_resampled: Resampled training feature set.
            - y_train_resampled: Resampled training target set.
    """
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def apply_scaling(x_train, x_test, x_val, df_prediction, config):
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

    #Choosing the type of scaler
    if model_name in ["LogisticRegression", "svm", "pca", "neural_network"]:
        print(f'{ut.get_time_now()} :: Running model: Applying StandardScaler')
        scaler = StandardScaler()
    elif model_name in ["knn"]:
        print(f'{ut.get_time_now()} :: Running model: Applying MinMaxScaler')
        scaler = MinMaxScaler()
    else:
        print(f'{ut.get_time_now()} :: Running model: No scaler required')
        return x_train, x_test, df_prediction  # If no scaling is needed, return unchanged

    # Check if datetime exists and delete before scaling
    for df in [x_train, x_test, df_prediction]:
        df.drop(columns=['datetime'], errors='ignore', inplace=True)

    # Select numeric columns only
    numeric_columns = x_train.select_dtypes(include=[np.number]).columns
    x_train_numeric = x_train[numeric_columns]
    x_test_numeric = x_test[numeric_columns]
    x_val_numeric = x_val[numeric_columns]
    df_prediction_numeric = df_prediction[numeric_columns]

    # Replace NaN with median before scaling
    x_train_numeric.fillna(x_train_numeric.median(), inplace=True)
    x_test_numeric.fillna(x_test_numeric.median(), inplace=True)
    x_val_numeric.fillna(x_val_numeric.median(), inplace=True)
    df_prediction_numeric.fillna(df_prediction_numeric.median(), inplace=True)

    # Adjust the scaler only with x_train and apply it to the others.
    x_train_scaled = scaler.fit_transform(x_train_numeric)
    x_test_scaled = scaler.transform(x_test_numeric)
    x_val_scaled = scaler.transform(x_val_numeric)
    df_prediction_scaled = scaler.transform(df_prediction_numeric)

    # Convert back to DataFrame with the same indexes and column names
    x_train_final = pd.DataFrame(x_train_scaled, columns=numeric_columns, index=x_train.index)
    x_test_final = pd.DataFrame(x_test_scaled, columns=numeric_columns, index=x_test.index)
    x_val_final = pd.DataFrame(x_val_scaled, columns=numeric_columns, index=x_test.index)
    df_prediction_final = pd.DataFrame(df_prediction_scaled, columns=numeric_columns, index=df_prediction.index)

    # Restore non-numeric columns
    x_train_final = x_train_final.join(x_train.drop(columns=numeric_columns, errors='ignore'))
    x_test_final = x_test_final.join(x_test.drop(columns=numeric_columns, errors='ignore'))
    x_val_final = x_val_final.join(df_prediction.drop(columns=numeric_columns, errors='ignore'))
    df_prediction_final = df_prediction_final.join(df_prediction.drop(columns=numeric_columns, errors='ignore'))

    # Check and refill possible NaNs after scaling
    for df in [x_train_final, x_test_final, x_val_final, df_prediction_final]:
        df.fillna(df.median(), inplace=True)

    print(f'{ut.get_time_now()} :: Running model: Scaling finished')

    return x_train_final, x_test_final, x_val_final, df_prediction_final