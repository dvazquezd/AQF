import utils.utils as ut
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve


def is_balance_needed(y_train):
    """
    Determines if balancing of class distribution in a dataset is needed.

    This function calculates the distribution of target labels in the given
    dataset and checks if the minimum percentage of any class is below 39%.
    If so, it indicates that balancing may be required.

    Args:
        y_train (pd.Series): The target variable containing class labels.

    Returns:
        bool: True if balancing is needed (minimum class percentage is below
        X%), otherwise False.
    """
    target_distribution = y_train.value_counts(normalize=True) * 100
    min_class_percentage = target_distribution.min()
    return min_class_percentage < 10

def needs_scaling(x_train, config):
    """
    Checks if scaling is required for the given training dataset based on characteristics of numerical columns,
    such as data range, standard deviation, and presence of outliers. The decision is influenced by the
    model requirements specified in the configuration.

    Parameters:
        x_train (DataFrame): Input training dataset.
        config (dict): Configuration dictionary containing applied model details and scaling requirements.

    Returns:
        bool: True if scaling is needed, False otherwise.

    Raises:
        None
    """
    print(f'{ut.get_time_now()} :: Running model: Checking if scaling is needed')

    if config['applied_model'].get('name') in config['models_need_scaling']:
        #Filtering only numeric columns to avoid errors with Timedelta
        numeric_columns = x_train.select_dtypes(include=[np.number])

        #Check if columns are left after filtering.
        if numeric_columns.empty:
            print(f'{ut.get_time_now()} :: Running model: No numeric columns in X_train,scaling will not be applied')
            return False

        #Calculate statistics on numerical data only
        ranges = numeric_columns.max() - numeric_columns.min()
        max_range = ranges.max()
        min_range = ranges.min()
        range_ratio = max_range / (min_range + 1e-6)  #Avoiding 0 division

        #Calculate standard deviation and detect outliers
        std_devs = numeric_columns.std()
        high_std_count = (std_devs > 10).sum()
        q1 = numeric_columns.quantile(0.25)
        q3 = numeric_columns.quantile(0.75)
        iqr = q3 - q1
        outliers_count = ((numeric_columns < (q1 - 1.5 * iqr)) | (numeric_columns > (q3 + 1.5 * iqr))).sum().sum()

        #Decision on scaling
        is_scaling_needed = (range_ratio > 100) or (high_std_count > 0) or (outliers_count > 0)

        print(f'{ut.get_time_now()} :: Running model: Is scaling needed analysis:')
        print(f'{ut.get_time_now()} :: Running model: Max range: {max_range}, Min range: {min_range}, Ratio: {range_ratio:.2f}')
        print(f'{ut.get_time_now()} :: Running model: Features with high STD (>10): {high_std_count}')
        print(f'{ut.get_time_now()} :: Running model: Detected outliers: {outliers_count}')
        print(f"{ut.get_time_now()} :: Running model: Used model: {config['applied_model'].get('name')}")
        print(f"{ut.get_time_now()} :: Running model: Is scaling needed?: {'Yes' if is_scaling_needed else  'No'}")

        return is_scaling_needed

    else:
        print(f'{ut.get_time_now()} :: Running model: Is scaling needed?: No')

def evaluate_model(x, y_prediction, dataset_name="testing"):
    """
    Evaluates the model with the data provided and prints the results.

    Parameters:
    - y: Actual labels of the dataset.
    - dataset_name: Name of the dataset to print in the evaluation (“testing” or “training”)
    """
    print(f'\nEvaluating the model based on {dataset_name}')
    print(f'{ut.get_time_now()} :: Running model: Evaluation Results:')
    print(f'Accuracy: {accuracy_score(x, y_prediction):.4f}')
    print(f'Classification Report:\n{classification_report(x, y_prediction)}')
    print(f'Confusion Matrix:\n{confusion_matrix(x, y_prediction)}')

def best_threshold(y_true, y_scores):
    """
    Determine the optimal threshold for classification based on the Youden J statistic.

    The function computes the Receiver Operating Characteristic (ROC) curve using provided true labels
    and prediction scores. Then, it calculates the Youden J statistic for each threshold, which helps
    to evaluate the effectiveness of a binary classification test. Finally, the function identifies and
    returns the threshold value that maximizes the Youden J statistic, aiding in determining the optimal
    balance between sensitivity and specificity.

    Parameters:
        y_true:
            The ground truth binary labels for the dataset.
        y_scores:
            The predicted scores or probabilities associated with the positive class.

    Returns:
        float:
            The threshold value that yields the maximum Youden J statistic.
    """
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculating Youden for each threshold
    youden_index = tpr - fpr

    # Threshold that maximizes Youden
    best_idx = np.argmax(youden_index)

    return thresholds[best_idx]