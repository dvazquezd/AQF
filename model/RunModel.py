import utils.utils as ut
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def is_balance_needed(y_train):
    """
    """
    target_distribution = y_train.value_counts(normalize=True) * 100
    min_class_percentage = target_distribution.min()
    return min_class_percentage < 39

def balance_dataset(x_train, y_train, config):
    """
    """
    print(f'{ut.get_time_now()} :: Running model: Verifying if dataset balancing is necessary')

    if is_balance_needed(y_train):
        if config['dataset_balance'].get('undersampling', False):
            print(f'{ut.get_time_now()} :: Running model: Applying undersampling')
            x_train, y_train = random_under_sampling(x_train, y_train)
        elif config['dataset_balance'].get('smote', False):
            print(f'{ut.get_time_now()} :: Running model: Applying SMOTE')
            x_train, y_train = smote_sampling(x_train, y_train)
        else:
            print(f'{ut.get_time_now()} :: Running model: No balancing method has been specified')

    return x_train, y_train

def random_under_sampling(x_train, y_train):
    """
    """
    rus = RandomUnderSampler(random_state=42)
    x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def smote_sampling(x_train, y_train):
    """
    """
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def split_dataset(df):
    """
    """
    x = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

def run_model(df):
    """
    """
    print(f'{ut.get_time_now()} :: Running model: Starting running model')

    # Loading config
    config = ut.load_config('model_config')

    # Splitting train and test
    x_train, x_test, y_train, y_test = split_dataset(df)

    # Applying balance if it is necessary
    x_train, y_train = balance_dataset(x_train, y_train, config)

    return x_train, x_test, y_train, y_test





