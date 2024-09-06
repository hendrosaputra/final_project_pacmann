import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from ml_from_scratch.tree import DecisionTreeClassifier
from ml_from_scratch.metrics import accuracy_score

def read_dataset(path):
  df = pd.read_csv(path)
  #df.set_index('id')

  print('Data Shape: ', df.shape)
  return df

def split_input_output(df, feature_cols, target_col):
  df = df.copy()
  x = df[feature_cols]
  y = df[target_col]
  print('X Shape:', x.shape)
  print('y Shape:', y.shape)
  return x, y


def split_train_test(x, y, test_size, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state
    )
    print('X Train Shape:', x_train.shape)
    print('y Train Shape:', y_train.shape)
    print('X Test Shape:', x_test.shape)
    print('y Test Shape:', y_test.shape)
    return x_train, x_test, y_train, y_test

def fit_scaler(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler

def transform_scaler(x, scaler):
    x_clean = pd.DataFrame(
        scaler.transform(x),
        index = x.index,
        columns = x.columns
    )
    return x_clean

def categorize_column(df, column_name, category_zero_values, new_column_name=None):
    """
    Mengkategorikan kolom menjadi biner, di mana 0 untuk nilai yang sesuai dengan category_zero_values, 
    dan 1 untuk nilai lainnya.

    Parameters:
    df (pd.DataFrame): DataFrame yang mengandung data.
    column_name (str): Nama kolom yang ingin dikategorikan.
    category_zero_values (list): Daftar nilai yang ingin dikategorikan sebagai 0.
    new_column_name (str, optional): Nama kolom baru untuk hasil yang dikategorikan. 
                                     Jika None, akan menggantikan kolom asli.

    Returns:
    pd.DataFrame: DataFrame dengan kolom yang sudah dikategorikan menjadi biner.
    """
    # Jika nama kolom baru tidak diberikan, gunakan nama kolom yang asli
    if new_column_name is None:
        new_column_name = column_name
    
    # Mengkategorikan nilai-nilai sesuai dengan kriteria yang diberikan
    df[new_column_name] = df[column_name].apply(lambda x: 0 if x in category_zero_values else 1)
    
    return df
    

def one_hot_encode_column(df, column_name, prefix=None):
    """
    Melakukan One-Hot Encoding pada kolom tertentu dalam DataFrame, dan mengonversi hasil menjadi 1 dan 0.

    Parameters:
    df (pd.DataFrame): DataFrame yang mengandung data.
    column_name (str): Nama kolom yang akan di-encode.
    prefix (str, optional): Prefix untuk nama kolom hasil encoding. 
                            Jika None, akan menggunakan nama kolom asli.

    Returns:
    pd.DataFrame: DataFrame dengan kolom hasil One-Hot Encoding sebagai 1 dan 0.
    """
    # Jika prefix tidak diberikan, gunakan nama kolom asli sebagai prefix
    if prefix is None:
        prefix = column_name
    
    # Melakukan One-Hot Encoding pada kolom yang dipilih
    df_encoded = pd.get_dummies(df, columns=[column_name], prefix=prefix, dtype=int, drop_first=True)
    
    return df_encoded

def label_encode_target(y):
    """
    Melakukan Label Encoding pada target yang sudah dipisahkan dari DataFrame.

    Parameters:
    y (pd.Series or np.array): Kolom target yang akan di-encode.

    Returns:
    y_encoded (np.array): Target yang sudah di-encode.
    label_encoder (LabelEncoder): Objek LabelEncoder yang bisa digunakan untuk decoding jika diperlukan.
    """
    # Inisialisasi LabelEncoder
    label_encoder = LabelEncoder()

    # Melakukan encoding pada target
    y_encoded = label_encoder.fit_transform(y)

    return y_encoded, label_encoder

def undersample_data(X, y, sampling_strategy='auto', random_state=42):
    """
    Melakukan undersampling pada dataset untuk menyeimbangkan kelas.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Data fitur (independen variabel).
    
    y : pd.Series or np.ndarray
        Data target (label atau kelas).

    sampling_strategy : str, dict, or float, default='auto'
        Strategi sampling untuk menyeimbangkan kelas. Bisa berupa:
        - 'auto': Menyimbangkan secara otomatis
        - float: Menentukan rasio antara minoritas dan mayoritas.
        - dict: Menentukan jumlah sampel untuk tiap kelas.
    
    random_state : int, default=42
        Seed untuk pengacakan.

    Returns:
    --------
    X_resampled : pd.DataFrame or np.ndarray
        Data fitur setelah undersampling.
    
    y_resampled : pd.Series or np.ndarray
        Data target setelah undersampling.
    """
    
    # Inisialisasi RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    
    # Fit dan resample dataset
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    return X_resampled, y_resampled

# Read Dataset
dataset = read_dataset('data/Air_Plane_Passenger_Data.csv')

# Data Preprocessing
dataset['Arrival Delay in Minutes'].fillna(dataset['Arrival Delay in Minutes'].mean(), inplace=True)

feature_cols = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
                'Flight Distance', 'Inflight wifi service',
                'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                'Inflight entertainment', 'On-board service', 'Leg room service',
                'Baggage handling', 'Checkin service', 'Inflight service',
                'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
target_col = 'satisfaction'

X, y = split_input_output(dataset, feature_cols=feature_cols, target_col=target_col)
X_resampled, y_resampled = undersample_data(X, y)

y_encoded, label_encoder = label_encode_target(y_resampled)
X_num = X_resampled.select_dtypes(include=['number'])
X_cat = X_resampled.select_dtypes(include=['object'])

for col in X_cat.columns:
    if X_cat[col].nunique() == 2:
        # Untuk kolom dengan 2 nilai unik, gunakan categorize_column
        X_cat = categorize_column(X_cat, col, category_zero_values=[X_cat[col].unique()[0]])
    elif X_cat[col].nunique() == 3:
        # Untuk kolom dengan 3 nilai unik, gunakan one_hot_encode_column
        X_cat = one_hot_encode_column(X_cat, col)

X_prepro = pd.concat([X_num, X_cat], axis=1)

X_train, X_test, y_train, y_test = split_train_test(X_prepro, y_encoded, test_size=0.2)

scaler = fit_scaler(X_train)

X_train_scaled = transform_scaler(X_train, scaler)
X_test_scaled = transform_scaler(X_test, scaler)


tree_classifier = DecisionTreeClassifier(
    criterion="gini",  # atau "entropy" atau "log_loss"
    max_depth=None,  # Sesuaikan sesuai kebutuhan
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.0
)

# Melatih model dengan data latih
tree_classifier.fit(X_train_scaled, y_train)

y_pred_train = tree_classifier.predict(X_train_scaled)
y_pred_test = tree_classifier.predict(X_test_scaled)
print(f"Accuracy score Train : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Accuracy score Test  : {accuracy_score(y_test, y_pred_test):.4f}")
print("")
print("")
