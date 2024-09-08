import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from ml_from_scratch.tree import DecisionTreeClassifier
from ml_from_scratch.metrics import accuracy_score

# Definisi konstanta untuk random_state yang digunakan berulang kali
RANDOM_STATE = 42

# Fungsi untuk membaca dataset dari file CSV
def read_dataset(path):
    """
    Membaca dataset dari file CSV dan mengembalikannya dalam bentuk DataFrame.
    Juga menangani kesalahan jika file tidak ditemukan.
    """
    try:
        df = pd.read_csv(path)
        print(f'Data Shape: {df.shape}')  # Menampilkan ukuran data
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan pada path {path}")
        return None

# Fungsi untuk memisahkan fitur (X) dan target (y)
def split_features_and_target(df, feature_cols, target_col):
    """
    Memisahkan dataset menjadi fitur dan target.
    
    Parameters:
    - df: DataFrame
    - feature_cols: Daftar kolom yang digunakan sebagai fitur
    - target_col: Kolom yang digunakan sebagai target
    
    Returns:
    - X: Data fitur
    - y: Data target
    """
    X = df[feature_cols]
    y = df[target_col]
    print(f'Features Shape: {X.shape}, Target Shape: {y.shape}')  # Menampilkan ukuran X dan y
    return X, y

# Fungsi untuk melakukan scaling pada data fitur
def scale_features(X_train, X_test):
    """
    Melakukan scaling pada data fitur menggunakan StandardScaler.
    
    Parameters:
    - X_train: Data fitur training
    - X_test: Data fitur testing
    
    Returns:
    - X_train_scaled: Data fitur training yang sudah diskalakan
    - X_test_scaled: Data fitur testing yang sudah diskalakan
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

# Fungsi untuk melakukan undersampling pada data untuk menangani ketidakseimbangan kelas
def undersample_data(X, y, strategy='auto'):
    """
    Melakukan undersampling pada data untuk menyeimbangkan kelas yang tidak seimbang.
    
    Parameters:
    - X: Data fitur
    - y: Data target
    - strategy: Strategi sampling, default adalah 'auto'
    
    Returns:
    - X_resampled: Data fitur setelah di-resample
    - y_resampled: Data target setelah di-resample
    """
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=RANDOM_STATE)
    return rus.fit_resample(X, y)

# Fungsi untuk meng-encode kolom kategori menjadi format biner atau one-hot encoding
def encode_categorical_columns(X):
    """
    Meng-encode kolom kategori menjadi biner jika ada 2 kategori, 
    atau menggunakan one-hot encoding jika lebih dari 2 kategori.
    
    Parameters:
    - X: DataFrame berisi kolom kategori
    
    Returns:
    - X_encoded: DataFrame dengan kolom kategori yang sudah di-encode
    """
    for col in X.columns:
        unique_values = X[col].nunique()
        if unique_values == 2:
            # Jika ada 2 nilai unik, lakukan kategorisasi biner
            X = categorize_column(X, col, category_zero_values=[X[col].unique()[0]])
        else:
            # Jika lebih dari 2 nilai unik, gunakan one-hot encoding
            X = one_hot_encode_column(X, col)
    return X

# Fungsi untuk melakukan kategorisasi biner pada kolom
def categorize_column(X, column_name, category_zero_values):
    X[column_name] = X[column_name].apply(lambda x: 0 if x in category_zero_values else 1)
    return X

# Fungsi untuk melakukan one-hot encoding
def one_hot_encode_column(X, column_name):
    X = pd.get_dummies(X, columns=[column_name], drop_first=True)
    return X

# Contoh penggunaan dari kode di atas

# 1. Membaca dataset
dataset = read_dataset('data/Air_Plane_Passenger_Data.csv')

# 2. Pastikan dataset tidak kosong (None)
if dataset is not None:
    # 3. Mengisi nilai yang hilang dengan rata-rata kolom
    dataset.fillna(dataset.mean(), inplace=True)
    
    # 4. Mendefinisikan kolom fitur dan target
    feature_cols = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
                    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
                    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
                    'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service',
                    'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']  # Sesuaikan dengan kolom pada dataset
    X, y = split_features_and_target(dataset, feature_cols, 'satisfaction')
    
    # 5. Melakukan undersampling untuk menangani ketidakseimbangan kelas
    X_resampled, y_resampled = undersample_data(X, y)
    
    # 6. Melakukan label encoding pada target (y)
    y_encoded = LabelEncoder().fit_transform(y_resampled)

    # 7. Memisahkan fitur numerik dan kategori
    X_numeric = X_resampled.select_dtypes(include=['number'])
    X_categorical = encode_categorical_columns(X_resampled.select_dtypes(include=['object']))
    
    # 8. Menggabungkan kembali data numerik dan kategori yang sudah di-encode
    X_prepro = pd.concat([X_numeric, X_categorical], axis=1)

    # 9. Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X_prepro, y_encoded, test_size=0.2, random_state=RANDOM_STATE)
    
    # 10. Melakukan scaling pada data fitur
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 11. Melatih model Decision Tree
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train_scaled, y_train)

    # 12. Membuat prediksi
    y_pred_train = classifier.predict(X_train_scaled)
    y_pred_test = classifier.predict(X_test_scaled)

    # 13. Menampilkan akurasi model
    print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
