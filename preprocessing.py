def clean_indicators(indicators_dict):
    cleaned_indicators = {}

    for ticker, df in indicators_dict.items():
        # Supprimer les lignes avec NaN
        # Les premiers jours ont des NaN à cause des indicateurs (ex: SMA_50 a besoin de 50 jours)
        df_clean = df.dropna()
        cleaned_indicators[ticker] = df_clean

    return cleaned_indicators

def prepare_train_test_split(X, y, test_size=0.2, val_size=0.1):
    """
    """
    n_samples = len(X)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_sequences(df, window_size=60, horizons=[1, 5, 30]):
    """
    Crée des séquences temporelles pour le LSTM
    Version plus flexible :
    - Adapte window_size si pas assez de données
    - Gère les NaN dans les targets
    - Validation plus souple des séquences
    """
    print(f"  Shape initial du DataFrame: {df.shape}")

    # Colonnes cibles
    target_cols = [f'Target_Return_{h}d' for h in horizons]
    target_cols = [col for col in target_cols if col in df.columns]

    if not target_cols:
        raise ValueError("Aucune colonne target trouvée dans le DataFrame")

    # Features (toutes les colonnes sauf les targets)
    feature_cols = [col for col in df.columns if col not in target_cols]
    print(f"  Nombre de features: {len(feature_cols)}")
    print(f"  Nombre de targets: {len(target_cols)}")

    if not feature_cols:
        raise ValueError("Aucune colonne feature trouvée dans le DataFrame")

    # Adapter window_size si nécessaire
    n_samples = len(df)
    min_window = 5  # taille minimum de fenêtre
    if n_samples < window_size:
        original_window = window_size
        # Utiliser 20% des données comme taille de fenêtre, minimum 5
        window_size = max(min_window, min(window_size, n_samples // 5))
        print(f"  Attention: window_size réduit de {original_window} à {window_size} (adaptation aux données disponibles)")

    # Préparation des données
    features = df[feature_cols].values
    targets = df[target_cols].values

    # Création des séquences
    n_sequences = len(df) - window_size
    print(f"  Nombre potentiel de séquences: {n_sequences}")

    if n_sequences <= 0:
        print("  Pas assez de données pour créer des séquences")
        return np.array([]), np.array([]), feature_cols, target_cols

    X = []
    y = []

    for i in range(n_sequences):
        seq_features = features[i:i+window_size]
        seq_target = targets[i+window_size-1]  # target de la dernière timestep

        # Une séquence est valide si elle a moins de 20% de NaN
        if (np.isnan(seq_features).sum() / seq_features.size < 0.2 and
            not np.isnan(seq_target).any()):
            X.append(seq_features)
            y.append(seq_target)

    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        print(f"  Séquences finales:")
        print(f"    X shape: {X.shape}, y shape: {y.shape}")
        print(f"    Window size utilisé: {window_size}")
    else:
        X = np.array([])
        y = np.array([])
        print("  Aucune séquence valide créée")

    return X, y, feature_cols, target_cols

def normalize_sequences_split(X_train, X_val, X_test, method='minmax'):
    """
    Normalise les séquences séparées (Train, Val, Test) en utilisant
    UNIQUEMENT le X_train pour le fit (ÉVITE LE DATA LEAKAGE).

    Args:
        X_train, X_val, X_test: Séquences 3D (samples, timesteps, features)
        method: 'minmax' ou 'standard'
    """

    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    n_samples, n_timesteps, n_features = X_train.shape

    # Reshape en 2D pour fitter (samples * timesteps, features)
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
