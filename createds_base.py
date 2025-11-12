tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'META', 'NVDA', 'JPM', 'JNJ', 'V',
    'AIR.PA', 'MC.PA', 'TTE.PA', 'SAN.PA', 'ASML.AS',
    'SIE.DE', 'HSBA.L', 'NESN.SW', 'BMW.DE', 'IBE.MC'
]

# Augmenter la période à 5 ans au lieu de 18 mois pour avoir plus de données
data = yf.download(tickers, start="2010-01-01", end="2025-01-01")

print("Colonnes disponibles:")
print(data.columns.levels[0])

# Fonction pour calculer les indicateurs pour chaque ticker
def add_technical_indicators(df, ticker):
    """
    Ajoute les indicateurs techniques à un DataFrame pour un ticker donné
    """
    # Extraction des données de clôture pour le ticker
    close = df['Close'][ticker].dropna()
    high = df['High'][ticker].dropna()
    low = df['Low'][ticker].dropna()
    volume = df['Volume'][ticker].dropna()

    common_index = close.index.intersection(high.index).intersection(low.index).intersection(volume.index)

    close = close.loc[common_index]
    high = high.loc[common_index]
    low = low.loc[common_index]
    volume = volume.loc[common_index]
    # DataFrame pour stocker les indicateurs
    indicators = pd.DataFrame(index=close.index)

    # Prix de clôture
    indicators['Close'] = close

    # RSI (Relative Strength Index) - 14 jours par défaut
    rsi = RSIIndicator(close=close, window=14)
    indicators['RSI'] = rsi.rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    indicators['MACD'] = macd.macd()
    indicators['MACD_Signal'] = macd.macd_signal()
    indicators['MACD_Diff'] = macd.macd_diff()

    # Moving Averages
    indicators['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
    indicators['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
    indicators['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
    indicators['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()

    # Bollinger Bands
    bollinger = BollingerBands(close=close, window=20, window_dev=2)
    indicators['BB_High'] = bollinger.bollinger_hband()
    indicators['BB_Low'] = bollinger.bollinger_lband()
    indicators['BB_Mid'] = bollinger.bollinger_mavg()

    # Indicateurs dérivés utiles pour LSTM
    indicators['Price_Change'] = close.pct_change(fill_method=None) # fill_method=None permet d'éviter les erreurs de warning
    indicators['Volume_Change'] = volume.pct_change(fill_method=None)

    # Distance du prix par rapport aux moyennes mobiles (normalisée) -- éviter division par zéro
    eps = 1e-8
    indicators['Distance_SMA20'] = (close - indicators['SMA_20']) / (indicators['SMA_20'] + eps)
    indicators['Distance_SMA50'] = (close - indicators['SMA_50']) / (indicators['SMA_50'] + eps)

    return indicators

# Calcul des indicateurs pour chaque ticker
def calculate_indic_ticker():
    all_indicators = {}
    for ticker in tickers:
        print(f"Calcul des indicateurs pour {ticker}...")
        indicators = add_technical_indicators(data, ticker)
        all_indicators[ticker] = indicators

        # Afficher un aperçu
        print(f"\n{ticker} - Dernières valeurs:")
        print(indicators.tail())
        print("\n" + "="*50 + "\n")
    return 'indicators for {ticker}'

