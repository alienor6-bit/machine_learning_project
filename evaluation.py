def evaluate_model(model, sequences_data, ticker):
    """Évalue le modèle sur les 3 splits et affiche les métriques"""
    print(f"\n{'='*60}\nÉvaluation pour {ticker}\n{'='*60}")

    data = sequences_data[ticker]

    for split_name in ['train', 'val', 'test']:
        X, y_true = data[f'X_{split_name}'], data[f'y_{split_name}']

        if len(X) == 0:
            print(f"\n{split_name.upper()}: Pas de données")
            continue

        print(f"\n{split_name.upper()}:")
        calculate_metrics(y_true, model.predict(X, verbose=0), ['1d', '5d', '30d'])

