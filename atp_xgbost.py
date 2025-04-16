import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import os

# Configuración inicial
DATA_PATH = './data/'
CURRENT_YEAR = datetime.now().year

def load_data():
    matches_files = [f for f in os.listdir(DATA_PATH) if f.startswith('atp_matches')]
    
    df_matches = pd.concat([
        pd.read_csv(
            os.path.join(DATA_PATH, f),
            dtype={
                'winner_id': 'str',
                'loser_id': 'str',
                'tourney_date': 'str',
                'winner_seed': 'str',
                'loser_seed': 'str',
                'winner_entry': 'str',
                'loser_entry': 'str',
                'surface': 'str',
                'winner_hand': 'str',
                'loser_hand': 'str'
            },
            low_memory=False
        ) for f in matches_files
    ])
    
    df_players = pd.read_csv(
        os.path.join(DATA_PATH, 'atp_players.csv'),
        dtype={'player_id': 'str', 'dob': 'str', 'wikidata_id': 'str'}
    )
    
    df_rankings = pd.read_csv(os.path.join(DATA_PATH, 'atp_rankings_current.csv'))
    
    return df_matches, df_players, df_rankings

def create_features(df_matches, df_players, df_rankings):
    df_matches['tourney_date'] = pd.to_datetime(
        df_matches['tourney_date'].str[:8], 
        format='%Y%m%d', 
        errors='coerce'
    )
    df_matches = df_matches.dropna(subset=['tourney_date'])
    
    for role in ['winner', 'loser']:
        players_renamed = df_players[
            ['player_id', 'hand', 'height', 'dob']
        ].rename(columns={
            'hand': f'{role}_hand',
            'height': f'{role}_height',
            'dob': f'{role}_dob'
        })
        
        df_matches = df_matches.merge(
            players_renamed,
            left_on=f'{role}_id',
            right_on='player_id',
            how='left'
        ).drop(columns=['player_id'])
        
        for col in [f'{role}_hand', f'{role}_height', f'{role}_dob']:
            if col not in df_matches.columns:
                df_matches[col] = np.nan
    
    for role in ['winner', 'loser']:
        df_matches[f'{role}_hand'] = df_matches[f'{role}_hand'].fillna('U')
        df_matches[f'{role}_height'] = df_matches[f'{role}_height'].fillna(180)
        df_matches[f'{role}_dob'] = df_matches[f'{role}_dob'].fillna('19700101')
    
    for role in ['winner', 'loser']:
        df_matches[f'{role}_dob'] = pd.to_datetime(
            df_matches[f'{role}_dob'], 
            format='%Y%m%d',
            errors='coerce'
        )
        df_matches[f'{role}_age'] = (
            (df_matches['tourney_date'] - df_matches[f'{role}_dob']).dt.days / 365.25
        )
    
    features = [
        'surface',
        'winner_hand', 'loser_hand',
        'winner_height', 'loser_height',
        'winner_age', 'loser_age',
        'winner_rank', 'loser_rank'
    ]
    
    for player in ['winner', 'loser']:
        df_matches[f'{player}_recent_wins'] = df_matches.groupby(f'{player}_id')['winner_id'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df_matches[f'{player}_surface_win_rate'] = df_matches.groupby(
            [f'{player}_id', 'surface']
        )['winner_id'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        features += [f'{player}_recent_wins', f'{player}_surface_win_rate']
    
    df_matches['pair_key'] = df_matches.apply(
        lambda x: '-'.join(sorted([str(x['winner_id']), str(x['loser_id'])])), axis=1)
    df_matches['h2h_win_rate'] = df_matches.groupby('pair_key')['winner_id'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    features.append('h2h_win_rate')
    
    df_matches['target'] = 1
    df_mirror = df_matches.copy()
    
    swap_columns = {
        'winner_id': 'loser_id',
        'loser_id': 'winner_id',
        'winner_hand': 'loser_hand',
        'loser_hand': 'winner_hand',
        'winner_height': 'loser_height',
        'loser_height': 'winner_height',
        'winner_age': 'loser_age',
        'loser_age': 'winner_age',
        'winner_rank': 'loser_rank',
        'loser_rank': 'winner_rank',
        'winner_recent_wins': 'loser_recent_wins',
        'loser_recent_wins': 'winner_recent_wins',
        'winner_surface_win_rate': 'loser_surface_win_rate',
        'loser_surface_win_rate': 'winner_surface_win_rate',
        'h2h_win_rate': lambda x: 1 - x['h2h_win_rate']
    }
    
    for col, new_col in swap_columns.items():
        if callable(new_col):
            df_mirror[col] = df_mirror.apply(new_col, axis=1)
        else:
            df_mirror[col] = df_mirror[new_col]
    
    df_mirror['target'] = 0
    df_balanced = pd.concat([df_matches, df_mirror], ignore_index=True)
    
    df_balanced = df_balanced[features + ['target', 'tourney_date']].dropna()
    df_balanced['target'] = df_balanced['target'].astype(int)
    
    categorical_cols = ['surface', 'winner_hand', 'loser_hand']
    df_balanced = pd.get_dummies(df_balanced, columns=categorical_cols)
    features = [col for col in df_balanced.columns if col not in ['target', 'tourney_date']]
    
    return df_balanced, features

def train_model(df, features):
    df = df.sort_values('tourney_date')
    
    split_year = CURRENT_YEAR - 3
    train = df[df['tourney_date'].dt.year < split_year]
    test = df[df['tourney_date'].dt.year >= split_year]
    
    if train.empty or test.empty:
        raise ValueError("Datos insuficientes para entrenar y validar")
    
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=50,
        use_label_encoder=False
    )
    
    model.fit(
        train[features],
        train['target'],
        eval_set=[(test[features], test['target'])],
        verbose=False
    )
    
    return model

def predict_match(model, df_players, df_rankings, player1, player2, surface):
    try:
        # Separar nombre y apellido
        p1_first, p1_last = player1.rsplit(' ', 1) if ' ' in player1 else ('', player1)
        p2_first, p2_last = player2.rsplit(' ', 1) if ' ' in player2 else ('', player2)  # Error corregido: usaba player1
        
        # Buscar por nombre y apellido con búsqueda más flexible
        if p1_first:
            p1_candidates = df_players[(df_players['name_first'].str.lower() == p1_first.lower()) & 
                                      (df_players['name_last'].str.lower() == p1_last.lower())]
            # Si no hay resultados exactos, buscar de forma parcial
            if p1_candidates.empty:
                p1_candidates = df_players[(df_players['name_first'].str.lower().str.contains(p1_first.lower())) & 
                                          (df_players['name_last'].str.lower().str.contains(p1_last.lower()))]
        else:
            p1_candidates = df_players[df_players['name_last'].str.lower() == p1_last.lower()]
            # Si no hay resultados exactos, buscar de forma parcial
            if p1_candidates.empty:
                p1_candidates = df_players[df_players['name_last'].str.lower().str.contains(p1_last.lower())]
            
        if p2_first:
            p2_candidates = df_players[(df_players['name_first'].str.lower() == p2_first.lower()) & 
                                      (df_players['name_last'].str.lower() == p2_last.lower())]
            # Si no hay resultados exactos, buscar de forma parcial
            if p2_candidates.empty:
                p2_candidates = df_players[(df_players['name_first'].str.lower().str.contains(p2_first.lower())) & 
                                          (df_players['name_last'].str.lower().str.contains(p2_last.lower()))]
        else:
            p2_candidates = df_players[df_players['name_last'].str.lower() == p2_last.lower()]
            # Si no hay resultados exactos, buscar de forma parcial
            if p2_candidates.empty:
                p2_candidates = df_players[df_players['name_last'].str.lower().str.contains(p2_last.lower())]
        
        if p1_candidates.empty:
            raise ValueError(f"No se encontró al jugador: {player1}")
        if len(p1_candidates) > 1:
            # Mostrar las primeras 5 opciones si hay muchas
            player_list = ", ".join([f"{row['name_first']} {row['name_last']}" for _, row in p1_candidates.iterrows()[:5]])
            if len(p1_candidates) > 5:
                player_list += f" y {len(p1_candidates) - 5} más"
            raise ValueError(f"Múltiples jugadores encontrados como '{player1}'. Opciones: {player_list}")
        
        if p2_candidates.empty:
            raise ValueError(f"No se encontró al jugador: {player2}")
        if len(p2_candidates) > 1:
            # Mostrar las primeras 5 opciones si hay muchas
            player_list = ", ".join([f"{row['name_first']} {row['name_last']}" for _, row in p2_candidates.iterrows()[:5]])
            if len(p2_candidates) > 5:
                player_list += f" y {len(p2_candidates) - 5} más"
            raise ValueError(f"Múltiples jugadores encontrados como '{player2}'. Opciones: {player_list}")
        
        p1 = p1_candidates.iloc[0]
        p2 = p2_candidates.iloc[0]
        
        # Verificar si están en el ranking
        p1_rankings = df_rankings[df_rankings['player'] == p1['player_id']]
        if p1_rankings.empty:
            raise ValueError(f"No se encontró ranking para el jugador: {p1['name_first']} {p1['name_last']}")
        
        p2_rankings = df_rankings[df_rankings['player'] == p2['player_id']]
        if p2_rankings.empty:
            raise ValueError(f"No se encontró ranking para el jugador: {p2['name_first']} {p2['name_last']}")
        
        rank1 = p1_rankings.iloc[0]
        rank2 = p2_rankings.iloc[0]
        
    except IndexError:
        raise ValueError("Error al acceder a los datos del jugador")

    input_dict = {
        'surface': surface,
        'winner_hand': p1['hand'],
        'loser_hand': p2['hand'],
        'winner_height': p1['height'],
        'loser_height': p2['height'],
        'winner_age': (datetime.now() - pd.to_datetime(p1['dob'], format='%Y%m%d')).days / 365.25,
        'loser_age': (datetime.now() - pd.to_datetime(p2['dob'], format='%Y%m%d')).days / 365.25,
        'winner_rank': rank1['rank'],
        'loser_rank': rank2['rank'],
        'h2h_win_rate': 0.5,
        'winner_recent_wins': df_matches[df_matches['winner_id'] == p1['player_id']]['winner_recent_wins'].mean(),
        'loser_recent_wins': df_matches[df_matches['winner_id'] == p2['player_id']]['winner_recent_wins'].mean(),
        'winner_surface_win_rate': df_matches[(df_matches['winner_id'] == p1['player_id']) & (df_matches['surface'] == surface)]['winner_surface_win_rate'].mean(),
        'loser_surface_win_rate': df_matches[(df_matches['winner_id'] == p2['player_id']) & (df_matches['surface'] == surface)]['winner_surface_win_rate'].mean()
    }
    
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    prob = model.predict_proba(input_df)[0][1]
    print(f"\n[Predicción] {p1['name_first']} {p1['name_last']} vs {p2['name_first']} {p2['name_last']} en {surface}")
    print(f"Probabilidad de victoria para {p1['name_first']} {p1['name_last']}: {prob*100:.1f}%")
    print(f"Probabilidad de victoria para {p2['name_first']} {p2['name_last']}: {(1-prob)*100:.1f}%")

if __name__ == "__main__":
    print("Cargando datos...")
    df_matches, df_players, df_rankings = load_data()
    
    print("Procesando características...")
    df_features, features = create_features(df_matches, df_players, df_rankings)
    
    print(f"Entrenando modelo con {len(df_features)} partidos...")
    model = train_model(df_features, features)
    
    print("\n--- Predictor de Partidos ATP ---")
    while True:
        try:
            p1 = input("\nJugador 1 (nombre y apellido): ").strip()
            p2 = input("Jugador 2 (nombre y apellido): ").strip()
            surface = input("Superficie (Hard/Clay/Grass): ").strip().capitalize()
            
            predict_match(model, df_players, df_rankings, p1, p2, surface)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Ejemplos válidos:")
            print("- Jugadores: Rafael Nadal, Novak Djokovic, Roger Federer")
            print("- Superficies: Hard, Clay, Grass")