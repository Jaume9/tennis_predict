import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configuración inicial
round_mapping = {
    'R128': 7, 'R64': 6, 'R32': 5, 'R16': 4, 
    'QF': 3, 'SF': 2, 'F': 1, 'RR': 0
}

def load_data():
    files = glob.glob('./data/atp_matches_*.csv')
    dfs = []
    for f in files:
        # Especificar dtype para columnas problemáticas
        dfs.append(pd.read_csv(f, low_memory=False))
    data = pd.concat(dfs, ignore_index=True)
    return data

def preprocess_data(data):
    # Crear dataset balanceado
    df1 = data.rename(columns={
        'winner_id': 'p1_id', 'loser_id': 'p2_id',
        'winner_seed': 'p1_seed', 'loser_seed': 'p2_seed',
        'winner_hand': 'p1_hand', 'loser_hand': 'p2_hand',
        'winner_ht': 'p1_ht', 'loser_ht': 'p2_ht',
        'winner_ioc': 'p1_ioc', 'loser_ioc': 'p2_ioc',
        'winner_age': 'p1_age', 'loser_age': 'p2_age',
        'winner_rank': 'p1_rank', 'loser_rank': 'p2_rank',
        'winner_rank_points': 'p1_rank_points', 'loser_rank_points': 'p2_rank_points'
    })
    df1['target'] = 1

    df2 = data.rename(columns={
        'loser_id': 'p1_id', 'winner_id': 'p2_id',
        'loser_seed': 'p1_seed', 'winner_seed': 'p2_seed',
        'loser_hand': 'p1_hand', 'winner_hand': 'p2_hand',
        'loser_ht': 'p1_ht', 'winner_ht': 'p2_ht',
        'loser_ioc': 'p1_ioc', 'winner_ioc': 'p2_ioc',
        'loser_age': 'p1_age', 'winner_age': 'p2_age',
        'loser_rank': 'p1_rank', 'winner_rank': 'p2_rank',
        'loser_rank_points': 'p1_rank_points', 'winner_rank_points': 'p2_rank_points'
    })
    df2['target'] = 0

    combined = pd.concat([df1, df2], ignore_index=True)

    # Convertir columnas categóricas a strings
    categorical_cols = ['surface', 'tourney_level', 
                       'p1_hand', 'p2_hand', 'p1_ioc', 'p2_ioc']    
    
    for col in categorical_cols:
        combined[col] = combined[col].astype(str).str.strip().replace({'nan': 'missing', '': 'missing'})
        
    # Convertir TODAS las columnas numéricas incluyendo las del torneo
    numeric_cols = [
        'draw_size', 'best_of', 'round',
        'p1_seed', 'p2_seed', 'p1_ht', 'p2_ht', 
        'p1_age', 'p2_age', 'p1_rank', 'p2_rank',
        'p1_rank_points', 'p2_rank_points'
    ]
    
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    # Manejar valores faltantes en columnas clave
    combined['p1_hand'] = combined['p1_hand'].fillna('U')
    combined['p2_hand'] = combined['p2_hand'].fillna('U')
    
    return combined

def train_model(data):
    # Preprocesamiento mejorado
    data['round'] = data['round'].replace(round_mapping).fillna(0).astype(int)
    data['draw_size'] = data['draw_size'].fillna(128).astype(int)
    data['best_of'] = data['best_of'].fillna(3).astype(int)
    categorical_features = ['surface', 'tourney_level', 
                           'p1_hand', 'p2_hand', 'p1_ioc', 'p2_ioc']
    
    for col in categorical_features:
        data[col] = data[col].astype(str)

    
    features = ['surface', 'tourney_level', 'draw_size', 'best_of', 'round',
                'p1_seed', 'p1_hand', 'p1_ht', 'p1_ioc', 'p1_age', 
                'p1_rank', 'p1_rank_points',
                'p2_seed', 'p2_hand', 'p2_ht', 'p2_ioc', 'p2_age',
                'p2_rank', 'p2_rank_points']

    numeric_features = ['draw_size', 'best_of', 'round',
                        'p1_ht', 'p1_age', 'p1_rank', 'p1_rank_points',
                        'p2_ht', 'p2_age', 'p2_rank', 'p2_rank_points']
    
    categorical_features = ['surface', 'tourney_level', 
                           'p1_hand', 'p2_hand', 'p1_ioc', 'p2_ioc']
    
    # Nuevo: Pipeline para semillas
    seed_features = ['p1_seed', 'p2_seed']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]), numeric_features),
            ('seed', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=999))
            ]), seed_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            verbose=1  # Para ver el progreso
        ))
    ])

    model.fit(data[features], data['target'])
    return model

# Simular torneo
def simulate_tournament(players, model, surface='Hard', tourney_level='G'):
    bracket = players.copy()
    round_number = 7  # Comienza en R128
    
    while len(bracket) > 1:
        print(f"\n--- Ronda Actual: {round_number} ({len(bracket)} jugadores) ---")
        next_round = []
        
        for i in range(0, len(bracket), 2):
            if i+1 >= len(bracket):
                next_round.append(bracket[i])
                continue
                
            p1 = bracket[i]
            p2 = bracket[i+1]
            
            # Crear features del partido
            match = pd.DataFrame([{
                'surface': surface,
                'tourney_level': tourney_level,
                'draw_size': len(bracket),
                'best_of': 3,
                'round': round_number,
                'p1_seed': p1['seed'],
                'p1_hand': p1['hand'],
                'p1_ht': p1['ht'],
                'p1_ioc': p1['ioc'],
                'p1_age': p1['age'],
                'p1_rank': p1['rank'],
                'p1_rank_points': p1['rank_points'],
                'p2_seed': p2['seed'],
                'p2_hand': p2['hand'],
                'p2_ht': p2['ht'],
                'p2_ioc': p2['ioc'],
                'p2_age': p2['age'],
                'p2_rank': p2['rank'],
                'p2_rank_points': p2['rank_points']
            }])
            
            # Predecir ganador
            prob = model.predict_proba(match)[0][1]
            winner = p1 if prob >= 0.5 else p2
            next_round.append(winner)
        
        bracket = next_round
        round_number -= 1  # Avanzar a la siguiente ronda
    
    return bracket[0]

if __name__ == "__main__":
    # Cargar datos y entrenar modelo
    print("Cargando datos y entrenando modelo...")
    historical_data = load_data()
    processed_data = preprocess_data(historical_data)
    
    # Limpieza final adicional
    processed_data = processed_data.dropna(subset=[
        'p1_rank', 'p2_rank', 
        'p1_rank_points', 'p2_rank_points'
    ])
    
    model = train_model(processed_data)
    
    # Preparar jugadores para el torneo (ejemplo con datos históricos)
    latest_data = historical_data.sort_values('tourney_date', ascending=False).iloc[0]
    players = []
    for i in range(96):
        players.append({
            'seed': i+1,
            'hand': 'R' if np.random.rand() > 0.2 else 'L',
            'ht': np.random.randint(170, 200),
            'ioc': 'USA' if np.random.rand() > 0.5 else 'ESP',
            'age': np.random.randint(18, 35),
            'rank': i+1,
            'rank_points': 10000 - (i*100)
        })
    
    # Simular torneo
    print("\nIniciando simulación del torneo...")
    champion = simulate_tournament(players, model)
    
    print(f"\n¡El campeón predicho es: Jugador #{champion['seed']}!")
    print(f"Ranking: {champion['rank']}")
    print(f"Edad: {champion['age']} años")
    print(f"Altura: {champion['ht']} cm")
    print(f"Nacionalidad: {champion['ioc']}")
    