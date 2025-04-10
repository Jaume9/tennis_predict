import pandas as pd
import numpy as np
import glob
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
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

def find_player_by_name(name, historical_data):
    """Busca un jugador por nombre en los datos históricos y retorna su información más reciente"""
    # Buscar en winner_name
    if 'winner_name' in historical_data.columns:
        winner_matches = historical_data[historical_data['winner_name'].str.lower() == name.lower()].sort_values('tourney_date', ascending=False)
        if not winner_matches.empty:
            latest_match = winner_matches.iloc[0]
            return {
                'name': name,
                'id': latest_match['winner_id'],
                'hand': latest_match['winner_hand'] if pd.notna(latest_match['winner_hand']) else 'R',
                'ht': float(latest_match['winner_ht']) if pd.notna(latest_match['winner_ht']) else 185.0,
                'ioc': latest_match['winner_ioc'] if pd.notna(latest_match['winner_ioc']) else 'UNK',
                'age': float(latest_match['winner_age']) if pd.notna(latest_match['winner_age']) else 25.0,
                'rank': int(latest_match['winner_rank']) if pd.notna(latest_match['winner_rank']) else 100,
                'rank_points': float(latest_match['winner_rank_points']) if pd.notna(latest_match['winner_rank_points']) else 500.0,
                'seed': 999  # Valor por defecto
            }
    
    # Buscar en loser_name
    if 'loser_name' in historical_data.columns:
        loser_matches = historical_data[historical_data['loser_name'].str.lower() == name.lower()].sort_values('tourney_date', ascending=False)
        if not loser_matches.empty:
            latest_match = loser_matches.iloc[0]
            return {
                'name': name,
                'id': latest_match['loser_id'],
                'hand': latest_match['loser_hand'] if pd.notna(latest_match['loser_hand']) else 'R',
                'ht': float(latest_match['loser_ht']) if pd.notna(latest_match['loser_ht']) else 185.0,
                'ioc': latest_match['loser_ioc'] if pd.notna(latest_match['loser_ioc']) else 'UNK',
                'age': float(latest_match['loser_age']) if pd.notna(latest_match['loser_age']) else 25.0,
                'rank': int(latest_match['loser_rank']) if pd.notna(latest_match['loser_rank']) else 150,
                'rank_points': float(latest_match['loser_rank_points']) if pd.notna(latest_match['loser_rank_points']) else 300.0,
                'seed': 999  # Valor por defecto
            }
    
    # Buscar coincidencia parcial si no hay match exacto
    if 'winner_name' in historical_data.columns:
        partial_matches = historical_data[historical_data['winner_name'].str.lower().str.contains(name.lower(), na=False)]
        if not partial_matches.empty:
            # Usar el primer match parcial
            latest_match = partial_matches.sort_values('tourney_date', ascending=False).iloc[0]
            return {
                'name': latest_match['winner_name'],  # Usar el nombre completo encontrado
                'id': latest_match['winner_id'],
                'hand': latest_match['winner_hand'] if pd.notna(latest_match['winner_hand']) else 'R',
                'ht': float(latest_match['winner_ht']) if pd.notna(latest_match['winner_ht']) else 185.0,
                'ioc': latest_match['winner_ioc'] if pd.notna(latest_match['winner_ioc']) else 'UNK',
                'age': float(latest_match['winner_age']) if pd.notna(latest_match['winner_age']) else 25.0,
                'rank': int(latest_match['winner_rank']) if pd.notna(latest_match['winner_rank']) else 100,
                'rank_points': float(latest_match['winner_rank_points']) if pd.notna(latest_match['winner_rank_points']) else 500.0,
                'seed': 999  # Valor por defecto
            }
    
    # Buscar por ID si es un valor numérico
    try:
        player_id = int(name)
        id_matches = historical_data[
            (historical_data['winner_id'] == player_id) | 
            (historical_data['loser_id'] == player_id)
        ].sort_values('tourney_date', ascending=False)
        
        if not id_matches.empty:
            latest_match = id_matches.iloc[0]
            if latest_match['winner_id'] == player_id:
                return {
                    'name': str(player_id),
                    'id': player_id,
                    'hand': latest_match['winner_hand'] if pd.notna(latest_match['winner_hand']) else 'R',
                    'ht': float(latest_match['winner_ht']) if pd.notna(latest_match['winner_ht']) else 185.0,
                    'ioc': latest_match['winner_ioc'] if pd.notna(latest_match['winner_ioc']) else 'UNK',
                    'age': float(latest_match['winner_age']) if pd.notna(latest_match['winner_age']) else 25.0,
                    'rank': int(latest_match['winner_rank']) if pd.notna(latest_match['winner_rank']) else 100,
                    'rank_points': float(latest_match['winner_rank_points']) if pd.notna(latest_match['winner_rank_points']) else 500.0,
                    'seed': 999  # Valor por defecto
                }
            else:
                return {
                    'name': str(player_id),
                    'id': player_id,
                    'hand': latest_match['loser_hand'] if pd.notna(latest_match['loser_hand']) else 'R',
                    'ht': float(latest_match['loser_ht']) if pd.notna(latest_match['loser_ht']) else 185.0,
                    'ioc': latest_match['loser_ioc'] if pd.notna(latest_match['loser_ioc']) else 'UNK',
                    'age': float(latest_match['loser_age']) if pd.notna(latest_match['loser_age']) else 25.0,
                    'rank': int(latest_match['loser_rank']) if pd.notna(latest_match['loser_rank']) else 150,
                    'rank_points': float(latest_match['loser_rank_points']) if pd.notna(latest_match['loser_rank_points']) else 300.0,
                    'seed': 999  # Valor por defecto
                }
    except ValueError:
        pass
    
    # Si no se encuentra, retornar None
    return None

def generate_random_player(name, not_found_count):
    """Genera datos aleatorios para un jugador no encontrado"""
    hands = ['R', 'L']
    countries = ['ESP', 'USA', 'FRA', 'GBR', 'ITA', 'GER', 'SRB', 'RUS', 'ARG', 'AUS']
    
    return {
        'name': name,
        'id': -not_found_count,  # ID negativo para identificar jugadores aleatorios
        'hand': random.choice(hands),
        'ht': random.uniform(175.0, 200.0),  # Altura entre 175-200 cm
        'ioc': random.choice(countries),
        'age': random.uniform(20.0, 35.0),  # Edad entre 20-35 años
        'rank': random.randint(50, 300),  # Ranking entre 50-300
        'rank_points': random.uniform(300.0, 1500.0),  # Puntos entre 300-1500
        'seed': 999  # Valor por defecto para no cabeza de serie
    }

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
                ('imputer', SimpleImputer(strategy='median')),
                ('float_converter', FunctionTransformer(lambda x: x.astype(np.float64)))
            ]), numeric_features),
            ('seed', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=999.0)),  # Cambiar a float
                ('float_converter', FunctionTransformer(lambda x: x.astype(np.float64)))
            ]), seed_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=20), categorical_features)
        ],
        remainder='drop',
        n_jobs=-1
    )

    # Modelo optimizado
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50,  # Reducir de 200 a 50
            max_depth=10,     # Limitar profundidad
            max_samples=0.2,  # Usar sólo 20% de datos por árbol
            n_jobs=-1,        # Usar todos los núcleos
            verbose=2,        # Más detalle en progreso
            random_state=42
        ))
    ])

    # Reducir dataset para entrenamiento inicial
    sample_data = data.sample(frac=0.3, random_state=42) if len(data) > 100000 else data
    
    print("Iniciando entrenamiento...")
    model.fit(sample_data[features], sample_data['target'])
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
                print(f"{bracket[i]['name']} avanza por bye")
                continue
                
            p1 = bracket[i]
            p2 = bracket[i+1]
            
            # Crear features del partido
            match = pd.DataFrame([{
                'surface': surface,
                'tourney_level': tourney_level,
                'draw_size': float(len(bracket)),  # Convertir a float
                'best_of': 3.0,  # float
                'round': float(round_number),  # float
                'p1_seed': float(p1['seed']),
                'p1_hand': p1['hand'],
                'p1_ht': float(p1['ht']),
                'p1_ioc': p1['ioc'],
                'p1_age': float(p1['age']),
                'p1_rank': float(p1['rank']),
                'p1_rank_points': float(p1['rank_points']),
                'p2_seed': float(p2['seed']),
                'p2_hand': p2['hand'],
                'p2_ht': float(p2['ht']),
                'p2_ioc': p2['ioc'],
                'p2_age': float(p2['age']),
                'p2_rank': float(p2['rank']),
                'p2_rank_points': float(p2['rank_points'])
            }])
            
            # Predecir ganador
            prob = model.predict_proba(match)[0][1]
            winner = p1 if prob >= 0.5 else p2
            loser = p2 if prob >= 0.5 else p1
            next_round.append(winner)
            print(f"{p1['name']} vs {p2['name']} => {winner['name']} gana (probabilidad: {prob:.2f if prob >= 0.5 else 1-prob:.2f})")
        
        bracket = next_round
        round_number -= 1  # Avanzar a la siguiente ronda
    
    return bracket[0]

if __name__ == "__main__":
    # Cargar datos y entrenar modelo
    print("Cargando datos...")
    historical_data = load_data()

    # Limitar a los últimos 5 años para prueba inicial
    historical_data = historical_data[historical_data['tourney_date'] >= 20180000]
    
    # Preprocesar datos para el entrenamiento
    print("Preprocesando datos...")
    processed_data = preprocess_data(historical_data)
    
    # Limpieza final adicional
    processed_data = processed_data.dropna(subset=[
        'p1_rank', 'p2_rank', 
        'p1_rank_points', 'p2_rank_points'
    ])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    model = train_model(processed_data)
    
    # Recoger información de los jugadores
    players = []
    not_found_count = 1
    
    print("\nIntroduce los nombres de los jugadores (uno por línea, 'fin' para terminar):")
    print("Ejemplo: 'Rafael Nadal', 'Novak Djokovic', etc.")
    
    while True:
        name = input(f"Jugador #{len(players)+1} (o 'fin' para terminar): ")
        if name.lower() == 'fin':
            break
            
        # Buscar datos del jugador
        player_data = find_player_by_name(name, historical_data)
        
        if player_data:
            # Si se encontró el jugador, asignar seed según el orden de ingreso
            player_data['seed'] = len(players) + 1
            print(f"Jugador encontrado: {player_data['name']} (Rank: {player_data['rank']}, {player_data['ioc']})")
            players.append(player_data)
        else:
            # Si no se encontró, generar datos aleatorios
            print(f"Jugador '{name}' no encontrado. Generando datos aleatorios.")
            random_player = generate_random_player(name, not_found_count)
            random_player['seed'] = len(players) + 1
            players.append(random_player)
            not_found_count += 1
            
        # Terminar si tenemos suficientes jugadores para un torneo
        if len(players) >= 128:
            print("Límite de 128 jugadores alcanzado.")
            break
    
    if len(players) < 2:
        print("Se necesitan al menos 2 jugadores para simular un torneo.")
    else:
        # Ajustar el número de jugadores a una potencia de 2
        player_count = 2
        while player_count < len(players):
            player_count *= 2
            
        if player_count > len(players):
            player_count //= 2
            
        if player_count < len(players):
            print(f"Usando los primeros {player_count} jugadores para el torneo.")
            players = players[:player_count]
        
        # Elegir superficie para el torneo
        surface = input("\nElige la superficie (Hard/Clay/Grass): ").strip().capitalize() or "Hard"
        if surface not in ["Hard", "Clay", "Grass"]:
            surface = "Hard"
            
        tourney_level = input("\nElige el nivel del torneo (G=Grand Slam, M=Masters, A=ATP): ").strip().upper() or "G"
        if tourney_level not in ["G", "M", "A"]:
            tourney_level = "G"
        
        # Simular torneo
        print(f"\nIniciando simulación del torneo ({surface}, {tourney_level}) con {len(players)} jugadores...")
        champion = simulate_tournament(players, model, surface, tourney_level)
        
        print(f"\n¡El campeón predicho es: {champion['name']}!")
        print(f"Ranking: {champion['rank']}")
        print(f"Edad: {champion['age']:.1f} años")
        print(f"Altura: {champion['ht']:.1f} cm")
        print(f"Nacionalidad: {champion['ioc']}")
        