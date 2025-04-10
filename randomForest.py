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

def load_data(years_back=5):
    """Carga sólo los datos de los últimos n años para mayor eficiencia"""
    current_year = 2025  # Ajustar según la fecha actual
    start_year = current_year - years_back
    
    # Sólo cargar archivos de los últimos años
    files = [f for f in glob.glob('./data/atp_matches_*.csv') 
             if f'atp_matches_{start_year}' in f or 
                any(f'atp_matches_{year}' in f for year in range(start_year+1, current_year+1))]
    
    if not files:  # Si no hay archivos específicos, cargar todos
        files = glob.glob('./data/atp_matches_*.csv')
    
    print(f"Cargando {len(files)} archivos de datos...")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    return data

def create_player_lookup_dict(data):
    """Crea un diccionario para búsqueda rápida de jugadores"""
    player_dict = {}
    
    # Procesar ganadores
    if 'winner_name' in data.columns:
        for _, row in data.iterrows():
            if pd.notna(row['winner_name']):
                name = row['winner_name'].lower()
                if name not in player_dict or row['tourney_date'] > player_dict[name]['date']:
                    player_dict[name] = {
                        'date': row['tourney_date'],
                        'name': row['winner_name'],
                        'id': row['winner_id'],
                        'hand': row['winner_hand'] if pd.notna(row['winner_hand']) else 'R',
                        'ht': float(row['winner_ht']) if pd.notna(row['winner_ht']) else 185.0,
                        'ioc': row['winner_ioc'] if pd.notna(row['winner_ioc']) else 'UNK',
                        'age': float(row['winner_age']) if pd.notna(row['winner_age']) else 25.0,
                        'rank': int(row['winner_rank']) if pd.notna(row['winner_rank']) else 100,
                        'rank_points': float(row['winner_rank_points']) if pd.notna(row['winner_rank_points']) else 500.0
                    }
    
    # Procesar perdedores (sólo si tienen datos más recientes)
    if 'loser_name' in data.columns:
        for _, row in data.iterrows():
            if pd.notna(row['loser_name']):
                name = row['loser_name'].lower()
                if name not in player_dict or row['tourney_date'] > player_dict[name]['date']:
                    player_dict[name] = {
                        'date': row['tourney_date'],
                        'name': row['loser_name'],
                        'id': row['loser_id'],
                        'hand': row['loser_hand'] if pd.notna(row['loser_hand']) else 'R',
                        'ht': float(row['loser_ht']) if pd.notna(row['loser_ht']) else 185.0,
                        'ioc': row['loser_ioc'] if pd.notna(row['loser_ioc']) else 'UNK',
                        'age': float(row['loser_age']) if pd.notna(row['loser_age']) else 25.0,
                        'rank': int(row['loser_rank']) if pd.notna(row['loser_rank']) else 150,
                        'rank_points': float(row['loser_rank_points']) if pd.notna(row['loser_rank_points']) else 300.0
                    }
    
    return player_dict

def find_player_by_name_fast(name, player_dict):
    """Busca un jugador usando el diccionario precompilado"""
    # Búsqueda exacta
    if name.lower() in player_dict:
        player_data = player_dict[name.lower()]
        return {
            'name': player_data['name'],
            'id': player_data['id'],
            'hand': player_data['hand'],
            'ht': player_data['ht'],
            'ioc': player_data['ioc'],
            'age': player_data['age'],
            'rank': player_data['rank'],
            'rank_points': player_data['rank_points'],
            'seed': 999  # Valor por defecto
        }
    
    # Búsqueda parcial
    for key, player_data in player_dict.items():
        if name.lower() in key:
            return {
                'name': player_data['name'],
                'id': player_data['id'],
                'hand': player_data['hand'],
                'ht': player_data['ht'],
                'ioc': player_data['ioc'],
                'age': player_data['age'],
                'rank': player_data['rank'],
                'rank_points': player_data['rank_points'],
                'seed': 999  # Valor por defecto
            }
    
    # No encontrado
    return None

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
    
    # Convertir todo a string una sola vez
    categorical_features = ['surface', 'tourney_level', 'p1_hand', 'p2_hand', 'p1_ioc', 'p2_ioc']
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
    
    # Para semillas
    seed_features = ['p1_seed', 'p2_seed']

    # Simplificar el preprocesador para ser más eficiente
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('seed', SimpleImputer(strategy='constant', fill_value=999.0), seed_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop',
        n_jobs=-1,
        verbose_feature_names_out=False  # Reduce overhead en nombres de columnas
    )

    # Modelo optimizado para velocidad
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=30,  # Reduce complejidad
            max_features='sqrt',   # Más eficiente
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            verbose=0,            # Reducir verbosidad para mayor velocidad
            warm_start=False      # Más rápido para un solo entrenamiento
        ))
    ])

    # Usar menos datos para entrenamiento rápido si el dataset es grande
    sample_size = min(100000, len(data))  # Máximo 100k muestras
    sample_data = data.sample(n=sample_size, random_state=42) if len(data) > sample_size else data
    
    print(f"Iniciando entrenamiento con {len(sample_data)} muestras...")
    model.fit(sample_data[features], sample_data['target'])
    return model

# Simular torneo
def simulate_tournament(players, model, surface='Hard', tourney_level='G'):
    bracket = players.copy()
    round_number = 7  # Comienza en R128
    
    while len(bracket) > 1:
        print(f"\n--- Ronda Actual: {round_number} ({len(bracket)} jugadores) ---")
        next_round = []
        batch_data = []
        match_pairs = []
        
        # Preparar todos los partidos de la ronda
        for i in range(0, len(bracket), 2):
            if i+1 >= len(bracket):
                next_round.append(bracket[i])
                print(f"{bracket[i]['name']} avanza por bye")
                continue
                
            p1 = bracket[i]
            p2 = bracket[i+1]
            match_pairs.append((p1, p2))
            
            batch_data.append({
                'surface': surface,
                'tourney_level': tourney_level,
                'draw_size': float(len(bracket)),
                'best_of': 3.0,
                'round': float(round_number),
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
            })
            
        # Hacer predicciones en lote si hay partidos
        if batch_data:
            batch_df = pd.DataFrame(batch_data)
            probabilities = model.predict_proba(batch_df)[:, 1]
            
            # Procesar los resultados
            for idx, ((p1, p2), prob) in enumerate(zip(match_pairs, probabilities)):
                winner = p1 if prob >= 0.5 else p2
                next_round.append(winner)
                print(f"{p1['name']} vs {p2['name']} => {winner['name']} gana (probabilidad: {(prob if prob >= 0.5 else 1 - prob):.2f})")
        
        bracket = next_round
        round_number -= 1  # Avanzar a la siguiente ronda
    
    return bracket[0]

if __name__ == "__main__":
    # Cargar datos de manera más eficiente
    print("Cargando datos...")
    historical_data = load_data(years_back=5)  # Solo últimos 5 años
    
    # Crear diccionario para búsqueda rápida
    print("Creando índice de jugadores...")
    player_dict = create_player_lookup_dict(historical_data)
    
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
    
    while True:
        name = input(f"Jugador #{len(players)+1} (o 'fin' para terminar): ")
        if name.lower() == 'fin':
            break
            
        # Buscar datos del jugador con el método optimizado
        player_data = find_player_by_name_fast(name, player_dict)
        
        if player_data:
            player_data['seed'] = len(players) + 1
            print(f"Jugador encontrado: {player_data['name']} (Rank: {player_data['rank']}, {player_data['ioc']})")
            players.append(player_data)
        else:
            print(f"Jugador '{name}' no encontrado. Generando datos aleatorios.")
            random_player = generate_random_player(name, not_found_count)
            random_player['seed'] = len(players) + 1
            players.append(random_player)
            not_found_count += 1
            
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
        