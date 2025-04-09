import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Configuración de directorios
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_atp_data(start_year=2010, end_year=2023):
    """Cargar datos ATP para un rango de años"""
    print(f"Cargando datos ATP de {start_year} a {end_year}...")
    all_files = []
    
    # Buscar archivos que coincidan con el patrón atp_matches_YYYY.csv
    for year in range(start_year, end_year + 1):
        pattern = DATA_DIR / f"atp_matches_{year}.csv"
        files = glob.glob(str(pattern))
        all_files.extend(files)
    
    if not all_files:
        raise ValueError(f"No se encontraron archivos para los años {start_year}-{end_year}")
    
    # Cargar y combinar archivos
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            print(f"Cargado: {os.path.basename(file)}, {df.shape[0]} registros")
            dfs.append(df)
        except Exception as e:
            print(f"Error cargando {file}: {str(e)}")
    
    # Combinar todos los dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Datos ATP combinados: {combined_df.shape[0]} partidos")
    
    return combined_df

def preprocess_data(df):
    """Limpiar y preprocesar datos para el análisis"""
    print("Preprocesando datos...")
    
    # Crear una copia para no modificar los datos originales
    processed_df = df.copy()
    
    # Convertir fechas a formato datetime
    if 'tourney_date' in processed_df.columns:
        processed_df['tourney_date'] = pd.to_datetime(processed_df['tourney_date'], format='%Y%m%d')
    
    # Eliminar filas con valores faltantes en campos críticos
    critical_columns = ['winner_id', 'loser_id', 'winner_rank', 'loser_rank', 'surface']
    critical_columns = [col for col in critical_columns if col in processed_df.columns]
    processed_df = processed_df.dropna(subset=critical_columns)
    
    # Crear columna de diferencia de ranking
    processed_df['rank_difference'] = processed_df['winner_rank'] - processed_df['loser_rank']
    
    # Normalizar nombres de superficies
    if 'surface' in processed_df.columns:
        surface_mapping = {
            'Hard': 'hard',
            'Clay': 'clay',
            'Grass': 'grass',
            'Carpet': 'carpet'
        }
        processed_df['surface'] = processed_df['surface'].map(lambda x: surface_mapping.get(x, x.lower()) if isinstance(x, str) else x)
    
    # Crear variable objetivo: 1 si gana el jugador con mejor ranking, 0 en caso contrario
    processed_df['better_rank_won'] = (processed_df['winner_rank'] < processed_df['loser_rank']).astype(int)
    
    # Ordenar por fecha si está disponible
    if 'tourney_date' in processed_df.columns:
        processed_df = processed_df.sort_values('tourney_date')
    
    print(f"Datos preprocesados: {processed_df.shape[0]} partidos")
    return processed_df

def calculate_elo_ratings(df, k=32, initial_elo=1500):
    """Calcular ratings ELO dinámicos para jugadores por superficie"""
    print("Calculando ELO ratings...")
    
    # Crear una copia para no modificar los datos originales
    df_elo = df.copy()
    
    # Inicializar diccionarios para tracking de ELO
    elo_ratings = {}  # ELO general
    surface_elo = {
        'clay': {},
        'grass': {},
        'hard': {},
        'carpet': {}
    }
    
    # Columnas para almacenar ELO
    df_elo['winner_elo'] = None
    df_elo['loser_elo'] = None
    df_elo['winner_surface_elo'] = None
    df_elo['loser_surface_elo'] = None
    
    # Procesar partidos cronológicamente
    for idx, match in df_elo.iterrows():
        if idx % 10000 == 0:
            print(f"Procesando partidos para ELO: {idx}")
        
        # Obtener IDs de jugadores
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        surface = match['surface'] if 'surface' in match and pd.notna(match['surface']) else 'hard'
        
        # Asegurar que el valor de superficie sea válido
        if surface not in surface_elo:
            surface = 'hard'
        
        # Asignar ELO inicial si el jugador es nuevo
        if winner_id not in elo_ratings:
            elo_ratings[winner_id] = initial_elo
        if loser_id not in elo_ratings:
            elo_ratings[loser_id] = initial_elo
            
        # Asignar ELO de superficie inicial si es nuevo
        if winner_id not in surface_elo[surface]:
            surface_elo[surface][winner_id] = initial_elo
        if loser_id not in surface_elo[surface]:
            surface_elo[surface][loser_id] = initial_elo
        
        # Guardar ELO actual
        df_elo.at[idx, 'winner_elo'] = elo_ratings[winner_id]
        df_elo.at[idx, 'loser_elo'] = elo_ratings[loser_id]
        df_elo.at[idx, 'winner_surface_elo'] = surface_elo[surface][winner_id]
        df_elo.at[idx, 'loser_surface_elo'] = surface_elo[surface][loser_id]
        
        # Calcular probabilidad esperada de victoria
        winner_expected = 1 / (1 + 10 ** ((elo_ratings[loser_id] - elo_ratings[winner_id]) / 400))
        loser_expected = 1 / (1 + 10 ** ((elo_ratings[winner_id] - elo_ratings[loser_id]) / 400))
        
        # Calcular probabilidad esperada de victoria en esta superficie
        winner_surface_expected = 1 / (1 + 10 ** ((surface_elo[surface][loser_id] - surface_elo[surface][winner_id]) / 400))
        loser_surface_expected = 1 / (1 + 10 ** ((surface_elo[surface][winner_id] - surface_elo[surface][loser_id]) / 400))
        
        # Actualizar ELO general
        elo_ratings[winner_id] += k * (1 - winner_expected)
        elo_ratings[loser_id] += k * (0 - loser_expected)
        
        # Actualizar ELO de superficie
        surface_elo[surface][winner_id] += k * 1.5 * (1 - winner_surface_expected)
        surface_elo[surface][loser_id] += k * 1.5 * (0 - loser_surface_expected)
    
    # Calcular diferencias de ELO
    df_elo['elo_difference'] = df_elo['winner_elo'] - df_elo['loser_elo']
    df_elo['surface_elo_difference'] = df_elo['winner_surface_elo'] - df_elo['loser_surface_elo']
    
    print("ELO ratings calculados")
    return df_elo

def calculate_recent_form(df, window=10):
    """Calcular forma reciente (últimos N partidos) para cada jugador"""
    print("Calculando forma reciente...")
    
    # Crear copia para no modificar el original
    df_form = df.copy()
    
    # Inicializar diccionarios para tracking de forma
    recent_matches = {}  # {player_id: list of recent results (1=win, 0=loss)}
    
    # Columnas para almacenar forma reciente
    df_form['winner_recent_winrate'] = None
    df_form['loser_recent_winrate'] = None
    
    # Procesar partidos cronológicamente
    for idx, match in df_form.iterrows():
        if idx % 10000 == 0:
            print(f"Procesando partidos para forma reciente: {idx}")
        
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        
        # Inicializar si el jugador es nuevo
        if winner_id not in recent_matches:
            recent_matches[winner_id] = []
        if loser_id not in recent_matches:
            recent_matches[loser_id] = []
        
        # Calcular tasa de victorias reciente
        winner_winrate = sum(recent_matches[winner_id]) / len(recent_matches[winner_id]) if recent_matches[winner_id] else 0.5
        loser_winrate = sum(recent_matches[loser_id]) / len(recent_matches[loser_id]) if recent_matches[loser_id] else 0.5
        
        # Guardar tasa de victorias
        df_form.at[idx, 'winner_recent_winrate'] = winner_winrate
        df_form.at[idx, 'loser_recent_winrate'] = loser_winrate
        
        # Actualizar resultados recientes
        recent_matches[winner_id].append(1)  # Victoria
        recent_matches[loser_id].append(0)   # Derrota
        
        # Mantener solo los últimos 'window' partidos
        if len(recent_matches[winner_id]) > window:
            recent_matches[winner_id] = recent_matches[winner_id][-window:]
        if len(recent_matches[loser_id]) > window:
            recent_matches[loser_id] = recent_matches[loser_id][-window:]
    
    # Calcular diferencia de forma reciente
    df_form['recent_form_difference'] = df_form['winner_recent_winrate'] - df_form['loser_recent_winrate']
    
    print("Forma reciente calculada")
    return df_form

def engineer_features(df):
    """Crear características para el modelado predictivo"""
    print("Realizando ingeniería de características...")
    
    # Aplicar cálculos de ELO
    df = calculate_elo_ratings(df)
    
    # Calcular forma reciente
    df = calculate_recent_form(df)
    
    # Seleccionar características relevantes para el modelado
    features = df[['rank_difference', 'elo_difference', 'surface_elo_difference', 
                   'recent_form_difference', 'better_rank_won']].copy()
    
    # Eliminar filas con valores faltantes
    features = features.dropna()
    
    print(f"Características generadas: {features.shape[0]} muestras")
    return features

def visualize_correlations(df):
    """Visualizar correlaciones entre características"""
    print("Generando visualizaciones...")
    
    # Matriz de correlación
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación entre Características')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Relación entre ELO y resultado
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='better_rank_won', y='elo_difference', data=df)
    plt.title('Diferencia de ELO vs Resultado del Partido')
    plt.xlabel('Ganó el jugador con mejor ranking (1=Sí, 0=No)')
    plt.ylabel('Diferencia de ELO (Ganador - Perdedor)')
    plt.tight_layout()
    plt.savefig('elo_vs_result.png')
    plt.close()
    
    print("Visualizaciones guardadas como archivos PNG")

def train_and_evaluate_models(features_df):
    """Entrenar y evaluar múltiples modelos predictivos"""
    print("Entrenando modelos predictivos...")
    
    # Separar características y variable objetivo
    X = features_df.drop('better_rank_won', axis=1)
    y = features_df['better_rank_won']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models_results = {}
    
    # 1. Baseline: Predicción basada en ELO
    print("\nEvaluando baseline (ELO)...")
    elo_predictions = (X_test['elo_difference'] > 0).astype(int)
    elo_accuracy = accuracy_score(y_test, elo_predictions)
    models_results['baseline_elo'] = {
        'accuracy': elo_accuracy,
        'report': classification_report(y_test, elo_predictions)
    }
    print(f"Baseline (ELO) Accuracy: {elo_accuracy:.4f}")
    
    # 2. Random Forest
    print("\nEntrenando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    models_results['random_forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'report': classification_report(y_test, rf_predictions)
    }
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # 3. XGBoost
    print("\nEntrenando XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    models_results['xgboost'] = {
        'model': xgb_model,
        'accuracy': xgb_accuracy,
        'report': classification_report(y_test, xgb_predictions)
    }
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Comparar resultados
    model_names = list(models_results.keys())
    accuracies = [models_results[model]['accuracy'] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    plt.bar(model_names, accuracies, color=colors)
    plt.title('Comparación de Modelos por Precisión')
    plt.xlabel('Modelo')
    plt.ylabel('Precisión')
    plt.ylim(0.5, 1.0)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Guardar el mejor modelo
    best_model_name = max(models_results, key=lambda k: models_results[k]['accuracy'] if 'model' in models_results[k] else 0)
    
    if 'model' in models_results[best_model_name]:
        best_model = models_results[best_model_name]['model']
        model_path = MODELS_DIR / 'best_tennis_model.pkl'
        joblib.dump(best_model, model_path)
        print(f"\nMejor modelo ({best_model_name}) guardado en: {model_path}")
    
    # Mostrar matriz de confusión para el mejor modelo
    if 'model' in models_results[best_model_name]:
        plt.figure(figsize=(8, 6))
        best_predictions = models_results[best_model_name]['model'].predict(X_test)
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {best_model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    # Mostrar importancia de características para Random Forest
    if 'model' in models_results['random_forest']:
        feature_importance = models_results['random_forest']['model'].feature_importances_
        features = X.columns
        
        plt.figure(figsize=(10, 6))
        sorted_idx = feature_importance.argsort()
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
        plt.title('Importancia de Características (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return models_results

def find_player_by_name(df, player_name):
    """
    Busca un jugador por su nombre en la base de datos y devuelve su ID
    """
    # Buscar en ganadores
    winner_matches = df[df['winner_name'].str.lower().str.contains(player_name.lower(), na=False)]
    if not winner_matches.empty:
        player_id = winner_matches.iloc[0]['winner_id']
        return player_id
    
    # Buscar en perdedores si no se encontró como ganador
    loser_matches = df[df['loser_name'].str.lower().str.contains(player_name.lower(), na=False)]
    if not loser_matches.empty:
        player_id = loser_matches.iloc[0]['loser_id']
        return player_id
    
    return None

def get_player_features(df, player_id):
    """
    Obtiene las características más recientes de un jugador
    """
    # Obtener últimos partidos donde el jugador participó
    player_matches = df[(df['winner_id'] == player_id) | (df['loser_id'] == player_id)]
    if player_matches.empty:
        return None
    
    # Ordenar por fecha (más recientes primero)
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)
    
    # Obtener el partido más reciente
    last_match = player_matches.iloc[0]
    
    # Extraer características
    player_data = {}
    
    if last_match['winner_id'] == player_id:
        player_data['name'] = last_match.get('winner_name', f"Jugador {player_id}")
        player_data['rank'] = last_match.get('winner_rank', 100)
        player_data['elo'] = last_match.get('winner_elo', 1500)
        player_data['surface_elo'] = last_match.get('winner_surface_elo', 1500)
        player_data['recent_winrate'] = last_match.get('winner_recent_winrate', 0.5)
    else:
        player_data['name'] = last_match.get('loser_name', f"Jugador {player_id}")
        player_data['rank'] = last_match.get('loser_rank', 100)
        player_data['elo'] = last_match.get('loser_elo', 1500)
        player_data['surface_elo'] = last_match.get('surface_elo', 1500)
        player_data['recent_winrate'] = last_match.get('loser_recent_winrate', 0.5)
    
    return player_data

def predict_match(model, player1, player2, surface='hard'):
    """
    Predice el ganador de un partido entre dos jugadores
    """
    # Calcular características para el partido
    features = {}
    features['rank_difference'] = player1['rank'] - player2['rank']
    features['elo_difference'] = player1['elo'] - player2['elo']
    features['surface_elo_difference'] = player1['surface_elo'] - player2['surface_elo']
    features['recent_form_difference'] = player1['recent_winrate'] - player2['recent_winrate']
    
    # Convertir a DataFrame
    match_features = pd.DataFrame([features])
    
    # Predecir
    prediction = model.predict(match_features)[0]
    probability = model.predict_proba(match_features)[0]
    
    # Si prediction es 1, gana el jugador con mejor ranking (menor número)
    if prediction == 1:
        # Verificar quién tiene mejor ranking
        if player1['rank'] < player2['rank']:
            winner = player1
            loser = player2
            win_prob = probability[1]
        else:
            winner = player2
            loser = player1
            win_prob = probability[1]
    else:
        # Verificar quién tiene peor ranking
        if player1['rank'] > player2['rank']:
            winner = player1
            loser = player2
            win_prob = probability[0]
        else:
            winner = player2
            loser = player1
            win_prob = probability[0]
    
    return {
        'winner': winner,
        'loser': loser,
        'probability': win_prob
    }

def simulate_tournament(processed_data, model, player_names, surface='hard'):
    """
    Simula un torneo de tenis con los jugadores especificados
    """
    print(f"\n=== SIMULACIÓN DE TORNEO DE TENIS ({surface.upper()}) ===\n")
    
    # Verificar que tengamos suficientes jugadores
    if len(player_names) != 28:
        raise ValueError("Se requieren exactamente 28 jugadores para el torneo")
    
    # Buscar IDs de los jugadores
    players = []
    for name in player_names:
        player_id = find_player_by_name(processed_data, name)
        if player_id:
            player_data = get_player_features(processed_data, player_id)
            if player_data:
                players.append(player_data)
                print(f"Jugador encontrado: {player_data['name']} (Rank: {player_data['rank']})")
            else:
                print(f"No se encontraron características para: {name}")
        else:
            print(f"Jugador no encontrado: {name}")
    
    if len(players) < 8:
        raise ValueError("No hay suficientes jugadores válidos para el torneo")
    
    # Organizar el torneo (28 jugadores)
    # Estructura: 4 partidos en primera ronda, los ganadores avanzan a R16
    print("\n=== PRIMERA RONDA (8 jugadores) ===")
    first_round_players = players[:8]
    first_round_winners = []
    
    for i in range(0, len(first_round_players), 2):
        p1 = first_round_players[i]
        p2 = first_round_players[i+1]
        print(f"\nPartido: {p1['name']} vs {p2['name']}")
        
        result = predict_match(model, p1, p2, surface)
        winner = result['winner']
        loser = result['loser']
        prob = result['probability']
        
        print(f"Predicción: {winner['name']} vence a {loser['name']} (probabilidad: {prob:.2f})")
        first_round_winners.append(winner)
    
    # Segunda ronda (16 jugadores: 4 ganadores de primera ronda + 12 jugadores restantes)
    print("\n=== SEGUNDA RONDA (16 jugadores) ===")
    second_round_players = first_round_winners + players[8:20]
    second_round_winners = []
    
    for i in range(0, len(second_round_players), 2):
        p1 = second_round_players[i]
        p2 = second_round_players[i+1]
        print(f"\nPartido: {p1['name']} vs {p2['name']}")
        
        result = predict_match(model, p1, p2, surface)
        winner = result['winner']
        loser = result['loser']
        prob = result['probability']
        
        print(f"Predicción: {winner['name']} vence a {loser['name']} (probabilidad: {prob:.2f})")
        second_round_winners.append(winner)
    
    # Cuartos de final (8 jugadores)
    print("\n=== CUARTOS DE FINAL (8 jugadores) ===")
    quarter_final_players = second_round_winners + players[20:28]
    quarter_final_winners = []
    
    for i in range(0, len(quarter_final_players), 2):
        p1 = quarter_final_players[i]
        p2 = quarter_final_players[i+1]
        print(f"\nPartido: {p1['name']} vs {p2['name']}")
        
        result = predict_match(model, p1, p2, surface)
        winner = result['winner']
        loser = result['loser']
        prob = result['probability']
        
        print(f"Predicción: {winner['name']} vence a {loser['name']} (probabilidad: {prob:.2f})")
        quarter_final_winners.append(winner)
    
    # Semifinales (4 jugadores)
    print("\n=== SEMIFINALES (4 jugadores) ===")
    semifinal_winners = []
    
    for i in range(0, len(quarter_final_winners), 2):
        p1 = quarter_final_winners[i]
        p2 = quarter_final_winners[i+1]
        print(f"\nPartido: {p1['name']} vs {p2['name']}")
        
        result = predict_match(model, p1, p2, surface)
        winner = result['winner']
        loser = result['loser']
        prob = result['probability']
        
        print(f"Predicción: {winner['name']} vence a {loser['name']} (probabilidad: {prob:.2f})")
        semifinal_winners.append(winner)
    
    # Final (2 jugadores)
    print("\n=== FINAL ===")
    p1 = semifinal_winners[0]
    p2 = semifinal_winners[1]
    print(f"\nPartido: {p1['name']} vs {p2['name']}")
    
    result = predict_match(model, p1, p2, surface)
    winner = result['winner']
    loser = result['loser']
    prob = result['probability']
    
    print(f"\nCAMPEÓN DEL TORNEO: {winner['name']} (probabilidad: {prob:.2f})")
    
    return winner

def run_tournament_simulation():
    """
    Ejecuta una simulación de torneo completa
    """
    print("=== SIMULACIÓN DE TORNEO DE TENIS ===")
    
    # 1. Cargar datos históricos
    try:
        data = load_atp_data(start_year=2010, end_year=2022)
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return
    
    # 2. Preprocesar datos
    processed_data = preprocess_data(data)
    
    # 3. Calcular características
    processed_data = calculate_elo_ratings(processed_data)
    processed_data = calculate_recent_form(processed_data)
    
    # 4. Cargar el modelo entrenado
    model_path = MODELS_DIR / 'best_tennis_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")
    else:
        print("No se encontró un modelo previamente entrenado. Entrenando nuevo modelo...")
        features_df = engineer_features(processed_data)
        models_results = train_and_evaluate_models(features_df)
        best_model_name = max(models_results, key=lambda k: models_results[k]['accuracy'] if 'model' in models_results[k] else 0)
        model = models_results[best_model_name]['model']
    
    # 5. Pedir nombres de los jugadores
    print("\nIngrese los nombres de 28 jugadores para el torneo (uno por línea):")
    player_names = []
    for i in range(28):
        while True:
            name = input(f"Jugador {i+1}: ")
            if name:
                player_names.append(name)
                break
            else:
                print("Nombre inválido, intente nuevamente.")
    
    # 6. Elegir superficie
    surfaces = ['hard', 'clay', 'grass', 'carpet']
    print("\nElija la superficie del torneo:")
    for i, surface in enumerate(surfaces):
        print(f"{i+1}. {surface.capitalize()}")
    
    while True:
        try:
            choice = int(input("Opción (1-4): "))
            if 1 <= choice <= 4:
                surface = surfaces[choice-1]
                break
            else:
                print("Opción inválida, intente nuevamente.")
        except ValueError:
            print("Por favor ingrese un número.")
    
    # 7. Simular el torneo
    champion = simulate_tournament(processed_data, model, player_names, surface)
    
    print(f"\n¡{champion['name']} es el campeón del torneo!")
    print("\n=== SIMULACIÓN COMPLETADA ===")

# Modificar la función main para incluir la opción de simular un torneo
def main():
    """Función principal con menú de opciones"""
    print("=== SISTEMA DE PREDICCIÓN DE PARTIDOS DE TENIS ===")
    print("\nOPCIONES:")
    print("1. Entrenar modelo con datos históricos")
    print("2. Simular un torneo")
    print("3. Salir")
    
    while True:
        try:
            choice = int(input("\nSeleccione una opción (1-3): "))
            if choice == 1:
                train_model()
                break
            elif choice == 2:
                run_tournament_simulation()
                break
            elif choice == 3:
                print("Saliendo...")
                break
            else:
                print("Opción inválida, intente nuevamente.")
        except ValueError:
            print("Por favor ingrese un número.")

def train_model():
    """Entrena el modelo predictivo con datos históricos"""
    # 1. Cargar datos
    try:
        data = load_atp_data(start_year=2010, end_year=2022)
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return
    
    # 2. Preprocesar datos
    processed_data = preprocess_data(data)
    
    # 3. Ingeniería de características
    features_df = engineer_features(processed_data)
    
    # 4. Análisis exploratorio y visualizaciones
    visualize_correlations(features_df)
    
    # 5. Entrenar y evaluar modelos
    models_results = train_and_evaluate_models(features_df)
    
    print("\n=== ENTRENAMIENTO COMPLETADO CON ÉXITO ===")
    print("Resultados guardados en archivos PNG y el mejor modelo guardado en /models")

if __name__ == "__main__":
    main()