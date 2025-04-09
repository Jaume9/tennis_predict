import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

# Configure directories
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_atp_data(start_year=2010, end_year=2023):
    """Load ATP match data for a range of years"""
    print(f"Loading ATP data from {start_year} to {end_year}...")
    all_files = []
    
    # Find files matching pattern atp_matches_YYYY.csv
    for year in range(start_year, end_year + 1):
        pattern = DATA_DIR / f"atp_matches_{year}.csv"
        files = glob.glob(str(pattern))
        all_files.extend(files)
    
    if not all_files:
        raise ValueError(f"No files found for years {start_year}-{end_year}")
    
    # Load and combine files
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            print(f"Loaded: {os.path.basename(file)}, {df.shape[0]} records")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined ATP data: {combined_df.shape[0]} matches")
    
    return combined_df

def preprocess_data(df):
    """Clean and preprocess data for analysis"""
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Convert dates to datetime format
    if 'tourney_date' in processed_df.columns:
        processed_df['tourney_date'] = pd.to_datetime(processed_df['tourney_date'], format='%Y%m%d')
    
    # Drop rows with missing values in critical columns
    critical_columns = ['winner_id', 'loser_id', 'winner_rank', 'loser_rank', 'surface']
    critical_columns = [col for col in critical_columns if col in processed_df.columns]
    processed_df = processed_df.dropna(subset=critical_columns)
    
    # Create ranking difference column
    processed_df['rank_difference'] = processed_df['winner_rank'] - processed_df['loser_rank']
    
    # Normalize surface names
    if 'surface' in processed_df.columns:
        surface_mapping = {
            'Hard': 'hard',
            'Clay': 'clay',
            'Grass': 'grass',
            'Carpet': 'carpet'
        }
        processed_df['surface'] = processed_df['surface'].map(lambda x: surface_mapping.get(x, x.lower()) if isinstance(x, str) else x)
    
    # Create target variable: 1 if player with better ranking wins, 0 otherwise
    processed_df['better_rank_won'] = (processed_df['winner_rank'] < processed_df['loser_rank']).astype(int)
    
    # Sort by date if available
    if 'tourney_date' in processed_df.columns:
        processed_df = processed_df.sort_values('tourney_date')
    
    print(f"Preprocessed data: {processed_df.shape[0]} matches")
    return processed_df

def calculate_elo_ratings(df, k=32, initial_elo=1500):
    """Calculate dynamic ELO ratings for players by surface"""
    print("Calculating ELO ratings...")
    
    # Create a copy to avoid modifying original data
    df_elo = df.copy()
    
    # Initialize dictionaries for ELO tracking
    elo_ratings = {}  # General ELO
    surface_elo = {
        'clay': {},
        'grass': {},
        'hard': {},
        'carpet': {}
    }
    
    # Columns to store ELO
    df_elo['winner_elo'] = None
    df_elo['loser_elo'] = None
    df_elo['winner_surface_elo'] = None
    df_elo['loser_surface_elo'] = None
    
    # Process matches chronologically
    for idx, match in df_elo.iterrows():
        if idx % 10000 == 0:
            print(f"Processing matches for ELO: {idx}")
        
        # Get player IDs
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        surface = match['surface'] if 'surface' in match and pd.notna(match['surface']) else 'hard'
        
        # Ensure surface value is valid
        if surface not in surface_elo:
            surface = 'hard'
        
        # Assign initial ELO if player is new
        if winner_id not in elo_ratings:
            elo_ratings[winner_id] = initial_elo
        if loser_id not in elo_ratings:
            elo_ratings[loser_id] = initial_elo
            
        # Assign initial surface ELO if new
        if winner_id not in surface_elo[surface]:
            surface_elo[surface][winner_id] = initial_elo
        if loser_id not in surface_elo[surface]:
            surface_elo[surface][loser_id] = initial_elo
        
        # Save current ELO
        df_elo.at[idx, 'winner_elo'] = elo_ratings[winner_id]
        df_elo.at[idx, 'loser_elo'] = elo_ratings[loser_id]
        df_elo.at[idx, 'winner_surface_elo'] = surface_elo[surface][winner_id]
        df_elo.at[idx, 'loser_surface_elo'] = surface_elo[surface][loser_id]
        
        # Calculate expected win probability
        winner_expected = 1 / (1 + 10 ** ((elo_ratings[loser_id] - elo_ratings[winner_id]) / 400))
        loser_expected = 1 / (1 + 10 ** ((elo_ratings[winner_id] - elo_ratings[loser_id]) / 400))
        
        # Calculate expected win probability on this surface
        winner_surface_expected = 1 / (1 + 10 ** ((surface_elo[surface][loser_id] - surface_elo[surface][winner_id]) / 400))
        loser_surface_expected = 1 / (1 + 10 ** ((surface_elo[surface][winner_id] - surface_elo[surface][loser_id]) / 400))
        
        # Update general ELO
        elo_ratings[winner_id] += k * (1 - winner_expected)
        elo_ratings[loser_id] += k * (0 - loser_expected)
        
        # Update surface ELO
        surface_elo[surface][winner_id] += k * 1.5 * (1 - winner_surface_expected)
        surface_elo[surface][loser_id] += k * 1.5 * (0 - loser_surface_expected)
    
    # Calculate ELO differences
    df_elo['elo_difference'] = df_elo['winner_elo'] - df_elo['loser_elo']
    df_elo['surface_elo_difference'] = df_elo['winner_surface_elo'] - df_elo['loser_surface_elo']
    
    print("ELO ratings calculated")
    return df_elo

def calculate_recent_form(df, window=10):
    """Calculate recent form (last N matches) for each player"""
    print("Calculating recent form...")
    
    # Create copy to avoid modifying original
    df_form = df.copy()
    
    # Initialize dictionaries for form tracking
    recent_matches = {}  # {player_id: list of recent results (1=win, 0=loss)}
    
    # Columns to store recent form
    df_form['winner_recent_winrate'] = None
    df_form['loser_recent_winrate'] = None
    
    # Process matches chronologically
    for idx, match in df_form.iterrows():
        if idx % 10000 == 0:
            print(f"Processing matches for recent form: {idx}")
        
        winner_id = match['winner_id']
        loser_id = match['loser_id']
        
        # Initialize if player is new
        if winner_id not in recent_matches:
            recent_matches[winner_id] = []
        if loser_id not in recent_matches:
            recent_matches[loser_id] = []
        
        # Calculate recent win rate
        winner_winrate = sum(recent_matches[winner_id]) / len(recent_matches[winner_id]) if recent_matches[winner_id] else 0.5
        loser_winrate = sum(recent_matches[loser_id]) / len(recent_matches[loser_id]) if recent_matches[loser_id] else 0.5
        
        # Save win rates
        df_form.at[idx, 'winner_recent_winrate'] = winner_winrate
        df_form.at[idx, 'loser_recent_winrate'] = loser_winrate
        
        # Update recent results
        recent_matches[winner_id].append(1)  # Win
        recent_matches[loser_id].append(0)   # Loss
        
        # Keep only the last 'window' matches
        if len(recent_matches[winner_id]) > window:
            recent_matches[winner_id] = recent_matches[winner_id][-window:]
        if len(recent_matches[loser_id]) > window:
            recent_matches[loser_id] = recent_matches[loser_id][-window:]
    
    # Calculate recent form difference
    df_form['recent_form_difference'] = df_form['winner_recent_winrate'] - df_form['loser_recent_winrate']
    
    print("Recent form calculated")
    return df_form

def engineer_features(df):
    """Create features for predictive modeling"""
    print("Performing feature engineering...")
    
    # Apply ELO calculations
    df = calculate_elo_ratings(df)
    
    # Calculate recent form
    df = calculate_recent_form(df)
    
    # Select relevant features for modeling
    features = df[['rank_difference', 'elo_difference', 'surface_elo_difference', 
                   'recent_form_difference', 'better_rank_won']].copy()
    
    # Drop rows with missing values
    features = features.dropna()
    
    print(f"Features generated: {features.shape[0]} samples")
    return features, df

def train_decision_tree_model(features_df):
    """Train a Decision Tree model for match prediction"""
    print("Training Decision Tree model...")
    
    # Separate features and target variable
    X = features_df.drop('better_rank_won', axis=1)
    y = features_df['better_rank_won']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Evaluate model
    dt_predictions = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, dt_predictions))
    
    # Save the model
    model_path = MODELS_DIR / 'decision_tree_model.pkl'
    joblib.dump(dt_model, model_path)
    print(f"Decision Tree model saved to: {model_path}")
    
    # Feature importance
    feature_importance = dt_model.feature_importances_
    features = X.columns
    
    plt.figure(figsize=(10, 6))
    sorted_idx = feature_importance.argsort()
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title('Feature Importance (Decision Tree)')
    plt.tight_layout()
    plt.savefig('dt_feature_importance.png')
    
    return dt_model

def find_player_by_name(df, player_name):
    """Find a player by name in the database and return their ID"""
    # Look in winners
    winner_matches = df[df['winner_name'].str.lower().str.contains(player_name.lower(), na=False)]
    if not winner_matches.empty:
        player_id = winner_matches.iloc[0]['winner_id']
        player_name = winner_matches.iloc[0]['winner_name']
        return player_id, player_name
    
    # Look in losers if not found as winner
    loser_matches = df[df['loser_name'].str.lower().str.contains(player_name.lower(), na=False)]
    if not loser_matches.empty:
        player_id = loser_matches.iloc[0]['loser_id']
        player_name = loser_matches.iloc[0]['loser_name']
        return player_id, player_name
    
    return None, None

def get_player_features(df, player_id):
    """Get the most recent features for a player"""
    # Get latest matches where the player participated
    player_matches = df[(df['winner_id'] == player_id) | (df['loser_id'] == player_id)]
    if player_matches.empty:
        return None
    
    # Sort by date (most recent first)
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)
    
    # Get the most recent match
    last_match = player_matches.iloc[0]
    
    # Extract features
    player_data = {}
    
    if last_match['winner_id'] == player_id:
        player_data['id'] = player_id
        player_data['name'] = last_match.get('winner_name', f"Player {player_id}")
        player_data['rank'] = last_match.get('winner_rank', 100)
        player_data['elo'] = last_match.get('winner_elo', 1500)
        player_data['surface_elo'] = last_match.get('winner_surface_elo', 1500)
        player_data['recent_winrate'] = last_match.get('winner_recent_winrate', 0.5)
    else:
        player_data['id'] = player_id
        player_data['name'] = last_match.get('loser_name', f"Player {player_id}")
        player_data['rank'] = last_match.get('loser_rank', 100)
        player_data['elo'] = last_match.get('loser_elo', 1500)
        player_data['surface_elo'] = last_match.get('loser_surface_elo', 1500)
        player_data['recent_winrate'] = last_match.get('loser_recent_winrate', 0.5)
    
    return player_data

def predict_match(model, player1, player2, surface='hard'):
    """Predict the winner of a match between two players"""
    # Calculate match features
    features = {}
    features['rank_difference'] = player1['rank'] - player2['rank']
    features['elo_difference'] = player1['elo'] - player2['elo']
    features['surface_elo_difference'] = player1['surface_elo'] - player2['surface_elo']
    features['recent_form_difference'] = player1['recent_winrate'] - player2['recent_winrate']
    
    # Convert to DataFrame
    match_features = pd.DataFrame([features])
    
    # Predict
    prediction = model.predict(match_features)[0]
    probability = model.predict_proba(match_features)[0]
    
    # If prediction is 1, the player with better ranking wins (lower number)
    if prediction == 1:
        # Check who has better ranking
        if player1['rank'] < player2['rank']:
            winner = player1
            loser = player2
            win_prob = probability[1]
        else:
            winner = player2
            loser = player1
            win_prob = probability[1]
    else:
        # Check who has worse ranking
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

def simulate_round(model, players, round_name, surface='hard'):
    """Simulate a tournament round and return winners"""
    print(f"\n=== {round_name} ===")
    winners = []
    matches = []
    
    # Ensure even number of players
    if len(players) % 2 != 0:
        raise ValueError(f"Number of players must be even, got {len(players)}")
    
    # Simulate matches
    for i in range(0, len(players), 2):
        p1 = players[i]
        p2 = players[i+1]
        
        match_str = f"{p1['name']} vs {p2['name']}"
        print(f"Match {i//2+1}: {match_str}")
        
        result = predict_match(model, p1, p2, surface)
        winner = result['winner']
        loser = result['loser']
        prob = result['probability']
        
        print(f"  Prediction: {winner['name']} defeats {loser['name']} (probability: {prob:.2f})")
        
        matches.append({
            'player1': p1,
            'player2': p2,
            'winner': winner,
            'loser': loser,
            'probability': prob
        })
        
        winners.append(winner)
    
    return winners, matches

def simulate_tournament(model, players, surface='hard'):
    """Simulate a full tennis tournament with 96 players"""
    tournament_results = {
        'first_round': [],
        'second_round': [],
        'round_of_32': [],
        'round_of_16': [],
        'quarterfinals': [],
        'semifinals': [],
        'final': []
    }
    
    # First round: 32 matches, 64 players
    print(f"\n=== TOURNAMENT SIMULATION ({surface.upper()}) ===")
    first_round_players = players[:64]
    first_round_winners, first_round_matches = simulate_round(
        model, first_round_players, "FIRST ROUND (64 players)", surface
    )
    tournament_results['first_round'] = first_round_matches
    
    # Second round: first_round_winners + remaining 32 players (seeded)
    second_round_players = first_round_winners + players[64:]
    second_round_winners, second_round_matches = simulate_round(
        model, second_round_players, "SECOND ROUND (64 players)", surface
    )
    tournament_results['second_round'] = second_round_matches
    
    # Round of 32: 16 matches, 32 winners
    r32_winners, r32_matches = simulate_round(
        model, second_round_winners, "ROUND OF 32", surface
    )
    tournament_results['round_of_32'] = r32_matches
    
    # Round of 16: 8 matches, 16 winners
    r16_winners, r16_matches = simulate_round(
        model, r32_winners, "ROUND OF 16", surface
    )
    tournament_results['round_of_16'] = r16_matches
    
    # Quarterfinals: 4 matches, 8 winners
    qf_winners, qf_matches = simulate_round(
        model, r16_winners, "QUARTERFINALS", surface
    )
    tournament_results['quarterfinals'] = qf_matches
    
    # Semifinals: 2 matches, 4 winners
    sf_winners, sf_matches = simulate_round(
        model, qf_winners, "SEMIFINALS", surface
    )
    tournament_results['semifinals'] = sf_matches
    
    # Final: 1 match, champion
    final_winners, final_match = simulate_round(
        model, sf_winners, "FINAL", surface
    )
    tournament_results['final'] = final_match
    
    champion = final_winners[0]
    print(f"\n=== TOURNAMENT CHAMPION: {champion['name']} ===")
    
    return tournament_results, champion

def print_tournament_bracket(results):
    """Print a formatted view of the tournament bracket"""
    print("\n===============================================================")
    print("                     TOURNAMENT BRACKET")
    print("===============================================================")
    
    rounds = [
        ("FIRST ROUND", results['first_round']),
        ("SECOND ROUND", results['second_round']),
        ("ROUND OF 32", results['round_of_32']),
        ("ROUND OF 16", results['round_of_16']),
        ("QUARTERFINALS", results['quarterfinals']),
        ("SEMIFINALS", results['semifinals']),
        ("FINAL", results['final'])
    ]
    
    for round_name, matches in rounds:
        print(f"\n{round_name}")
        print("-" * len(round_name))
        
        for i, match in enumerate(matches):
            p1 = match['player1']['name']
            p2 = match['player2']['name']
            winner = match['winner']['name']
            prob = match['probability']
            
            print(f"{i+1}. {p1} vs {p2} => {winner} ({prob:.2f})")
    
    champion = results['final'][0]['winner']['name']
    print("\n===============================================================")
    print(f"                  CHAMPION: {champion}")
    print("===============================================================")

def main():
    """Main function to run the ATP bracket predictor"""
    print("=== ATP TOURNAMENT BRACKET PREDICTOR ===")
    
    # 1. Load historical data
    try:
        data = load_atp_data(start_year=2010, end_year=2023)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # 2. Preprocess data
    processed_data = preprocess_data(data)
    
    # 3. Engineer features
    features_df, enriched_data = engineer_features(processed_data)
    
    # 4. Load or train Decision Tree model
    model_path = MODELS_DIR / 'decision_tree_model.pkl'
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                model = joblib.load(model_path)
                print(f"Model loaded from: {model_path}")
            except (EOFError, Exception) as e:
                print(f"Error loading model: {str(e)}")
                print("Training new model...")
                model = train_decision_tree_model(features_df)
        else:
            print("No pre-trained model found or file is empty. Training new model...")
            model = train_decision_tree_model(features_df)
    except Exception as e:
        print(f"Error handling model: {str(e)}")
        return
    
    # 5. Get player names
    print("\nEnter the names of 96 players for the tournament (one per line):")
    player_names = []
    for i in range(96):
        while True:
            name = input(f"Player {i+1}: ")
            if name:
                player_names.append(name)
                break
            else:
                print("Invalid name, please try again.")
    
    # 6. Choose surface
    surfaces = ['hard', 'clay', 'grass', 'carpet']
    print("\nSelect tournament surface:")
    for i, surface in enumerate(surfaces):
        print(f"{i+1}. {surface.capitalize()}")
    
    while True:
        try:
            choice = int(input("Option (1-4): "))
            if 1 <= choice <= 4:
                surface = surfaces[choice-1]
                break
            else:
                print("Invalid option, please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # 7. Find players in database
    print("\nLooking up players in database...")
    players = []
    not_found_count = 0
    
    for name in player_names:
        player_id, full_name = find_player_by_name(enriched_data, name)
        if player_id:
            player_data = get_player_features(enriched_data, player_id)
            if player_data:
                players.append(player_data)
                print(f"Found: {player_data['name']} (Rank: {player_data['rank']})")
            else:
                # Create synthetic player if features not found
                print(f"Player {name} found but no recent matches, using default stats")
                player_data = {
                    'id': player_id,
                    'name': full_name or name,
                    'rank': 100,
                    'elo': 1500,
                    'surface_elo': 1500,
                    'recent_winrate': 0.5
                }
                players.append(player_data)
        else:
            # Create synthetic player
            not_found_count += 1
            print(f"Player not found: {name}, using default stats")
            player_data = {
                'id': -not_found_count,
                'name': name,
                'rank': 150,
                'elo': 1450,
                'surface_elo': 1450,
                'recent_winrate': 0.45
            }
            players.append(player_data)
    
    # 8. Simulate tournament
    tournament_results, champion = simulate_tournament(model, players, surface)
    
    # 9. Print detailed bracket
    print_tournament_bracket(tournament_results)
    
    print(f"\n{champion['name']} is the tournament champion!")
    print("\n=== SIMULATION COMPLETED ===")

if __name__ == "__main__":
    main()