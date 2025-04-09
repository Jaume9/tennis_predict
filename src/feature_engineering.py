import pandas as pd
import numpy as np

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