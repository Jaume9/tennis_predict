import pandas as pd

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
            player1 = match['player1']['name']
            player2 = match['player2']['name']
            winner = match['winner']['name']
            prob = match['probability']
            print(f"{i+1}. {player1} vs {player2} → {winner} ({prob:.2f})")
    
    champion = results['final'][0]['winner']['name']
    print("\n===============================================================")
    print(f"                  CHAMPION: {champion}")
    print("===============================================================")

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