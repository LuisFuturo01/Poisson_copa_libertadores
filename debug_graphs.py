import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback

try:
    print("Reading CSV...")
    df_partidos = pd.read_csv('data/libertadores-results-ds.csv', encoding='utf-8')
    print(f"Rows: {len(df_partidos)}")
    print(f"Columns: {df_partidos.columns.tolist()}")
    
    df_partidos.columns = [c.strip() for c in df_partidos.columns]
    df_partidos.rename(columns={'AwayScore': 'Away Score', 'HomeScore': 'Home Score'}, inplace=True)
    
    print("Columns after rename:", df_partidos.columns.tolist())
    
    df_partidos['Home Club'] = df_partidos['Home Club'].astype(str).str.strip().replace('', np.nan)
    df_partidos['Away Club'] = df_partidos['Away Club'].astype(str).str.strip().replace('', np.nan)
    df_partidos['Home Score'] = pd.to_numeric(df_partidos['Home Score'], errors='coerce')
    df_partidos['Away Score'] = pd.to_numeric(df_partidos['Away Score'], errors='coerce')
    
    print("Home Score NaNs:", df_partidos['Home Score'].isna().sum())
    
    equipos = pd.Index(df_partidos['Home Club'].dropna().unique()).union(
        pd.Index(df_partidos['Away Club'].dropna().unique())
    ).sort_values()
    
    print(f"Equipos found: {len(equipos)}")
    
    # Check the boxplot data
    home_prob_victoria = df_partidos.groupby('Home Club').apply(lambda x: (x['Home Score'] > x['Away Score']).sum() / len(x)).reindex(equipos)
    away_prob_victoria = df_partidos.groupby('Away Club').apply(lambda x: (x['Away Score'] > x['Home Score']).sum() / len(x)).reindex(equipos)
    
    prob_local = home_prob_victoria.dropna().values
    prob_visita = away_prob_victoria.dropna().values
    
    print(f"Prob local len: {len(prob_local)}")
    print(f"Prob visita len: {len(prob_visita)}")
    
    if len(prob_local) == 0 or len(prob_visita) == 0:
        print("WARNING: Empty probability data for boxplot")
        
    # Try to plot
    print("Attempting plot...")
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([prob_local, prob_visita],
                    labels=['Prob. ganar (local)', 'Prob. ganar (visitante)'],
                    patch_artist=True,
                    showfliers=False)
    print("Plot created successfully (not shown)")
    
except Exception:
    traceback.print_exc()
