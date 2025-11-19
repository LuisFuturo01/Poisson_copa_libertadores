import pandas as pd
from scipy.stats import poisson
import numpy as np
# 1. Medias de Goles Globales del Torneo (2011-2023)
df_partidos=pd.read_csv('data/libertadores-results-ds.csv')
# Media Global de Goles Anotados por el LOCAL (Home Score)
mu_local_anotados = df_partidos['Home Score'].mean()
# Media Global de Goles Anotados por el VISITANTE (Away Score)
mu_visitante_anotados = df_partidos['AwayScore'].mean()
print(f"Media Global Goles Local: {mu_local_anotados:.3f}")
print(f"Media Global Goles Visitante: {mu_visitante_anotados:.3f}")
# Goles Anotados Local (GAL)
gal = df_partidos.groupby('Home Club')['Home Score'].mean().rename('GAL')
# Goles Recibidos Local (GRL). Son los goles que anota el visitante (Away Score)
grl = df_partidos.groupby('Home Club')['AwayScore'].mean().rename('GRL')
# Combinar en un DataFrame inicial
df_promedios_equipo = pd.concat([gal, grl], axis=1)
print(df_promedios_equipo.head())
# Goles Anotados Visitante (GAV)
gav = df_partidos.groupby('Away Club')['AwayScore'].mean().rename('GAV')

# Goles Recibidos Visitante (GRV). Son los goles que anota el local (Home Score)
grv = df_partidos.groupby('Away Club')['Home Score'].mean().rename('GRV')

# Agregar las columnas al DataFrame de promedios, uniendo por el nombre del club
df_promedios_equipo = df_promedios_equipo.merge(
    pd.concat([gav, grv], axis=1),
    left_index=True,
    right_index=True,
    how='outer'
)

# El índice del DataFrame resultante es el nombre del equipo
df_promedios_equipo.index.name = 'Equipo'
print(df_promedios_equipo.head())




# 2. Cálculo de Fuerzas de Ataque (FA)

# Fuerza de Ataque Local (FA_L): GAL / Media Global Local
df_promedios_equipo['FA_L'] = df_promedios_equipo['GAL'] / mu_local_anotados

# Fuerza de Ataque Visitante (FA_V): GAV / Media Global Visitante
df_promedios_equipo['FA_V'] = df_promedios_equipo['GAV'] / mu_visitante_anotados

# 3. Cálculo de Fuerzas de Defensa (FD)

# Fuerza de Defensa Local (FD_L): GRL / Media Global Visitante
# Un equipo que defiende en casa lo hace contra el ataque del visitante.
df_promedios_equipo['FD_L'] = df_promedios_equipo['GRL'] / mu_visitante_anotados

# Fuerza de Defensa Visitante (FD_V): GRV / Media Global Local
# Un equipo que defiende fuera lo hace contra el ataque del local.
df_promedios_equipo['FD_V'] = df_promedios_equipo['GRV'] / mu_local_anotados
print(df_promedios_equipo.head())

equipo_Local='The Strongest'
equipo_Visitante='Bolívar'

# --- Extracción de Fuerzas ---
# FA_L de A (Ataque Local de River)
FA_L_A = df_promedios_equipo.loc[equipo_Local, 'FA_L']
# FD_V de B (Defensa Visitante de Boca)
FD_V_B = df_promedios_equipo.loc[equipo_Visitante, 'FD_V']
# FA_V de B (Ataque Visitante de Boca)
FA_V_B = df_promedios_equipo.loc[equipo_Visitante, 'FA_V']
# FD_L de A (Defensa Local de River)
FD_L_A = df_promedios_equipo.loc[equipo_Local, 'FD_L']

# --- Cálculo de Lambdas ---
# Goles esperados para el Equipo A (Local)
lambda_A = FA_L_A * FD_V_B * mu_local_anotados

# Goles esperados para el Equipo B (Visitante)
lambda_B = FA_V_B * FD_L_A * mu_visitante_anotados

print(f"\n{equipo_Local} (Local) -> Goles Esperados (λA): {lambda_A:.3f}")
print(f"{equipo_Visitante} (Visitante) -> Goles Esperados (λB): {lambda_B:.3f}")




# Definimos el rango de goles (0 a 7 es un buen límite, ya que P(k>7) es casi cero)
MAX_GOLES = 7
rango_goles = np.arange(MAX_GOLES + 1)

# 1. Calcular las probabilidades marginales
# Probabilidad de que A anote 0, 1, 2, ..., 7 goles
prob_A = poisson.pmf(rango_goles, lambda_A)
# Probabilidad de que B anote 0, 1, 2, ..., 7 goles
prob_B = poisson.pmf(rango_goles, lambda_B)

# 2. Generar la Matriz de Probabilidad Conjunta (P(A=i y B=j))
# Se usa np.outer para multiplicar todas las combinaciones (independencia)
matriz_probabilidad = np.outer(prob_A, prob_B)

# Opcional: Mostrar la probabilidad de un marcador específico (ejemplo 1-1)
P_1_1 = matriz_probabilidad[1, 1]
print(f"\nProbabilidad del marcador 1-1: {P_1_1 * 100:.2f}%")


# 1. Probabilidad de Victoria del Local A
# Suma de elementos donde la fila (goles de A) > columna (goles de B)
# Esto corresponde a la suma de los elementos bajo la diagonal principal (k=-1)
prob_gana_Local = np.sum(np.tril(matriz_probabilidad, k=-1))

# Probabilidad de Empate
# Suma de elementos donde la fila = columna (Ej: 0-0, 1-1, 2-2, etc.)
# Esto corresponde a la suma de los elementos en la diagonal principal
prob_empate = np.sum(np.diag(matriz_probabilidad))

# 3. Probabilidad de Victoria del Visitante B
# Suma de elementos donde la fila (goles de A) < columna (goles de B)
# Esto corresponde a la suma de los elementos sobre la diagonal principal (k=1)
prob_gana_Visitante = np.sum(np.triu(matriz_probabilidad, k=1))

print("\n--- Resultados del Modelo ---")
print(f"Probabilidad de que Gane {equipo_Local}: {prob_gana_Local * 100:.2f}%")
print(f"Probabilidad de Empate: {prob_empate * 100:.2f}%")
print(f"Probabilidad de que Gane {equipo_Visitante}: {prob_gana_Visitante * 100:.2f}%")