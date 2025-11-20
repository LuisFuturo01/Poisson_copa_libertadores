import pandas as pd
from scipy.stats import poisson
import numpy as np

df_partidos=pd.read_csv('data/libertadores-results-ds.csv')

mu_local_anotados = df_partidos['Home Score'].mean()

mu_visitante_anotados = df_partidos['AwayScore'].mean()
print(f"Media Global Goles Local: {mu_local_anotados:.3f}")
print(f"Media Global Goles Visitante: {mu_visitante_anotados:.3f}")

gal = df_partidos.groupby('Home Club')['Home Score'].mean().rename('GAL')

grl = df_partidos.groupby('Home Club')['AwayScore'].mean().rename('GRL')

df_promedios_equipo = pd.concat([gal, grl], axis=1)
print(df_promedios_equipo.head())

gav = df_partidos.groupby('Away Club')['AwayScore'].mean().rename('GAV')


grv = df_partidos.groupby('Away Club')['Home Score'].mean().rename('GRV')


df_promedios_equipo = df_promedios_equipo.merge(
    pd.concat([gav, grv], axis=1),
    left_index=True,
    right_index=True,
    how='outer'
)


df_promedios_equipo.index.name = 'Equipo'
print(df_promedios_equipo.head())







df_promedios_equipo['FA_L'] = df_promedios_equipo['GAL'] / mu_local_anotados


df_promedios_equipo['FA_V'] = df_promedios_equipo['GAV'] / mu_visitante_anotados





df_promedios_equipo['FD_L'] = df_promedios_equipo['GRL'] / mu_visitante_anotados



df_promedios_equipo['FD_V'] = df_promedios_equipo['GRV'] / mu_local_anotados
print(df_promedios_equipo.head())

equipo_Local='The Strongest'
equipo_Visitante='Bolívar'



FA_L_A = df_promedios_equipo.loc[equipo_Local, 'FA_L']

FD_V_B = df_promedios_equipo.loc[equipo_Visitante, 'FD_V']

FA_V_B = df_promedios_equipo.loc[equipo_Visitante, 'FA_V']

FD_L_A = df_promedios_equipo.loc[equipo_Local, 'FD_L']



lambda_A = FA_L_A * FD_V_B * mu_local_anotados


lambda_B = FA_V_B * FD_L_A * mu_visitante_anotados

print(f"\n{equipo_Local} (Local) -> Goles Esperados (λA): {lambda_A:.3f}")
print(f"{equipo_Visitante} (Visitante) -> Goles Esperados (λB): {lambda_B:.3f}")





MAX_GOLES = 7
rango_goles = np.arange(MAX_GOLES + 1)



prob_A = poisson.pmf(rango_goles, lambda_A)

prob_B = poisson.pmf(rango_goles, lambda_B)



matriz_probabilidad = np.outer(prob_A, prob_B)


P_1_1 = matriz_probabilidad[1, 1]
print(f"\nProbabilidad del marcador 1-1: {P_1_1 * 100:.2f}%")





prob_gana_Local = np.sum(np.tril(matriz_probabilidad, k=-1))




prob_empate = np.sum(np.diag(matriz_probabilidad))




prob_gana_Visitante = np.sum(np.triu(matriz_probabilidad, k=1))

print("\n--- Resultados del Modelo ---")
print(f"Probabilidad de que Gane {equipo_Local}: {prob_gana_Local * 100:.2f}%")
print(f"Probabilidad de Empate: {prob_empate * 100:.2f}%")
print(f"Probabilidad de que Gane {equipo_Visitante}: {prob_gana_Visitante * 100:.2f}%")