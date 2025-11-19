import pandas as pd
from scipy.stats import poisson
import numpy as np
import unicodedata

# --- CARGA DE DATOS E INICIALIZACIÓN (Se ejecuta una sola vez al importar) ---
try:
    df_partidos = pd.read_csv('data/libertadores-results-ds.csv')
except FileNotFoundError:
    # Fallback para evitar que rompa si no encuentra el archivo al inicio
    print("Error: No se encontró 'data/libertadores-results-ds.csv'")
    df_partidos = pd.DataFrame(columns=['Home Club', 'Away Club', 'Home Score', 'AwayScore'])

# Medias Globales
mu_local_anotados = df_partidos['Home Score'].mean()
mu_visitante_anotados = df_partidos['AwayScore'].mean()

# Agrupaciones
gal = df_partidos.groupby('Home Club')['Home Score'].mean().rename('GAL')
grl = df_partidos.groupby('Home Club')['AwayScore'].mean().rename('GRL')
gav = df_partidos.groupby('Away Club')['AwayScore'].mean().rename('GAV')
grv = df_partidos.groupby('Away Club')['Home Score'].mean().rename('GRV')

# DataFrame consolidado de fuerzas
df_promedios_equipo = pd.concat([gal, grl], axis=1)
df_promedios_equipo = df_promedios_equipo.merge(
    pd.concat([gav, grv], axis=1),
    left_index=True, right_index=True, how='outer'
)
df_promedios_equipo.index.name = 'Equipo'

# Cálculo de Fuerzas (Ataque y Defensa)
if not df_partidos.empty:
    df_promedios_equipo['FA_L'] = df_promedios_equipo['GAL'] / mu_local_anotados
    df_promedios_equipo['FA_V'] = df_promedios_equipo['GAV'] / mu_visitante_anotados
    df_promedios_equipo['FD_L'] = df_promedios_equipo['GRL'] / mu_visitante_anotados
    df_promedios_equipo['FD_V'] = df_promedios_equipo['GRV'] / mu_local_anotados

# --- FUNCIONES PÚBLICAS PARA LA API ---

def obtener_lista_equipos():
    """Devuelve la lista de equipos ordenada alfabéticamente."""
    if df_promedios_equipo.empty:
        return []
    # Usamos el índice del dataframe procesado para asegurar que el equipo tiene estadísticas
    return sorted(df_promedios_equipo.index.tolist())

def obtener_prediccion(equipo_Local, equipo_Visitante):
    """
    Calcula probabilidades Poisson y devuelve un diccionario
    listo para ser consumido por el frontend.
    """
    # Verificar existencia
    if equipo_Local not in df_promedios_equipo.index or equipo_Visitante not in df_promedios_equipo.index:
        raise ValueError("Uno de los equipos no se encuentra en la base de datos.")

    # Extracción de Fuerzas
    FA_L_A = df_promedios_equipo.loc[equipo_Local, 'FA_L']
    FD_V_B = df_promedios_equipo.loc[equipo_Visitante, 'FD_V']
    FA_V_B = df_promedios_equipo.loc[equipo_Visitante, 'FA_V']
    FD_L_A = df_promedios_equipo.loc[equipo_Local, 'FD_L']

    # Cálculo de Lambdas
    lambda_A = FA_L_A * FD_V_B * mu_local_anotados
    lambda_B = FA_V_B * FD_L_A * mu_visitante_anotados

    # Poisson y Matriz
    MAX_GOLES = 7
    rango_goles = np.arange(MAX_GOLES + 1)
    prob_A = poisson.pmf(rango_goles, lambda_A)
    prob_B = poisson.pmf(rango_goles, lambda_B)
    matriz_probabilidad = np.outer(prob_A, prob_B)

    # Sumatorias de victoria/empate
    prob_gana_Local = np.sum(np.tril(matriz_probabilidad, k=-1))
    prob_empate = np.sum(np.diag(matriz_probabilidad))
    prob_gana_Visitante = np.sum(np.triu(matriz_probabilidad, k=1))

    # Retornamos las claves exactas que el JS espera
    return {
        'prob_local': round(prob_gana_Local * 100, 2),
        'prob_empate': round(prob_empate * 100, 2),
        'prob_visitante': round(prob_gana_Visitante * 100, 2)
    }