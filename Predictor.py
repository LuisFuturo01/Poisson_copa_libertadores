import pandas as pd
import numpy as np
from scipy.stats import poisson

# Leer CSV y normalizar nombres de columnas
df_partidos = pd.read_csv('data/libertadores-results-ds.csv', encoding='utf-8')
df_partidos.columns = [c.strip() for c in df_partidos.columns]

# Normalizar nombres de columnas de puntuación si vienen en otro formato
df_partidos.rename(columns={
    'AwayScore': 'Away Score',
    'HomeScore': 'Home Score'
}, inplace=True)

# Asegurar tipos y limpiar nombres de equipos
df_partidos['Home Club'] = df_partidos['Home Club'].astype(str).str.strip()
df_partidos['Away Club'] = df_partidos['Away Club'].astype(str).str.strip()
df_partidos['Home Score'] = pd.to_numeric(df_partidos['Home Score'], errors='coerce')
df_partidos['Away Score'] = pd.to_numeric(df_partidos['Away Score'], errors='coerce')

# Tratar cadenas vacías como NaN para evitar equipos vacíos
df_partidos['Home Club'] = df_partidos['Home Club'].replace('', np.nan)
df_partidos['Away Club'] = df_partidos['Away Club'].replace('', np.nan)

# Medias globales de goles (local y visitante)
media_goles_local_global = df_partidos['Home Score'].mean()
media_goles_visitante_global = df_partidos['Away Score'].mean()

# Índice con la unión de todos los equipos (sin NaN)
equipos = pd.Index(df_partidos['Home Club'].dropna().unique()).union(
    pd.Index(df_partidos['Away Club'].dropna().unique())
).sort_values()

# Calcular los promedios por equipo:
# - GOL_AN_L: goles anotados en casa (media cuando el equipo es local)
# - GOL_REC_L: goles recibidos en casa (media de goles del rival cuando el equipo es local)
# - GOL_AN_V: goles anotados como visitante (media cuando el equipo es visitante)
# - GOL_REC_V: goles recibidos como visitante (media de goles del rival cuando el equipo es visitante)
prom_goles_local_anotados = df_partidos.groupby('Home Club')['Home Score'].mean().rename('GOL_AN_L').reindex(equipos)
prom_goles_local_recibidos = df_partidos.groupby('Home Club')['Away Score'].mean().rename('GOL_REC_L').reindex(equipos)
prom_goles_visitante_anotados = df_partidos.groupby('Away Club')['Away Score'].mean().rename('GOL_AN_V').reindex(equipos)
prom_goles_visitante_recibidos = df_partidos.groupby('Away Club')['Home Score'].mean().rename('GOL_REC_V').reindex(equipos)

# DataFrame con los 4 promedios (mantener NaN donde no hay datos)
df_4_promedios = pd.concat([
    prom_goles_local_anotados,
    prom_goles_local_recibidos,
    prom_goles_visitante_anotados,
    prom_goles_visitante_recibidos
], axis=1)
df_4_promedios.index.name = 'Equipo'

# Rellenar NaN con medias globales sólo para el cálculo de fuerzas (no altera df_4_promedios original)
df_promedios_completos = df_4_promedios.fillna({
    'GOL_AN_L': media_goles_local_global,
    'GOL_AN_V': media_goles_visitante_global,
    'GOL_REC_L': media_goles_visitante_global,
    'GOL_REC_V': media_goles_local_global
})

# Calcular fuerzas relativas (ataque/defensa casa/visita)
df_promedios_completos['fuerza_ataque_local'] = df_promedios_completos['GOL_AN_L'] / media_goles_local_global
df_promedios_completos['fuerza_ataque_visita'] = df_promedios_completos['GOL_AN_V'] / media_goles_visitante_global
df_promedios_completos['fuerza_defensa_local'] = df_promedios_completos['GOL_REC_L'] / media_goles_visitante_global
df_promedios_completos['fuerza_defensa_visita'] = df_promedios_completos['GOL_REC_V'] / media_goles_local_global

# Ajustes de display para consola
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Mostrar resultados: los 4 promedios (con NaN donde falte) y las fuerzas calculadas
print("\n--- 4 promedios por equipo (GOL_AN_L, GOL_REC_L, GOL_AN_V, GOL_REC_V) ---")
print(df_4_promedios.round(3))

print("\n--- Fuerzas por equipo (fuerza_ataque_local, fuerza_ataque_visita, fuerza_defensa_local, fuerza_defensa_visita) ---")
print(df_promedios_completos[['fuerza_ataque_local','fuerza_ataque_visita','fuerza_defensa_local','fuerza_defensa_visita']].round(3))
