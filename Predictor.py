import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =========================
# Carga y limpieza de datos
# =========================
df_partidos = pd.read_csv('data/libertadores-results-ds.csv', encoding='utf-8')
df_partidos.columns = [c.strip() for c in df_partidos.columns]
df_partidos.rename(columns={'AwayScore': 'Away Score', 'HomeScore': 'Home Score'}, inplace=True)

# Limpiar textos y tipos
df_partidos['Home Club'] = df_partidos['Home Club'].astype(str).str.strip().replace('', np.nan)
df_partidos['Away Club'] = df_partidos['Away Club'].astype(str).str.strip().replace('', np.nan)
df_partidos['Home Score'] = pd.to_numeric(df_partidos['Home Score'], errors='coerce')
df_partidos['Away Score'] = pd.to_numeric(df_partidos['Away Score'], errors='coerce')

# Medias globales
media_goles_local_global = df_partidos['Home Score'].mean()
media_goles_visitante_global = df_partidos['Away Score'].mean()

# Lista de equipos únicos (sin NaN)
equipos = pd.Index(df_partidos['Home Club'].dropna().unique()).union(
    pd.Index(df_partidos['Away Club'].dropna().unique())
).sort_values()

# =========================
# Cálculo de promedios y fuerzas
# =========================
prom_goles_local_anotados = df_partidos.groupby('Home Club')['Home Score'].mean().rename('GOL_AN_L').reindex(equipos)
prom_goles_local_recibidos = df_partidos.groupby('Home Club')['Away Score'].mean().rename('GOL_REC_L').reindex(equipos)
prom_goles_visitante_anotados = df_partidos.groupby('Away Club')['Away Score'].mean().rename('GOL_AN_V').reindex(equipos)
prom_goles_visitante_recibidos = df_partidos.groupby('Away Club')['Home Score'].mean().rename('GOL_REC_V').reindex(equipos)

df_4_promedios = pd.concat([
    prom_goles_local_anotados,
    prom_goles_local_recibidos,
    prom_goles_visitante_anotados,
    prom_goles_visitante_recibidos
], axis=1)
df_4_promedios.index.name = 'Equipo'

# Completar NaN solo para fuerzas (no altera df_4_promedios)
df_promedios_completos = df_4_promedios.fillna({
    'GOL_AN_L': media_goles_local_global,
    'GOL_AN_V': media_goles_visitante_global,
    'GOL_REC_L': media_goles_visitante_global,
    'GOL_REC_V': media_goles_local_global
})

df_promedios_completos['fuerza_ataque_local'] = df_promedios_completos['GOL_AN_L'] / media_goles_local_global
df_promedios_completos['fuerza_ataque_visita'] = df_promedios_completos['GOL_AN_V'] / media_goles_visitante_global
df_promedios_completos['fuerza_defensa_local'] = df_promedios_completos['GOL_REC_L'] / media_goles_visitante_global
df_promedios_completos['fuerza_defensa_visita'] = df_promedios_completos['GOL_REC_V'] / media_goles_local_global

# =========================
# Preparar series por equipo
# =========================
metricas_nombres = [
    'GOL_AN_L (local anotado)',
    'GOL_REC_L (local recibido)',
    'GOL_AN_V (visita anotado)',
    'GOL_REC_V (visita recibido)'
]

datos_por_metrica = {m: [] for m in metricas_nombres}
partidos_por_equipo = {}  # para filtrar equipos con pocos partidos

for equipo in equipos:
    gol_an_l = df_partidos.loc[df_partidos['Home Club'] == equipo, 'Home Score'].dropna().values
    gol_rec_l = df_partidos.loc[df_partidos['Home Club'] == equipo, 'Away Score'].dropna().values
    gol_an_v = df_partidos.loc[df_partidos['Away Club'] == equipo, 'Away Score'].dropna().values
    gol_rec_v = df_partidos.loc[df_partidos['Away Club'] == equipo, 'Home Score'].dropna().values

    datos_por_metrica[metricas_nombres[0]].append(gol_an_l if len(gol_an_l) > 0 else np.array([np.nan]))
    datos_por_metrica[metricas_nombres[1]].append(gol_rec_l if len(gol_rec_l) > 0 else np.array([np.nan]))
    datos_por_metrica[metricas_nombres[2]].append(gol_an_v if len(gol_an_v) > 0 else np.array([np.nan]))
    datos_por_metrica[metricas_nombres[3]].append(gol_rec_v if len(gol_rec_v) > 0 else np.array([np.nan]))

    # Conteo total de partidos del equipo (local + visita)
    partidos_por_equipo[equipo] = len(gol_an_l) + len(gol_an_v)

# Colores para las 4 métricas
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# =========================
# Funciones de graficación
# =========================
def plot_boxplots_chunked(equipos_lista, chunk_size=10, min_partidos=0):
    """
    Muestra boxplots en chunks de 'chunk_size' equipos.
    min_partidos: oculta equipos con menos partidos que el umbral (para mejorar legibilidad).
    """
    # Filtrar por mínimo de partidos si se requiere
    if min_partidos > 0:
        equipos_filtrados = [e for e in equipos_lista if partidos_por_equipo.get(e, 0) >= min_partidos]
    else:
        equipos_filtrados = list(equipos_lista)

    n_equipos = len(equipos_filtrados)
    if n_equipos == 0:
        print("No hay equipos que cumplan el filtro de partidos.")
        return

    for start in range(0, n_equipos, chunk_size):
        end = min(start + chunk_size, n_equipos)
        equipos_chunk = equipos_filtrados[start:end]
        indices = np.arange(len(equipos_chunk))
        fig, ax = plt.subplots(figsize=(max(10, len(equipos_chunk) * 0.9), 6))
        ancho = 0.15

        # Construir los datos para el chunk
        for j, met in enumerate(metricas_nombres):
            series_chunk = [datos_por_metrica[met][equipos.get_loc(e)] for e in equipos_chunk]
            posiciones = indices + (j - 1.5) * ancho
            bp = ax.boxplot(series_chunk,
                            positions=posiciones,
                            widths=ancho * 0.9,
                            patch_artist=True,
                            showfliers=False)
            for elemento in bp['boxes']:
                elemento.set(facecolor=colores[j], alpha=0.6)
            for elemento in bp['whiskers'] + bp['medians'] + bp['caps']:
                elemento.set(color='black')

        ax.set_xticks(indices)
        ax.set_xticklabels(equipos_chunk, rotation=90, fontsize=8)
        ax.set_title(f'Distribución de goles (equipos {start+1}-{end} de {n_equipos})')
        ax.set_ylabel('Goles')

        parches = [mpatches.Patch(color=colores[i], label=metricas_nombres[i]) for i in range(len(metricas_nombres))]
        ax.legend(handles=parches, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

def plot_forces_chunked(df_fuerzas, equipos_lista, chunk_size=10, min_partidos=0):
    """
    Muestra barras de fuerzas en chunks de 'chunk_size' equipos.
    min_partidos: oculta equipos con menos partidos que el umbral.
    """
    cols_fuerzas = ['fuerza_ataque_local', 'fuerza_ataque_visita', 'fuerza_defensa_local', 'fuerza_defensa_visita']

    # Filtrar por mínimo de partidos si se requiere
    if min_partidos > 0:
        equipos_filtrados = [e for e in equipos_lista if partidos_por_equipo.get(e, 0) >= min_partidos]
    else:
        equipos_filtrados = list(equipos_lista)

    n_equipos = len(equipos_filtrados)
    if n_equipos == 0:
        print("No hay equipos que cumplan el filtro de partidos.")
        return

    for start in range(0, n_equipos, chunk_size):
        end = min(start + chunk_size, n_equipos)
        equipos_chunk = equipos_filtrados[start:end]
        indices = np.arange(len(equipos_chunk))

        fig, ax = plt.subplots(figsize=(max(10, len(equipos_chunk) * 0.9), 6))
        bar_width = 0.18

        for i, col in enumerate(cols_fuerzas):
            posiciones = indices + (i - 1.5) * bar_width
            ax.bar(posiciones,
                   df_fuerzas.loc[equipos_chunk, col].values,
                   bar_width, label=col, color=colores[i], alpha=0.8)

        ax.set_xticks(indices)
        ax.set_xticklabels(equipos_chunk, rotation=90, fontsize=8)
        ax.set_title(f'Fuerzas por equipo (equipos {start+1}-{end} de {n_equipos})')
        ax.set_ylabel('Multiplicador de fuerza (relativo)')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

# =========================
# Uso: graficar sin límites
# =========================
# df_fuerzas desde df_promedios_completos (relleno NaN a 0 solo para graficar)
df_fuerzas = df_promedios_completos[['fuerza_ataque_local', 'fuerza_ataque_visita',
                                     'fuerza_defensa_local', 'fuerza_defensa_visita']].fillna(0)

# Ajusta a tu gusto:
chunk_size = 10        # equipos por figura
min_partidos = 1       # oculta equipos con menos de 3 partidos (pon 0 si no quieres filtrar)

# Boxplots en chunks (todos los equipos, sin recortes)
plot_boxplots_chunked(equipos_lista=equipos, chunk_size=chunk_size, min_partidos=min_partidos)

# Gráficas de fuerzas en chunks (todos los equipos, sin recortes)
plot_forces_chunked(df_fuerzas=df_fuerzas, equipos_lista=equipos, chunk_size=chunk_size, min_partidos=min_partidos)
