import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




filtros = {
    "chunk_size": 10,            
    "min_partidos": 0,           
    "mostrar_boxplots": True,    
    "mostrar_fuerzas": True,     
    "orden_por": None,           
    "seleccion_equipos": None    
}




df_partidos = pd.read_csv('data/libertadores-results-ds.csv', encoding='utf-8')
df_partidos.columns = [c.strip() for c in df_partidos.columns]
df_partidos.rename(columns={'AwayScore': 'Away Score', 'HomeScore': 'Home Score'}, inplace=True)

df_partidos['Home Club'] = df_partidos['Home Club'].astype(str).str.strip().replace('', np.nan)
df_partidos['Away Club'] = df_partidos['Away Club'].astype(str).str.strip().replace('', np.nan)
df_partidos['Home Score'] = pd.to_numeric(df_partidos['Home Score'], errors='coerce')
df_partidos['Away Score'] = pd.to_numeric(df_partidos['Away Score'], errors='coerce')


media_goles_local_global = df_partidos['Home Score'].mean()
media_goles_visitante_global = df_partidos['Away Score'].mean()


equipos = pd.Index(df_partidos['Home Club'].dropna().unique()).union(
    pd.Index(df_partidos['Away Club'].dropna().unique())
).sort_values()




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


cols_fuerzas = ['fuerza_ataque_local', 'fuerza_ataque_visita', 'fuerza_defensa_local', 'fuerza_defensa_visita']
df_fuerzas = df_promedios_completos[cols_fuerzas].reindex(equipos).fillna(0)




metricas_nombres = [
    'GOL_AN_L (local anotado)',
    'GOL_REC_L (local recibido)',
    'GOL_AN_V (visita anotado)',
    'GOL_REC_V (visita recibido)'
]

datos_por_metrica = {m: [] for m in metricas_nombres}
partidos_por_equipo = {}

for equipo in equipos:
    gol_an_l = df_partidos.loc[df_partidos['Home Club'] == equipo, 'Home Score'].dropna().values
    gol_rec_l = df_partidos.loc[df_partidos['Home Club'] == equipo, 'Away Score'].dropna().values
    gol_an_v = df_partidos.loc[df_partidos['Away Club'] == equipo, 'Away Score'].dropna().values
    gol_rec_v = df_partidos.loc[df_partidos['Away Club'] == equipo, 'Home Score'].dropna().values

    datos_por_metrica[metricas_nombres[0]].append(gol_an_l if len(gol_an_l) > 0 else np.array([np.nan]))
    datos_por_metrica[metricas_nombres[1]].append(gol_rec_l if len(gol_rec_l) > 0 else np.array([np.nan]))
    datos_por_metrica[metricas_nombres[2]].append(gol_an_v if len(gol_an_v) > 0 else np.array([np.nan]))
    datos_por_metrica[metricas_nombres[3]].append(gol_rec_v if len(gol_rec_v) > 0 else np.array([np.nan]))

    partidos_por_equipo[equipo] = len(gol_an_l) + len(gol_an_v)


colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']




def obtener_lista_equipos(equipos_idx, filtros, df_fuerzas):
    if filtros.get("seleccion_equipos"):
        equipos_seleccion = [e for e in filtros["seleccion_equipos"] if e in equipos_idx]
    else:
        equipos_seleccion = list(equipos_idx)

    min_part = max(0, int(filtros.get("min_partidos", 0)))
    equipos_filtrados = [e for e in equipos_seleccion if partidos_por_equipo.get(e, 0) >= min_part]

    if len(equipos_filtrados) == 0:
        equipos_filtrados = equipos_seleccion

    orden_col = filtros.get("orden_por")
    if orden_col in df_fuerzas.columns:
        equipos_filtrados = sorted(equipos_filtrados, key=lambda e: df_fuerzas.loc[e, orden_col], reverse=True)

    return equipos_filtrados







def plot_boxplots_chunked(equipos_lista, filtros):
    equipos_filtrados = obtener_lista_equipos(equipos_lista, filtros, df_fuerzas)
    n_equipos = len(equipos_filtrados)
    if n_equipos == 0:
        print("No hay equipos para mostrar en boxplots.")
        return

    chunk_size = max(1, int(filtros.get("chunk_size", 10)))

    for start in range(0, n_equipos, chunk_size):
        end = min(start + chunk_size, n_equipos)
        equipos_chunk = equipos_filtrados[start:end]
        indices = np.arange(len(equipos_chunk))
        fig, ax = plt.subplots(figsize=(max(10, len(equipos_chunk) * 0.9), 6))
        ancho = 0.15

        
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

def plot_forces_chunked(df_fuerzas, equipos_lista, filtros):
    equipos_filtrados = obtener_lista_equipos(equipos_lista, filtros, df_fuerzas)
    n_equipos = len(equipos_filtrados)
    if n_equipos == 0:
        print("No hay equipos para mostrar en fuerzas.")
        return

    chunk_size = max(1, int(filtros.get("chunk_size", 10)))
    cols_fuerzas = ['fuerza_ataque_local', 'fuerza_ataque_visita', 'fuerza_defensa_local', 'fuerza_defensa_visita']

    for start in range(0, n_equipos, chunk_size):
        end = min(start + chunk_size, n_equipos)
        equipos_chunk = equipos_filtrados[start:end]
        indices = np.arange(len(equipos_chunk))

        fig, ax = plt.subplots(figsize=(max(10, len(equipos_chunk) * 0.9), 6))
        bar_width = 0.18

        for i, col in enumerate(cols_fuerzas):
            posiciones = indices + (i - 1.5) * bar_width
            valores = df_fuerzas.loc[equipos_chunk, col].values
            ax.bar(posiciones, valores, bar_width, label=col, color=colores[i], alpha=0.8)

        ax.set_xticks(indices)
        ax.set_xticklabels(equipos_chunk, rotation=90, fontsize=8)
        ax.set_title(f'Fuerzas por equipo (equipos {start+1}-{end} de {n_equipos})')
        ax.set_ylabel('Multiplicador de fuerza (relativo)')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()






goles_anotados = pd.concat([
    df_partidos['Home Score'].dropna(),
    df_partidos['Away Score'].dropna()
])
goles_recibidos = pd.concat([
    df_partidos['Away Score'].dropna(),
    df_partidos['Home Score'].dropna()
])

fig, ax = plt.subplots(figsize=(6, 6))
bp = ax.boxplot([goles_anotados, goles_recibidos],
                labels=['Goles anotados (global)', 'Goles recibidos (global)'],
                patch_artist=True,
                showfliers=False)

colores_globales = ['#1f77b4', '#d62728']
for patch, color in zip(bp['boxes'], colores_globales):
    patch.set(facecolor=color, alpha=0.6)

ax.set_title('Distribución global de goles anotados vs recibidos')
ax.set_ylabel('Goles')
plt.tight_layout()
plt.show()


if filtros.get("mostrar_boxplots", True):
    plot_boxplots_chunked(equipos_lista=equipos, filtros=filtros)

if filtros.get("mostrar_fuerzas", True):
    plot_forces_chunked(df_fuerzas=df_fuerzas, equipos_lista=equipos, filtros=filtros)
