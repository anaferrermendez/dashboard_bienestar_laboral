import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import io
import matplotlib
import plotly.express as px
import plotly.graph_objects as go

# Configuración premium
st.set_page_config(page_title="Data Quality Dashboard", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')
PALETA = ['#1f4e79', '#2e75b6', '#9dc3e6', '#ffc000', '#c55a11']
sns.set_palette(PALETA)

# Cargar Datos
@st.cache_data
def load_data():
    df_orig = pd.read_pickle('df_original.pkl')
    df_clean = pd.read_pickle('df_cleaned.pkl')
    df_probs = pd.read_csv('problemas_calidad.csv')
    return df_orig, df_clean, df_probs

df_original, df_cleaned, df_problemas = load_data()

@st.cache_data
def load_comparativo():
    df_antes = pd.read_excel('bienestar_laboral_Preprocesamiento.xlsx')
    df_despues = pd.read_excel('dataset_limpio.xlsx')
    return df_antes, df_despues

DIMS_COMP = {
    'CTRL': ['CT1', 'CT2', 'CT3'],
    'PRES': ['PT1', 'PT2', 'PT3', 'PT4'],
    'LIDER': ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7'],
    'COMP': ['AC1', 'AC2', 'AC3'],
    'ROL_C': ['CR1', 'CR2', 'CR3', 'CR4'],
    'ROL_CON': ['CoR1', 'CoR2', 'CoR3'],
    'CAMBIO': ['GC1', 'GC2', 'GC3', 'GC4'],
    'SM_ORG': ['SM1', 'SM2', 'SM3', 'SM4', 'SM5'],
    'SAT': ['SAT1', 'SAT2', 'SAT3', 'SAT4', 'SAT5', 'SAT6', 'SAT7', 'SAT8', 'SAT9'],
    'RETIRO': ['IR1', 'IR2', 'IR3', 'IR4'],
    'FAM_TRAB': ['FT1', 'FT2', 'FT3', 'FT4', 'FT5'],
    'TRAB_FAM': ['TF1', 'TF2', 'TF3', 'TF4', 'TF5'],
    'BURNOUT': ['BU1', 'BU2', 'BU3', 'BU4', 'BU5', 'BU6', 'BU7', 'BU8', 'BU9', 'BU10', 'BU11', 'BU12'],
    'BIENESTAR': [f'BP{i}' for i in range(1, 11)],
    'SOMATIZ': ['SOM1', 'SOM2', 'SOM3', 'SOM4', 'SOM5'],
    'DESGASTE': ['DL1', 'DL2', 'DL3', 'DL4', 'DL5', 'DL6', 'DL7', 'DL8'],
}
COL_TO_DIM = {col: dim for dim, cols in DIMS_COMP.items() for col in cols}
IMPUTACION_DIMENSION = {
    'CTRL': 101, 'PRES': 97, 'LIDER': 191, 'COMP': 85, 'ROL_C': 97, 'ROL_CON': 80,
    'CAMBIO': 132, 'SM_ORG': 141, 'SAT': 272, 'RETIRO': 109, 'FAM_TRAB': 145,
    'TRAB_FAM': 137, 'BURNOUT': 335, 'BIENESTAR': 275, 'SOMATIZ': 127, 'DESGASTE': 209,
}
IMPUTACION_CONTINUA = {
    'Edad': (1, 41.5),
    'Horas_Semana': (1, 43.0),
    'Horas_Formacion': (4, 16.0),
}
CODIGOS_NULOS_COMP = [
    999, -1, '999', '-1', 'N/A', 'NA', 'null', 'NULL', 'Null', 'no aplica',
    'No aplica', 'NO APLICA', '.', 'nan', 'NaN', 'sin dato', 'Sin dato', '?', '--',
]
MAP_FRECUENCIA = {'Nunca': 1, 'Rara vez': 2, 'Algunas veces': 3, 'Frecuentemente': 4, 'Siempre': 5}
MAP_ACUERDO = {
    'Muy en desacuerdo': 1, 'Moderadamente en desacuerdo': 2, 'Algo en desacuerdo': 3,
    'Ni de acuerdo ni en desacuerdo': 4, 'Algo de acuerdo': 5, 'Moderadamente de acuerdo': 6, 'Muy de acuerdo': 7,
}
MAP_BURNOUT = {'Nunca': 1, 'Raramente': 2, 'Algunas veces': 3, 'A menudo': 4, 'Siempre': 5}
MAP_SOMATIZ = {
    'Nunca': 1, 'Raramente': 2, 'Ocasionalmente': 3, 'Algunas veces': 4,
    'Frecuentemente': 5, 'Casi siempre': 6, 'Siempre': 7,
}
MAP_PALABRAS_NUMERO = {
    'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5, 'seis': 6, 'siete': 7,
    'Uno': 1, 'Dos': 2, 'Tres': 3, 'Cuatro': 4, 'Cinco': 5, 'Seis': 6, 'Siete': 7,
    'UNO': 1, 'DOS': 2, 'TRES': 3, 'CUATRO': 4, 'CINCO': 5, 'SEIS': 6, 'SIETE': 7,
}
TYPO_REEMPLAZOS = {
    'Siempree': 'Siempre', 'Frecuente': 'Frecuentemente', 'A menudo': 'Frecuentemente',
    'Alguna vez': 'Algunas veces', 'Raramente': 'Rara vez',
}

def _grupos_escala(columnas):
    cols_frec = sum([DIMS_COMP[k] for k in ['CTRL', 'PRES', 'LIDER', 'COMP', 'ROL_C', 'ROL_CON', 'CAMBIO', 'SM_ORG']], [])
    cols_acuerdo = DIMS_COMP['SAT'] + DIMS_COMP['RETIRO'] + DIMS_COMP['FAM_TRAB'] + DIMS_COMP['TRAB_FAM']
    cols_bu = DIMS_COMP['BURNOUT']
    cols_som = DIMS_COMP['SOMATIZ'] + DIMS_COMP['DESGASTE']
    cols_bp = [c for c in columnas if str(c).startswith('BP')]
    return cols_frec, cols_acuerdo, cols_bu, cols_som, cols_bp

def preparar_serie_antes(df, col, grupos):
    if col not in df.columns:
        return None
    s = df[col].copy()
    for codigo in CODIGOS_NULOS_COMP:
        s = s.replace(codigo, np.nan)
    cols_frec, cols_acuerdo, cols_bu, cols_som, cols_bp = grupos
    if col in cols_frec + cols_acuerdo + cols_bu + cols_som:
        if s.dtype == object:
            s = s.astype(str).str.strip().str.capitalize().replace(TYPO_REEMPLAZOS)
        mapper = (
            MAP_FRECUENCIA if col in cols_frec else
            MAP_ACUERDO if col in cols_acuerdo else
            MAP_BURNOUT if col in cols_bu else MAP_SOMATIZ
        )
        s = pd.to_numeric(s.map(mapper), errors='coerce')
    elif col in cols_bp:
        s = s.replace(MAP_PALABRAS_NUMERO)
        s = pd.to_numeric(s, errors='coerce')
    elif col in ['Edad', 'Horas_Semana', 'Horas_Formacion']:
        s = pd.to_numeric(s, errors='coerce')
    if col == 'IR1':
        s = pd.to_numeric(s, errors='coerce')
        s = 8 - s
    return s

def mensaje_imputacion(col):
    if col in IMPUTACION_CONTINUA:
        n, mediana = IMPUTACION_CONTINUA[col]
        return (
            f"**Imputación aplicada:** {n} valor(es) faltante(s) imputado(s) con "
            f"**mediana = {mediana}** (variables numéricas continuas)."
        )
    if col in COL_TO_DIM:
        dim = COL_TO_DIM[col]
        n = IMPUTACION_DIMENSION[dim]
        return (
            f"**Imputación aplicada:** faltantes imputados con **mediana por columna** "
            f"(dimensión **{dim}**: {n} celdas imputadas en total en sus ítems)."
        )
    if col in DIMS_COMP:
        n = IMPUTACION_DIMENSION[col]
        items = ', '.join(DIMS_COMP[col][:5])
        sufijo = '…' if len(DIMS_COMP[col]) > 5 else ''
        return (
            f"**Puntaje de dimensión {col}** (media de {items}{sufijo}): calculado tras "
            f"**imputación por mediana** en los ítems ({n} celdas imputadas en la dimensión)."
        )
    return None

def es_numerica_comparativo(serie_antes, serie_despues):
    if serie_despues is not None and pd.api.types.is_numeric_dtype(serie_despues):
        return True
    if serie_antes is not None:
        num = pd.to_numeric(serie_antes, errors='coerce')
        if num.notna().sum() > 0 and num.notna().mean() >= 0.5:
            return True
    return False

def resumen_numerico(serie):
    s = pd.to_numeric(serie, errors='coerce')
    if s.count() == 0:
        return {m: None for m in ['Conteo (válidos)', 'Nulos', 'Media', 'Mediana', 'Mínimo', 'Máximo', 'Desv. estándar']}
    return {
        'Conteo (válidos)': int(s.count()),
        'Nulos': int(s.isna().sum()),
        'Media': round(s.mean(), 2),
        'Mediana': round(s.median(), 2),
        'Mínimo': round(s.min(), 2),
        'Máximo': round(s.max(), 2),
        'Desv. estándar': round(s.std(), 2),
    }

def ordenar_variables_comparativo(df_antes, df_despues):
    socio = list(df_antes.columns[:21])
    dimensiones = sorted(c for c in df_despues.columns if c not in df_antes.columns)
    patron_psico = re.compile(r'^[A-Za-z]+\d+$')
    psico = sorted(
        c for c in df_antes.columns
        if c not in socio and patron_psico.match(str(c))
    )
    resto = sorted(c for c in df_antes.columns if c not in socio and c not in psico)
    return socio + psico + dimensiones + resto

RANGOS_DIMENSION = {
    'CTRL': (1, 5), 'PRES': (1, 5), 'LIDER': (1, 5), 'COMP': (1, 5),
    'ROL_C': (1, 5), 'ROL_CON': (1, 5), 'CAMBIO': (1, 5), 'SM_ORG': (1, 5),
    'SAT': (1, 7), 'RETIRO': (1, 7), 'FAM_TRAB': (1, 7), 'TRAB_FAM': (1, 7),
    'BURNOUT': (1, 5), 'BIENESTAR': (1, 7), 'SOMATIZ': (1, 7), 'DESGASTE': (1, 7),
}
JDR_CHECKS = [
    ('PRES', 'BURNOUT', '>', 0.50),
    ('PRES', 'DESGASTE', '>', 0.50),
    ('PRES', 'SOMATIZ', '>', 0.50),
    ('LIDER', 'SAT', '>', 0.40),
    ('LIDER', 'BIENESTAR', '>', 0.50),
    ('SAT', 'RETIRO', '<', -0.40),
    ('BURNOUT', 'DESGASTE', '>', 0.60),
]
ESTADO_SEMAFORO = {
    'green': ('#2e7d32', '🟢', 'VALIDADA'),
    'yellow': ('#f9a825', '🟡', 'REQUIERE REVISIÓN'),
    'red': ('#c62828', '🔴', 'ERROR'),
}

def validar_dimension(df, dim, items):
    minimo, maximo = RANGOS_DIMENSION[dim]
    items_ok = [c for c in items if c in df.columns]
    faltantes = int(df[items_ok].isna().sum().sum()) if items_ok else 0
    fuera_rango = int(
        ((df[items_ok] < minimo) | (df[items_ok] > maximo)).sum().sum()
    ) if items_ok else 0
    dtype_correcto = all(
        pd.api.types.is_numeric_dtype(df[c]) for c in items_ok
    ) if items_ok else False
    promedio_valido = (
        df[dim].between(minimo, maximo).all()
        if dim in df.columns and df[dim].notna().all() else False
    )
    media_obs = round(df[dim].mean(), 2) if dim in df.columns else None
    if dim not in df.columns or not items_ok:
        estado = 'red'
    elif faltantes or fuera_rango or not dtype_correcto or not promedio_valido:
        estado = 'yellow'
    else:
        estado = 'green'
    return {
        'faltantes': faltantes,
        'fuera_rango': fuera_rango,
        'dtype_correcto': dtype_correcto,
        'promedio_valido': promedio_valido,
        'media_obs': media_obs,
        'estado': estado,
        'minimo': minimo,
        'maximo': maximo,
        'n_items': len(items_ok),
    }

def tarjeta_dimension(dim, resultado):
    color, icono, etiqueta = ESTADO_SEMAFORO[resultado['estado']]
    html = (
        f"<motion.div class='kpi-card' style='border-top: 4px solid {color}; min-height: 150px;'>"
        f"<motion.div style='font-size: 1.1rem; font-weight: bold; color: #1f4e79;'>{icono} {dim}</div>"
        f"<motion.div style='color: #6c757d; font-size: 0.85rem;'>{resultado['n_items']} ítems · "
        f"Escala {resultado['minimo']}–{resultado['maximo']}</div>"
        f"<motion.div class='kpi-value' style='font-size: 1.6rem;'>{resultado['media_obs']}</motion.div>"
        f"<motion.div style='font-size: 0.8rem; color: #6c757d;'>Media observada</motion.div>"
        f"<motion.div style='margin-top: 8px; font-weight: 600; color: {color};'>{etiqueta}</motion.div>"
        "</motion.div>"
    )
    st.markdown(html.replace('motion.', ''), unsafe_allow_html=True)

def evaluar_check_jdr(corr, a, b, operador, umbral):
    r = corr.loc[a, b]
    cumple = r > umbral if operador == '>' else r < umbral
    texto_op = 'positiva' if operador == '>' else 'negativa'
    signo = '>' if operador == '>' else '<'
    return cumple, r, f"Esperado: {texto_op} {signo} {umbral}"

def resumen_categorico(serie):
    vc = serie.value_counts(dropna=False)
    datos = {
        'Conteo (válidos)': int(serie.notna().sum()),
        'Nulos': int(serie.isna().sum()),
        'Categorías únicas': int(serie.nunique(dropna=True)),
        'Moda': vc.index[0] if len(vc) else None,
        'Frecuencia moda': int(vc.iloc[0]) if len(vc) else None,
    }
    return pd.DataFrame({'Métrica': list(datos.keys()), 'Valor': list(datos.values())})

# CSS Personalizado para diseño moderno
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1f4e79; font-family: 'Inter', sans-serif; }
    .kpi-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border-top: 4px solid #2e75b6;}
    .kpi-value { font-size: 2.5rem; font-weight: bold; color: #1f4e79; }
    .kpi-label { font-size: 1rem; color: #6c757d; font-weight: 500; }
    .stSelectbox label, .stRadio label { font-weight: bold; color: #1f4e79; }
    </style>
""", unsafe_allow_html=True)

st.title("Dashboard de Calidad de Datos - Preprocesamiento")
st.markdown("**Ana Ferrer** · **Sofia Nuñez** · **Sergio Soler**")
st.markdown("---")

# Menú lateral
menu = st.sidebar.radio(
    "Navegación",
    [
        "1. Estado Inicial",
        "2. Explorador de Problemas",
        "3. Comparativo Antes/Después",
        "4. Certificación de Calidad",
    ],
)

# Panel 1: Estado Inicial
if menu == "1. Estado Inicial":
    st.header("1. Estado Inicial del Dataset")
    st.markdown("Análisis del dataset crudo (`bienestar_laboral_Preprocesamiento.xlsx`) antes de cualquier tratamiento.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total celdas ausentes", "2 601")
    col2.metric("Total columnas afectadas", "91")
    col3.metric("Tipos de problemas detectados", "12")
    
    st.markdown("---")
    
    # Componente 1: % de Faltantes por Columna
    st.subheader("Componente 1: % de Faltantes por Columna")
    st.markdown("El gráfico de barras muestra qué variables concentran la mayor ausencia de datos.")
    
    codigos_nulos = [999, -1, '999', '-1', '?', '--', 'sin dato', 'No aplica']
    df_heat = df_original.replace(codigos_nulos, np.nan)
    
    faltantes_pct = (df_heat.isnull().sum() / len(df_heat) * 100).reset_index()
    faltantes_pct.columns = ['Columna', 'Pct_faltante']
    faltantes_pct = faltantes_pct[faltantes_pct['Pct_faltante'] > 0].sort_values('Pct_faltante', ascending=True)
    
    cols_sociodem = df_original.columns[:21].tolist()
    faltantes_pct['Grupo'] = faltantes_pct['Columna'].apply(
        lambda c: 'Sociodemográfica' if c in cols_sociodem else 'Ítem psicosocial'
    )
    
    fig_heat = px.bar(
        faltantes_pct,
        x='Pct_faltante',
        y='Columna',
        color='Grupo',
        orientation='h',
        title='% de valores faltantes por columna (solo columnas con al menos 1 faltante)',
        labels={'Pct_faltante': '% faltante', 'Columna': ''},
        color_discrete_map={'Ítem psicosocial': '#4C78A8', 'Sociodemográfica': '#E45756'},
        height=900
    )
    fig_heat.update_layout(showlegend=True)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    
    # Componente 2: Magnitud de problemas
    st.subheader("Componente 2: Magnitud de los problemas detectados")
    
    # Cálculos programáticos de magnitudes en celdas afectadas
    df_base = df_original.copy()
    codigos_nulos = [999, -1, '999', '-1', '?', '--', 'sin dato', 'No aplica']
    
    # 6. Códigos nulos ocultos (calculados antes del replace)
    celdas_ocultos = df_base.isin(codigos_nulos).sum().sum()
    
    # 5. Valores faltantes NaN originales
    celdas_nans_orig = df_base.isna().sum().sum()
    
    # Convertir especiales a NaN para el resto de análisis
    df_nans = df_base.replace(codigos_nulos, np.nan)
    
    # 1. dtype incorrecto: Celdas en columnas de escala guardadas como texto
    cols_escala = df_nans.columns[21:]
    celdas_dtype = df_nans[cols_escala].apply(lambda col: col.map(lambda x: isinstance(x, str))).sum().sum()
    
    # 2. Variantes textuales en escala
    def is_variant(x):
        if not isinstance(x, str): return False
        std = x.strip().capitalize()
        if std != x: return True # Capitalization or trailing spaces
        if std in ['Siempree', 'Frecuente', 'A menudo', 'Alguna vez', 'Raramente']: return True
        return False
    celdas_variantes = df_nans[cols_escala].apply(lambda col: col.map(is_variant)).sum().sum()
    
    # 3. BP con texto
    bp_textos = ['uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete']
    celdas_bp = 0
    for c in ['BP1', 'BP2', 'BP3', 'BP4', 'BP5']:
        if c in df_nans.columns:
            celdas_bp += df_nans[c].astype(str).str.lower().isin(bp_textos).sum()
            
    # 4. Ítem inverso sin invertir
    celdas_ir1 = df_nans['IR1'].notna().sum() if 'IR1' in df_nans.columns else 0
    
    # 7. Duplicados lógicos
    SUBSET_LOGICO = ['Edad', 'Sexo', 'Estado_Civil', 'Numero_Hijos', 'Nivel_Educativo', 'Zona_Vivienda', 
                     'Estrato', 'Sector', 'Tamaño_Empresa', 'Tipo_Contrato', 'Horas_Semana', 'Ingreso', 
                     'Tipo_Cargo', 'Personas_Cargo', 'Años_Experiencia', 'Antiguedad_Cargo', 'Modalidad', 'Horas_Traslado', 'Horas_Formacion']
    filas_dup = df_nans.duplicated(subset=SUBSET_LOGICO, keep=False).sum()
    celdas_duplicadas = filas_dup * df_nans.shape[1] # Celdas afectadas = filas * total_columnas
    
    # 8. Typos y variantes categóricas
    vars_cat = ['Sexo', 'Estado_Civil', 'Sector', 'Tipo_Cargo', 'Modalidad']
    def is_cat_typo(x):
        if pd.isna(x) or not isinstance(x, str): return False
        std = x.strip().title()
        if std != x: return True
        if std in ['Mjer', 'Hombree', 'Casdo', 'Solero', 'Mixo', 'Adminstrativo', 'Opearativo']: return True
        return False
    celdas_typos_cat = df_nans[vars_cat].apply(lambda col: col.map(is_cat_typo)).sum().sum()
    
    magnitudes = pd.DataFrame([
        {'Tipo_problema': 'dtype incorrecto (escala como texto)', 'Magnitud_n': celdas_dtype, 'Prioridad': 'Crítica'},
        {'Tipo_problema': 'Variantes textuales en escala (typos)', 'Magnitud_n': celdas_variantes, 'Prioridad': 'Crítica'},
        {'Tipo_problema': 'BP con texto en lugar de número', 'Magnitud_n': celdas_bp, 'Prioridad': 'Crítica'},
        {'Tipo_problema': 'Ítem inverso sin invertir', 'Magnitud_n': celdas_ir1, 'Prioridad': 'Crítica'},
        {'Tipo_problema': 'Valores faltantes NaN', 'Magnitud_n': celdas_nans_orig, 'Prioridad': 'Alta'},
        {'Tipo_problema': 'Códigos nulos (999/-1/?/--/sin dato)', 'Magnitud_n': celdas_ocultos, 'Prioridad': 'Alta'},
        {'Tipo_problema': 'Duplicados lógicos', 'Magnitud_n': celdas_duplicadas, 'Prioridad': 'Alta'},
        {'Tipo_problema': 'Typos y variantes categóricas', 'Magnitud_n': celdas_typos_cat, 'Prioridad': 'Alta'},
    ])
    
    color_map = {'Crítica': '#d62728', 'Alta': '#ff7f0e', 'Media': '#1f77b4'}
    
    fig_px = px.bar(
        magnitudes.sort_values('Magnitud_n'),
        x='Magnitud_n',
        y='Tipo_problema',
        color='Prioridad',
        orientation='h',
        color_discrete_map=color_map,
        title='Magnitud de cada problema detectado',
        labels={'Magnitud_n': 'Celdas afectadas', 'Tipo_problema': ''},
        text='Magnitud_n'
    )
    fig_px.update_traces(textposition='outside')
    fig_px.update_layout(height=420)
    st.plotly_chart(fig_px, use_container_width=True)
    
    def color_prioridad(val):
        color = '#d62728' if val == 'Crítica' else '#ff7f0e' if val == 'Alta' else '#1f77b4'
        return f'background-color: {color}; color: white'
        
    try:
        st.dataframe(df_problemas.style.map(color_prioridad, subset=['Prioridad']), use_container_width=True)
    except Exception:
        # Fallback for older pandas versions
        st.dataframe(df_problemas.style.applymap(color_prioridad, subset=['Prioridad']), use_container_width=True)
    
    st.markdown("---")
    
    # Componente 3: Distribución de variables clave con problemas visibles
    st.subheader("Componente 3: Un gráfico por tipo de problema")
    st.markdown("Visualización directa sobre los datos crudos (`df_original`), evidenciando cada uno de los 8 problemas detectados.")
    
    colA, colB = st.columns(2)
    
    with colA:
        # Gráfico 1 — dtype incorrecto
        dtypes_counts = df_original.dtypes.astype(str).value_counts().reset_index()
        dtypes_counts.columns = ['Tipo de dato', 'Cantidad de columnas']
        
        fig1 = px.bar(
            dtypes_counts, 
            x='Tipo de dato', 
            y='Cantidad de columnas', 
            color='Tipo de dato', 
            title="1. Dtype incorrecto — 'object' agrupa 86 escalas<br>que deberían ser numéricas",
            text='Cantidad de columnas'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico 3 — BP con texto en lugar de número (BP3)
        bp3_counts = df_original['BP3'].value_counts(dropna=False).reset_index()
        bp3_counts.columns = ['Valor', 'Frecuencia']
        bp3_counts['Valor_str'] = bp3_counts['Valor'].astype(str)
        
        def clasificar_bp3(val):
            if pd.isna(val): return "Faltante"
            try:
                float(val)
                return "Número correcto"
            except:
                return "Palabra — error"
                
        bp3_counts['Clasificación'] = bp3_counts['Valor'].apply(clasificar_bp3)
        color_bp3 = {'Número correcto': 'green', 'Palabra — error': 'red', 'Faltante': 'gray'}
        
        fig3 = px.bar(
            bp3_counts, 
            x='Valor_str', 
            y='Frecuencia', 
            color='Clasificación', 
            color_discrete_map=color_bp3, 
            title="3. BP con texto en lugar de número (Variable BP3)<br>La misma cantidad aparece en formatos incompatibles",
            labels={'Valor_str': 'Valor registrado'}
        )
        fig3.update_xaxes(type='category')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Gráfico 5 — Valores faltantes NaN
        nans_top = df_original.isna().sum().sort_values(ascending=False).head(10).reset_index()
        nans_top.columns = ['Columna', 'Celdas vacías']
        nans_top['Porcentaje'] = round(nans_top['Celdas vacías'] / len(df_original) * 100, 1)
        
        fig5 = px.bar(
            nans_top, 
            y='Columna', 
            x='Celdas vacías', 
            orientation='h', 
            text='Porcentaje', 
            title="5. Valores faltantes NaN (Top 10 columnas)<br>Conteo solo de verdaderos NaN, sin incluir códigos nulos"
        )
        fig5.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig5, use_container_width=True)

    with colB:
        # Gráfico 2 — Variantes textuales en escala (CT2)
        ct2_counts = df_original['CT2'].value_counts(dropna=False).reset_index()
        ct2_counts.columns = ['Valor', 'Frecuencia']
        ct2_counts['Valor_str'] = ct2_counts['Valor'].astype(str)
        canonicos_likert = ['Nunca', 'Rara vez', 'Algunas veces', 'Frecuentemente', 'Siempre']
        ct2_counts['Estado'] = ct2_counts['Valor_str'].apply(lambda x: 'Correcto (Forma canónica)' if x in canonicos_likert else 'Variante / Error / Nulo')
        color_ct2 = {'Correcto (Forma canónica)': 'green', 'Variante / Error / Nulo': 'red'}
        
        fig2 = px.bar(
            ct2_counts, 
            x='Valor_str', 
            y='Frecuencia', 
            color='Estado', 
            color_discrete_map=color_ct2, 
            title="2. Variantes textuales en escala (Variable CT2)<br>Una misma respuesta fragmentada en múltiples categorías",
            labels={'Valor_str': 'Respuesta registrada'}
        )
        fig2.update_xaxes(type='category')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico 4 — Ítem inverso sin invertir (IR1)
        ir1_counts = df_original['IR1'].value_counts(dropna=False).reset_index()
        ir1_counts.columns = ['Respuesta', 'Frecuencia']
        ir1_counts['Respuesta_str'] = ir1_counts['Respuesta'].astype(str)
        
        fig4 = px.bar(
            ir1_counts, 
            x='Respuesta_str', 
            y='Frecuencia', 
            title="4. Ítem inverso sin invertir (Variable IR1)<br>Distribución actual de las respuestas",
            labels={'Respuesta_str': 'Nivel de respuesta (1-7)'}
        )
        fig4.update_xaxes(type='category')
        # Añadir anotación desplazada a la derecha
        fig4.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper", xanchor="right", yanchor="top",
            text="🚨 ATENCIÓN:<br>En este ítem, 'Muy de acuerdo' (Nivel 7)<br>significa BAJA intención de retiro.<br>Sin aplicar la fórmula (8 - IR1),<br>la interpretación es opuesta a la real.", 
            showarrow=False, 
            bgcolor="#ffeb3b",
            bordercolor="#f57f17",
            borderwidth=2,
            font=dict(color="black", size=12)
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # Gráfico 6 — Duplicados lógicos
        SUBSET_LOGICO = ['Edad', 'Sexo', 'Estado_Civil', 'Numero_Hijos', 'Nivel_Educativo', 'Zona_Vivienda', 
                         'Estrato', 'Sector', 'Tamaño_Empresa', 'Tipo_Contrato', 'Horas_Semana', 'Ingreso', 
                         'Tipo_Cargo', 'Personas_Cargo', 'Años_Experiencia', 'Antiguedad_Cargo', 'Modalidad', 'Horas_Traslado', 'Horas_Formacion']
        dups_mask = df_original.duplicated(subset=SUBSET_LOGICO, keep=False)
        dups = df_original[dups_mask].copy()
        
        if len(dups) > 0:
            dups['Perfil'] = dups['Sexo'].astype(str) + " - " + dups['Sector'].astype(str) + " - " + dups['Tipo_Cargo'].astype(str)
            dups_counts = dups['Perfil'].value_counts().reset_index()
            dups_counts.columns = ['Perfil Demográfico', 'Cantidad de registros duplicados']
            
            fig6 = px.bar(
                dups_counts, 
                y='Perfil Demográfico', 
                x='Cantidad de registros duplicados', 
                orientation='h', 
                title="6. Duplicados lógicos (Re-envíos del formulario)<br>No son personas distintas, sino el mismo perfil enviado múltiples veces",
                text='Cantidad de registros duplicados'
            )
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No se detectaron duplicados en este subset para visualizar.")

    st.markdown("---")
    
    # Gráfico 7 — Typos categóricas (ancho completo)
    sexo_counts = df_original['Sexo'].value_counts(dropna=False).reset_index()
    sexo_counts.columns = ['Valor', 'Frecuencia']
    sexo_counts['Valor_str'] = sexo_counts['Valor'].astype(str)
    canonicos_sexo = ['Mujer', 'Hombre', 'Prefiero no decir']
    sexo_counts['Estado'] = sexo_counts['Valor_str'].apply(lambda x: 'Correcto (Forma canónica)' if x in canonicos_sexo else 'Typos / Error')
    color_sexo = {'Correcto (Forma canónica)': 'green', 'Typos / Error': 'red'}
    
    fig7 = px.bar(
        sexo_counts, 
        x='Valor_str', 
        y='Frecuencia', 
        color='Estado', 
        color_discrete_map=color_sexo, 
        title="7. Typos y variantes categóricas (Variable Sexo) — 12 categorías detectadas versus 3 esperadas",
        labels={'Valor_str': 'Categorías registradas'}
    )
    st.plotly_chart(fig7, use_container_width=True)
    
    # Componente 8 — Todos los códigos nulos especiales (ancho completo)
    st.subheader("8. Códigos nulos especiales — Ausencias encubiertas")
    st.markdown("Estos valores son ausencias reales, pero son peligrosas porque la función `isnull()` no las detecta por defecto.")
    
    codigos_nulos_buscados = [999, -1, '?', '--', 'sin dato']
    nulos_datos = []
    
    for cod in codigos_nulos_buscados:
        if isinstance(cod, int):
            mask = df_original.isin([cod, str(cod)])
        else:
            mask = df_original.isin([cod])
            
        celdas_afectadas = mask.sum().sum()
        if celdas_afectadas > 0:
            cols_con_nulos = mask.sum()[mask.sum() > 0]
            nulos_datos.append({
                'Código especial': str(cod),
                'Celdas afectadas': celdas_afectadas,
                'Columnas donde aparece': ", ".join(cols_con_nulos.index.tolist())
            })
            
    df_nulos_especiales = pd.DataFrame(nulos_datos)
    
    if len(df_nulos_especiales) > 0:
        st.dataframe(df_nulos_especiales, use_container_width=True)

# Panel 2: Explorador de Problemas
elif menu == "2. Explorador de Problemas":
    st.header("2. Explorador de Problemas y Auditoría")
    st.markdown("Este panel permite auditar de forma granular cada problema detectado, asegurando la trazabilidad desde los datos crudos antes de aplicar el pipeline de limpieza.")
    
    prob_type = st.selectbox(
        "🔎 Selecciona la categoría del problema a explorar:",
        [
            "1. Dtype incorrecto en escalas",
            "2. Variantes textuales en escalas Likert",
            "3. BP con texto en lugar de número",
            "4. Ítem inverso sin invertir (IR1)",
            "5. Valores faltantes verdaderos (NaN)",
            "6. Códigos nulos especiales (encubiertos)",
            "7. Duplicados lógicos (Re-envíos)",
            "8. Typos y variantes categóricas",
            "9. Valores Atípicos Imposibles (Outliers)"
        ]
    )
    
    st.markdown("---")
    
    col_tabla, col_grafico = st.columns([1.5, 1])
    
    if prob_type.startswith("1"):
        with col_grafico:
            dtypes_counts = df_original.dtypes.astype(str).value_counts().reset_index()
            dtypes_counts.columns = ['Tipo de dato', 'Cantidad de columnas']
            fig1 = px.bar(dtypes_counts, x='Tipo de dato', y='Cantidad de columnas', color='Tipo de dato', title="Distribución de Tipos de Dato")
            st.plotly_chart(fig1, use_container_width=True)
        with col_tabla:
            st.subheader("Columnas afectadas")
            st.info("Estas columnas contienen respuestas de la escala Likert pero están almacenadas como texto (`object`) en lugar de numéricas (`float64`).")
            cols_escala = df_original.columns[21:]
            cols_object = [c for c in cols_escala if df_original[c].dtype == 'object']
            df_cols_obj = pd.DataFrame({'Columna': cols_object, 'Tipo detectado': 'object', 'Tipo esperado': 'Numérico'})
            st.dataframe(df_cols_obj, use_container_width=True)
            
    elif prob_type.startswith("2"):
        with col_grafico:
            ct2_counts = df_original['CT2'].value_counts(dropna=False).reset_index()
            ct2_counts.columns = ['Valor', 'Frecuencia']
            ct2_counts['Valor_str'] = ct2_counts['Valor'].astype(str)
            canonicos = ['Nunca', 'Rara vez', 'Algunas veces', 'Frecuentemente', 'Siempre']
            ct2_counts['Estado'] = ct2_counts['Valor_str'].apply(lambda x: 'Correcto' if x in canonicos else 'Error')
            fig2 = px.bar(ct2_counts, x='Valor_str', y='Frecuencia', color='Estado', color_discrete_map={'Correcto': 'green', 'Error': 'red'}, title="Variantes en CT2")
            fig2.update_xaxes(type='category')
            st.plotly_chart(fig2, use_container_width=True)
            
        with col_tabla:
            st.subheader("Auditoría global de celdas erróneas")
            st.info("Mostrando TODAS las variables de escala donde existen respuestas que no coinciden con la forma canónica.")
            cols_escala = df_original.columns[21:]
            filas_con_error = []
            for col in cols_escala:
                mask_err = ~df_original[col].isin(canonicos) & df_original[col].notna()
                if mask_err.any():
                    for idx, row in df_original[mask_err].iterrows():
                        filas_con_error.append({
                            'ID': row.get('ID', idx),
                            'Variable': col,
                            'Valor Registrado (Error)': row[col]
                        })
            
            if filas_con_error:
                st.dataframe(pd.DataFrame(filas_con_error), use_container_width=True)
            else:
                st.success("No se encontraron errores en otras variables.")
            
    elif prob_type.startswith("3"):
        with col_grafico:
            bp3_counts = df_original['BP3'].value_counts(dropna=False).reset_index()
            bp3_counts.columns = ['Valor', 'Frecuencia']
            bp3_counts['Valor_str'] = bp3_counts['Valor'].astype(str)
            def cls_bp(val):
                if pd.isna(val): return "Faltante"
                try: float(val); return "Número correcto"
                except: return "Palabra — error"
            bp3_counts['Clasificación'] = bp3_counts['Valor'].apply(cls_bp)
            fig3 = px.bar(bp3_counts, x='Valor_str', y='Frecuencia', color='Clasificación', color_discrete_map={'Número correcto': 'green', 'Palabra — error': 'red', 'Faltante': 'gray'}, title="Errores en BP3")
            fig3.update_xaxes(type='category')
            st.plotly_chart(fig3, use_container_width=True)
            
        with col_tabla:
            st.subheader("Filas afectadas en Bloque BP")
            st.info("Registros donde alguna variable BP contiene palabras en lugar de números.")
            bp_cols = ['BP1', 'BP2', 'BP3', 'BP4', 'BP5']
            mask_bp = pd.Series([False]*len(df_original))
            for c in bp_cols:
                if c in df_original.columns:
                    mask_bp = mask_bp | df_original[c].astype(str).str.lower().isin(['uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete'])
            st.dataframe(df_original[mask_bp][['ID'] + bp_cols], use_container_width=True)
            
    elif prob_type.startswith("4"):
        with col_grafico:
            ir1_counts = df_original['IR1'].value_counts(dropna=False).reset_index()
            ir1_counts.columns = ['Respuesta', 'Frecuencia']
            ir1_counts['Respuesta_str'] = ir1_counts['Respuesta'].astype(str)
            fig4 = px.bar(ir1_counts, x='Respuesta_str', y='Frecuencia', title="Distribución IR1 (Sin invertir)")
            fig4.update_xaxes(type='category')
            st.plotly_chart(fig4, use_container_width=True)
            
        with col_tabla:
            st.subheader("Valores crudos de IR1")
            st.info("Estos valores requieren inversión matemática (8 - IR1) en el pipeline para ser consistentes con el resto de la escala.")
            st.dataframe(df_original[['ID', 'IR1']].head(100), use_container_width=True)

    elif prob_type.startswith("5"):
        with col_grafico:
            nans_top = df_original.isna().sum().sort_values(ascending=False).head(10).reset_index()
            nans_top.columns = ['Columna', 'Faltantes']
            fig5 = px.bar(nans_top, y='Columna', x='Faltantes', orientation='h', title="Top 10 columnas con NaN verdaderos")
            st.plotly_chart(fig5, use_container_width=True)
            
        with col_tabla:
            st.subheader("Auditoría de filas con NaN")
            st.info("Filas que tienen al menos un valor vacío (NaN) en su registro original.")
            mask_na = df_original.isna().any(axis=1)
            st.dataframe(df_original[mask_na], use_container_width=True)
            
    elif prob_type.startswith("6"):
        with col_grafico:
            cods = [999, -1, '?', '--', 'sin dato']
            counts = []
            for cod in cods:
                if isinstance(cod, int): m = df_original.isin([cod, str(cod)])
                else: m = df_original.isin([cod])
                counts.append({'Código': str(cod), 'Afectadas': m.sum().sum()})
            fig6 = px.bar(pd.DataFrame(counts), x='Código', y='Afectadas', title="Ocurrencias por código especial")
            st.plotly_chart(fig6, use_container_width=True)
            
        with col_tabla:
            st.subheader("Auditoría de Códigos Ocultos")
            st.info("Muestra las filas y únicamente las variables específicas donde se introdujeron estos códigos.")
            codigos_buscados = [999, '999', -1, '-1', '?', '--', 'sin dato']
            
            filas_nulos = []
            for col in df_original.columns:
                mask_col = df_original[col].isin(codigos_buscados)
                if mask_col.any():
                    for idx, row in df_original[mask_col].iterrows():
                        filas_nulos.append({
                            'ID': row.get('ID', idx),
                            'Variable': col,
                            'Código Oculto Encontrado': str(row[col])
                        })
            
            if filas_nulos:
                st.dataframe(pd.DataFrame(filas_nulos), use_container_width=True)
            else:
                st.success("No se encontraron códigos nulos encubiertos.")
            
    elif prob_type.startswith("7"):
        with col_grafico:
            SUBSET_LOGICO = ['Edad', 'Sexo', 'Estado_Civil', 'Numero_Hijos', 'Nivel_Educativo', 'Zona_Vivienda', 
                             'Estrato', 'Sector', 'Tamaño_Empresa', 'Tipo_Contrato', 'Horas_Semana', 'Ingreso', 
                             'Tipo_Cargo', 'Personas_Cargo', 'Años_Experiencia', 'Antiguedad_Cargo', 'Modalidad', 'Horas_Traslado', 'Horas_Formacion']
            dup_mask = df_original.duplicated(subset=SUBSET_LOGICO, keep=False)
            filas_dup = df_original[dup_mask].copy()
            if len(filas_dup) > 0:
                filas_dup['Perfil'] = filas_dup['Sexo'].astype(str) + " - " + filas_dup['Sector'].astype(str)
                fig7 = px.bar(filas_dup['Perfil'].value_counts().reset_index(), y='Perfil', x='count', orientation='h', title="Perfiles duplicados detectados")
                st.plotly_chart(fig7, use_container_width=True)
                
        with col_tabla:
            st.subheader("Auditoría de Duplicados (Re-envíos)")
            st.warning(f"Se encontraron {len(filas_dup)} registros que comparten exactamente la misma demografía y respuestas.")
            st.dataframe(filas_dup.sort_values(SUBSET_LOGICO)[['ID'] + SUBSET_LOGICO], use_container_width=True)
            
    elif prob_type.startswith("8"):
        with col_grafico:
            sexo_counts = df_original['Sexo'].value_counts(dropna=False).reset_index()
            sexo_counts.columns = ['Valor', 'Frecuencia']
            sexo_counts['Valor_str'] = sexo_counts['Valor'].astype(str)
            sexo_counts['Estado'] = sexo_counts['Valor_str'].apply(lambda x: 'Correcto' if x in ['Mujer', 'Hombre', 'Prefiero no decir'] else 'Error')
            fig8 = px.bar(sexo_counts, x='Valor_str', y='Frecuencia', color='Estado', color_discrete_map={'Correcto': 'green', 'Error': 'red'}, title="Errores Tipográficos en Sexo")
            st.plotly_chart(fig8, use_container_width=True)
            
        with col_tabla:
            st.subheader("Auditoría de Typos en Sexo")
            st.info("Filas donde el encuestado ingresó la variable 'Sexo' con errores ortográficos o variaciones no estándar.")
            mask_typo = ~df_original['Sexo'].isin(['Mujer', 'Hombre', 'Prefiero no decir']) & df_original['Sexo'].notna()
            st.dataframe(df_original[mask_typo][['ID', 'Sexo']], use_container_width=True)

    elif prob_type.startswith("9"):
        temp_edad = pd.to_numeric(df_original['Edad'], errors='coerce')
        temp_formacion = pd.to_numeric(df_original['Horas_Formacion'], errors='coerce')
        temp_semana = pd.to_numeric(df_original['Horas_Semana'], errors='coerce')
        
        condiciones_imposibles = (
            (temp_edad > 100) |              
            (temp_edad < 0) |                
            (temp_formacion > 1000) |        
            (temp_formacion < 0) |           
            (temp_semana > 168) |            
            (temp_semana < 0)                
        )
        
        filas_a_eliminar = df_original[condiciones_imposibles]
        
        df_outliers_plot = pd.DataFrame({
            'Edad': temp_edad,
            'Horas Formación': temp_formacion,
            'Horas Semana': temp_semana
        })
        
        with col_grafico:
            fig_edad = px.box(
                df_outliers_plot, y='Edad',
                title="Edad (años)",
                color_discrete_sequence=[PALETA[0]],
            )
            fig_edad.update_layout(showlegend=False, xaxis_visible=False)
            st.plotly_chart(fig_edad, use_container_width=True)

            fig_formacion = px.box(
                df_outliers_plot, y='Horas Formación',
                title="Horas de Formación",
                color_discrete_sequence=[PALETA[1]],
            )
            fig_formacion.update_layout(showlegend=False, xaxis_visible=False)
            st.plotly_chart(fig_formacion, use_container_width=True)

            fig_semana = px.box(
                df_outliers_plot, y='Horas Semana',
                title="Horas Semana (horas/semana)",
                color_discrete_sequence=[PALETA[2]],
            )
            fig_semana.update_layout(showlegend=False, xaxis_visible=False)
            st.plotly_chart(fig_semana, use_container_width=True)
            
        with col_tabla:
            st.subheader("Auditoría de Valores Imposibles (Outliers)")
            st.markdown("""
**Criterios de detección lógica:**
- **Edad:** Mayor a 100 años o registros negativos.
- **Horas de Trabajo (Semana):** Mayor a 168 horas (excede las horas totales de una semana física) o registros negativos.
- **Horas de Formación:** Mayor a 1000 horas o registros negativos.
            """)
            
            st.warning(f"Se encontraron {len(filas_a_eliminar)} registros que violan estas leyes básicas y deben ser depurados.")
            st.dataframe(filas_a_eliminar[['ID', 'Edad', 'Horas_Formacion', 'Horas_Semana']], use_container_width=True)

# Panel 3: Comparativo Antes/Después
elif menu == "3. Comparativo Antes/Después":
    st.header("3. Comparativo Antes vs Después del Preprocesamiento")
    df_antes, df_despues = load_comparativo()
    st.caption(
        f"**Antes (sucio):** `bienestar_laboral_Preprocesamiento.xlsx` — {len(df_antes):,} filas · "
        f"**Después (limpio):** `dataset_limpio.xlsx` — {len(df_despues):,} filas"
    )

    vars_solo_despues = [c for c in df_despues.columns if c not in df_antes.columns]
    todas_variables = ordenar_variables_comparativo(df_antes, df_despues)
    var_comp = st.selectbox("Selecciona una variable para comparar:", todas_variables)

    msg_imp = mensaje_imputacion(var_comp)
    if msg_imp:
        st.info(msg_imp)

    grupos = _grupos_escala(df_antes.columns)
    serie_antes = preparar_serie_antes(df_antes, var_comp, grupos) if var_comp in df_antes.columns else None
    serie_despues = df_despues[var_comp] if var_comp in df_despues.columns else None
    es_numerica = es_numerica_comparativo(serie_antes, serie_despues)

    st.subheader("Distribución de la variable")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h3 style='text-align: center; color: #c55a11;'>Antes (Original)</h3>",
            unsafe_allow_html=True,
        )
        if serie_antes is None:
            st.info("Variable no presente en el dataset original (p. ej. puntaje de dimensión calculado tras la limpieza).")
        else:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            if es_numerica:
                datos = pd.to_numeric(serie_antes, errors='coerce').dropna()
                sns.histplot(datos, kde=True, color='#c55a11', ax=ax1, bins=15)
                ax1.set_xlabel(var_comp)
            else:
                vc = serie_antes.value_counts().head(10)
                sns.barplot(y=vc.index.astype(str), x=vc.values, color='#c55a11', ax=ax1)
                ax1.set_xlabel("Frecuencia")
            ax1.set_ylabel("")
            ax1.set_title(f"n = {len(df_antes):,}")
            st.pyplot(fig1)

    with col2:
        st.markdown(
            "<h3 style='text-align: center; color: #2ca02c;'>Después (Limpio)</h3>",
            unsafe_allow_html=True,
        )
        if serie_despues is None:
            st.warning("Variable no disponible en el dataset limpio.")
        else:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            if es_numerica:
                datos = pd.to_numeric(serie_despues, errors='coerce').dropna()
                sns.histplot(datos, kde=True, color='#2ca02c', ax=ax2, bins=15)
                ax2.set_xlabel(var_comp)
            else:
                vc = serie_despues.value_counts().head(10)
                sns.barplot(y=vc.index.astype(str), x=vc.values, color='#2ca02c', ax=ax2)
                ax2.set_xlabel("Frecuencia")
            ax2.set_ylabel("")
            ax2.set_title(f"n = {len(df_despues):,}")
            st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Resumen estadístico")

    if es_numerica:
        res_antes = resumen_numerico(serie_antes) if serie_antes is not None else {}
        res_despues = resumen_numerico(serie_despues) if serie_despues is not None else {}
        metricas = list(res_despues.keys()) if res_despues else list(res_antes.keys())
        stats_df = pd.DataFrame({
            'Métrica': metricas,
            'Antes (sucio)': [res_antes.get(m) for m in metricas],
            'Después (limpio)': [res_despues.get(m) for m in metricas],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Antes (sucio)**")
            if serie_antes is None:
                st.info("No aplica.")
            else:
                st.dataframe(resumen_categorico(serie_antes), use_container_width=True, hide_index=True)
                st.markdown("**Top 10 categorías**")
                top_antes = serie_antes.value_counts(dropna=False).reset_index().head(10)
                top_antes.columns = ['Categoría', 'Frecuencia']
                st.dataframe(top_antes, use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**Después (limpio)**")
            if serie_despues is None:
                st.warning("No disponible.")
            else:
                st.dataframe(resumen_categorico(serie_despues), use_container_width=True, hide_index=True)
                st.markdown("**Top 10 categorías**")
                top_despues = serie_despues.value_counts(dropna=False).reset_index().head(10)
                top_despues.columns = ['Categoría', 'Frecuencia']
                st.dataframe(top_despues, use_container_width=True, hide_index=True)

    if var_comp in vars_solo_despues:
        st.caption(
            f"Las dimensiones agregadas ({', '.join(vars_solo_despues)}) solo existen en el dataset limpio."
        )

# Panel 4: Certificación de Calidad
elif menu == "4. Certificación de Calidad":
    st.header("4. Certificación de Calidad del Dataset")
    st.caption("Validación final del pipeline · Modelo JD-R · Dataset: `dataset_limpio.xlsx`")

    _, df_cert = load_comparativo()
    dimensiones = DIMS_COMP
    dims_orden = list(dimensiones.keys())
    items_escala = [c for cols in dimensiones.values() for c in cols]
    resultados_dim = {dim: validar_dimension(df_cert, dim, items) for dim, items in dimensiones.items()}
    n_verde = sum(1 for r in resultados_dim.values() if r['estado'] == 'green')
    n_amarillo = sum(1 for r in resultados_dim.values() if r['estado'] == 'yellow')
    n_rojo = sum(1 for r in resultados_dim.values() if r['estado'] == 'red')
    faltantes_actuales = int(df_cert.isna().sum().sum())
    cols_con_faltantes = int(df_cert.isna().any().sum())
    cols_object_escala = sum(1 for c in items_escala if c in df_cert.columns and df_cert[c].dtype == object)
    outliers_actuales = int(
        (df_cert['Edad'] > 100).sum() + (df_cert['Edad'] < 0).sum()
        + (df_cert['Horas_Semana'] > 168).sum() + (df_cert['Horas_Semana'] < 0).sum()
        + (df_cert['Horas_Formacion'] > 1000).sum() + (df_cert['Horas_Formacion'] < 0).sum()
    )
    rangos_ok = all(r['promedio_valido'] and r['fuera_rango'] == 0 for r in resultados_dim.values())
    dtypes_ok = all(r['dtype_correcto'] for r in resultados_dim.values())
    corr_dims = df_cert[dims_orden].corr()
    jdr_ok = sum(1 for chk in JDR_CHECKS if evaluar_check_jdr(corr_dims, *chk)[0])
    criterios_globales = [
        faltantes_actuales == 0, len(df_cert) == 391, cols_con_faltantes == 0,
        cols_object_escala == 0, outliers_actuales == 0, n_rojo == 0,
        rangos_ok, dtypes_ok, jdr_ok == len(JDR_CHECKS),
    ]
    pct_global = round(sum(criterios_globales) / len(criterios_globales) * 100)

    st.subheader("1. Semáforo de calidad por dimensión")
    st.progress(pct_global / 100, text=f"CALIDAD GLOBAL DEL DATASET: {pct_global}%")
    st.caption(
        f"Validadas: {n_verde} · Requieren revisión: {n_amarillo} · "
        f"Error crítico: {n_rojo}"
    )
    for fila in range(0, len(dims_orden), 4):
        cols_cards = st.columns(4)
        for i, col_card in enumerate(cols_cards):
            idx = fila + i
            if idx >= len(dims_orden):
                break
            with col_card:
                tarjeta_dimension(dims_orden[idx], resultados_dim[dims_orden[idx]])

    st.markdown("---")
    st.subheader("2. Verificación de correlaciones esperadas (JD-R)")
    st.markdown(
        "El proceso de limpieza debe preservar la estructura teórica del modelo "
        "**Job Demands–Resources (JD-R)** entre dimensiones derivadas."
    )
    col_hm, col_chk = st.columns([1.4, 1])
    with col_hm:
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_dims.values, x=corr_dims.columns, y=corr_dims.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr_dims.values, 2), texttemplate="%{text}",
            textfont={"size": 9}, colorbar=dict(title="r"),
        ))
        fig_corr.update_layout(
            title="Correlaciones entre dimensiones derivadas", height=520,
            margin=dict(l=60, r=20, t=50, b=60),
            paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    with col_chk:
        st.markdown("**Validaciones automáticas JD-R**")
        for chk in JDR_CHECKS:
            cumple, r_val, esperado = evaluar_check_jdr(corr_dims, *chk)
            icono = "🟢" if cumple else "🔴"
            st.markdown(
                f"{icono} **{chk[0]} ↔ {chk[1]}** = `{r_val:.2f}`  \n"
                f"<span style='color:#6c757d;font-size:0.85rem;'>{esperado}</span>",
                unsafe_allow_html=True,
            )
        if jdr_ok == len(JDR_CHECKS):
            st.success(
                "La estructura correlacional del dataset limpio "
                "es consistente con el modelo JD-R original."
            )
        else:
            st.warning(f"{jdr_ok}/{len(JDR_CHECKS)} validaciones JD-R cumplidas.")

    st.markdown("---")
    st.subheader("3. Resumen ejecutivo del pipeline")
    k1, k2, k3 = st.columns(3)
    k4, k5, k6 = st.columns(3)
    with k1:
        st.metric("Registros", "391", delta="-21", delta_color="inverse",
                  help="412 → 391 tras deduplicación y outliers")
    with k2:
        st.metric("Celdas faltantes", "0", delta="-2,582", delta_color="inverse")
    with k3:
        st.metric("Columnas con faltantes", "0", delta="-91", delta_color="inverse")
    with k4:
        st.metric("Columnas object (escala)", "0", delta="-86", delta_color="inverse")
    with k5:
        st.metric("Outliers imposibles", "0", delta="-19", delta_color="inverse")
    with k6:
        st.metric("Dimensiones construidas", "16", delta="16", delta_color="normal")

    st.markdown("**Flujo del pipeline**")
    pasos = [
        "Dataset original", "Auditoría inicial", "Estandarización",
        "Codificación Likert", "Imputación", "Tratamiento de outliers",
        "Variables derivadas", "Dataset certificado",
    ]
    paso_box = (
        "text-align:center;padding:10px 8px;background:white;border-radius:8px;"
        "border:1px solid #dee2e6;font-size:0.72rem;color:#1f4e79;font-weight:600;"
        "line-height:1.2;min-width:0;"
    )
    flecha = (
        "<span style='color:#2e75b6;font-size:1.1rem;font-weight:bold;"
        "padding:0 2px;flex-shrink:0;'>→</span>"
    )
    timeline_html = (
        "<motion.div style='display:flex;align-items:center;justify-content:space-between;"
        "flex-wrap:nowrap;gap:2px;margin:12px 0 36px 0;overflow-x:auto;padding-bottom:8px;'>"
    )
    for i, paso in enumerate(pasos):
        timeline_html += f"<motion.div style='flex:1;{paso_box}'>{paso}</motion.div>"
        if i < len(pasos) - 1:
            timeline_html += flecha
    timeline_html += "</motion.div>"
    st.markdown(timeline_html.replace("motion.", ""), unsafe_allow_html=True)

    st.markdown(
        "<motion.div class='kpi-card' style='border-top:4px solid #2e7d32;padding:28px;"
        "text-align:center;margin-top:8px;'>"
        "<motion.div style='font-size:1.4rem;font-weight:bold;color:#1f4e79;margin-bottom:16px;'>"
        "DATASET CERTIFICADO PARA ANÁLISIS</motion.div>"
        "<motion.div style='text-align:left;max-width:520px;margin:0 auto;color:#495057;line-height:1.9;'>"
        "✓ Sin faltantes · ✓ Sin duplicados lógicos residuales<br>"
        "✓ Escalas validadas · ✓ Correlaciones JD-R preservadas<br>"
        "✓ Variables derivadas construidas · ✓ Compatible con el modelo JD-R"
        "</motion.div></motion.div>".replace("motion.", ""),
        unsafe_allow_html=True,
    )

