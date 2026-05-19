import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_excel('bienestar_laboral_Preprocesamiento.xlsx')
    df_original = df.copy()
    
    # Identify variables
    COLS_SOCIODEM = df.columns[:21].tolist()
    COLS_BP = [c for c in df.columns if c.startswith('BP')]
    DIMS = {
        'CTRL':    ['CT1', 'CT2', 'CT3'],
        'PRES':    ['PT1', 'PT2', 'PT3', 'PT4'],
        'LIDER':   ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6', 'CL7'],
        'COMP':    ['AC1', 'AC2', 'AC3'],
        'ROL_C':   ['CR1', 'CR2', 'CR3', 'CR4'],
        'ROL_CON': ['CoR1', 'CoR2', 'CoR3'],
        'CAMBIO':  ['GC1', 'GC2', 'GC3', 'GC4'],
        'SM_ORG':  ['SM1', 'SM2', 'SM3', 'SM4', 'SM5'],
        'SAT':     ['SAT1', 'SAT2', 'SAT3', 'SAT4', 'SAT5', 'SAT6', 'SAT7', 'SAT8', 'SAT9'],
        'RETIRO':  ['IR1', 'IR2', 'IR3', 'IR4'],
        'FAM_TRAB':['FT1', 'FT2', 'FT3', 'FT4', 'FT5'],
        'TRAB_FAM':['TF1', 'TF2', 'TF3', 'TF4', 'TF5'],
        'BURNOUT': ['BU1', 'BU2', 'BU3', 'BU4', 'BU5', 'BU6', 'BU7', 'BU8', 'BU9', 'BU10', 'BU11', 'BU12'],
        'BIENESTAR':COLS_BP,
        'SOMATIZ': ['SOM1', 'SOM2', 'SOM3', 'SOM4', 'SOM5'],
        'DESGASTE':['DL1', 'DL2', 'DL3', 'DL4', 'DL5', 'DL6', 'DL7', 'DL8'],
    }
    COLS_ESCALA = []
    for items in DIMS.values():
        COLS_ESCALA.extend(items)
        
    COLS_FREC    = (DIMS['CTRL'] + DIMS['PRES'] + DIMS['LIDER'] + DIMS['COMP'] +
                    DIMS['ROL_C'] + DIMS['ROL_CON'] + DIMS['CAMBIO'] + DIMS['SM_ORG'])
    COLS_ACUERDO = DIMS['SAT'] + DIMS['RETIRO'] + DIMS['FAM_TRAB'] + DIMS['TRAB_FAM']
    COLS_BU_FREC = DIMS['BURNOUT']
    COLS_SOMATIZ = DIMS['SOMATIZ'] + DIMS['DESGASTE']

    # Step 1: Null codes -> NaN
    CODIGOS_NULOS_REPLACE = [999, -1, '999', '-1', 'N/A', 'NA', 'null', 'NULL',
                              'Null', 'no aplica', 'No aplica', 'NO APLICA',
                              '.', 'nan', 'NaN', 'sin dato', 'Sin dato', '?', '--']
    for codigo in CODIGOS_NULOS_REPLACE:
        df = df.replace(codigo, np.nan)
    for col in ['Edad', 'Horas_Semana', 'Horas_Formacion']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 2: Remove Logical Duplicates
    SUBSET_LOGICO = ['Edad', 'Sexo', 'Estado_Civil', 'Numero_Hijos', 'Nivel_Educativo', 'Zona_Vivienda', 
                     'Estrato', 'Sector', 'Tamaño_Empresa', 'Tipo_Contrato', 'Horas_Semana', 'Ingreso', 
                     'Tipo_Cargo', 'Personas_Cargo', 'Años_Experiencia', 'Antiguedad_Cargo', 'Modalidad', 
                     'Horas_Traslado', 'Horas_Formacion']
    dup_mask = df.duplicated(subset=SUBSET_LOGICO, keep=False)
    items_likert = [c for c in df.columns if c not in COLS_SOCIODEM]
    filas_dup = df[dup_mask].sort_values(SUBSET_LOGICO + ['ID']).reset_index(drop=True)
    pares_confirmados = []
    for i in range(len(filas_dup) - 1):
        fila_a = filas_dup.iloc[i]
        fila_b = filas_dup.iloc[i + 1]
        mismo_grupo = all((pd.isna(fila_a[col]) and pd.isna(fila_b[col])) or fila_a[col] == fila_b[col] for col in SUBSET_LOGICO)
        if not mismo_grupo: continue
        coincidencias = (fila_a[items_likert] == fila_b[items_likert]).sum()
        validas = (fila_a[items_likert].notna() & fila_b[items_likert].notna()).sum()
        pct = (coincidencias / validas * 100) if validas > 0 else 0
        if pct == 100.0:
            pares_confirmados.append(max(fila_a['ID'], fila_b['ID']))
    
    df = df[~df['ID'].isin(pares_confirmados)].reset_index(drop=True)

    # Step 3: Categorical standardizing and typos
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    for col in obj_cols:
        df[col] = df[col].str.strip()
    
    VARS_TITLE = ['Sexo', 'Estado_Civil', 'Sector', 'Tipo_Cargo', 'Modalidad', 'Zona_Vivienda', 'Trabajo_Turnos', 'Tipo_Contrato', 'Personas_Cargo']
    for col in VARS_TITLE:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.title()
            
    DICT_CORRECCIONES = {
        'Sexo': {'Mjer': 'Mujer', 'Hombree': 'Hombre', 'Prefiero No Decir': 'Prefiero no decir'},
        'Estado_Civil': {'Casdo': 'Casado', 'Solero': 'Soltero', 'Unión Libre': 'Unión libre'},
        'Sector': {'Mixo': 'Mixto', 'Publico': 'Público'},
        'Tipo_Cargo': {'Adminstrativo': 'Administrativo', 'Opearativo': 'Operativo'},
    }
    for col, mapeo in DICT_CORRECCIONES.items():
        if col in df.columns:
            df[col] = df[col].replace(mapeo)

    # Step 4: Scales mappings
    todas_likert = COLS_FREC + COLS_BU_FREC + COLS_SOMATIZ + DIMS['SAT'] + DIMS['RETIRO'] + DIMS['FAM_TRAB'] + DIMS['TRAB_FAM']
    for col in todas_likert:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.capitalize()
            
    df = df.replace({
        'Siempree': 'Siempre', 'Frecuente': 'Frecuentemente', 'A menudo': 'Frecuentemente', 
        'Alguna vez': 'Algunas veces', 'Raramente': 'Rara vez'
    })

    MAP_FRECUENCIA = {'Nunca': 1, 'Rara vez': 2, 'Algunas veces': 3, 'Frecuentemente': 4, 'Siempre': 5}
    MAP_ACUERDO = {'Muy en desacuerdo': 1, 'Moderadamente en desacuerdo': 2, 'Algo en desacuerdo': 3, 
                   'Ni de acuerdo ni en desacuerdo': 4, 'Algo de acuerdo': 5, 'Moderadamente de acuerdo': 6, 'Muy de acuerdo': 7}
    MAP_BURNOUT = {'Nunca': 1, 'Raramente': 2, 'Algunas veces': 3, 'A menudo': 4, 'Siempre': 5}
    MAP_SOMATIZ = {'Nunca': 1, 'Raramente': 2, 'Ocasionalmente': 3, 'Algunas veces': 4, 'Frecuentemente': 5, 'Casi siempre': 6, 'Siempre': 7}

    for col in COLS_FREC: df[col] = pd.to_numeric(df[col].map(MAP_FRECUENCIA), errors='coerce')
    for col in COLS_ACUERDO: df[col] = pd.to_numeric(df[col].map(MAP_ACUERDO), errors='coerce')
    for col in COLS_BU_FREC: df[col] = pd.to_numeric(df[col].map(MAP_BURNOUT), errors='coerce')
    for col in COLS_SOMATIZ: df[col] = pd.to_numeric(df[col].map(MAP_SOMATIZ), errors='coerce')

    # Fix BP
    MAP_PALABRAS_NUMERO = {'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5, 'seis': 6, 'siete': 7, 
                           'Uno': 1, 'Dos': 2, 'Tres': 3, 'Cuatro': 4, 'Cinco': 5, 'Seis': 6, 'Siete': 7, 
                           'UNO': 1, 'DOS': 2, 'TRES': 3, 'CUATRO': 4, 'CINCO': 5, 'SEIS': 6, 'SIETE': 7}
    for col in ['BP1', 'BP2', 'BP3', 'BP4', 'BP5']:
        df[col] = df[col].replace(MAP_PALABRAS_NUMERO)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['BP6', 'BP7', 'BP8', 'BP9', 'BP10']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Invert IR1
    if 'IR1' in df.columns:
        df['IR1'] = 8 - df['IR1']

    # Step 5: Imputation
    cols_cat_falt = [c for c in COLS_SOCIODEM if df[c].dtype == object and df[c].isnull().any()]
    if cols_cat_falt:
        for col in cols_cat_falt:
            df[col] = df[col].fillna(df[col].mode()[0])

    for dim, items in DIMS.items():
        items_en_df = [c for c in items if c in df.columns]
        for col in items_en_df:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
    for col in ['Edad', 'Horas_Semana', 'Horas_Formacion']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Eliminate impossible outliers
    condiciones_imposibles = (
        (df['Edad'] > 100) | (df['Edad'] < 0) |
        (df['Horas_Formacion'] > 1000) | (df['Horas_Formacion'] < 0) |
        (df['Horas_Semana'] > 168) | (df['Horas_Semana'] < 0)
    )
    df = df[~condiciones_imposibles].reset_index(drop=True)

    # Compute Dimension Scores
    for dim, items in DIMS.items():
        df[dim] = df[items].mean(axis=1)

    print("Data cleaning completed.")
    
    df_original.to_pickle('df_original.pkl')
    df.to_pickle('df_cleaned.pkl')
    
    # Save the quality issues table found in the notebook
    problemas = pd.DataFrame([
        {'Columna_afectada': 'CT1…DL8 (86 cols)', 'Tipo_problema': 'dtype incorrecto', 'Magnitud': '86 columnas', 'Descripcion': 'Columnas de escala almacenadas como object en lugar de numérico', 'Impacto_potencial': 'Imposible calcular estadísticos, correlaciones o imputar con sklearn', 'Prioridad': 'Crítica'},
        {'Columna_afectada': '91 columnas de escala', 'Tipo_problema': 'Valores faltantes NaN', 'Magnitud': '2 582 celdas (5.6%)', 'Descripcion': 'Celdas vacías verdaderas detectadas con df.isnull()', 'Impacto_potencial': 'Promedios de dimensión sesgados, modelos con errores si no se imputa', 'Prioridad': 'Alta'},
        {'Columna_afectada': 'Edad, Horas_Semana, IR1, SOM4, SM5', 'Tipo_problema': 'Código nulo 999', 'Magnitud': '5 celdas', 'Descripcion': 'El valor 999 representa ausencia pero no es NaN; no detectado por isnull()', 'Impacto_potencial': 'Media de Edad sesgada, 999 mapeado a NaN en escala', 'Prioridad': 'Alta'},
        {'Columna_afectada': 'Horas_Formacion', 'Tipo_problema': 'Código nulo -1', 'Magnitud': '4 celdas', 'Descripcion': 'El valor -1 representa horas no aplica pero contamina estadísticos', 'Impacto_potencial': 'Estadísticos erróneos de Horas_Formacion', 'Prioridad': 'Media'},
        {'Columna_afectada': 'CT2, IR4, BU10', 'Tipo_problema': "Código nulo '?'", 'Magnitud': '3 celdas', 'Descripcion': 'Signo de interrogación usado como código de ausencia', 'Impacto_potencial': 'Falla el mapeo de escala, queda como NaN sin registro', 'Prioridad': 'Media'},
        {'Columna_afectada': 'IR3, IR4, BU10, DL7', 'Tipo_problema': "Código nulo '--'", 'Magnitud': '5 celdas', 'Descripcion': 'Doble guión usado como código de ausencia', 'Impacto_potencial': 'Falla el mapeo de escala', 'Prioridad': 'Media'},
        {'Columna_afectada': 'CT2, IR4', 'Tipo_problema': "Código nulo 'sin dato'", 'Magnitud': '2 celdas', 'Descripcion': 'Texto literal que representa ausencia', 'Impacto_potencial': 'Falla el mapeo de escala', 'Prioridad': 'Media'},
        {'Columna_afectada': 'Múltiples columnas', 'Tipo_problema': 'Duplicados lógicos', 'Magnitud': '18 filas involucradas', 'Descripcion': 'Re-envíos de formulario con IDs diferentes pero mismo perfil', 'Impacto_potencial': 'Inflación de ciertos grupos demográficos, sesgo en promedios', 'Prioridad': 'Alta'},
        {'Columna_afectada': 'Sexo, Estado_Civil, Sector, Tipo_Cargo, Modalidad', 'Tipo_problema': 'Typos y variantes de capitalización', 'Magnitud': '5 variables', 'Descripcion': "Ej: 'Mjer', 'Casdo', 'Mixo', 'Adminstrativo', '  Presencial  '", 'Impacto_potencial': 'Categorías fantasma, análisis de frecuencias incorrecto', 'Prioridad': 'Alta'},
        {'Columna_afectada': 'CT2, CT3, PT1, PT3, CL3, CL4, CL7, AC1, AC2, CR4, SM1, SM3', 'Tipo_problema': 'Variantes textuales en escala', 'Magnitud': '>12 columnas', 'Descripcion': "Ej: 'NUNCA','nunca', 'Frecuente','A menudo','Siempree','Alguna vez'", 'Impacto_potencial': 'NaN al mapear → subcobertura del instrumento', 'Prioridad': 'Crítica'},
        {'Columna_afectada': 'BP1, BP2, BP3, BP4, BP5', 'Tipo_problema': 'BP con texto en lugar de número', 'Magnitud': '5 columnas, ~30% de celdas', 'Descripcion': "Valores como 'cuatro','cinco','seis' en lugar de 4,5,6", 'Impacto_potencial': 'Toda la dimensión BIENESTAR no se puede analizar', 'Prioridad': 'Crítica'},
        {'Columna_afectada': 'IR1', 'Tipo_problema': 'Ítem inverso sin invertir', 'Magnitud': '1 columna, 412 filas', 'Descripcion': 'IR1 redactado en sentido opuesto — requiere transformación 8 - IR1', 'Impacto_potencial': 'Promedio de dimensión RETIRO invalida la interpretación', 'Prioridad': 'Crítica'}
    ])
    problemas['ID_problema'] = range(1, len(problemas)+1)
    cols_orden = ['ID_problema', 'Tipo_problema', 'Columna_afectada', 'Magnitud', 'Descripcion', 'Impacto_potencial', 'Prioridad']
    problemas = problemas[cols_orden]
    problemas.to_csv('problemas_calidad.csv', index=False)
    
    print("Files saved: df_original.pkl, df_cleaned.pkl, problemas_calidad.csv")

if __name__ == "__main__":
    main()
