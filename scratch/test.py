import pandas as pd
import plotly.express as px
df = pd.read_pickle('df_original.pkl')
c = df['BP3'].value_counts(dropna=False).reset_index()
c.columns=['Valor', 'Frecuencia']
c['Valor_str'] = c['Valor'].astype(str)
def cls(v):
    if pd.isna(v): return 'Faltante'
    try:
        float(v); return 'Número correcto'
    except:
        return 'Palabra — error'
c['Clasificación'] = c['Valor'].apply(cls)
fig = px.bar(c, x='Valor_str', y='Frecuencia', color='Clasificación', color_discrete_map={'Número correcto': 'green', 'Palabra — error': 'red', 'Faltante': 'gray'})
fig.write_image('scratch/bp3_test.png')
print('done')
