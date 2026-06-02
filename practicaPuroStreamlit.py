import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import re
import datetime
from scipy import stats
from scipy.stats import gaussian_kde
 
#@st.cache_data es un decorador que permite almacenar en caché los resultados de una función para mejorar el rendimiento
@st.cache_data
def load_csv(uploaded_file):
    """Lee un archivo CSV subido por Streamlit y devuelve un DataFrame o None si falla."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return None

def freq_table_simple(series):
    #reset_index() convierte el resultado de value_counts() 
    # en un DataFrame con índices numéricos
    frec = series.value_counts().reset_index()
    # series.name es el nombre de la columna original, 
    # se asigna a la primera columna de la tabla de frecuencias
    frec.columns = [series.name, 'frecuencia']
    frec['porcentaje'] = frec['frecuencia'] / frec['frecuencia'].sum() * 100
    # La función cumsum() calcula la frecuencia acumulada sumando 
    # sucesivamente las frecuencias
    frec['frecuencia_acumulada'] = frec['frecuencia'].cumsum()
    # La frecuencia relativa acumulada se calcula sumando 
    # sucesivamente los porcentajes
    frec['porcentaje_acumulado'] = frec['porcentaje'].cumsum()

    # ordenamos la tabla de frecuencias de mayor a menor frecuencia 
    return frec.sort_values(by='frecuencia', ascending=False)

#devuelve una tabla de frecuencia para datos agrupados 
# bins significa el número de intervalos o los bordes de los intervalos
# Usando pd.cut para agrupar los datos en intervalos
# Esta versión usa el mismo método directo que en tu practicaPuro.py.
def freq_table_agrupada(series, bins=None):
    datos = series.dropna()
    n = len(datos)
    if bins is None:
        # La regla de Sturges: k = 1 + 3.322 * log₁₀(n)
        # Donde: k = número de intervalos, n = número de observaciones
        # Ejemplo: Si n=100, entonces k = 1 + 3.322*log₁₀(100) = 1 + 3.322*2 = 7.644 → redondeado a 8 intervalos
        if n > 0:
            bins = int(np.ceil(1 + 3.322 * np.log10(n)))
        else:
            bins = 1

    # Creamos los intervalos con pd.cut, usando right=False para que el intervalo sea [a,b).
    cut_data = pd.cut(datos, bins=bins, include_lowest=True, right=False)

    # Contamos cuántos datos caen en cada intervalo y ordenamos.
    df = cut_data.value_counts().sort_index().reset_index()
    df.columns = ['intervalo', 'frecuencia']

    # Guardamos el intervalo original para calcular la marca de clase.
    intervalos_original = df['intervalo']

    # Formateamos los intervalos como texto con números y corchetes en formato [li,ls).
    def formato_intervalo(intervalo):
        li = intervalo.left
        ls = intervalo.right
        #float.is_integer() verifica si el número es un entero, 
        # si es así lo convierte a int para evitar decimales innecesarios
        if float(li).is_integer():
            li = int(li)
        else:
            li = round(li, 2)
        if float(ls).is_integer():
            ls = int(ls)
        else:
            ls = round(ls, 2)
        return f"[{li},{ls})"
    #con apply se aplica la función formato_intervalo 
    # a cada elemento de la columna 'intervalo'
    df['intervalo'] = df['intervalo'].apply(formato_intervalo)

    # Calculamos la marca de clase como el punto medio del intervalo original
    # lambda se usa para aplicar una función a cada elemento de la columna 'intervalo'
    df['marca_de_clase'] = intervalos_original.apply(lambda x: x.mid)

    # Frecuencia relativa y porcentaje.
    df['hi'] = df['frecuencia'] / len(datos)
    df['porcentaje'] = df['hi'] * 100
    df['Fi'] = df['frecuencia'].cumsum()
    df['Hi'] = df['hi'].cumsum()

    # Retornamos las columnas académicas para la tabla agrupada.
    return df[['intervalo', 'marca_de_clase', 'frecuencia', 'hi', 'porcentaje', 'Fi', 'Hi']]

# A continuación, se definen funciones para graficar los datos según el tipo de variable.
def plot_baston(series):
    fig, ax = plt.subplots(figsize=(6,4))
    values = series.value_counts().sort_index()
    #range(len(values)) genera una secuencia de números 
    # desde 0 hasta el número de valores únicos - 1,
    x = range(len(values))
    ax.stem(x, values.values, basefmt=" ", use_line_collection=True)
    ax.set_xticks(x)
    ax.set_xticklabels(values.index, rotation=45)
    ax.set_xlabel(series.name)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Gráfico de Bastón - {series.name}')
    plt.tight_layout()
    return fig


def plot_histogram_polygon(series, bins=None):
    datos = series.dropna()
    n = len(datos)
    if bins is None:
        bins = int(np.ceil(1 + 3.322 * np.log10(n))) if n > 0 else 1
    fig, ax = plt.subplots(figsize=(6,4))
    counts, edges, patches = ax.hist(datos, bins=bins, color='skyblue', edgecolor='black')
    mid = (edges[:-1] + edges[1:]) / 2
    ax.plot(mid, counts, marker='o', color='red', linestyle='-')
    ax.set_xlabel(series.name)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Histograma y Polígono de Frecuencia - {series.name}')
    plt.tight_layout()
    return fig

# Función para calcular estadísticas descriptivas completas
def compute_descriptive_stats(series):
    """Calcula estadísticas descriptivas: media, mediana, moda, varianza, desv.est., CV, 
    asimetría (Fisher, Pearson, Bowley), cuartiles y curtosis."""
    datos = series.dropna()
    if len(datos) == 0:
        return None
    
    stats_dict = {}
    
    # Tendencia central
    stats_dict['Media'] = np.mean(datos)
    stats_dict['Mediana'] = np.median(datos)
    
    # Moda (puede haber múltiples modas)
    try:
        moda = stats.mode(datos, keepdims=True).mode[0]
        stats_dict['Moda'] = moda
    except:
        stats_dict['Moda'] = np.nan
    
    # Dispersión
    stats_dict['Varianza'] = np.var(datos, ddof=1)  # ddof=1 para varianza muestral
    stats_dict['Desv. Estándar'] = np.std(datos, ddof=1)
    
    # Coeficiente de Variación (CV%)
    media = stats_dict['Media']
    if media != 0:
        stats_dict['Coef. Variación (%)'] = (stats_dict['Desv. Estándar'] / media) * 100
    else:
        stats_dict['Coef. Variación (%)'] = np.nan
    
    # Cuartiles
    stats_dict['Q1 (25%)'] = np.percentile(datos, 25)
    stats_dict['Q2 (50%)'] = np.percentile(datos, 50)  # Mediana
    stats_dict['Q3 (75%)'] = np.percentile(datos, 75)
    
    # Asimetría
    # Fisher (Skewness clásico)
    stats_dict['Asimetría Fisher'] = stats.skew(datos)
    
    # Pearson: 3*(media - mediana) / desv.est.
    mediana = stats_dict['Mediana']
    desv_est = stats_dict['Desv. Estándar']
    if desv_est != 0:
        stats_dict['Asimetría Pearson'] = 3 * (media - mediana) / desv_est
    else:
        stats_dict['Asimetría Pearson'] = np.nan
    
    # Bowley: (Q3 + Q1 - 2*Q2) / (Q3 - Q1)
    q1 = stats_dict['Q1 (25%)']
    q2 = stats_dict['Q2 (50%)']
    q3 = stats_dict['Q3 (75%)']
    if (q3 - q1) != 0:
        stats_dict['Asimetría Bowley'] = (q3 + q1 - 2*q2) / (q3 - q1)
    else:
        stats_dict['Asimetría Bowley'] = np.nan
    
    # Curtosis
    stats_dict['Curtosis'] = stats.kurtosis(datos)
    
    return stats_dict

# Función para crear tres gráficos específicos: tendencia central, dispersión y forma
def plot_descriptive_stats(series, stats_dict):
    """Genera tres figuras: central (media/mediana/moda), dispersión (var, sd, CV, cuartiles), y forma (asimetrías y curtosis)."""
    datos = series.dropna()

    # --- Gráfico 1: Medidas de tendencia central ---
    fig_central, axc = plt.subplots(figsize=(8, 4))
    axc.hist(datos, bins=12, color='#330000', alpha=0.25, edgecolor='black')
    m = stats_dict['Media']
    med = stats_dict['Mediana']
    mod = stats_dict['Moda']
    axc.axvline(m, color='#FFD700', linestyle='-', linewidth=2.5, label=f'Media: {m:.2f}')
    axc.axvline(med, color='#00FF00', linestyle='--', linewidth=2.5, label=f'Mediana: {med:.2f}')
    if not np.isnan(mod):
        axc.axvline(mod, color='#FF00FF', linestyle=':', linewidth=2.5, label=f'Moda: {mod:.2f}')
    axc.set_title('Medidas de Tendencia Central', color='white', fontsize=12, fontweight='bold')
    axc.set_xlabel('Valor', color='white')
    axc.set_ylabel('Frecuencia', color='white')
    axc.legend()
    axc.tick_params(colors='white')
    axc.set_facecolor('#1a1a1a')
    fig_central.patch.set_facecolor('#0d0d0d')

    # --- Gráfico 2: Dispersión (barra) + visual Q1-Q3 ---
    fig_disp, (axd1, axd2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1]})
    # Barras para Varianza, Desv.Est., CV%
    vals = [stats_dict['Varianza'], stats_dict['Desv. Estándar'], stats_dict['Coef. Variación (%)']]
    labels = ['Varianza', 'Desv. Estándar', 'Coef. Var (%)']
    colors = ['#a00000', '#ff7f50', '#ffd700']
    axd1.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        try:
            axd1.text(i, v, f'{v:.2f}', ha='center', va='bottom', color='white')
        except:
            axd1.text(i, 0, 'NaN', ha='center', va='bottom', color='white')
    axd1.set_title('Medidas de Dispersión', color='white', fontsize=12, fontweight='bold')
    axd1.tick_params(colors='white')
    axd1.set_facecolor('#1a1a1a')

    # Visualización de cuartiles como banda
    q1 = stats_dict['Q1 (25%)']
    q2 = stats_dict['Q2 (50%)']
    q3 = stats_dict['Q3 (75%)']
    axd2.hlines(1, q1, q3, lw=18, color='#8B0000', alpha=0.7)
    # Dibujar puntos individuales con marcadores distintos (no pasar una lista a `marker`)
    for x, mk in zip([q1, q2, q3], ['o', 'D', 'o']):
        axd2.plot(x, 1, marker=mk, color='white', markersize=8, linestyle='')
    axd2.text(q1, 1.05, f'Q1: {q1:.2f}', ha='center', color='white')
    axd2.text(q2, 1.05, f'Q2: {q2:.2f}', ha='center', color='white')
    axd2.text(q3, 1.05, f'Q3: {q3:.2f}', ha='center', color='white')
    axd2.set_ylim(0.8, 1.3)
    axd2.set_yticks([])
    axd2.set_title('Cuartiles (Q1 - Q3)', color='white', fontsize=12, fontweight='bold')
    axd2.tick_params(colors='white')
    axd2.set_facecolor('#1a1a1a')
    fig_disp.patch.set_facecolor('#0d0d0d')

    # --- Gráfico 3: Forma (asimetrías y curtosis) ---
    fig_shape, axs = plt.subplots(figsize=(8, 4))
    shape_keys = ['Asimetría Fisher', 'Asimetría Pearson', 'Asimetría Bowley', 'Curtosis']
    shape_vals = [stats_dict.get(k, np.nan) for k in shape_keys]
    bar_colors = []
    for k, v in zip(shape_keys, shape_vals):
        if 'Curtosis' in k:
            bar_colors.append('#ffd700' if abs(v) < 0.5 else ('#ff4500' if v > 0.5 else '#1e90ff'))
        else:
            bar_colors.append('#00ff00' if abs(v) < 0.5 else ('#ff4500' if v > 0 else '#1e90ff'))
    axs.bar(shape_keys, shape_vals, color=bar_colors)
    axs.axhline(0, color='white', linewidth=1)
    for i, v in enumerate(shape_vals):
        try:
            axs.text(i, v + (0.02 * np.sign(v) + 0.02), f'{v:.2f}', ha='center', color='white')
        except:
            axs.text(i, 0, 'NaN', ha='center', color='white')
    axs.set_title('Medidas de Forma y Curtosis', color='white', fontsize=12, fontweight='bold')
    axs.tick_params(colors='white')
    axs.set_facecolor('#1a1a1a')
    fig_shape.patch.set_facecolor('#0d0d0d')

    plt.tight_layout()
    return fig_central, fig_disp, fig_shape

# `st.number_input` es una función de Streamlit para crear un campo de entrada numérica.
# Úsala directamente donde necesites un input numérico, por ejemplo:
# `valor = st.number_input('Valor', min_value=0, max_value=100, value=1)`
def main():
    # initial_sidebar_state es una funcion que permite ocultar la barra lateral
    # collapsed es una funcion que permite ocultar la barra lateral
    # expanded es una funcion que permite mostrar la barra lateral
    st.set_page_config(page_title="Ejemplo de estadistica II",page_icon="👓", layout="wide",initial_sidebar_state="expanded")    
    # Estilos personalizados: degradé rojo -> negro y ajustes de colores
    st.markdown(
        """
        <style>
        /* Fondo general de la app */
        .stApp {
            background: linear-gradient(180deg, #8B0000 0%, #000000 100%);
            color: #ffffff;
        }
        /* Sidebar (navegación) */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #5a0000 0%, #000000 100%);
            color: #ffffff;
        }
        /* Texto y títulos */
        .css-1d391kg, .css-10trblm, .stMarkdown, h1, h2, h3, h4, .stText {
            color: #fff !important;
        }
        /* Botones y controles */
        button[kind="primary"], .stButton>button {
            background: linear-gradient(180deg, #a00000, #200000) !important;
            color: #fff !important;
            border: none !important;
        }
        /* DataFrame ligero sobre fondo oscuro */
        .stDataFrame table { background: rgba(255,255,255,0.02); color: #fff; }
        /* Ajustes para métricas */
        .css-1hynsf4 .css-10trblm { color: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # =========================================================================
    # 🛠️ PANEL IZQUIERDO (SIDEBAR): CONTROL TOTAL DE INGESTACIÓN Y MENÚS
    # =========================================================================
    # Uploader ubicado en el cuerpo principal (más visible)
    archivo_cargado = st.file_uploader("📂 Cargar archivo de datos (.csv):", type=["csv"]) 
    with st.sidebar:
        try:
            logo = Image.open("images.jpg")
            st.image(logo, width=120, caption="Universidad Amazónica de Pando")
        except FileNotFoundError:
            st.image("https://cdn-icons-png.flaticon.com/512/3112/3112946.png", width=60)
        st.title("Control General")
        st.markdown("---")
        # inicializamos variables para el menú de navegación (valores por defecto)
        modulo_principal = "🏠 Panel de Datos Base"
        variable_seleccionada = None
        opcion_grafico = None
        selected_simple = None
        selected_agrupada = None
        tipo_procesamiento = None
       
        # El menú de navegación solo se dibuja si el archivo ya fue cargado con éxito
        df_est = None
    if archivo_cargado is not None:
        df_est = load_csv(archivo_cargado)
        if df_est is None:
            st.error("No se pudo leer el archivo CSV. Verifica que el archivo esté en formato CSV válido.")
            return
        #define las columnas cuantitativas y cualitativas para el menú de navegación
        columnas_cuantitativas = list(df_est.select_dtypes(include=[np.number]).columns)
        columnas_cualitativas = list(df_est.select_dtypes(include=['object', 'category']).columns)
        #Filtra variables cuantitativas tabla de frecuencia simple o agruapa
        columnas_simples=[]
        columnas_agrupadas=[]
        for col in columnas_cuantitativas:
            #st.write(f"Variable: {col} - Unicos: {df_est[col].nunique()} - Tipo: {df_est[col].dtype}")
            if df_est[col].nunique() < 20 and not df_est[col].dtype == 'int64':
                
                columnas_simples.append(col)
            else:
                columnas_agrupadas.append(col)
        # =========================================================================
        # ⚙️ CONSTRUCCIÓN DEL MENÚ INTERACTIVO EN EL SIDEBAR
        # =========================================================================
        with st.sidebar:
            modulo_principal = st.radio(
            "Selecciona la fase de análisis:",
            ["🏠 Panel de Datos Base", "📊 Variables Cualitativas", "📈 Variables Cuantitativas"])
            st.markdown("---")
            # LÓGICA DEL SUB-MENÚ PARA CUALITATIVAS
            if "Cualitativas" in modulo_principal:
                if len(columnas_cualitativas) > 0:
                    variable_seleccionada = st.selectbox("🔤 Variable Cualitativa:", columnas_cualitativas)
                    opcion_grafico = st.selectbox("🎨 Formato de Gráfico:", ["Gráfico de Barras", "Gráfico de Torta"])
                    tipo_procesamiento = "cualitativa"
                else:
                    st.warning("No se detectaron variables cualitativas.")
                
            # LÓGICA DEL SUB-MENÚ PARA CUANTITATIVAS (Tu regla exacta)
            elif "Cuantitativas" in modulo_principal:
                if len(columnas_simples) > 0 and len(columnas_agrupadas) > 0:
                    menu_cuantitativas = st.radio(
                        "Elige la variable cuantitativa a analizar",
                        ["Datos Simples (Discretos < 20 únicos)", "Datos Agrupados (Continuos / Rango Amplio)"]
                    )
                    if menu_cuantitativas.startswith("Datos Simples"):
                        selected_simple = st.selectbox("🔢 Datos Simples (Discretos < 20 únicos):", columnas_simples)
                        tipo_procesamiento = "cuant_simple"
                    else:
                        selected_agrupada = st.selectbox("📐 Datos Agrupados (Continuos / Rango Amplio):", columnas_agrupadas)
                        tipo_procesamiento = "cuant_agrupada"
                elif len(columnas_simples) > 0:
                    selected_simple = st.selectbox("🔢 Datos Simples (Discretos < 20 únicos):", columnas_simples)
                    tipo_procesamiento = "cuant_simple"
                elif len(columnas_agrupadas) > 0:
                    selected_agrupada = st.selectbox("📐 Datos Agrupados (Continuos / Rango Amplio):", columnas_agrupadas)
                    tipo_procesamiento = "cuant_agrupada"
                else:
                    st.warning("No se detectaron variables cuantitativas.")              
            st.caption("Estudiante: JORGE CHURA FERNANDEZ | RU:43417 ")
# =========================================================================
# 🖥️ CONTENIDO CENTRAL: REACCIÓN DINÁMICA A LOS DATOS
# =========================================================================
    st.title("📈 Plataforma Automatizada de Procesamiento Estadístico",text_alignment="center")
    st.markdown("### *Universidad Amazónica de Pando - Área de Tecnologías - Ingenieria de Sistemas*",text_alignment="center")
    if archivo_cargado is not None:
    #============================================================================
    # Conteo de columnas cuantitativas y cualitativas    
    # select_dtypes incluye np.number solo columnas numéricas (enteros y decimales)
        columnas_cuantitativas = df_est.select_dtypes(include=[np.number]).columns
        cant_cuantitativas = len(columnas_cuantitativas)
    
    # select_dtypes con 'object' o 'category' para identificar columnas de texto
        columnas_cualitativas = df_est.select_dtypes(include=['object', 'category']).columns
        cant_cualitativas = len(columnas_cualitativas)
    #============================================================================
        
        st.header("Vista Previa de los Datos Cargados")
        col1,col2,col3,col4=st.columns(4)
        col1.metric("Número de registros", len(df_est))
        col2.metric("Total de columnas detectadas", len(df_est.columns))
        col3.metric("Total Columnas cualitativas", cant_cualitativas)
        col4.metric("Total Columnas cuantitativas", cant_cuantitativas)
        st.dataframe(df_est, use_container_width=True)


    #carga el dataframe
    
    #st.dataframe(df_est,use_container_width=True)  
    #explicar sample para quesirve al programador
    #sample es una funcion que permite seleccionar una muestra aleatoria de un dataframe
    #n es el tamaño de la muestra
    #random_state es el estado de la semilla aleatoria
    #que significa semila aleatoria

        st.header("Análisis de la variable seleccionada")
        if df_est is None:
            st.warning("No hay datos cargados. Sube un archivo CSV para ver los análisis.")
            return

        # Mostrar resultado según tipo de procesamiento elegido en el sidebar
        #locals() devuelve un diccionario con las variables locales de la función main(),
        #esto permite verificar si las variables tipo_procesamiento, variable_seleccionada, 
        # opcion_grafico, selected_simple y selected_agrupada existen antes de usarlas
        if 'tipo_procesamiento' in locals() and tipo_procesamiento == 'cualitativa' and variable_seleccionada:
            #astype(str) convierte la serie a tipo string para evitar problemas con variables numéricas que se tratan como cualitativas
            serie = df_est[variable_seleccionada].astype(str)
            frec = freq_table_simple(serie)
            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader(f'Tabla de frecuencias - {variable_seleccionada}')
                st.dataframe(frec, use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(6,4))
                if opcion_grafico and "Barras" in opcion_grafico:
                    ax.bar(frec[variable_seleccionada], frec['frecuencia'], color=sns.color_palette("pastel"))
                    ax.set_xlabel(variable_seleccionada)
                    ax.set_ylabel('Frecuencia')
                    ax.set_title(f'Gráfico de Barras - {variable_seleccionada}')
                    for i,v in enumerate(frec['frecuencia']):
                        ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                else:
                    ax.pie(frec['porcentaje'], labels=frec[variable_seleccionada], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                    ax.set_title(f'Gráfico de Pastel - {variable_seleccionada}')
                st.pyplot(fig)

        elif 'tipo_procesamiento' in locals() and tipo_procesamiento == 'cuant_simple' and selected_simple:
            #pd.to_numeric convierte la serie a tipo numérico, errors='coerce' 
            # errors='coerce' convierte los valores no convertibles en NaN
            serie = pd.to_numeric(df_est[selected_simple], errors='coerce')
            frec = freq_table_simple(serie)
            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader(f'Tabla de frecuencias - {selected_simple}')
                st.dataframe(frec, use_container_width=True)
            with col2:
                fig = plot_baston(serie)
                st.pyplot(fig)
            
            # Mostrar estadísticas descriptivas con visualizaciones
            st.subheader("📊 Estadísticas Descriptivas")
            stats_result = compute_descriptive_stats(serie)
            if stats_result:
                # Crear tabs para organizar las estadísticas
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Gráficos", "📋 Medidas Centrales", "📐 Dispersión", "🔄 Forma"])
                
                with tab1:
                    # Mostrar los gráficos
                    fig1, fig2, fig3 = plot_descriptive_stats(serie, stats_result)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig1)
                    with col2:
                        st.pyplot(fig2)
                    st.pyplot(fig3)
                
                with tab2:
                    # Medidas de tendencia central
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📍 Media", f"{stats_result['Media']:.4f}", delta=None)
                        st.info("Promedio aritmético de todos los valores", icon="ℹ️")
                        st.info(f"Interpretación: La media es {stats_result['Media']:.4f}. Representa el valor promedio de la variable; compárala con la desviación estándar para juzgar la dispersión relativa.")
                    with col2:
                        st.metric("📊 Mediana", f"{stats_result['Mediana']:.4f}", delta=None)
                        st.info("Valor central que divide los datos en dos mitades", icon="ℹ️")
                        st.info(f"Interpretación: La mediana es {stats_result['Mediana']:.4f}. Si la mediana difiere mucho de la media, hay asimetría o valores extremos.")
                    with col3:
                        st.metric("🎯 Moda", f"{stats_result['Moda']:.4f}", delta=None)
                        st.info("Valor más frecuente en los datos", icon="ℹ️")
                        st.info(f"Interpretación: La moda es {stats_result['Moda']:.4f}. Indica el valor que aparece con mayor frecuencia; puede ser útil para variables discretas.")
                
                with tab3:
                    # Medidas de dispersión
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Varianza", f"{stats_result['Varianza']:.4f}", delta=None)
                        st.info("Promedio de las desviaciones al cuadrado", icon="ℹ️")
                        st.info(f"Interpretación: La varianza es {stats_result['Varianza']:.4f}. Mide la dispersión en unidades al cuadrado; su raíz es la desviación estándar.")
                    with col2:
                        st.metric("📊 Desv. Estándar", f"{stats_result['Desv. Estándar']:.4f}", delta=None)
                        st.info("Raíz cuadrada de la varianza; mide dispersión", icon="ℹ️")
                        st.info(f"Interpretación: La desviación estándar es {stats_result['Desv. Estándar']:.4f}. Indica, en promedio, cuánto se alejan los datos de la media (mismas unidades que la variable).")
                    with col3:
                        st.metric("📈 Coef. Variación (%)", f"{stats_result['Coef. Variación (%)']:.4f}", delta=None)
                        st.info("Desv.Est. relativa a la media (variabilidad %)", icon="ℹ️")
                        st.info(f"Interpretación: El coeficiente de variación es {stats_result['Coef. Variación (%)']:.4f}%. Valores mayores indican mayor variabilidad relativa respecto a la media.")
                    
                    # Cuartiles
                    st.markdown("**Cuartiles (Percentiles):**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Q1 (25%)", f"{stats_result['Q1 (25%)']:.4f}")
                        st.caption("25% de datos por debajo")
                        st.info(f"Interpretación: Q1 = {stats_result['Q1 (25%)']:.4f}. 25% de los valores son menores o iguales a este número.")
                    with col2:
                        st.metric("Q2 (50%)", f"{stats_result['Q2 (50%)']:.4f}")
                        st.caption("50% de datos por debajo (Mediana)")
                        st.info(f"Interpretación: Q2 (mediana) = {stats_result['Q2 (50%)']:.4f}. Divide la muestra en dos mitades.")
                    with col3:
                        st.metric("Q3 (75%)", f"{stats_result['Q3 (75%)']:.4f}")
                        st.caption("75% de datos por debajo")
                        st.info(f"Interpretación: Q3 = {stats_result['Q3 (75%)']:.4f}. 75% de los valores son menores o iguales a este número.")
                
                with tab4:
                    # Medidas de forma
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("⚖️ Asimetría Fisher", f"{stats_result['Asimetría Fisher']:.4f}", delta=None)
                        if abs(stats_result['Asimetría Fisher']) < 0.5:
                            st.success("✓ Distribución simétrica")
                        elif stats_result['Asimetría Fisher'] > 0:
                            st.warning("⚠ Asimétrica a la derecha (cola derecha)")
                        else:
                            st.warning("⚠ Asimétrica a la izquierda (cola izquierda)")
                        st.info(f"Interpretación: Asimetría Fisher = {stats_result['Asimetría Fisher']:.4f}. Valores próximos a 0 indican simetría; positivos indican sesgo a la derecha.")
                    
                    with col2:
                        st.metric("⚖️ Asimetría Pearson", f"{stats_result['Asimetría Pearson']:.4f}", delta=None)
                        st.info("Método alternativo usando media, mediana y desv.est.", icon="ℹ️")
                        st.info(f"Interpretación: Asimetría Pearson = {stats_result['Asimetría Pearson']:.4f}. Similar a Fisher, útil como comparación rápida.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🔽 Asimetría Bowley", f"{stats_result['Asimetría Bowley']:.4f}", delta=None)
                        st.info("Método basado en cuartiles", icon="ℹ️")
                        st.info(f"Interpretación: Asimetría Bowley = {stats_result['Asimetría Bowley']:.4f}. Valores positivos indican sesgo a la derecha; negativos, a la izquierda.")
                    
                    with col2:
                        st.metric("📊 Curtosis", f"{stats_result['Curtosis']:.4f}", delta=None)
                        if abs(stats_result['Curtosis']) < 0.5:
                            st.info("✓ Distribución mesocúrtica (normal)", icon="ℹ️")
                        elif stats_result['Curtosis'] > 0.5:
                            st.info("⚠ Distribución leptocúrtica (colas pesadas)", icon="ℹ️")
                        else:
                            st.info("⚠ Distribución platicúrtica (colas ligeras)", icon="ℹ️")

        elif 'tipo_procesamiento' in locals() and tipo_procesamiento == 'cuant_agrupada' and selected_agrupada:
            serie = pd.to_numeric(df_est[selected_agrupada], errors='coerce')
            df_freq = freq_table_agrupada(serie)
            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader(f'Tabla de frecuencias agrupadas - {selected_agrupada}')
                st.dataframe(df_freq, use_container_width=True)
            with col2:
                fig = plot_histogram_polygon(serie)
                st.pyplot(fig)
            
            # Mostrar estadísticas descriptivas con visualizaciones
            st.subheader("📊 Estadísticas Descriptivas")
            stats_result = compute_descriptive_stats(serie)
            if stats_result:
                # Crear tabs para organizar las estadísticas
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Gráficos", "📋 Medidas Centrales", "📐 Dispersión", "🔄 Forma"])
                
                with tab1:
                    # Mostrar los gráficos
                    fig1, fig2, fig3 = plot_descriptive_stats(serie, stats_result)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig1)
                    with col2:
                        st.pyplot(fig2)
                    st.pyplot(fig3)
                
                with tab2:
                    # Medidas de tendencia central
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📍 Media", f"{stats_result['Media']:.4f}", delta=None)
                        st.info("Promedio aritmético de todos los valores", icon="ℹ️")
                        st.info(f"Interpretación: La media es {stats_result['Media']:.4f}. Representa el valor promedio de la variable; compárala con la desviación estándar para juzgar la dispersión relativa.")
                    with col2:
                        st.metric("📊 Mediana", f"{stats_result['Mediana']:.4f}", delta=None)
                        st.info("Valor central que divide los datos en dos mitades", icon="ℹ️")
                        st.info(f"Interpretación: La mediana es {stats_result['Mediana']:.4f}. Si la mediana difiere mucho de la media, hay asimetría o valores extremos.")
                    with col3:
                        st.metric("🎯 Moda", f"{stats_result['Moda']:.4f}", delta=None)
                        st.info("Valor más frecuente en los datos", icon="ℹ️")
                        st.info(f"Interpretación: La moda es {stats_result['Moda']:.4f}. Indica el valor que aparece con mayor frecuencia; puede ser útil para variables discretas.")
                
                with tab3:
                    # Medidas de dispersión
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Varianza", f"{stats_result['Varianza']:.4f}", delta=None)
                        st.info("Promedio de las desviaciones al cuadrado", icon="ℹ️")
                        st.info(f"Interpretación: La varianza es {stats_result['Varianza']:.4f}. Mide la dispersión en unidades al cuadrado; su raíz es la desviación estándar.")
                    with col2:
                        st.metric("📊 Desv. Estándar", f"{stats_result['Desv. Estándar']:.4f}", delta=None)
                        st.info("Raíz cuadrada de la varianza; mide dispersión", icon="ℹ️")
                        st.info(f"Interpretación: La desviación estándar es {stats_result['Desv. Estándar']:.4f}. Indica, en promedio, cuánto se alejan los datos de la media (mismas unidades que la variable).")
                    with col3:
                        st.metric("📈 Coef. Variación (%)", f"{stats_result['Coef. Variación (%)']:.4f}", delta=None)
                        st.info("Desv.Est. relativa a la media (variabilidad %)", icon="ℹ️")
                        st.info(f"Interpretación: El coeficiente de variación es {stats_result['Coef. Variación (%)']:.4f}%. Valores mayores indican mayor variabilidad relativa respecto a la media.")
                    
                    # Cuartiles
                    st.markdown("**Cuartiles (Percentiles):**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Q1 (25%)", f"{stats_result['Q1 (25%)']:.4f}")
                        st.caption("25% de datos por debajo")
                        st.info(f"Interpretación: Q1 = {stats_result['Q1 (25%)']:.4f}. 25% de los valores son menores o iguales a este número.")
                    with col2:
                        st.metric("Q2 (50%)", f"{stats_result['Q2 (50%)']:.4f}")
                        st.caption("50% de datos por debajo (Mediana)")
                        st.info(f"Interpretación: Q2 (mediana) = {stats_result['Q2 (50%)']:.4f}. Divide la muestra en dos mitades.")
                    with col3:
                        st.metric("Q3 (75%)", f"{stats_result['Q3 (75%)']:.4f}")
                        st.caption("75% de datos por debajo")
                        st.info(f"Interpretación: Q3 = {stats_result['Q3 (75%)']:.4f}. 75% de los valores son menores o iguales a este número.")
                
                with tab4:
                    # Medidas de forma
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("⚖️ Asimetría Fisher", f"{stats_result['Asimetría Fisher']:.4f}", delta=None)
                        if abs(stats_result['Asimetría Fisher']) < 0.5:
                            st.success("✓ Distribución simétrica")
                        elif stats_result['Asimetría Fisher'] > 0:
                            st.warning("⚠ Asimétrica a la derecha (cola derecha)")
                        else:
                            st.warning("⚠ Asimétrica a la izquierda (cola izquierda)")
                    
                    with col2:
                        st.metric("⚖️ Asimetría Pearson", f"{stats_result['Asimetría Pearson']:.4f}", delta=None)
                        st.info("Método alternativo usando media, mediana y desv.est.", icon="ℹ️")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🔽 Asimetría Bowley", f"{stats_result['Asimetría Bowley']:.4f}", delta=None)
                        st.info("Método basado en cuartiles", icon="ℹ️")
                    
                    with col2:
                        st.metric("📊 Curtosis", f"{stats_result['Curtosis']:.4f}", delta=None)
                        if abs(stats_result['Curtosis']) < 0.5:
                            st.info("✓ Distribución mesocúrtica (normal)", icon="ℹ️")
                        elif stats_result['Curtosis'] > 0.5:
                            st.info("⚠ Distribución leptocúrtica (colas pesadas)", icon="ℹ️")
                        else:
                            st.info("⚠ Distribución platicúrtica (colas ligeras)", icon="ℹ️")

        else:
            st.info("Selecciona una variable en el menú lateral para ver la tabla de frecuencias y el gráfico correspondiente.")
    else:
        st.warning("Por favor, carga un archivo CSV para visualizar los datos y gráficos.")
         
if  __name__ == "__main__":
    main()