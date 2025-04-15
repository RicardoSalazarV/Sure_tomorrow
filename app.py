# app.py
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

# Configuración general de la página
st.set_page_config(page_title="Sure Tomorrow", layout="wide")
st.title("🛡️ Sure Tomorrow - Explorador de Datos de Seguros")

# Función robusta para cargar datos
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Datos cargados desde el archivo subido por el usuario.")
        else:
            # Rutas posibles
            possible_paths = [
                "datasets/insurance_us.csv",
                "insurance_us.csv",
                "./insurance_us.csv",
                os.path.join("datasets", "insurance_us.csv")
            ]

            # Ruta local para desarrollo (ajústala a tu entorno si necesitas)
            if os.path.exists("C:\\Users\\ricar\\Desktop\\insurance_project\\datasets\\insurance_us.csv"):
                possible_paths.insert(0, "C:\\Users\\ricar\\Desktop\\insurance_project\\datasets\\insurance_us.csv")

            df = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.success(f"✅ Datos cargados correctamente desde: {path}")
                    break
                except FileNotFoundError:
                    continue

            if df is None:
                st.info("⚠️ No se encontró el archivo de datos. Por favor, carga un archivo CSV.")
                return pd.DataFrame()

        # Renombrar columnas para consistencia
        df = df.rename(columns={
            'Gender': 'gender',
            'Age': 'age',
            'Salary': 'income',
            'Family members': 'family_members',
            'Insurance benefits': 'insurance_benefits'
        })

        return df

    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return pd.DataFrame()

# --- Carga de datos ---
st.sidebar.title("Carga de datos")
uploaded_file = st.sidebar.file_uploader("📂 Cargar archivo CSV personalizado", type=["csv"])
df = load_data(uploaded_file)

# --- Interfaz principal ---
if df.empty:
    st.warning("Por favor, carga un archivo de datos válido para continuar.")
else:
    st.sidebar.title("Opciones de análisis")
    show_data = st.sidebar.checkbox("🔍 Mostrar datos")
    show_stats = st.sidebar.checkbox("📈 Estadísticas")
    show_plot = st.sidebar.checkbox("🎨 Visualización")
    show_model = st.sidebar.checkbox("🤖 Modelo interactivo")

    if show_data:
        st.subheader("🔍 Vista previa del dataset")
        st.dataframe(df.head(10))

    if show_stats:
        st.subheader("📈 Estadísticas descriptivas")
        st.write(df.describe())

    if show_plot:
        st.subheader("🎨 Matriz de correlación")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    if show_model:
        st.subheader("🤖 Simulador de usuario - Sugerencias similares")

        age = st.slider("Edad", int(df['age'].min()), int(df['age'].max()), 35)
        income = st.slider("Ingresos", int(df['income'].min()), int(df['income'].max()), 50000)
        family_members = st.slider("Miembros de familia", int(df['family_members'].min()), int(df['family_members'].max()), 3)

        user_input = pd.DataFrame([[age, income, family_members]], columns=['age', 'income', 'family_members'])

        scaler = MaxAbsScaler()
        features = df[['age', 'income', 'family_members']]
        features_scaled = scaler.fit_transform(features)

        model = NearestNeighbors(n_neighbors=5)
        model.fit(features_scaled)

        user_scaled = scaler.transform(user_input)
        distances, indices = model.kneighbors(user_scaled)

        st.write("Usuarios más similares:")
        st.dataframe(df.iloc[indices[0]])

    st.markdown("---")
    st.caption("Desarrollado por [Tu Nombre] • Proyecto de portafolio con Streamlit")
