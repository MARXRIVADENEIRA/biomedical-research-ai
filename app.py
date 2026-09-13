import streamlit as st
import anthropic
from pypdf import PdfReader

# Configuración de página
st.set_page_config(page_title="BioMedical Research AI", layout="wide", page_icon="🔬")

# Sidebar - Configuración de API Key
st.sidebar.title("🛠️ Configuración")
api_key = st.sidebar.text_input("Ingrese su Claude (Anthropic) API Key:", type="password")

if not api_key:
    st.info("👈 Por favor, ingrese su API Key de Anthropic en el panel izquierdo para comenzar.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# Pestañas principales de la App
tab1, tab2, tab3 = st.tabs(["📄 Evaluador Crítico (Múltiples PDFs)", "🔍 Búsqueda PubMed (MeSH)", "✍️ Redactor Biomédico"])

# ==========================================
# MÓDULO 1: EVALUADOR CRÍTICO DE MÚLTIPLES PDF
# ==========================================
with tab1:
    st.header("Módulo 1: Lectura Crítica y Evaluación de Sesgos (Multi-Archivo)")
    
    # Habilitado para subir múltiples archivos a la vez
    uploaded_files = st.file_uploader(
        "Cargue los artículos científicos en formato PDF (puede seleccionar varios a la vez)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Se han cargado {len(uploaded_files)} archivo(s) PDF correctamente.")
        
        if st.button("Ejecutar Evaluación Metodológica Multi-Archivo"):
            for i, uploaded_file in enumerate(uploaded_files, 1):
                st.subheader(f"📄 Artículo {i}: {uploaded_file.name}")
                
                with st.spinner(f"Procesando y analizando {uploaded_file.name}..."):
                    reader = PdfReader(uploaded_file)
                    pdf_text = ""
                    for page in reader.pages:
                        pdf_text += page.extract_text() or ""
                    
                    prompt = f"""
                    Eres un epidemiólogo y metodólogo experto en lectura crítica.
                    Analiza el siguiente texto extraído del archivo '{uploaded_file.name}' y genera una evaluación rigurosa:

                    1. Clasifica el diseño de estudio (Reporte de Caso, Serie de Casos, Cohorte, Ecológico, Ensayo Clínico, etc.).
                    2. Dependiendo del diseño, aplica el checklist correspondiente (JBI 8 ítems para Reporte de Caso, JBI 10 ítems para Serie de Casos, STROBE para analíticos, CARE para reportes, MInCir o CASPe). Muestra cada ítem y evalúa si cumple (Sí / No / No Claro / No Aplica) extrayendo la cita/evidencia textual breve del PDF que lo justifica.
                    3. Aplica los 4 dominios de Sesgo según Hassan y Wu (Selección, Diagnóstico/Verificación, Causación, Reporte).
                    4. Emite un Dictamen Final: Puntos fuertes, debilidades metodológicas y si es recomendable usarlo en una investigación.

                    TEXTO DEL ARTÍCULO:
                    {pdf_text[:15000]}
                    """
                    
                    response = client.messages.create(
                        model="claude-3-5-sonnet-latest",
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    st.markdown(response.content[0].text)
                
                st.divider()

# ==========================================
# MÓDULO 2: BÚSQUEDA AVANZADA EN PUBMED
# ==========================================
with tab2:
    st.header("Módulo 2: Estrategia de Búsqueda Autónoma (PubMed)")
    user_query = st.text_input("Ingrese la idea o tema de investigación:", "Hospitalizaciones prevenibles en adultos mayores en Chile")
    
    if st.button("Generar Ecuación Booleana MeSH"):
        with st.spinner("Traduciendo a términos MeSH y construyendo operadores booleanos..."):
            prompt_mesh = f"""
            Toma la siguiente consulta de investigación médica: "{user_query}"
            1. Genera los términos estandarizados MeSH (Medical Subject Headings) y DeCS equivalentes.
            2. Construye una sintaxis de búsqueda avanzada para PubMed usando operadores booleanos (AND, OR, NOT) y términos MeSH/textuales.
            3. Devuelve ÚNICAMENTE la cadena de búsqueda en un bloque de código.
            """
            
            response_mesh = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt_mesh}]
            )
            
            st.markdown(response_mesh.content[0].text)

# ==========================================
# MÓDULO 3: REDACCIÓN Y CITAS VANCOUVER
# ==========================================
with tab3:
    st.header("Módulo 3: Asistente de Redacción Biomédica")
    instruccion_redaccion = st.text_area("¿Qué sección o texto necesitas redactar?", "Escribe la introducción para un estudio ecológico longitudinal sobre hospitalizaciones evitables y desigualdad en Chile.")
    
    if st.button("Redactar con Estilo Académico"):
        with st.spinner("Redactando texto con estructura en embudo e insertando citas estilo Vancouver/Medline..."):
            prompt_redaccion = f"""
            Actúa como un investigador médico sénior. Redacta el siguiente apartado con rigor académico de posgrado:
            "{instruccion_redaccion}"
            
            - Utiliza una estructura clara de párrafos (formato embudo para Introducción).
            - Coloca marcadores de citación tipo Vancouver [1], [2] donde sea conceptualmente necesario respaldar la afirmación.
            - Al final, agrega una sección de bibliografía de ejemplo ajustada al formato Medline/Vancouver.
            """
            
            response_redaccion = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt_redaccion}]
            )
            
            st.markdown(response_redaccion.content[0].text)
