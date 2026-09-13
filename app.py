import streamlit as st
from google import genai
from pypdf import PdfReader

# Configuración de la página
st.set_page_config(page_title="BioMedical Research AI", layout="wide", page_icon="🔬")

# ==========================================
# SIDEBAR - CONFIGURACIÓN DE API
# ==========================================
st.sidebar.title("🛠️ Configuración de API")
gemini_key = st.sidebar.text_input("Ingresa tu Gemini (Google) API Key:", type="password")

if not gemini_key:
    st.info("👈 Ingresa tu API Key de Google Gemini en el panel izquierdo para comenzar.")
    st.stop()

# Inicialización del cliente de Gemini
try:
    g_client = genai.Client(api_key=gemini_key)
except Exception as e:
    st.error(f"Error al inicializar Gemini: {e}")
    st.stop()

# Usamos el modelo recomendado para tareas de razonamiento avanzado y lectura amplia
MODEL_GEMINI = "gemini-2.5-flash" 

# Memoria compartida para los PDFs
if "notebook_context" not in st.session_state:
    st.session_state["notebook_context"] = ""

tab1, tab2, tab3 = st.tabs([
    "📄 Evaluador Crítico (Múltiples PDFs)", 
    "🔍 Búsqueda PubMed (MeSH)", 
    "✍️ Redactor Científico"
])

# ==========================================
# MÓDULO 1: EVALUADOR CRÍTICO (GEMINI)
# ==========================================
with tab1:
    st.header("Módulo 1: Lectura Crítica (Gemini)")
    
    uploaded_files = st.file_uploader(
        "Cargue los artículos científicos en PDF", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Se han cargado {len(uploaded_files)} archivo(s) PDF.")
        
        if st.button("Procesar Archivos y Guardar en Contexto"):
            st.session_state["notebook_context"] = ""
            
            for i, uploaded_file in enumerate(uploaded_files, 1):
                st.subheader(f"📄 Artículo {i}: {uploaded_file.name}")
                
                with st.spinner(f"Analizando {uploaded_file.name} con Gemini..."):
                    try:
                        reader = PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in reader.pages:
                            pdf_text += page.extract_text() or ""
                        
                        st.session_state["notebook_context"] += f"\n\n--- DOCUMENTO {i}: {uploaded_file.name} ---\n" + pdf_text
                        
                        prompt = f"""
                        Eres un epidemiólogo experto y revisor de literatura médica. Analiza el siguiente artículo '{uploaded_file.name}':

                        1. Clasifica el diseño de estudio.
                        2. Aplica el checklist correspondiente (JBI, STROBE, CARE, CONSORT, etc.) indicando si cumple (Sí/No/No Claro) con citas breves del texto.
                        3. Evalúa los dominios de Sesgo (Selección, Medición, Confusión, Reporte).
                        4. Dictamen Final (Fortalezas, Debilidades y Relevancia Clínica).

                        TEXTO DEL ARTÍCULO:
                        {pdf_text[:150000]}
                        """
                        
                        res = g_client.models.generate_content(
                            model=MODEL_GEMINI,
                            contents=prompt
                        )
                        st.markdown(res.text)
                    except Exception as e:
                        st.error(f"Error al procesar {uploaded_file.name}: {str(e)}")
                st.divider()

# ==========================================
# MÓDULO 2: BÚSQUEDA EN PUBMED
# ==========================================
with tab2:
    st.header("Módulo 2: Búsqueda Autónoma PubMed")
    user_query = st.text_input("Idea o tema de investigación:", "Hospitalizaciones prevenibles en adultos mayores en Chile")
    
    if st.button("Generar Ecuación Booleana"):
        with st.spinner("Construyendo sintaxis MeSH..."):
            prompt_mesh = f"Actúa como bibliotecólogo biomédico experto. Genera una estrategia de búsqueda completa para PubMed sobre: '{user_query}' con términos MeSH, texto libre (tiab) y ecuación booleana con AND/OR."
            try:
                res_mesh = g_client.models.generate_content(
                    model=MODEL_GEMINI,
                    contents=prompt_mesh
                )
                st.markdown(res_mesh.text)
            except Exception as e:
                st.error(f"Error al generar ecuación: {str(e)}")

# ==========================================
# MÓDULO 3: REDACCIÓN CIENTÍFICA (GEMINI)
# ==========================================
with tab3:
    st.header("Módulo 3: Redactor Basado en los PDFs Cargados")
    
    if st.session_state["notebook_context"]:
        st.info("📚 Cuaderno Activo: Se utilizarán los PDFs procesados en el Módulo 1 como fuente exclusiva.")
    else:
        st.warning("⚠️ No has procesado PDFs en el Módulo 1.")
        
    instruccion_redaccion = st.text_area(
        "¿Qué deseas redactar?", 
        "Escribe la sección de Introducción/Justificación respondiendo a la evidencia de los PDFs cargados."
    )
    
    if st.button("Redactar Sección con Gemini"):
        if not st.session_state["notebook_context"]:
            st.error("No hay contexto de PDFs guardado. Procesa primero tus artículos en el Módulo 1.")
        else:
            with st.spinner("Sintetizando información..."):
                prompt_redactor = f"""
                Actúa como un investigador médico sénior.
                Utiliza ÚNICAMENTE la siguiente BASE DE CONOCIMIENTO (extraída de los PDFs) para redactar el texto solicitado.
                Incluye citas tipo Vancouver [1], [2].

                BASE DE CONOCIMIENTO:
                {st.session_state['notebook_context']}

                SOLICITUD:
                {instruccion_redaccion}
                """
                try:
                    res_redactor = g_client.models.generate_content(
                        model=MODEL_GEMINI,
                        contents=prompt_redactor
                    )
                    st.markdown(res_redactor.text)
                except Exception as e:
                    st.error(f"Error con Gemini API: {str(e)}")
