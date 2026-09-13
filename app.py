import streamlit as st
import anthropic
from google import genai
from pypdf import PdfReader

# Configuración de página
st.set_page_config(page_title="BioMedical Research AI", layout="wide", page_icon="🔬")

# Sidebar - Configuración de claves API
st.sidebar.title("🛠️ Configuración de APIs")
anthropic_key = st.sidebar.text_input("1. Claude (Anthropic) API Key:", type="password")
gemini_key = st.sidebar.text_input("2. Gemini (Google) API Key (Opcional):", type="password")

if not anthropic_key:
    st.info("👈 Ingresa al menos tu API Key de Anthropic en el panel izquierdo para comenzar.")
    st.stop()

# Clientes
claude_client = anthropic.Anthropic(api_key=anthropic_key)
MODEL_CLAUDE = "claude-3-5-sonnet-20240620"

# Inicializar estado para guardar el texto de los PDFs cargados (Estilo NotebookLM)
if "notebook_context" not in st.session_state:
    st.session_state["notebook_context"] = ""

tab1, tab2, tab3 = st.tabs(["📄 Evaluador Crítico (Múltiples PDFs)", "🔍 Búsqueda PubMed (MeSH)", "✍️ Redactor (Estilo NotebookLM)"])

# ==========================================
# MÓDULO 1: EVALUADOR CRÍTICO Y LECTURA
# ==========================================
with tab1:
    st.header("Módulo 1: Lectura Crítica (Claude 3.5 Sonnet)")
    
    uploaded_files = st.file_uploader(
        "Cargue los artículos científicos en PDF (Se guardarán también en el cuaderno de redacción)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Se han cargado {len(uploaded_files)} archivo(s) PDF.")
        
        if st.button("Procesar Archivos y Guardar en Contexto"):
            st.session_state["notebook_context"] = "" # Limpiar contexto previo
            
            for i, uploaded_file in enumerate(uploaded_files, 1):
                st.subheader(f"📄 Artículo {i}: {uploaded_file.name}")
                
                with st.spinner(f"Leyendo y analizando {uploaded_file.name}..."):
                    try:
                        reader = PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in reader.pages:
                            pdf_text += page.extract_text() or ""
                        
                        # Guardar texto extraído en la memoria compartida de la App
                        st.session_state["notebook_context"] += f"\n\n--- DOCUMENTO {i}: {uploaded_file.name} ---\n" + pdf_text
                        
                        prompt = f"""
                        Eres un epidemiólogo experto. Analiza el artículo '{uploaded_file.name}':
                        1. Clasifica el diseño de estudio.
                        2. Aplica el checklist correspondiente (JBI 8/10 ítems, STROBE, CARE, etc.) indicando si cumple (Sí/No/No Claro) con la cita breve del PDF.
                        3. Evalúa los 4 dominios de Sesgo según Hassan y Wu.
                        4. Dictamen Final (Fortalezas y Debilidades).

                        TEXTO:
                        {pdf_text[:15000]}
                        """
                        
                        response = claude_client.messages.create(
                            model=MODEL_CLAUDE,
                            max_tokens=4000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(response.content[0].text)
                    except Exception as e:
                        st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
                st.divider()

# ==========================================
# MÓDULO 2: BÚSQUEDA EN PUBMED
# ==========================================
with tab2:
    st.header("Módulo 2: Búsqueda Autónoma PubMed")
    user_query = st.text_input("Idea de investigación:", "Hospitalizaciones prevenibles en adultos mayores en Chile")
    
    if st.button("Generar Ecuación Booleana"):
        with st.spinner("Construyendo sintaxis MeSH..."):
            prompt_mesh = f"Crea la estrategia MeSH y ecuación booleana para PubMed de: '{user_query}'"
            response_mesh = claude_client.messages.create(
                model=MODEL_CLAUDE,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt_mesh}]
            )
            st.markdown(response_mesh.content[0].text)

# ==========================================
# MÓDULO 3: REDACCIÓN ESTILO NOTEBOOKLM (GEMINI / CLAUDE)
# ==========================================
with tab3:
    st.header("Módulo 3: Redactor Basado en los PDFs Cargados")
    
    # Muestra el estado del cuaderno de la app
    if st.session_state["notebook_context"]:
        st.info(f"📚 Cuaderno Activo: Se utilizarán los PDFs procesados en el Módulo 1 como fuente exclusiva.")
    else:
        st.warning("⚠️ No has cargado PDFs en el Módulo 1. La redacción se basará en conocimiento general.")
        
    instruccion_redaccion = st.text_area("¿Qué deseas redactar?", "Escribe la introducción para la investigación respetando únicamente los hallazgos de los documentos cargados.")
    
    motor_eleccion = st.radio("Selecciona el motor de redacción:", ["Google Gemini (Estilo NotebookLM)", "Claude 3.5 Sonnet"])
    
    if st.button("Redactar Sección"):
        with st.spinner("Redactando y vinculando evidencia de los PDFs..."):
            
            prompt_redactor = f"""
            Actúa como un investigador médico sénior y redactor científico.
            Utiliza la siguiente BASE DE CONOCIMIENTO (extraída de los PDFs cargados) para redactar el texto solicitado.
            NO inventes datos fuera de este texto. Incluye citas tipo Vancouver [1], [2] referenciando los documentos.

            BASE DE CONOCIMIENTO:
            {st.session_state['notebook_context'][:30000]}

            SOLICITUD:
            {instruccion_redaccion}
            """
            
            if motor_eleccion == "Google Gemini (Estilo NotebookLM)":
                if not gemini_key:
                    st.error("Por favor, ingresa tu API Key de Google Gemini en el panel izquierdo para usar este motor.")
                else:
                    try:
                        g_client = genai.Client(api_key=gemini_key)
                        res_gemini = g_client.models.generate_content(
                            model="gemini-1.5-pro",
                            contents=prompt_redactor
                        )
                        st.markdown(res_gemini.text)
                    except Exception as e:
                        st.error(f"Error con Gemini API: {str(e)}")
            else:
                try:
                    res_claude = claude_client.messages.create(
                        model=MODEL_CLAUDE,
                        max_tokens=3500,
                        messages=[{"role": "user", "content": prompt_redactor}]
                    )
                    st.markdown(res_claude.content[0].text)
                except Exception as e:
                    st.error(f"Error con Claude API: {str(e)}")
