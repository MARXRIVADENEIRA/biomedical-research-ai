import streamlit as st
import anthropic
from google import genai
from pypdf import PdfReader

# Configuración de la página
st.set_page_config(page_title="BioMedical Research AI", layout="wide", page_icon="🔬")

# ==========================================
# SIDEBAR - CONFIGURACIÓN DE APIS
# ==========================================
st.sidebar.title("🛠️ Configuración de APIs")
anthropic_key = st.sidebar.text_input("1. Claude (Anthropic) API Key:", type="password")
gemini_key = st.sidebar.text_input("2. Gemini (Google) API Key (Opcional):", type="password")

if not anthropic_key:
    st.info("👈 Ingresa al menos tu API Key de Anthropic en el panel izquierdo para comenzar.")
    st.stop()

# Inicialización de clientes
claude_client = anthropic.Anthropic(api_key=anthropic_key)
MODEL_CLAUDE = "claude-3-5-sonnet-20240620"

# Memoria compartida para los PDFs dentro de la sesión de Streamlit
if "notebook_context" not in st.session_state:
    st.session_state["notebook_context"] = ""

tab1, tab2, tab3 = st.tabs([
    "📄 Evaluador Crítico (Múltiples PDFs)", 
    "🔍 Búsqueda PubMed (MeSH)", 
    "✍️ Redactor (Estilo NotebookLM)"
])

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
            # Limpiar contexto previo antes de procesar el lote actual
            st.session_state["notebook_context"] = ""
            
            for i, uploaded_file in enumerate(uploaded_files, 1):
                st.subheader(f"📄 Artículo {i}: {uploaded_file.name}")
                
                with st.spinner(f"Leyendo y analizando {uploaded_file.name}..."):
                    try:
                        reader = PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in reader.pages:
                            pdf_text += page.extract_text() or ""
                        
                        # Guardar el texto extraído en el contexto compartido
                        st.session_state["notebook_context"] += f"\n\n--- DOCUMENTO {i}: {uploaded_file.name} ---\n" + pdf_text
                        
                        # Prompt ajustado para permitir lectura amplia sin recortes agresivos
                        prompt = f"""
                        Eres un epidemiólogo experto y revisor de literatura médica. Analiza el siguiente artículo '{uploaded_file.name}':

                        1. Clasifica el diseño de estudio.
                        2. Aplica el checklist correspondiente (JBI, STROBE, CARE, CONSORT, etc.) indicando si cumple (Sí/No/No Claro) con citas breves del texto.
                        3. Evalúa los dominios de Sesgo (Selección, Medición, Confusión, Reporte).
                        4. Dictamen Final (Fortalezas, Debilidades y Relevancia Clínica).

                        TEXTO DEL ARTÍCULO:
                        {pdf_text[:120000]}
                        """
                        
                        response = claude_client.messages.create(
                            model=MODEL_CLAUDE,
                            max_tokens=4000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(response.content[0].text)
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
            prompt_mesh = f"""
            Actúa como bibliotecólogo biomédico experto.
            Genera una estrategia de búsqueda completa para PubMed sobre el siguiente tema: '{user_query}'
            
            Incluye:
            1. Términos MeSH principales con sus variaciones (Entry Terms).
            2. Términos en texto libre (tiab).
            3. Ecuación booleana combinada con AND, OR y comillas lista para copiar y pegar en PubMed.
            """
            response_mesh = claude_client.messages.create(
                model=MODEL_CLAUDE,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt_mesh}]
            )
            st.markdown(response_mesh.content[0].text)

# ==========================================
# MÓDULO 3: REDACCIÓN ESTILO NOTEBOOKLM
# ==========================================
with tab3:
    st.header("Módulo 3: Redactor Basado en los PDFs Cargados")
    
    # Verificación del contexto activo
    if st.session_state["notebook_context"]:
        st.info("📚 Cuaderno Activo: Se utilizarán los PDFs procesados en el Módulo 1 como fuente exclusiva.")
    else:
        st.warning("⚠️ No has procesado PDFs en el Módulo 1. Ve al Módulo 1 y presiona 'Procesar Archivos' para alimentar el cuaderno.")
        
    instruccion_redaccion = st.text_area(
        "¿Qué deseas redactar?", 
        "Escribe la sección de Introducción/Justificación respondiendo a la evidencia de los PDFs cargados."
    )
    
    motor_eleccion = st.radio("Selecciona el motor de redacción:", ["Google Gemini (Estilo NotebookLM)", "Claude 3.5 Sonnet"])
    
    if st.button("Redactar Sección"):
        if not st.session_state["notebook_context"]:
            st.error("No hay contexto de PDFs guardado. Procesa primero tus artículos en el Módulo 1.")
        else:
            with st.spinner("Sintetizando información y redactando con citas..."):
                
                prompt_redactor = f"""
                Actúa como un investigador médico sénior y redactor científico.
                Utiliza ÚNICAMENTE la siguiente BASE DE CONOCIMIENTO (extraída de los PDFs cargados) para redactar el texto solicitado.
                
                REGLAS ESTRICTAS:
                - No inventes datos fuera de este texto.
                - Incluye citas tipo Vancouver [1], [2] o por autor/año haciendo referencia explícita a los documentos cargados.
                - Mantén un tono académico, riguroso y formal.

                BASE DE CONOCIMIENTO:
                {st.session_state['notebook_context']}

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
                                model="gemini-2.5-pro",
                                contents=prompt_redactor
                            )
                            st.markdown(res_gemini.text)
                        except Exception as e:
                            st.error(f"Error con Gemini API: {str(e)}")
                else:
                    try:
                        res_claude = claude_client.messages.create(
                            model=MODEL_CLAUDE,
                            max_tokens=4000,
                            messages=[{"role": "user", "content": prompt_redactor}]
                        )
                        st.markdown(res_claude.content[0].text)
                    except Exception as e:
                        st.error(f"Error con Claude API: {str(e)}")
