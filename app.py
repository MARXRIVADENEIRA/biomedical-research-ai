"""
Herramienta de Investigación Biomédica - Streamlit + Gemini (google-genai, capa gratuita)
============================================================================================
Módulo 1: Evaluador Crítico y Lectura de PDFs
Módulo 2: Búsqueda Autónoma en PubMed (Estrategias MeSH) + búsqueda real vía Entrez
Módulo 3: Redactor Científico (estilo Cuaderno de Investigación)
"""

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from Bio import Entrez
import io
import json

# --------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------------

st.set_page_config(page_title="Asistente de Investigación Biomédica", layout="wide")

# Google cambia/retira nombres de modelo con frecuencia (por eso los 404 que
# tenías). En vez de fijar un solo nombre, probamos una lista en orden de
# prioridad y usamos el primero que responda. Si el día de mañana alguno de
# estos deja de existir, la app sigue funcionando con el siguiente.
MODELOS_CANDIDATOS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]


def get_client():
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("GEMINI_API_KEY", None) or st.secrets.get("GOOGLE_API_KEY", None)
    try:
        if api_key:
            return genai.Client(api_key=api_key)
        return genai.Client()  # usa GEMINI_API_KEY / GOOGLE_API_KEY del entorno
    except Exception as e:
        st.error(
            "No se pudo inicializar el cliente de Gemini. Revisa que tengas "
            f"configurada la variable GEMINI_API_KEY. Detalle: {e}"
        )
        st.stop()


client = get_client()

# --------------------------------------------------------------------------
# ESTADO DE SESIÓN (persiste entre pestañas)
# --------------------------------------------------------------------------

if "notebook_context" not in st.session_state:
    st.session_state["notebook_context"] = []  # [{"filename": str, "text": str}]

if "evaluaciones" not in st.session_state:
    st.session_state["evaluaciones"] = {}  # {filename: texto_evaluacion}

if "modelo_activo" not in st.session_state:
    st.session_state["modelo_activo"] = None  # se fija apenas una llamada funcione


# --------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------------

def extraer_texto_pdf(uploaded_file) -> str:
    """Extrae texto de un PDF de forma segura. Nunca rompe el flujo:
    si una página falla, la salta y continúa con las demás."""
    texto_paginas = []
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        for i, page in enumerate(reader.pages):
            try:
                contenido = page.extract_text() or ""
                texto_paginas.append(contenido)
            except Exception as e:
                texto_paginas.append(f"[No se pudo leer la página {i + 1}: {e}]")
    except Exception as e:
        st.error(f"Error al abrir '{uploaded_file.name}': {e}")
        return ""

    texto_final = "\n".join(texto_paginas).strip()
    if not texto_final:
        st.warning(
            f"'{uploaded_file.name}' no arrojó texto extraíble. "
            "Puede ser un PDF escaneado (imagen) sin OCR."
        )
    return texto_final


def llamar_gemini(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Wrapper único para todas las llamadas a la API. Prueba la lista de
    modelos candidatos en orden hasta que uno responda; recuerda cuál
    funcionó para no reintentar todos en cada llamada posterior."""
    orden = MODELOS_CANDIDATOS
    if st.session_state["modelo_activo"]:
        orden = [st.session_state["modelo_activo"]] + [
            m for m in MODELOS_CANDIDATOS if m != st.session_state["modelo_activo"]
        ]

    config_kwargs = {"system_instruction": system_prompt}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    errores = []
    for modelo in orden:
        try:
            respuesta = client.models.generate_content(
                model=modelo,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            st.session_state["modelo_activo"] = modelo
            return respuesta.text or ""
        except ClientError as e:
            # 404 = el modelo ya no existe/no está disponible -> probamos el siguiente
            # 429 = cuota agotada -> también probamos el siguiente por si acaso
            errores.append(f"{modelo}: {e}")
            continue
        except Exception as e:
            errores.append(f"{modelo}: {e}")
            continue

    return (
        "⚠️ No se pudo obtener respuesta de ningún modelo de Gemini. "
        "Detalle de intentos:\n" + "\n".join(errores)
    )


def construir_bloque_contexto() -> str:
    """Arma el bloque de fuentes numeradas (para citas Vancouver)."""
    bloques = []
    for i, doc in enumerate(st.session_state["notebook_context"], start=1):
        bloques.append(f"[{i}] Documento: {doc['filename']}\n{doc['text']}")
    return "\n\n---\n\n".join(bloques)


# --------------------------------------------------------------------------
# INTERFAZ: PESTAÑAS
# --------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📄 Módulo 1: Evaluador Crítico",
    "🔍 Módulo 2: Búsqueda PubMed (MeSH)",
    "✍️ Módulo 3: Redactor Científico",
])

# ==========================================================================
# MÓDULO 1: EVALUADOR CRÍTICO Y LECTURA DE PDFs
# ==========================================================================
with tab1:
    st.header("Evaluador Crítico de Artículos Científicos")
    st.caption(
        "Sube uno o varios PDFs. El texto extraído se guarda en memoria de "
        "sesión y queda disponible para el Módulo 3 (Redactor Científico)."
    )

    archivos = st.file_uploader(
        "Sube tus artículos en PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader_modulo1",
    )

    SYSTEM_EVALUADOR = """Eres un metodólogo experto en epidemiología y lectura crítica de literatura científica.
Para el artículo que se te entregue, realiza un dictamen estructurado que incluya:

1. Clasificación del diseño de estudio (observacional/experimental, tipo específico).
2. Lista de chequeo aplicable según el diseño (STROBE, CARE, CONSORT o JBI) y evaluación ítem por ítem de lo que SÍ cumple y lo que NO.
3. Evaluación de sesgos: selección, medición/información, confusión, y otros relevantes al diseño.
4. Fortalezas metodológicas.
5. Debilidades y limitaciones.
6. Dictamen final sobre la calidad y aplicabilidad de la evidencia.

Basa tu análisis ÚNICAMENTE en el texto proporcionado. Si el texto está incompleto o corrupto en alguna sección, indícalo explícitamente en vez de inventar contenido."""

    if archivos:
        if st.button("Procesar y evaluar artículos", type="primary"):
            for archivo in archivos:
                ya_procesado = any(
                    d["filename"] == archivo.name
                    for d in st.session_state["notebook_context"]
                )
                with st.spinner(f"Extrayendo texto de {archivo.name}..."):
                    texto = extraer_texto_pdf(archivo)

                if not texto:
                    continue

                if not ya_procesado:
                    st.session_state["notebook_context"].append(
                        {"filename": archivo.name, "text": texto}
                    )

                with st.spinner(f"Evaluando {archivo.name} con IA..."):
                    evaluacion = llamar_gemini(
                        SYSTEM_EVALUADOR,
                        f"Artículo: {archivo.name}\n\nTexto completo:\n{texto}",
                    )
                st.session_state["evaluaciones"][archivo.name] = evaluacion

    if st.session_state["evaluaciones"]:
        st.subheader("Resultados de evaluación")
        for nombre, evaluacion in st.session_state["evaluaciones"].items():
            with st.expander(f"📄 {nombre}"):
                st.markdown(evaluacion)

    if st.session_state["notebook_context"]:
        st.divider()
        st.success(
            f"📚 {len(st.session_state['notebook_context'])} documento(s) "
            "en memoria, disponibles para el Módulo 3."
        )
        if st.button("🗑️ Limpiar memoria de documentos"):
            st.session_state["notebook_context"] = []
            st.session_state["evaluaciones"] = {}
            st.rerun()

# ==========================================================================
# MÓDULO 2: BÚSQUEDA AUTÓNOMA EN PUBMED (MESH) + BÚSQUEDA REAL (biopython)
# ==========================================================================
with tab2:
    st.header("Constructor de Estrategias de Búsqueda PubMed")
    st.caption(
        "Describe tu tema y la IA arma la ecuación booleana. Después puedes "
        "ejecutarla de una vez contra PubMed (búsqueda real, gratis, vía NCBI Entrez)."
    )

    correo_ncbi = st.text_input(
        "Tu correo (requerido por NCBI para usar Entrez, no se envían datos a Google)",
        placeholder="tu_correo@ejemplo.com",
    )

    tema = st.text_area(
        "Tema o idea de investigación",
        placeholder="Ej: Efecto de la metformina en la prevención de diabetes gestacional en mujeres con obesidad",
        height=100,
    )

    SYSTEM_BIBLIOTECOLOGO = """Eres un bibliotecólogo biomédico experto en estrategias de búsqueda para PubMed/MEDLINE.

Responde ÚNICAMENTE con un objeto JSON válido (sin texto adicional, sin markdown, sin backticks) con esta estructura exacta:

{
  "conceptos_pico": ["concepto 1", "concepto 2", "..."],
  "terminos_mesh": ["\\"Termino\\"[Mesh]", "..."],
  "terminos_libres": ["termino[tiab]", "..."],
  "ecuacion_final": "ecuación booleana completa lista para pegar en PubMed, con AND/OR/comillas/paréntesis"
}"""

    if "resultado_mesh" not in st.session_state:
        st.session_state["resultado_mesh"] = None

    if st.button("Generar estrategia de búsqueda", type="primary", disabled=not tema):
        with st.spinner("Construyendo estrategia MeSH..."):
            respuesta_raw = llamar_gemini(SYSTEM_BIBLIOTECOLOGO, tema, json_mode=True)
        try:
            st.session_state["resultado_mesh"] = json.loads(respuesta_raw)
        except json.JSONDecodeError:
            st.session_state["resultado_mesh"] = None
            st.error("La IA no devolvió un JSON válido. Respuesta cruda:")
            st.code(respuesta_raw)

    if st.session_state["resultado_mesh"]:
        r = st.session_state["resultado_mesh"]
        st.subheader("Conceptos clave (PICO)")
        st.write(", ".join(r.get("conceptos_pico", [])))

        st.subheader("Términos MeSH")
        st.write(", ".join(r.get("terminos_mesh", [])))

        st.subheader("Términos de texto libre")
        st.write(", ".join(r.get("terminos_libres", [])))

        st.subheader("Ecuación booleana final")
        st.code(r.get("ecuacion_final", ""), language="text")

        st.divider()
        if st.button("🔎 Ejecutar esta búsqueda en PubMed ahora", disabled=not correo_ncbi):
            Entrez.email = correo_ncbi
            try:
                with st.spinner("Consultando PubMed vía Entrez..."):
                    handle = Entrez.esearch(
                        db="pubmed",
                        term=r.get("ecuacion_final", ""),
                        retmax=15,
                        sort="relevance",
                    )
                    resultado_busqueda = Entrez.read(handle)
                    handle.close()
                    ids = resultado_busqueda.get("IdList", [])

                st.write(
                    f"**{resultado_busqueda.get('Count', '0')}** resultados totales "
                    f"en PubMed. Mostrando los primeros {len(ids)}:"
                )

                if ids:
                    with st.spinner("Obteniendo detalles de los artículos..."):
                        handle_sum = Entrez.esummary(db="pubmed", id=",".join(ids))
                        resumenes = Entrez.read(handle_sum)
                        handle_sum.close()

                    for item in resumenes:
                        pmid = item.get("Id", "")
                        titulo = item.get("Title", "Sin título")
                        revista = item.get("Source", "")
                        anio = item.get("PubDate", "")
                        st.markdown(
                            f"**[{titulo}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)**  \n"
                            f"{revista} · {anio} · PMID: {pmid}"
                        )
            except Exception as e:
                st.error(f"Error al consultar PubMed: {e}")

        if not correo_ncbi:
            st.caption("⬆️ Ingresa tu correo arriba para poder ejecutar la búsqueda real.")

# ==========================================================================
# MÓDULO 3: REDACTOR CIENTÍFICO (ESTILO CUADERNO DE INVESTIGACIÓN)
# ==========================================================================
with tab3:
    st.header("Redactor Científico basado en tus fuentes")

    if not st.session_state["notebook_context"]:
        st.info(
            "⚠️ Aún no hay documentos cargados. Ve al Módulo 1 y sube al "
            "menos un PDF para poder redactar con citas Vancouver."
        )
    else:
        st.success(
            f"Usando {len(st.session_state['notebook_context'])} documento(s) "
            "como fuente exclusiva:"
        )
        for i, doc in enumerate(st.session_state["notebook_context"], start=1):
            st.write(f"[{i}] {doc['filename']}")

        seccion = st.selectbox(
            "Sección a redactar",
            ["Introducción", "Justificación", "Marco Teórico", "Discusión", "Conclusiones"],
        )
        instrucciones_extra = st.text_area(
            "Instrucciones adicionales (opcional)",
            placeholder="Ej: Enfatizar la brecha de conocimiento en población pediátrica",
        )

        SYSTEM_REDACTOR = """Eres un redactor científico académico especializado en estilo Vancouver.

Reglas estrictas:
1. Usa EXCLUSIVAMENTE la información contenida en las fuentes numeradas que se te entregan. Está PROHIBIDO inventar datos, cifras o afirmaciones que no estén en las fuentes.
2. Cada afirmación factual debe llevar su cita correspondiente en formato Vancouver: [1], [2], etc., según el número de fuente.
3. Si la información disponible es insuficiente para cubrir algún punto de la sección solicitada, dilo explícitamente en vez de rellenar con contenido genérico o inventado.
4. Mantén un tono académico, objetivo y en español."""

        if st.button("Redactar sección", type="primary"):
            contexto = construir_bloque_contexto()
            prompt_usuario = f"""Redacta la sección "{seccion}" de un artículo científico.

Instrucciones adicionales: {instrucciones_extra or "Ninguna"}

FUENTES DISPONIBLES (usa citas Vancouver [n] referenciando el número de cada una):

{contexto}"""
            with st.spinner("Redactando con IA..."):
                texto_redactado = llamar_gemini(SYSTEM_REDACTOR, prompt_usuario)
            st.markdown(texto_redactado)

            st.divider()
            st.caption("Referencias (formato Vancouver)")
            for i, doc in enumerate(st.session_state["notebook_context"], start=1):
                st.write(f"{i}. {doc['filename']}")
