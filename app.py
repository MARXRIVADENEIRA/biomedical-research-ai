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
import pandas as pd
import requests
import io
import json
import re

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

if "fuentes_externas" not in st.session_state:
    # Artículos encontrados por API (PubMed/Europe PMC/Scopus/IEEE) y
    # aprobados por el usuario para usarse en el Módulo 3.
    # [{"id","fuente","titulo","autores","revista","anio","resumen","nivel_evidencia","url"}]
    st.session_state["fuentes_externas"] = []


# --------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------------

def extraer_texto_pdf(uploaded_file) -> str:
    """Extrae texto de un PDF de forma segura, marcando cada página con
    '--- Página N ---' para que la IA pueda citar la ubicación exacta de
    cada hallazgo. Nunca rompe el flujo: si una página falla, la salta y
    continúa con las demás."""
    texto_paginas = []
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        for i, page in enumerate(reader.pages):
            try:
                contenido = page.extract_text() or ""
                texto_paginas.append(f"--- Página {i + 1} ---\n{contenido}")
            except Exception as e:
                texto_paginas.append(f"--- Página {i + 1} ---\n[No se pudo leer esta página: {e}]")
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


# Pirámide de evidencia: filtros de tipo de publicación de PubMed, de mayor
# a menor jerarquía. La búsqueda priorizada consulta cada nivel por separado
# para que el investigador vea primero la evidencia más fuerte.
PIRAMIDE_EVIDENCIA = [
    ("1. Metaanálisis", '"Meta-Analysis"[Publication Type]'),
    ("2. Revisión Sistemática", '"Systematic Review"[Publication Type]'),
    ("3. Ensayo Clínico", '("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial"[Publication Type])'),
    ("4. Cohorte / Ecológico", '("Cohort Studies"[Mesh] OR "Ecological Studies"[Mesh])'),
]


def buscar_pubmed_nivel(query_base: str, filtro_pt: str, retmax: int = 5) -> list:
    """Busca en PubMed (gratis, vía NCBI Entrez) restringiendo por tipo de
    publicación a un nivel específico de la pirámide de evidencia."""
    query_completa = f"({query_base}) AND {filtro_pt}"
    resultados = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query_completa, retmax=retmax, sort="relevance")
        busqueda = Entrez.read(handle)
        handle.close()
        ids = busqueda.get("IdList", [])
        if not ids:
            return []

        handle_sum = Entrez.esummary(db="pubmed", id=",".join(ids))
        resumenes = Entrez.read(handle_sum)
        handle_sum.close()

        for item in resumenes:
            autores_lista = item.get("AuthorList", [])
            resultados.append({
                "fuente": "PubMed",
                "id": item.get("Id", ""),
                "titulo": item.get("Title", "Sin título"),
                "autores": ", ".join(autores_lista[:3]) + (" et al." if len(autores_lista) > 3 else ""),
                "revista": item.get("Source", ""),
                "anio": str(item.get("PubDate", ""))[:4],
                "resumen": "",  # se completa bajo demanda con efetch (más pesado)
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{item.get('Id', '')}/",
            })
    except Exception as e:
        st.warning(f"No se pudo consultar PubMed para este nivel: {e}")
    return resultados


def obtener_abstract_pubmed(pmid: str) -> str:
    """Trae el resumen completo de un artículo puntual vía efetch."""
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        texto = handle.read()
        handle.close()
        return texto.strip()
    except Exception as e:
        return f"[No se pudo obtener el resumen: {e}]"


def buscar_europepmc(query: str, retmax: int = 10) -> list:
    """Busca en Europe PMC (gratis, sin API key) — buena fuente complementaria
    a PubMed, incluye preprints y en varios casos el resumen completo."""
    resultados = []
    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": retmax, "resultType": "core"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("resultList", {}).get("result", []):
            resultados.append({
                "fuente": "Europe PMC",
                "id": item.get("id", ""),
                "titulo": item.get("title", "Sin título"),
                "autores": item.get("authorString", ""),
                "revista": item.get("journalTitle", ""),
                "anio": item.get("pubYear", ""),
                "resumen": item.get("abstractText", ""),
                "url": f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}",
            })
    except Exception as e:
        st.warning(f"No se pudo consultar Europe PMC: {e}")
    return resultados


def buscar_scopus(query: str, api_key: str, retmax: int = 10) -> list:
    """Busca en Scopus (Elsevier). Requiere una API Key institucional propia
    del investigador — nunca se guarda ni se comparte, solo se usa en la
    llamada directa a la API de Elsevier."""
    resultados = []
    try:
        resp = requests.get(
            "https://api.elsevier.com/content/search/scopus",
            headers={"X-ELS-APIKey": api_key, "Accept": "application/json"},
            params={"query": query, "count": retmax},
            timeout=20,
        )
        resp.raise_for_status()
        entradas = resp.json().get("search-results", {}).get("entry", [])
        for item in entradas:
            resultados.append({
                "fuente": "Scopus",
                "id": item.get("dc:identifier", ""),
                "titulo": item.get("dc:title", "Sin título"),
                "autores": item.get("dc:creator", ""),
                "revista": item.get("prism:publicationName", ""),
                "anio": (item.get("prism:coverDate", "") or "")[:4],
                "resumen": item.get("dc:description", ""),
                "url": item.get("prism:url", ""),
            })
    except Exception as e:
        st.warning(f"No se pudo consultar Scopus (revisa tu API Key institucional): {e}")
    return resultados


def buscar_ieee(query: str, api_key: str, retmax: int = 10) -> list:
    """Busca en IEEE Xplore. Requiere API Key institucional propia."""
    resultados = []
    try:
        resp = requests.get(
            "http://ieeexploreapi.ieee.org/api/v1/search/articles",
            params={"apikey": api_key, "querytext": query, "max_records": retmax},
            timeout=20,
        )
        resp.raise_for_status()
        articulos = resp.json().get("articles", [])
        for item in articulos:
            resultados.append({
                "fuente": "IEEE",
                "id": item.get("article_number", ""),
                "titulo": item.get("title", "Sin título"),
                "autores": ", ".join(a.get("full_name", "") for a in item.get("authors", {}).get("authors", [])),
                "revista": item.get("publication_title", ""),
                "anio": str(item.get("publication_year", "")),
                "resumen": item.get("abstract", ""),
                "url": item.get("html_url", ""),
            })
    except Exception as e:
        st.warning(f"No se pudo consultar IEEE Xplore (revisa tu API Key institucional): {e}")
    return resultados


def formatear_referencia_vancouver(doc: dict) -> str:
    """Construye una línea de referencia en formato Vancouver a partir de
    los metadatos disponibles; si faltan, cae de vuelta al nombre de archivo."""
    autores = (doc.get("autores") or "").strip()
    titulo = (doc.get("titulo") or "").strip()
    revista = (doc.get("revista") or "").strip()
    anio = (doc.get("anio") or "").strip()

    if titulo:
        titulo_fmt = titulo if titulo.endswith(".") else f"{titulo}."
        partes = [p for p in [autores, titulo_fmt, revista, anio] if p]
        return ". ".join(partes).replace("..", ".")
    return doc.get("filename", "Fuente sin identificar")


def construir_bloque_contexto() -> str:
    """Arma el bloque de fuentes numeradas (para citas Vancouver) combinando
    los PDFs analizados en el Módulo 1 y los artículos aprobados en el
    Módulo 2 (búsqueda por API)."""
    bloques = []
    contador = 1
    for doc in st.session_state["notebook_context"]:
        etiqueta = formatear_referencia_vancouver(doc) if doc.get("titulo") else doc["filename"]
        bloques.append(f"[{contador}] Fuente: {etiqueta}\nTexto completo del documento:\n{doc['text']}")
        contador += 1
    for fx in st.session_state["fuentes_externas"]:
        etiqueta = formatear_referencia_vancouver(fx)
        bloques.append(
            f"[{contador}] Fuente ({fx.get('fuente', '')}): {etiqueta}\n"
            f"Resumen/abstract:\n{fx.get('resumen') or '(sin resumen disponible)'}"
        )
        contador += 1
    return "\n\n---\n\n".join(bloques)


def listar_fuentes_numeradas() -> list:
    """Lista plana de referencias Vancouver ya numeradas, en el mismo orden
    que construir_bloque_contexto(), para mostrar la bibliografía final."""
    referencias = []
    contador = 1
    for doc in st.session_state["notebook_context"]:
        etiqueta = formatear_referencia_vancouver(doc) if doc.get("titulo") else doc["filename"]
        referencias.append(f"{contador}. {etiqueta}")
        contador += 1
    for fx in st.session_state["fuentes_externas"]:
        referencias.append(f"{contador}. {formatear_referencia_vancouver(fx)} [{fx.get('fuente', '')}]")
        contador += 1
    return referencias


# --------------------------------------------------------------------------
# INTERFAZ: PESTAÑAS
# --------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📄 Módulo 1: Evaluador Crítico",
    "🔍 Módulo 2: Búsqueda PubMed (MeSH)",
    "✍️ Módulo 3: Redactor Científico",
])

# ==========================================================================
# MÓDULO 1: EL EVALUADOR METODOLÓGICO
# ==========================================================================

with tab1:
    if "analisis_metodologico" not in st.session_state:
        st.session_state["analisis_metodologico"] = {}  # {filename: dict con el JSON de la IA}

    if "checklist_validado" not in st.session_state:
        st.session_state["checklist_validado"] = {}  # {filename: DataFrame validado por el usuario}


    def extraer_json(texto: str) -> dict:
        """El modelo a veces envuelve el JSON en texto o backticks pese a
        pedírselo limpio; esta función intenta rescatarlo de todas formas."""
        texto = texto.strip()
        texto = re.sub(r"^```json\s*|^```\s*|```$", "", texto, flags=re.MULTILINE).strip()
        try:
            return json.loads(texto)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise


    SYSTEM_METODOLOGO = """Eres un metodólogo experto en epidemiología clínica y lectura crítica de literatura científica.

Se te entrega el texto COMPLETO de un artículo científico —todas sus páginas, de la primera a la última, marcadas con "--- Página N ---"—. Debes leer y analizar el documento entero (portada, resumen, introducción, métodos, resultados, discusión y referencias), NO solo el resumen o la primera página. El usuario te indica cuál es su OBJETIVO de análisis: evaluar la CALIDAD DEL REPORTE (qué tan completo está escrito) o evaluar el RIESGO DE SESGO (qué tan confiables son sus resultados).

Sigue este proceso obligatorio:

PASO 0 - METADATOS: Identifica en la portada/encabezado del artículo (o en la sección de referencias si se cita a sí mismo) los datos bibliográficos: autores (apellido e iniciales, hasta 3 y "et al." si hay más), título exacto, nombre de la revista, y año de publicación.

PASO 1 - CLASIFICACIÓN DE DISEÑO: Lee el resumen y la sección de métodos para clasificar el diseño exacto del estudio, eligiendo entre: Reporte de Caso, Serie de Casos, Estudio de Cohorte, Estudio de Casos y Controles, Estudio Transversal, Ensayo Clínico, o Estudio Ecológico.

PASO 2 - SELECCIÓN DE HERRAMIENTA (según el objetivo indicado por el usuario):
- Si el objetivo es "calidad de reporte": elige CARE (si es Reporte de Caso), MInCir (si es Serie de Casos), o STROBE (si es cualquier diseño observacional: cohorte, casos y controles, transversal o ecológico). Para ensayos clínicos usa CONSORT.
- Si el objetivo es "riesgo de sesgo": elige la herramienta JBI específica para ese diseño (con opciones de respuesta: "Sí", "No", "No claro", "No aplica"), la escala de Hassan & Wu (organizada en sus 4 dominios: selección, comparabilidad, exposición/resultado y análisis estadístico), o CASPe (con opciones "Sí", "No", "No sé"). Elige la que mejor se ajuste al diseño identificado y justifica por qué.

PASO 3 - EXTRACCIÓN DE EVIDENCIA: Para cada ítem del checklist elegido, no te limites a responder Sí/No: busca en el texto la cita textual exacta (breve, máximo 25 palabras) y la ubicación (número de página según los marcadores "--- Página N ---" y la sección, ej. "Métodos") que justifica tu respuesta. Si el artículo no aborda ese ítem, responde "No claro" o "No aplica" y dilo explícitamente en vez de inventar una cita.

Responde ÚNICAMENTE con un objeto JSON válido (sin texto adicional, sin markdown, sin backticks) con esta estructura exacta:

{
  "referencia_bibliografica": {
    "autores": "Apellido AB, Apellido CD, et al.",
    "titulo": "título exacto del artículo",
    "revista": "nombre de la revista",
    "anio": "año de publicación (solo el número, ej. 2023)"
  },
  "diseno_identificado": "nombre del diseño",
  "justificacion_diseno": "breve explicación de por qué se clasificó así",
  "herramienta_seleccionada": "nombre de la herramienta (ej. STROBE, JBI para cohortes, CASPe, etc.)",
  "justificacion_seleccion": "por qué esta herramienta es la adecuada para este diseño y objetivo",
  "checklist": [
    {
      "numero": 1,
      "item": "texto del criterio o pregunta del checklist",
      "respuesta_ia": "Sí / No / No claro / No aplica",
      "cita_textual": "cita literal breve extraída del artículo, o vacío si no aplica",
      "ubicacion": "ej. Página 3, sección Métodos",
      "justificacion": "por qué se responde así, en 1-2 frases"
    }
  ],
  "fortalezas": ["fortaleza 1", "fortaleza 2"],
  "debilidades": ["debilidad 1", "debilidad 2"],
  "dictamen_final": "párrafo breve con el dictamen general de calidad/aplicabilidad de la evidencia"
}

Incluye TODOS los ítems del checklist de la herramienta seleccionada, no un subconjunto. Basa todo el análisis ÚNICAMENTE en el texto proporcionado."""


    st.header("📄 Módulo 1: El Evaluador Metodológico")
    st.caption(
        "Clasificación de diseño → selección de herramienta → extracción de "
        "evidencia con cita textual → checklist interactivo que puedes validar."
    )

    objetivo = st.radio(
        "¿Cuál es tu objetivo de análisis?",
        [
            "Calidad de reporte (CARE / MInCir / STROBE / CONSORT)",
            "Riesgo de sesgo (JBI / Hassan & Wu / CASPe)",
        ],
        horizontal=False,
    )

    archivos = st.file_uploader(
        "Sube tus artículos en PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader_modulo1",
    )

    if archivos:
        if st.button("🔬 Analizar artículo(s)", type="primary"):
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

                with st.spinner(f"Clasificando diseño y evaluando {archivo.name}..."):
                    prompt_usuario = (
                        f"Objetivo de análisis: {objetivo}\n\n"
                        f"Artículo: {archivo.name}\n\nTexto completo:\n{texto}"
                    )
                    respuesta_raw = llamar_gemini(SYSTEM_METODOLOGO, prompt_usuario, json_mode=True)

                try:
                    resultado = extraer_json(respuesta_raw)
                    st.session_state["analisis_metodologico"][archivo.name] = resultado
                    # Al llegar un análisis nuevo, se descarta cualquier validación previa
                    st.session_state["checklist_validado"].pop(archivo.name, None)

                    # Volcamos los metadatos bibliográficos al documento en memoria
                    # para poder citarlo en formato Vancouver real en el Módulo 3.
                    ref = resultado.get("referencia_bibliografica", {}) or {}
                    for doc in st.session_state["notebook_context"]:
                        if doc["filename"] == archivo.name:
                            doc["autores"] = ref.get("autores", "")
                            doc["titulo"] = ref.get("titulo", "")
                            doc["revista"] = ref.get("revista", "")
                            doc["anio"] = ref.get("anio", "")
                            break
                except (json.JSONDecodeError, AttributeError):
                    st.error(f"La IA no devolvió un JSON válido para '{archivo.name}'. Respuesta cruda:")
                    st.code(respuesta_raw)

    # --------------------------------------------------------------------
    # Resultados por artículo
    # --------------------------------------------------------------------
    if st.session_state["analisis_metodologico"]:
        st.subheader("Resultados del análisis")

        for nombre, resultado in st.session_state["analisis_metodologico"].items():
            with st.expander(f"📄 {nombre}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Paso 1 · Diseño identificado:** {resultado.get('diseno_identificado', '—')}")
                    st.caption(resultado.get("justificacion_diseno", ""))
                with col2:
                    st.markdown(f"**Paso 2 · Herramienta seleccionada:** {resultado.get('herramienta_seleccionada', '—')}")
                    st.caption(resultado.get("justificacion_seleccion", ""))

                st.markdown("**Paso 3 y 4 · Checklist interactivo (evidencia + validación)**")
                st.caption(
                    "La columna 'Respuesta IA' y la cita/ubicación las propone el modelo. "
                    "Edita 'Respuesta validada' y 'Comentario' según tu propio criterio experto."
                )

                checklist = resultado.get("checklist", [])
                if checklist:
                    df = pd.DataFrame(checklist)
                    columnas_esperadas = ["numero", "item", "respuesta_ia", "cita_textual", "ubicacion", "justificacion"]
                    for col in columnas_esperadas:
                        if col not in df.columns:
                            df[col] = ""
                    df = df[columnas_esperadas].rename(columns={
                        "numero": "N°",
                        "item": "Criterio",
                        "respuesta_ia": "Respuesta IA",
                        "cita_textual": "Cita textual (evidencia)",
                        "ubicacion": "Ubicación",
                        "justificacion": "Justificación IA",
                    })

                    # Si ya existe una validación previa del usuario, la reutilizamos como base
                    if nombre in st.session_state["checklist_validado"]:
                        df_base = st.session_state["checklist_validado"][nombre]
                    else:
                        df_base = df.copy()
                        df_base["Respuesta validada"] = df_base["Respuesta IA"]
                        df_base["Comentario del validador"] = ""

                    df_editado = st.data_editor(
                        df_base,
                        key=f"editor_{nombre}",
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Respuesta validada": st.column_config.SelectboxColumn(
                                options=["Sí", "No", "No claro / No sé", "No aplica"],
                                required=True,
                            ),
                            "Cita textual (evidencia)": st.column_config.TextColumn(width="large"),
                            "Criterio": st.column_config.TextColumn(width="large", disabled=True),
                            "Respuesta IA": st.column_config.TextColumn(disabled=True),
                            "Ubicación": st.column_config.TextColumn(disabled=True),
                            "Justificación IA": st.column_config.TextColumn(disabled=True),
                            "N°": st.column_config.NumberColumn(disabled=True),
                        },
                    )
                    st.session_state["checklist_validado"][nombre] = df_editado
                else:
                    st.warning("La IA no devolvió ítems de checklist para este artículo.")

                colf, cold = st.columns(2)
                with colf:
                    st.markdown("**Fortalezas**")
                    for f in resultado.get("fortalezas", []):
                        st.write(f"✅ {f}")
                with cold:
                    st.markdown("**Debilidades**")
                    for d in resultado.get("debilidades", []):
                        st.write(f"⚠️ {d}")

                st.info(f"**Dictamen final:** {resultado.get('dictamen_final', '—')}")

    if st.session_state["notebook_context"]:
        st.divider()
        st.success(
            f"📚 {len(st.session_state['notebook_context'])} documento(s) "
            "en memoria, disponibles para el Módulo 3."
        )
        if st.button("🗑️ Limpiar memoria de documentos"):
            st.session_state["notebook_context"] = []
            st.session_state["evaluaciones"] = {}
            st.session_state["analisis_metodologico"] = {}
            st.session_state["checklist_validado"] = {}
            st.rerun()

# ==========================================================================
# MÓDULO 2: EL RETO DE LAS BASES DE DATOS Y EL EXTENSOR (BÚSQUEDA AUTÓNOMA)
# ==========================================================================
with tab2:
    st.header("🔍 Módulo 2: Búsqueda Autónoma en Bases de Datos")
    st.caption(
        "La IA no navega como humano: se conecta por API a las bases de datos. "
        "Primero arma la ecuación booleana, luego busca de verdad y prioriza "
        "por pirámide de evidencia (Metaanálisis → Revisiones Sistemáticas → "
        "Ensayos Clínicos → Cohorte/Ecológico)."
    )

    correo_ncbi = st.text_input(
        "Tu correo (requerido por NCBI para usar la API de PubMed — gratis, no se envía a Google)",
        placeholder="tu_correo@ejemplo.com",
        key="correo_ncbi_input",
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

    if st.button("1️⃣ Generar estrategia de búsqueda", type="primary", disabled=not tema):
        with st.spinner("Construyendo estrategia MeSH..."):
            respuesta_raw = llamar_gemini(SYSTEM_BIBLIOTECOLOGO, tema, json_mode=True)
        try:
            st.session_state["resultado_mesh"] = extraer_json(respuesta_raw)
        except (json.JSONDecodeError, AttributeError):
            st.session_state["resultado_mesh"] = None
            st.error("La IA no devolvió un JSON válido. Respuesta cruda:")
            st.code(respuesta_raw)

    if st.session_state["resultado_mesh"]:
        r = st.session_state["resultado_mesh"]
        with st.expander("Ver estrategia MeSH generada", expanded=False):
            st.write("**Conceptos PICO:**", ", ".join(r.get("conceptos_pico", [])))
            st.write("**Términos MeSH:**", ", ".join(r.get("terminos_mesh", [])))
            st.write("**Términos de texto libre:**", ", ".join(r.get("terminos_libres", [])))
            st.code(r.get("ecuacion_final", ""), language="text")

        ecuacion = r.get("ecuacion_final", "")

        st.divider()
        st.subheader("2️⃣ Bases de datos abiertas (gratis)")

        col_a, col_b = st.columns(2)
        with col_a:
            usar_pubmed = st.checkbox("PubMed (priorizado por pirámide de evidencia)", value=True)
        with col_b:
            usar_europepmc = st.checkbox("Europe PMC (complementario, incluye preprints)", value=False)

        retmax_nivel = st.slider("Máximo de resultados por nivel/fuente", 3, 15, 5)

        if st.button("🔎 Ejecutar búsqueda priorizada", disabled=not (correo_ncbi and (usar_pubmed or usar_europepmc))):
            Entrez.email = correo_ncbi

            if usar_pubmed:
                for nombre_nivel, filtro_pt in PIRAMIDE_EVIDENCIA:
                    with st.spinner(f"Buscando en PubMed — {nombre_nivel}..."):
                        resultados_nivel = buscar_pubmed_nivel(ecuacion, filtro_pt, retmax_nivel)
                    st.markdown(f"#### {nombre_nivel}")
                    if not resultados_nivel:
                        st.caption("Sin resultados en este nivel.")
                        continue
                    for res in resultados_nivel:
                        ya_agregado = any(
                            f.get("id") == res["id"] and f.get("fuente") == "PubMed"
                            for f in st.session_state["fuentes_externas"]
                        )
                        with st.container(border=True):
                            st.markdown(f"**[{res['titulo']}]({res['url']})**")
                            st.caption(f"{res['autores']} · {res['revista']} · {res['anio']} · PMID {res['id']}")
                            st.button(
                                "✅ Ya agregado a fuentes" if ya_agregado else "➕ Agregar a fuentes (Módulo 3)",
                                key=f"add_pm_{nombre_nivel}_{res['id']}",
                                disabled=ya_agregado,
                                on_click=lambda r=res, n=nombre_nivel: (
                                    r.update({"resumen": obtener_abstract_pubmed(r["id"]), "nivel_evidencia": n}),
                                    st.session_state["fuentes_externas"].append(r),
                                ),
                            )

            if usar_europepmc:
                with st.spinner("Buscando en Europe PMC..."):
                    resultados_epmc = buscar_europepmc(ecuacion, retmax_nivel)
                st.markdown("#### Europe PMC")
                if not resultados_epmc:
                    st.caption("Sin resultados.")
                for res in resultados_epmc:
                    ya_agregado = any(
                        f.get("id") == res["id"] and f.get("fuente") == "Europe PMC"
                        for f in st.session_state["fuentes_externas"]
                    )
                    with st.container(border=True):
                        st.markdown(f"**[{res['titulo']}]({res['url']})**")
                        st.caption(f"{res['autores']} · {res['revista']} · {res['anio']}")
                        st.button(
                            "✅ Ya agregado a fuentes" if ya_agregado else "➕ Agregar a fuentes (Módulo 3)",
                            key=f"add_epmc_{res['id']}",
                            disabled=ya_agregado,
                            on_click=lambda r=res: (
                                r.update({"nivel_evidencia": "No clasificado"}),
                                st.session_state["fuentes_externas"].append(r),
                            ),
                        )

        st.divider()
        st.subheader("3️⃣ Bases de datos institucionales (opcional, requieren tu propia API Key)")
        st.caption(
            "Scopus, Web of Science e IEEE cobran suscripción; si tu universidad "
            "tiene acceso, pega aquí tu token/API Key personal para desbloquear "
            "la búsqueda. Tu key nunca se guarda ni se envía a Google — solo se "
            "usa para llamar directo a la API del proveedor correspondiente."
        )
        with st.expander("Configurar fuentes institucionales"):
            col_s, col_i = st.columns(2)
            with col_s:
                scopus_key = st.text_input("API Key de Scopus (Elsevier)", type="password", key="scopus_key")
                if st.button("Buscar en Scopus", disabled=not scopus_key):
                    with st.spinner("Consultando Scopus..."):
                        resultados_scopus = buscar_scopus(ecuacion, scopus_key, retmax_nivel)
                    for res in resultados_scopus:
                        with st.container(border=True):
                            st.markdown(f"**{res['titulo']}**")
                            st.caption(f"{res['autores']} · {res['revista']} · {res['anio']}")
                            st.button(
                                "➕ Agregar a fuentes (Módulo 3)",
                                key=f"add_scopus_{res['id']}",
                                on_click=lambda r=res: (
                                    r.update({"nivel_evidencia": "No clasificado"}),
                                    st.session_state["fuentes_externas"].append(r),
                                ),
                            )
            with col_i:
                ieee_key = st.text_input("API Key de IEEE Xplore", type="password", key="ieee_key")
                if st.button("Buscar en IEEE", disabled=not ieee_key):
                    with st.spinner("Consultando IEEE Xplore..."):
                        resultados_ieee = buscar_ieee(ecuacion, ieee_key, retmax_nivel)
                    for res in resultados_ieee:
                        with st.container(border=True):
                            st.markdown(f"**{res['titulo']}**")
                            st.caption(f"{res['autores']} · {res['revista']} · {res['anio']}")
                            st.button(
                                "➕ Agregar a fuentes (Módulo 3)",
                                key=f"add_ieee_{res['id']}",
                                on_click=lambda r=res: (
                                    r.update({"nivel_evidencia": "No clasificado"}),
                                    st.session_state["fuentes_externas"].append(r),
                                ),
                            )

    if not correo_ncbi:
        st.caption("⬆️ Ingresa tu correo arriba para poder ejecutar búsquedas reales en PubMed/Entrez.")

    if st.session_state["fuentes_externas"]:
        st.divider()
        st.subheader("📚 Fuentes aprobadas para el Módulo 3")
        for i, fx in enumerate(st.session_state["fuentes_externas"]):
            col_ref, col_del = st.columns([6, 1])
            with col_ref:
                st.write(f"**[{fx.get('nivel_evidencia', '')}]** {formatear_referencia_vancouver(fx)} — *{fx.get('fuente', '')}*")
            with col_del:
                if st.button("🗑️", key=f"del_fx_{i}"):
                    st.session_state["fuentes_externas"].pop(i)
                    st.rerun()

# ==========================================================================
# MÓDULO 3: REDACTOR CIENTÍFICO (ESTILO CUADERNO DE INVESTIGACIÓN)
# ==========================================================================
with tab3:
    st.header("✍️ Módulo 3: Redacción Asistida y Citación (estilo NotebookLM + Medline)")

    total_fuentes = len(st.session_state["notebook_context"]) + len(st.session_state["fuentes_externas"])

    if total_fuentes == 0:
        st.info(
            "⚠️ Aún no hay fuentes disponibles. Ve al Módulo 1 (sube y analiza PDFs) "
            "y/o al Módulo 2 (busca y aprueba artículos por API) para poder redactar "
            "con citas Medline/Vancouver."
        )
    else:
        st.success(f"Usando **{total_fuentes}** fuente(s) aprobadas, en este orden numerado:")
        for linea in listar_fuentes_numeradas():
            st.write(linea)

        seccion = st.selectbox(
            "Sección a redactar",
            ["Introducción", "Métodos", "Justificación", "Marco Teórico", "Discusión", "Conclusiones"],
        )
        instrucciones_extra = st.text_area(
            "Instrucciones adicionales (opcional)",
            placeholder="Ej: Enfatizar la brecha de conocimiento en población pediátrica",
        )

        SYSTEM_REDACTOR = """Eres un redactor científico académico especializado en estilo Medline/Vancouver, al estilo de un cuaderno de investigación (NotebookLM).

Reglas estrictas:
1. Usa EXCLUSIVAMENTE la información contenida en las fuentes numeradas que se te entregan (documentos completos del Módulo 1 y resúmenes/abstracts aprobados del Módulo 2). Está PROHIBIDO inventar datos, cifras o afirmaciones que no estén en las fuentes.
2. Basa tus afirmaciones en el contenido COMPLETO de cada fuente (todas las secciones del documento, no solo su resumen inicial), ya que se te entrega el texto íntegro.
3. Cada afirmación factual debe llevar su cita correspondiente insertada en el lugar exacto del texto, en formato de superíndice HTML: <sup>1</sup>, <sup>2</sup>, etc., según el número de fuente. NO uses corchetes [1]; usa siempre la etiqueta <sup>.
4. Si necesitas citar dos fuentes juntas, usa <sup>1,2</sup>.
5. Si la información disponible es insuficiente para cubrir algún punto de la sección solicitada, dilo explícitamente en vez de rellenar con contenido genérico o inventado.
6. Mantén un tono académico, objetivo y en español. No generes tú mismo la lista de referencias al final: eso ya lo construye la plataforma."""

        if st.button("Redactar sección", type="primary"):
            contexto = construir_bloque_contexto()
            prompt_usuario = f"""Redacta la sección "{seccion}" de un artículo científico.

Instrucciones adicionales: {instrucciones_extra or "Ninguna"}

FUENTES DISPONIBLES (usa <sup>n</sup> referenciando el número de cada una, según el orden en que aparecen abajo):

{contexto}"""
            with st.spinner("Redactando con IA..."):
                texto_redactado = llamar_gemini(SYSTEM_REDACTOR, prompt_usuario)

            st.markdown(texto_redactado, unsafe_allow_html=True)

            st.divider()
            st.caption("Bibliografía (formato Vancouver, generada automáticamente — no editada por la IA)")
            for linea in listar_fuentes_numeradas():
                st.write(linea)
