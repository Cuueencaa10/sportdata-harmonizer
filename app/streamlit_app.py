from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN E IMPORTACIÓN DEL MOTOR
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROJECT_ROOT_STRING = str(
    PROJECT_ROOT
)

if PROJECT_ROOT_STRING not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT_STRING,
    )


from src import harmonizer_core as core


st.set_page_config(
    page_title=(
        "SportData Harmonizer"
    ),
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. FUNCIONES AUXILIARES DE LA INTERFAZ
# ============================================================

@st.cache_data(
    show_spinner=False,
)
def process_payloads_cached(
    payloads: tuple[
        tuple[str, bytes],
        ...
    ],
) -> dict:
    """
    Ejecuta el motor y almacena temporalmente el resultado.
    """
    return core.process_payloads(
        payloads
    )


def collect_sample_payloads() -> list[
    tuple[str, bytes]
]:
    """
    Recupera los CSV incluidos en data/raw.
    """
    raw_directory = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    if not raw_directory.exists():
        return []

    sample_paths = sorted(
        raw_directory.glob("*.csv")
    )

    return [
        (
            path.name,
            path.read_bytes(),
        )
        for path in sample_paths
    ]


def build_payload_signature(
    payloads: list[
        tuple[str, bytes]
    ],
) -> tuple[
    tuple[str, str],
    ...
]:
    """
    Genera una firma para detectar cambios en los archivos.
    """
    return tuple(
        (
            filename,
            core.calculate_sha256(
                content
            ),
        )
        for filename, content
        in payloads
    )


def safe_dataframe(
    dataframe,
) -> pd.DataFrame:
    """
    Devuelve siempre un DataFrame válido para la interfaz.
    """
    if isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return dataframe

    return pd.DataFrame()


def dataframe_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convierte un DataFrame en CSV UTF-8 con BOM.
    """
    return core.dataframe_to_csv_bytes(
        dataframe
    )


def build_results_zip(
    results: dict,
) -> bytes:
    """
    Crea un ZIP con los principales productos del proceso.
    """
    zip_buffer = BytesIO()

    inventory = safe_dataframe(
        results.get(
            "inventario"
        )
    )

    profile = safe_dataframe(
        results.get(
            "perfil_archivos"
        )
    )

    schema_presence = safe_dataframe(
        results.get(
            "presencia_columnas"
        )
    )

    inferred_types = safe_dataframe(
        results.get(
            "tipos_inferidos"
        )
    )

    issues = safe_dataframe(
        results.get(
            "incidencias"
        )
    )

    coverage = safe_dataframe(
        results.get(
            "cobertura"
        )
    )

    quality_summary = safe_dataframe(
        results.get(
            "resumen_calidad"
        )
    )

    master = safe_dataframe(
        results.get(
            "dataset_maestro"
        )
    )

    harmonization_summary = safe_dataframe(
        results.get(
            "resumen_armonizacion"
        )
    )

    rejected_files = safe_dataframe(
        results.get(
            "archivos_rechazados"
        )
    )

    method_comparison = safe_dataframe(
        results.get(
            "comparacion_metodos"
        )
    )

    manifest = results.get(
        "manifiesto",
        {},
    )

    files_to_include = {
        "dataset_maestro.csv": master,
        "inventario_archivos.csv": inventory,
        "perfil_archivos.csv": profile,
        "presencia_columnas.csv": (
            schema_presence.reset_index()
        ),
        "tipos_inferidos.csv": (
            inferred_types.reset_index()
        ),
        "incidencias_calidad.csv": issues,
        "cobertura_variables.csv": coverage,
        "resumen_calidad.csv": quality_summary,
        "resumen_armonizacion.csv": (
            harmonization_summary
        ),
        "archivos_rechazados.csv": (
            rejected_files
        ),
        "comparacion_metodos.csv": (
            method_comparison
        ),
    }

    with ZipFile(
        zip_buffer,
        mode="w",
        compression=ZIP_DEFLATED,
    ) as zip_file:
        for filename, dataframe in (
            files_to_include.items()
        ):
            zip_file.writestr(
                filename,
                dataframe_csv_bytes(
                    dataframe
                ),
            )

        zip_file.writestr(
            "manifiesto_proceso.json",
            core.dictionary_to_json_bytes(
                manifest
            ),
        )

        zip_file.writestr(
            "LEEME.txt",
            (
                "SportData Harmonizer\n"
                "====================\n\n"
                "Este paquete contiene el dataset maestro, "
                "los controles de calidad, la cobertura, "
                "los archivos rechazados y el manifiesto "
                "del procesamiento realizado en Streamlit.\n\n"
                "Los archivos originales no han sido "
                "modificados ni incluidos en este paquete.\n"
            ).encode(
                "utf-8"
            ),
        )

    return zip_buffer.getvalue()


def format_integer(
    value,
) -> str:
    """
    Formatea números enteros con punto de millares.
    """
    try:
        return f"{int(value):,}".replace(
            ",",
            ".",
        )
    except (
        TypeError,
        ValueError,
    ):
        return "0"


# ============================================================
# 3. CABECERA
# ============================================================

st.title(
    "⚽ SportData Harmonizer"
)

st.subheader(
    "Validación, armonización y consolidación "
    "auditable de archivos deportivos"
)

st.markdown(
    """
    La aplicación revisa múltiples archivos CSV antes de
    consolidarlos. Detecta diferencias de esquema, problemas
    de calidad, limitaciones de cobertura e incidencias
    deportivas, manteniendo la trazabilidad de cada registro.
    """
)

st.caption(
    "Los archivos cargados se procesan temporalmente. "
    "No se modifican ni se almacenan de forma permanente."
)


# ============================================================
# 4. BARRA LATERAL Y SELECCIÓN DE ARCHIVOS
# ============================================================

st.sidebar.header(
    "Configuración del proceso"
)

input_mode = st.sidebar.radio(
    "Modo de entrada",
    options=[
        "Muestra del proyecto",
        "Archivos propios",
    ],
    index=0,
)

payloads: list[
    tuple[str, bytes]
] = []


if input_mode == "Muestra del proyecto":
    payloads = (
        collect_sample_payloads()
    )

    if payloads:
        st.sidebar.success(
            f"Se han localizado "
            f"{len(payloads)} archivos de muestra."
        )

        with st.sidebar.expander(
            "Archivos incluidos",
            expanded=False,
        ):
            for filename, _ in payloads:
                st.write(
                    f"• {filename}"
                )

    else:
        st.sidebar.error(
            "No se han encontrado CSV en data/raw."
        )


else:
    uploaded_files = (
        st.sidebar.file_uploader(
            "Selecciona uno o varios CSV",
            type=["csv"],
            accept_multiple_files=True,
            help=(
                "Los archivos deben contener como mínimo "
                "Date, HomeTeam, AwayTeam, FTHG, FTAG y FTR."
            ),
        )
    )

    payloads = [
        (
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
        for uploaded_file
        in uploaded_files
    ]

    if payloads:
        st.sidebar.info(
            f"Archivos seleccionados: "
            f"{len(payloads)}"
        )


st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "La competición y la temporada se intentan inferir "
    "a partir del nombre del archivo. Cuando no es posible, "
    "se registran como «No indicada»."
)


# ============================================================
# 5. CONTROL DEL ESTADO DE LA CARGA
# ============================================================

current_signature = (
    build_payload_signature(
        payloads
    )
)

stored_signature = (
    st.session_state.get(
        "payload_signature"
    )
)

if (
    stored_signature is not None
    and stored_signature
    != current_signature
):
    st.session_state.pop(
        "harmonizer_results",
        None,
    )

    st.session_state.pop(
        "payload_signature",
        None,
    )


process_button = (
    st.sidebar.button(
        "Procesar archivos",
        type="primary",
        use_container_width=True,
    )
)


if process_button:
    if not payloads:
        st.sidebar.error(
            "Debes seleccionar al menos un archivo."
        )

    else:
        with st.spinner(
            "Validando, armonizando y consolidando..."
        ):
            results = (
                process_payloads_cached(
                    tuple(payloads)
                )
            )

        st.session_state[
            "harmonizer_results"
        ] = results

        st.session_state[
            "payload_signature"
        ] = current_signature


results = st.session_state.get(
    "harmonizer_results"
)


# ============================================================
# 6. ESTADO INICIAL SIN PROCESAMIENTO
# ============================================================

if results is None:
    st.info(
        "Selecciona el modo de entrada y pulsa "
        "«Procesar archivos» para comenzar."
    )

    st.markdown(
        "### Esquema mínimo admitido"
    )

    minimum_schema = pd.DataFrame(
        {
            "columna": [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
            ],
            "descripcion": [
                "Fecha del partido",
                "Equipo local",
                "Equipo visitante",
                "Goles finales del local",
                "Goles finales del visitante",
                "Resultado: H, D o A",
            ],
        }
    )

    st.dataframe(
        minimum_schema,
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


# ============================================================
# 7. RECUPERACIÓN DE RESULTADOS
# ============================================================

inventory = safe_dataframe(
    results.get(
        "inventario"
    )
)

profile = safe_dataframe(
    results.get(
        "perfil_archivos"
    )
)

schema_presence = safe_dataframe(
    results.get(
        "presencia_columnas"
    )
)

inferred_types = safe_dataframe(
    results.get(
        "tipos_inferidos"
    )
)

issues = safe_dataframe(
    results.get(
        "incidencias"
    )
)

coverage = safe_dataframe(
    results.get(
        "cobertura"
    )
)

quality_summary = safe_dataframe(
    results.get(
        "resumen_calidad"
    )
)

master = safe_dataframe(
    results.get(
        "dataset_maestro"
    )
)

harmonization_summary = safe_dataframe(
    results.get(
        "resumen_armonizacion"
    )
)

rejected_files = safe_dataframe(
    results.get(
        "archivos_rechazados"
    )
)

method_comparison = safe_dataframe(
    results.get(
        "comparacion_metodos"
    )
)

manifest = results.get(
    "manifiesto",
    {},
)


received_files = int(
    len(inventory)
)

loaded_files = (
    int(
        inventory[
            "estado_carga"
        ]
        .eq("OK")
        .sum()
    )
    if (
        not inventory.empty
        and "estado_carga"
        in inventory.columns
    )
    else 0
)

harmonized_files = int(
    len(
        results.get(
            "dataframes_armonizados",
            {},
        )
    )
)

rejected_file_count = int(
    len(rejected_files)
)

master_rows = int(
    len(master)
)

critical_issues = (
    int(
        issues[
            "severidad"
        ]
        .eq("CRITICA")
        .sum()
    )
    if (
        not issues.empty
        and "severidad"
        in issues.columns
    )
    else 0
)

error_issues = (
    int(
        issues[
            "severidad"
        ]
        .eq("ERROR")
        .sum()
    )
    if (
        not issues.empty
        and "severidad"
        in issues.columns
    )
    else 0
)

warning_issues = (
    int(
        issues[
            "severidad"
        ]
        .eq("ADVERTENCIA")
        .sum()
    )
    if (
        not issues.empty
        and "severidad"
        in issues.columns
    )
    else 0
)

warning_records = (
    int(
        master[
            "numero_advertencias"
        ]
        .gt(0)
        .sum()
    )
    if (
        not master.empty
        and "numero_advertencias"
        in master.columns
    )
    else 0
)


# ============================================================
# 8. MENSAJE GLOBAL DEL PROCESO
# ============================================================

if master.empty:
    st.error(
        "No se ha generado un dataset maestro. "
        "Revisa los errores y los archivos rechazados."
    )

elif (
    critical_issues > 0
    or error_issues > 0
):
    st.warning(
        "El proceso ha finalizado, pero algunos archivos "
        "contienen errores y no se han consolidado."
    )

elif warning_issues > 0:
    st.success(
        "La consolidación se ha completado. "
        "Existen advertencias que deben revisarse, "
        "pero no se han corregido ni excluido silenciosamente."
    )

else:
    st.success(
        "La consolidación se ha completado sin errores "
        "ni advertencias."
    )


# ============================================================
# 9. PESTAÑAS PRINCIPALES
# ============================================================

(
    summary_tab,
    structure_tab,
    quality_tab,
    master_tab,
    download_tab,
) = st.tabs(
    [
        "Resumen",
        "Inventario y esquema",
        "Calidad",
        "Dataset maestro",
        "Comparación y descargas",
    ]
)


# ============================================================
# 10. PESTAÑA DE RESUMEN
# ============================================================

with summary_tab:
    metric_columns = st.columns(
        6
    )

    metric_columns[0].metric(
        "Archivos recibidos",
        format_integer(
            received_files
        ),
    )

    metric_columns[1].metric(
        "Archivos legibles",
        format_integer(
            loaded_files
        ),
    )

    metric_columns[2].metric(
        "Archivos armonizados",
        format_integer(
            harmonized_files
        ),
    )

    metric_columns[3].metric(
        "Registros consolidados",
        format_integer(
            master_rows
        ),
    )

    metric_columns[4].metric(
        "Advertencias",
        format_integer(
            warning_issues
        ),
    )

    metric_columns[5].metric(
        "Registros para revisión",
        format_integer(
            warning_records
        ),
    )

    st.markdown(
        "### Estado de calidad por archivo"
    )

    if quality_summary.empty:
        st.info(
            "No existe un resumen de calidad."
        )
    else:
        st.dataframe(
            quality_summary,
            use_container_width=True,
            hide_index=True,
        )

    if not profile.empty:
        coverage_chart_data = (
            profile[
                [
                    "archivo_origen",
                    "porcentaje_ausencia",
                ]
            ]
            .copy()
        )

        coverage_chart_data[
            "cobertura_pct"
        ] = (
            100
            - coverage_chart_data[
                "porcentaje_ausencia"
            ]
        )

        coverage_figure = px.bar(
            coverage_chart_data,
            x="archivo_origen",
            y="cobertura_pct",
            text_auto=".1f",
            labels={
                "archivo_origen": (
                    "Archivo"
                ),
                "cobertura_pct": (
                    "Cobertura (%)"
                ),
            },
            title=(
                "Cobertura global de las "
                "columnas recibidas"
            ),
        )

        coverage_figure.update_yaxes(
            range=[
                0,
                105,
            ]
        )

        coverage_figure.update_layout(
            xaxis_tickangle=-25,
        )

        st.plotly_chart(
            coverage_figure,
            use_container_width=True,
        )

    st.markdown(
        "### Resultado de la armonización"
    )

    if harmonization_summary.empty:
        st.info(
            "Ningún archivo ha podido armonizarse."
        )
    else:
        st.dataframe(
            harmonization_summary,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 11. PESTAÑA DE INVENTARIO Y ESQUEMA
# ============================================================

with structure_tab:
    st.markdown(
        "### Inventario de carga"
    )

    st.dataframe(
        inventory,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Perfil estructural"
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Matriz de presencia de columnas",
        expanded=False,
    ):
        st.dataframe(
            schema_presence,
            use_container_width=True,
        )

    with st.expander(
        "Matriz de tipos inferidos",
        expanded=False,
    ):
        st.dataframe(
            inferred_types,
            use_container_width=True,
        )


# ============================================================
# 12. PESTAÑA DE CALIDAD
# ============================================================

with quality_tab:
    quality_metric_columns = (
        st.columns(4)
    )

    quality_metric_columns[0].metric(
        "Críticas",
        format_integer(
            critical_issues
        ),
    )

    quality_metric_columns[1].metric(
        "Errores",
        format_integer(
            error_issues
        ),
    )

    quality_metric_columns[2].metric(
        "Advertencias",
        format_integer(
            warning_issues
        ),
    )

    quality_metric_columns[3].metric(
        "Archivos rechazados",
        format_integer(
            rejected_file_count
        ),
    )

    st.markdown(
        "### Incidencias registradas"
    )

    if issues.empty:
        st.success(
            "No se han detectado incidencias."
        )

    else:
        filter_columns = st.columns(
            2
        )

        severity_options = sorted(
            issues[
                "severidad"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_severities = (
            filter_columns[0]
            .multiselect(
                "Severidad",
                options=severity_options,
                default=severity_options,
            )
        )

        rule_options = sorted(
            issues[
                "regla"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_rules = (
            filter_columns[1]
            .multiselect(
                "Regla",
                options=rule_options,
                default=rule_options,
            )
        )

        filtered_issues = (
            issues.loc[
                issues[
                    "severidad"
                ].isin(
                    selected_severities
                )
                & issues[
                    "regla"
                ].isin(
                    selected_rules
                )
            ]
        )

        st.dataframe(
            filtered_issues,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### Cobertura de variables"
    )

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Archivos rechazados"
    )

    if rejected_files.empty:
        st.success(
            "No se ha rechazado ningún archivo."
        )
    else:
        st.dataframe(
            rejected_files,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 13. PESTAÑA DEL DATASET MAESTRO
# ============================================================

with master_tab:
    if master.empty:
        st.warning(
            "No existe un dataset maestro para mostrar."
        )

    else:
        master_metric_columns = (
            st.columns(4)
        )

        master_metric_columns[0].metric(
            "Filas",
            format_integer(
                len(master)
            ),
        )

        master_metric_columns[1].metric(
            "Columnas",
            format_integer(
                master.shape[1]
            ),
        )

        master_metric_columns[2].metric(
            "Archivos de origen",
            format_integer(
                master[
                    "archivo_origen"
                ]
                .nunique()
            ),
        )

        master_metric_columns[3].metric(
            "Registros advertidos",
            format_integer(
                warning_records
            ),
        )

        master_filter_columns = (
            st.columns(2)
        )

        file_options = sorted(
            master[
                "archivo_origen"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_files = (
            master_filter_columns[0]
            .multiselect(
                "Archivo de origen",
                options=file_options,
                default=file_options,
            )
        )

        status_options = sorted(
            master[
                "estado_registro"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_statuses = (
            master_filter_columns[1]
            .multiselect(
                "Estado del registro",
                options=status_options,
                default=status_options,
            )
        )

        filtered_master = (
            master.loc[
                master[
                    "archivo_origen"
                ].isin(
                    selected_files
                )
                & master[
                    "estado_registro"
                ].isin(
                    selected_statuses
                )
            ]
        )

        show_all_columns = st.toggle(
            "Mostrar las 37 columnas",
            value=False,
        )

        operational_columns = [
            "id_registro",
            "fecha",
            "competicion",
            "temporada",
            "equipo_local",
            "equipo_visitante",
            "goles_local_ft",
            "goles_visitante_ft",
            "resultado_ft",
            "datos_descanso_disponibles",
            "estadisticas_partido_disponibles",
            "arbitro_disponible",
            "numero_advertencias",
            "reglas_incidencia",
            "estado_registro",
            "archivo_origen",
            "fila_csv_origen",
        ]

        columns_to_display = (
            master.columns.tolist()
            if show_all_columns
            else [
                column
                for column
                in operational_columns
                if column
                in master.columns
            ]
        )

        st.dataframe(
            filtered_master[
                columns_to_display
            ],
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        st.caption(
            f"Registros mostrados: "
            f"{format_integer(len(filtered_master))}"
        )

        st.markdown(
            "### Registros con advertencias"
        )

        warning_dataframe = (
            master.loc[
                master[
                    "numero_advertencias"
                ].gt(0)
            ]
        )

        if warning_dataframe.empty:
            st.success(
                "No existen registros con advertencias."
            )

        else:
            st.dataframe(
                warning_dataframe[
                    [
                        "id_registro",
                        "fecha",
                        "equipo_local",
                        "equipo_visitante",
                        "numero_advertencias",
                        "reglas_incidencia",
                        "archivo_origen",
                        "fila_csv_origen",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 14. PESTAÑA DE COMPARACIÓN Y DESCARGAS
# ============================================================

with download_tab:
    st.markdown(
        "### Comparación con la unión convencional"
    )

    st.dataframe(
        method_comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Archivos descargables"
    )

    download_columns_1 = (
        st.columns(3)
    )

    download_columns_1[0].download_button(
        label="Descargar dataset maestro",
        data=dataframe_csv_bytes(
            master
        ),
        file_name=(
            "sportdata_harmonizer_"
            "dataset_maestro.csv"
        ),
        mime="text/csv",
        use_container_width=True,
        disabled=master.empty,
    )

    download_columns_1[1].download_button(
        label="Descargar incidencias",
        data=dataframe_csv_bytes(
            issues
        ),
        file_name=(
            "sportdata_harmonizer_"
            "incidencias.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    download_columns_1[2].download_button(
        label="Descargar inventario",
        data=dataframe_csv_bytes(
            inventory
        ),
        file_name=(
            "sportdata_harmonizer_"
            "inventario.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    download_columns_2 = (
        st.columns(3)
    )

    download_columns_2[0].download_button(
        label="Descargar resumen de calidad",
        data=dataframe_csv_bytes(
            quality_summary
        ),
        file_name=(
            "sportdata_harmonizer_"
            "resumen_calidad.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    download_columns_2[1].download_button(
        label="Descargar manifiesto",
        data=(
            core.dictionary_to_json_bytes(
                manifest
            )
        ),
        file_name=(
            "sportdata_harmonizer_"
            "manifiesto.json"
        ),
        mime="application/json",
        use_container_width=True,
    )

    download_columns_2[2].download_button(
        label="Descargar paquete completo",
        data=build_results_zip(
            results
        ),
        file_name=(
            "sportdata_harmonizer_"
            "resultados.zip"
        ),
        mime="application/zip",
        use_container_width=True,
    )

    st.caption(
        "El paquete completo incluye el dataset maestro, "
        "inventario, perfiles, incidencias, cobertura, "
        "resúmenes, comparación de métodos y manifiesto."
    )


# ============================================================
# 15. PIE DE PÁGINA
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "SportData Harmonizer · Actividad Colaborativa del "
    "Módulo 8 · Aitor Cuenca Retuerto"
)
