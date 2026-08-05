from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Iterable
import hashlib
import json
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


# ============================================================
# 1. ESQUEMA DE ORIGEN Y ESQUEMA CANÓNICO
# ============================================================

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]

OPTIONAL_COLUMNS = [
    "HTHG",
    "HTAG",
    "HTR",
    "Referee",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

SOURCE_COLUMNS = (
    REQUIRED_COLUMNS
    + OPTIONAL_COLUMNS
)

COLUMN_MAPPING = {
    "Date": "fecha",
    "HomeTeam": "equipo_local",
    "AwayTeam": "equipo_visitante",
    "FTHG": "goles_local_ft",
    "FTAG": "goles_visitante_ft",
    "FTR": "resultado_ft",
    "HTHG": "goles_local_ht",
    "HTAG": "goles_visitante_ht",
    "HTR": "resultado_ht",
    "Referee": "arbitro",
    "HS": "tiros_local",
    "AS": "tiros_visitante",
    "HST": "tiros_puerta_local",
    "AST": "tiros_puerta_visitante",
    "HF": "faltas_local",
    "AF": "faltas_visitante",
    "HC": "corners_local",
    "AC": "corners_visitante",
    "HY": "amarillas_local",
    "AY": "amarillas_visitante",
    "HR": "rojas_local",
    "AR": "rojas_visitante",
}

NUMERIC_SOURCE_COLUMNS = [
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

CANONICAL_NUMERIC_COLUMNS = [
    COLUMN_MAPPING[column]
    for column in NUMERIC_SOURCE_COLUMNS
]

CANONICAL_TEXT_COLUMNS = [
    "equipo_local",
    "equipo_visitante",
    "resultado_ft",
    "resultado_ht",
    "arbitro",
]

MATCH_STAT_COLUMNS = [
    "tiros_local",
    "tiros_visitante",
    "tiros_puerta_local",
    "tiros_puerta_visitante",
    "faltas_local",
    "faltas_visitante",
    "corners_local",
    "corners_visitante",
    "amarillas_local",
    "amarillas_visitante",
    "rojas_local",
    "rojas_visitante",
]

VALID_RESULTS = {
    "H",
    "D",
    "A",
}

CANONICAL_COLUMN_ORDER = [
    "id_registro",
    "fecha",
    "competicion",
    "temporada",
    "equipo_local",
    "equipo_visitante",
    "goles_local_ft",
    "goles_visitante_ft",
    "resultado_ft",
    "goles_local_ht",
    "goles_visitante_ht",
    "resultado_ht",
    "arbitro",
    "tiros_local",
    "tiros_visitante",
    "tiros_puerta_local",
    "tiros_puerta_visitante",
    "faltas_local",
    "faltas_visitante",
    "corners_local",
    "corners_visitante",
    "amarillas_local",
    "amarillas_visitante",
    "rojas_local",
    "rojas_visitante",
    "datos_descanso_disponibles",
    "estadisticas_partido_disponibles",
    "arbitro_disponible",
    "numero_incidencias",
    "numero_advertencias",
    "reglas_incidencia",
    "estado_registro",
    "codigo_fuente",
    "archivo_origen",
    "indice_origen",
    "fila_csv_origen",
    "sha256_archivo",
]

ISSUE_COLUMNS = [
    "archivo_origen",
    "indice_origen",
    "fila_csv",
    "fecha_original",
    "equipo_local",
    "equipo_visitante",
    "regla",
    "categoria",
    "severidad",
    "columna",
    "valor_observado",
    "valor_esperado",
    "descripcion",
]


# ============================================================
# 2. UTILIDADES GENERALES
# ============================================================

def calculate_sha256(
    content: bytes,
) -> str:
    """Calcula el hash SHA-256 de un contenido binario."""
    return hashlib.sha256(content).hexdigest()


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """Convierte a string anulable y elimina espacios externos."""
    return (
        series
        .astype("string")
        .str.strip()
    )


def parse_dates(
    series: pd.Series,
) -> pd.Series:
    """Interpreta fechas mixtas priorizando día-mes-año."""
    return pd.to_datetime(
        normalize_text(series),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )


def derive_result(
    home_goals: pd.Series,
    away_goals: pd.Series,
) -> pd.Series:
    """Deriva H, D o A a partir de los goles."""
    result = pd.Series(
        pd.NA,
        index=home_goals.index,
        dtype="string",
    )

    valid_mask = (
        home_goals.notna()
        & away_goals.notna()
    )

    result.loc[
        valid_mask
        & home_goals.gt(away_goals)
    ] = "H"

    result.loc[
        valid_mask
        & home_goals.eq(away_goals)
    ] = "D"

    result.loc[
        valid_mask
        & home_goals.lt(away_goals)
    ] = "A"

    return result


def safe_value(
    dataframe: pd.DataFrame,
    row_index,
    column: str,
):
    """Recupera un valor sin provocar errores."""
    if (
        row_index is None
        or column not in dataframe.columns
        or row_index not in dataframe.index
    ):
        return pd.NA

    return dataframe.at[
        row_index,
        column,
    ]


def infer_context_from_filename(
    filename: str,
) -> tuple[str, str]:
    """Infiere competición y temporada cuando el nombre lo permite."""
    stem = Path(filename).stem.lower()

    if "premier" in stem:
        competition = "Premier League"
    elif "la_liga" in stem or "laliga" in stem:
        competition = "La Liga"
    elif "bundesliga" in stem:
        competition = "Bundesliga"
    else:
        competition = "No indicada"

    full_year_match = re.search(
        r"((?:19|20)\d{2})[_-]((?:19|20)\d{2})",
        stem,
    )

    if full_year_match:
        first_year = full_year_match.group(1)
        second_year = full_year_match.group(2)
        season = (
            f"{first_year}/"
            f"{second_year[-2:]}"
        )
        return competition, season

    compact_match = re.search(
        r"(?:season[_-])?(\d{2})(\d{2})$",
        stem,
    )

    if compact_match:
        first_short = int(
            compact_match.group(1)
        )
        second_short = compact_match.group(2)

        first_year = (
            1900 + first_short
            if first_short >= 80
            else 2000 + first_short
        )

        season = (
            f"{first_year}/"
            f"{second_short}"
        )
        return competition, season

    return competition, "No indicada"


def build_source_code(
    filename: str,
) -> str:
    """Construye un código legible a partir del nombre del archivo."""
    stem = Path(filename).stem.upper()

    source_code = re.sub(
        r"[^A-Z0-9]+",
        "_",
        stem,
    ).strip("_")

    return (
        source_code[:48]
        if source_code
        else "ARCHIVO_SIN_NOMBRE"
    )


def build_record_id(
    filename: str,
    original_index: int,
    date_value,
    home_team,
    away_team,
) -> str:
    """Genera un identificador reproducible por registro."""
    identifier_text = "|".join(
        [
            filename,
            str(original_index),
            "" if pd.isna(date_value) else str(date_value),
            "" if pd.isna(home_team) else str(home_team),
            "" if pd.isna(away_team) else str(away_team),
        ]
    )

    return hashlib.sha256(
        identifier_text.encode("utf-8")
    ).hexdigest()[:24]


def read_csv_content(
    content: bytes,
) -> tuple[pd.DataFrame, str]:
    """Lee un CSV probando codificaciones habituales."""
    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                low_memory=False,
            )

            return dataframe, encoding

        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise ValueError(
        "No se pudo interpretar el contenido como CSV."
    )


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """Convierte un DataFrame en un CSV descargable."""
    return dataframe.to_csv(
        index=False,
        date_format="%Y-%m-%d",
    ).encode("utf-8-sig")


def dictionary_to_json_bytes(
    payload: dict,
) -> bytes:
    """Convierte un diccionario en JSON descargable."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    ).encode("utf-8")


# ============================================================
# 3. CARGA E INVENTARIO
# ============================================================

def load_payloads(
    payloads: Iterable[
        tuple[str, bytes]
    ],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Carga una colección de nombres y contenidos binarios.

    Los nombres duplicados, archivos vacíos o extensiones
    distintas de CSV se registran como errores.
    """
    payload_list = list(payloads)

    normalized_names = [
        str(name).casefold()
        for name, _ in payload_list
    ]

    duplicated_names = {
        name
        for name, count
        in Counter(normalized_names).items()
        if count > 1
    }

    dataframes = {}
    inventory_records = []

    for filename, content in payload_list:
        record = {
            "archivo_origen": filename,
            "extension": Path(
                filename
            ).suffix.lower(),
            "tamano_bytes": (
                len(content)
                if isinstance(
                    content,
                    bytes,
                )
                else 0
            ),
            "sha256": "",
            "codificacion": "",
            "filas": pd.NA,
            "columnas": pd.NA,
            "estado_carga": "ERROR",
            "mensaje": "",
        }

        try:
            if (
                filename.casefold()
                in duplicated_names
            ):
                raise ValueError(
                    "El nombre del archivo está duplicado."
                )

            if (
                Path(filename).suffix.lower()
                != ".csv"
            ):
                raise ValueError(
                    "La extensión admitida es CSV."
                )

            if not isinstance(
                content,
                bytes,
            ):
                raise TypeError(
                    "El contenido debe recibirse como bytes."
                )

            if not content.strip():
                raise ValueError(
                    "El archivo está vacío."
                )

            dataframe, encoding = (
                read_csv_content(content)
            )

            if dataframe.empty:
                raise ValueError(
                    "El CSV no contiene registros."
                )

            dataframes[filename] = dataframe

            record.update(
                {
                    "sha256": calculate_sha256(
                        content
                    ),
                    "codificacion": encoding,
                    "filas": int(
                        dataframe.shape[0]
                    ),
                    "columnas": int(
                        dataframe.shape[1]
                    ),
                    "estado_carga": "OK",
                    "mensaje": (
                        "Archivo leído correctamente."
                    ),
                }
            )

        except (
            UnicodeDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            record["mensaje"] = str(exc)

        inventory_records.append(
            record
        )

    return (
        dataframes,
        pd.DataFrame(
            inventory_records
        ),
    )


# ============================================================
# 4. PERFILADO ESTRUCTURAL
# ============================================================

def profile_dataframes(
    dataframes: dict[
        str,
        pd.DataFrame,
    ],
) -> dict[str, pd.DataFrame]:
    """Genera inventario y matrices de esquema y tipos."""
    profile_records = []

    all_columns = list(
        dict.fromkeys(
            column
            for dataframe
            in dataframes.values()
            for column
            in dataframe.columns
        )
    )

    for filename, dataframe in (
        dataframes.items()
    ):
        total_cells = int(
            dataframe.size
        )

        missing_cells = int(
            dataframe.isna()
            .sum()
            .sum()
        )

        empty_columns = [
            str(column)
            for column in dataframe.columns
            if dataframe[column]
            .isna()
            .all()
        ]

        profile_records.append(
            {
                "archivo_origen": filename,
                "filas": int(
                    dataframe.shape[0]
                ),
                "columnas": int(
                    dataframe.shape[1]
                ),
                "celdas_totales": total_cells,
                "celdas_ausentes": missing_cells,
                "porcentaje_ausencia": round(
                    (
                        missing_cells
                        / total_cells
                        * 100
                    )
                    if total_cells
                    else 0.0,
                    2,
                ),
                "filas_duplicadas_exactas": int(
                    dataframe.duplicated()
                    .sum()
                ),
                "encabezados_duplicados": int(
                    dataframe.columns
                    .duplicated()
                    .sum()
                ),
                "columnas_totalmente_vacias": int(
                    len(empty_columns)
                ),
                "lista_columnas_vacias": (
                    ", ".join(
                        empty_columns
                    )
                ),
            }
        )

    schema_presence = pd.DataFrame(
        {
            filename: [
                column in dataframe.columns
                for column in all_columns
            ]
            for filename, dataframe
            in dataframes.items()
        },
        index=all_columns,
    )

    schema_presence.index.name = (
        "columna"
    )

    inferred_types = pd.DataFrame(
        {
            filename: [
                (
                    str(
                        dataframe[
                            column
                        ].dtype
                    )
                    if column
                    in dataframe.columns
                    else "AUSENTE"
                )
                for column in all_columns
            ]
            for filename, dataframe
            in dataframes.items()
        },
        index=all_columns,
    )

    inferred_types.index.name = (
        "columna"
    )

    return {
        "perfil_archivos": pd.DataFrame(
            profile_records
        ),
        "presencia_columnas": (
            schema_presence
        ),
        "tipos_inferidos": (
            inferred_types
        ),
    }


# ============================================================
# 5. VALIDACIÓN DE CALIDAD
# ============================================================

def register_issue(
    records: list[dict],
    dataframe: pd.DataFrame,
    filename: str,
    rule: str,
    category: str,
    severity: str,
    description: str,
    row_index=None,
    column: str = "",
    observed_value=pd.NA,
    expected_value=pd.NA,
) -> None:
    """Registra una incidencia con trazabilidad."""
    if row_index is None:
        original_index = pd.NA
        csv_row = pd.NA
    else:
        original_index = int(
            row_index
        )
        csv_row = int(
            row_index
        ) + 2

    records.append(
        {
            "archivo_origen": filename,
            "indice_origen": (
                original_index
            ),
            "fila_csv": csv_row,
            "fecha_original": safe_value(
                dataframe,
                row_index,
                "Date",
            ),
            "equipo_local": safe_value(
                dataframe,
                row_index,
                "HomeTeam",
            ),
            "equipo_visitante": safe_value(
                dataframe,
                row_index,
                "AwayTeam",
            ),
            "regla": rule,
            "categoria": category,
            "severidad": severity,
            "columna": column,
            "valor_observado": (
                observed_value
            ),
            "valor_esperado": (
                expected_value
            ),
            "descripcion": description,
        }
    )


def validate_dataframes(
    dataframes: dict[
        str,
        pd.DataFrame,
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Aplica las reglas de calidad sobre cada archivo."""
    issue_records = []
    coverage_records = []

    for filename, dataframe in (
        dataframes.items()
    ):
        # ----------------------------------------------------
        # Cobertura del esquema
        # ----------------------------------------------------

        for column in SOURCE_COLUMNS:
            variable_role = (
                "OBLIGATORIA"
                if column
                in REQUIRED_COLUMNS
                else "OPCIONAL"
            )

            if column not in dataframe.columns:
                coverage_status = (
                    "AUSENTE"
                )
                non_null_count = 0
                null_count = int(
                    len(dataframe)
                )

                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule="COLUMNA_AUSENTE",
                    category="ESQUEMA",
                    severity=(
                        "CRITICA"
                        if variable_role
                        == "OBLIGATORIA"
                        else "INFORMACION"
                    ),
                    description=(
                        f"La columna {column} "
                        "no existe en el archivo."
                    ),
                    column=column,
                    observed_value="AUSENTE",
                    expected_value=(
                        variable_role
                    ),
                )

            else:
                series = dataframe[
                    column
                ]

                non_null_count = int(
                    series.notna().sum()
                )

                null_count = int(
                    series.isna().sum()
                )

                if non_null_count == 0:
                    coverage_status = (
                        "NO_DISPONIBLE"
                    )
                elif null_count == 0:
                    coverage_status = (
                        "COMPLETA"
                    )
                else:
                    coverage_status = (
                        "PARCIAL"
                    )

            coverage_records.append(
                {
                    "archivo_origen": filename,
                    "columna": column,
                    "rol_variable": (
                        variable_role
                    ),
                    "estado_cobertura": (
                        coverage_status
                    ),
                    "valores_no_nulos": (
                        non_null_count
                    ),
                    "valores_nulos": (
                        null_count
                    ),
                    "porcentaje_nulos": round(
                        (
                            null_count
                            / len(dataframe)
                            * 100
                        )
                        if len(dataframe)
                        else 0.0,
                        2,
                    ),
                }
            )

        missing_required_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column
            not in dataframe.columns
        ]

        if missing_required_columns:
            continue

        # ----------------------------------------------------
        # Series auxiliares
        # ----------------------------------------------------

        parsed_dates = parse_dates(
            dataframe["Date"]
        )

        home_team = normalize_text(
            dataframe["HomeTeam"]
        )

        away_team = normalize_text(
            dataframe["AwayTeam"]
        )

        full_time_result = (
            normalize_text(
                dataframe["FTR"]
            )
            .str.upper()
        )

        half_time_result = (
            normalize_text(
                dataframe["HTR"]
            )
            .str.upper()
            if "HTR"
            in dataframe.columns
            else pd.Series(
                pd.NA,
                index=dataframe.index,
                dtype="string",
            )
        )

        numeric_data = {
            column: pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
            for column
            in NUMERIC_SOURCE_COLUMNS
            if column
            in dataframe.columns
        }

        # ----------------------------------------------------
        # Valores esenciales ausentes
        # ----------------------------------------------------

        for column in REQUIRED_COLUMNS:
            series = dataframe[
                column
            ]

            if column in {
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTR",
            }:
                normalized_series = (
                    normalize_text(series)
                )

                missing_mask = (
                    series.isna()
                    | normalized_series.eq("")
                )

            else:
                missing_mask = (
                    series.isna()
                )

            for row_index in dataframe.index[
                missing_mask.fillna(True)
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule=(
                        "VALOR_OBLIGATORIO_AUSENTE"
                    ),
                    category="COMPLETITUD",
                    severity="ERROR",
                    description=(
                        "Falta un valor esencial "
                        "del partido."
                    ),
                    row_index=row_index,
                    column=column,
                    observed_value=safe_value(
                        dataframe,
                        row_index,
                        column,
                    ),
                    expected_value=(
                        "VALOR NO NULO"
                    ),
                )

        # ----------------------------------------------------
        # Fechas
        # ----------------------------------------------------

        invalid_date_mask = (
            dataframe["Date"].notna()
            & parsed_dates.isna()
        )

        for row_index in dataframe.index[
            invalid_date_mask
        ]:
            register_issue(
                records=issue_records,
                dataframe=dataframe,
                filename=filename,
                rule="FECHA_NO_INTERPRETABLE",
                category="VALIDEZ",
                severity="ERROR",
                description=(
                    "La fecha no puede "
                    "interpretarse."
                ),
                row_index=row_index,
                column="Date",
                observed_value=dataframe.at[
                    row_index,
                    "Date",
                ],
                expected_value=(
                    "FECHA RECONOCIBLE"
                ),
            )

        # ----------------------------------------------------
        # Equipos
        # ----------------------------------------------------

        same_team_mask = (
            home_team.notna()
            & away_team.notna()
            & home_team.ne("")
            & away_team.ne("")
            & home_team.str.casefold().eq(
                away_team.str.casefold()
            )
        )

        for row_index in dataframe.index[
            same_team_mask.fillna(False)
        ]:
            register_issue(
                records=issue_records,
                dataframe=dataframe,
                filename=filename,
                rule=(
                    "MISMO_EQUIPO_LOCAL_VISITANTE"
                ),
                category=(
                    "COHERENCIA_DEPORTIVA"
                ),
                severity="ERROR",
                description=(
                    "El mismo equipo aparece "
                    "como local y visitante."
                ),
                row_index=row_index,
                column=(
                    "HomeTeam|AwayTeam"
                ),
                observed_value=(
                    f"{dataframe.at[row_index, 'HomeTeam']} "
                    f"| {dataframe.at[row_index, 'AwayTeam']}"
                ),
                expected_value=(
                    "EQUIPOS DIFERENTES"
                ),
            )

        # ----------------------------------------------------
        # Duplicados
        # ----------------------------------------------------

        exact_duplicate_mask = (
            dataframe.duplicated(
                keep=False
            )
        )

        for row_index in dataframe.index[
            exact_duplicate_mask
        ]:
            register_issue(
                records=issue_records,
                dataframe=dataframe,
                filename=filename,
                rule=(
                    "FILA_DUPLICADA_EXACTA"
                ),
                category="INTEGRIDAD",
                severity="ERROR",
                description=(
                    "La fila completa está "
                    "duplicada."
                ),
                row_index=row_index,
                expected_value=(
                    "REGISTRO ÚNICO"
                ),
            )

        match_key = pd.DataFrame(
            {
                "fecha": parsed_dates,
                "local": (
                    home_team.str.casefold()
                ),
                "visitante": (
                    away_team.str.casefold()
                ),
            },
            index=dataframe.index,
        )

        valid_key_mask = (
            match_key.notna().all(axis=1)
            & home_team.ne("")
            & away_team.ne("")
        )

        duplicate_key_mask = (
            valid_key_mask
            & match_key.duplicated(
                keep=False
            )
        )

        for row_index in dataframe.index[
            duplicate_key_mask.fillna(
                False
            )
        ]:
            register_issue(
                records=issue_records,
                dataframe=dataframe,
                filename=filename,
                rule=(
                    "CLAVE_PARTIDO_DUPLICADA"
                ),
                category="INTEGRIDAD",
                severity="ERROR",
                description=(
                    "La fecha, el local y el "
                    "visitante están repetidos."
                ),
                row_index=row_index,
                column=(
                    "Date|HomeTeam|AwayTeam"
                ),
                expected_value=(
                    "CLAVE ÚNICA"
                ),
            )

        # ----------------------------------------------------
        # Variables numéricas
        # ----------------------------------------------------

        for column, numeric_series in (
            numeric_data.items()
        ):
            original_series = (
                dataframe[column]
            )

            invalid_numeric_mask = (
                original_series.notna()
                & numeric_series.isna()
            )

            for row_index in dataframe.index[
                invalid_numeric_mask
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule="VALOR_NO_NUMERICO",
                    category="VALIDEZ",
                    severity="ERROR",
                    description=(
                        "La estadística no puede "
                        "convertirse a número."
                    ),
                    row_index=row_index,
                    column=column,
                    observed_value=dataframe.at[
                        row_index,
                        column,
                    ],
                    expected_value=(
                        "NÚMERO O AUSENTE"
                    ),
                )

            non_integer_mask = (
                numeric_series.notna()
                & numeric_series.mod(1).ne(0)
            )

            for row_index in dataframe.index[
                non_integer_mask
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule="VALOR_NO_ENTERO",
                    category="VALIDEZ",
                    severity="ERROR",
                    description=(
                        "Una estadística de "
                        "recuento no es entera."
                    ),
                    row_index=row_index,
                    column=column,
                    observed_value=(
                        numeric_series.at[
                            row_index
                        ]
                    ),
                    expected_value=(
                        "NÚMERO ENTERO"
                    ),
                )

            negative_mask = (
                numeric_series.notna()
                & numeric_series.lt(0)
            )

            for row_index in dataframe.index[
                negative_mask
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule="VALOR_NEGATIVO",
                    category=(
                        "COHERENCIA_DEPORTIVA"
                    ),
                    severity="ERROR",
                    description=(
                        "La estadística contiene "
                        "un valor negativo."
                    ),
                    row_index=row_index,
                    column=column,
                    observed_value=(
                        numeric_series.at[
                            row_index
                        ]
                    ),
                    expected_value=(
                        "VALOR >= 0"
                    ),
                )

        # ----------------------------------------------------
        # Dominio de resultados
        # ----------------------------------------------------

        invalid_ftr_mask = (
            full_time_result.notna()
            & full_time_result.ne("")
            & ~full_time_result.isin(
                VALID_RESULTS
            )
        )

        for row_index in dataframe.index[
            invalid_ftr_mask.fillna(False)
        ]:
            register_issue(
                records=issue_records,
                dataframe=dataframe,
                filename=filename,
                rule=(
                    "RESULTADO_FINAL_INVALIDO"
                ),
                category="VALIDEZ",
                severity="ERROR",
                description=(
                    "El resultado final no "
                    "pertenece a H, D o A."
                ),
                row_index=row_index,
                column="FTR",
                observed_value=dataframe.at[
                    row_index,
                    "FTR",
                ],
                expected_value="H, D O A",
            )

        if "HTR" in dataframe.columns:
            invalid_htr_mask = (
                half_time_result.notna()
                & half_time_result.ne("")
                & ~half_time_result.isin(
                    VALID_RESULTS
                )
            )

            for row_index in dataframe.index[
                invalid_htr_mask.fillna(
                    False
                )
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule=(
                        "RESULTADO_DESCANSO_INVALIDO"
                    ),
                    category="VALIDEZ",
                    severity="ERROR",
                    description=(
                        "El resultado al descanso "
                        "no pertenece a H, D o A."
                    ),
                    row_index=row_index,
                    column="HTR",
                    observed_value=dataframe.at[
                        row_index,
                        "HTR",
                    ],
                    expected_value=(
                        "H, D, A O AUSENTE"
                    ),
                )

        # ----------------------------------------------------
        # Coherencia del resultado final
        # ----------------------------------------------------

        home_goals_ft = numeric_data[
            "FTHG"
        ]

        away_goals_ft = numeric_data[
            "FTAG"
        ]

        expected_ftr = derive_result(
            home_goals_ft,
            away_goals_ft,
        )

        inconsistent_ftr_mask = (
            expected_ftr.notna()
            & full_time_result.isin(
                VALID_RESULTS
            )
            & expected_ftr.ne(
                full_time_result
            )
        )

        for row_index in dataframe.index[
            inconsistent_ftr_mask.fillna(
                False
            )
        ]:
            register_issue(
                records=issue_records,
                dataframe=dataframe,
                filename=filename,
                rule=(
                    "RESULTADO_FINAL_INCOHERENTE"
                ),
                category=(
                    "COHERENCIA_DEPORTIVA"
                ),
                severity="ERROR",
                description=(
                    "El resultado final no "
                    "coincide con los goles."
                ),
                row_index=row_index,
                column="FTR",
                observed_value=(
                    full_time_result.at[
                        row_index
                    ]
                ),
                expected_value=(
                    expected_ftr.at[
                        row_index
                    ]
                ),
            )

        # ----------------------------------------------------
        # Coherencia al descanso
        # ----------------------------------------------------

        if all(
            column in numeric_data
            for column in [
                "HTHG",
                "HTAG",
            ]
        ):
            home_goals_ht = (
                numeric_data["HTHG"]
            )

            away_goals_ht = (
                numeric_data["HTAG"]
            )

            expected_htr = derive_result(
                home_goals_ht,
                away_goals_ht,
            )

            inconsistent_htr_mask = (
                expected_htr.notna()
                & half_time_result.isin(
                    VALID_RESULTS
                )
                & expected_htr.ne(
                    half_time_result
                )
            )

            for row_index in dataframe.index[
                inconsistent_htr_mask.fillna(
                    False
                )
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule=(
                        "RESULTADO_DESCANSO_INCOHERENTE"
                    ),
                    category=(
                        "COHERENCIA_DEPORTIVA"
                    ),
                    severity="ADVERTENCIA",
                    description=(
                        "El resultado al descanso "
                        "no coincide con los goles."
                    ),
                    row_index=row_index,
                    column="HTR",
                    observed_value=(
                        half_time_result.at[
                            row_index
                        ]
                    ),
                    expected_value=(
                        expected_htr.at[
                            row_index
                        ]
                    ),
                )

            ht_ft_rules = [
                (
                    home_goals_ht,
                    home_goals_ft,
                    "HTHG|FTHG",
                    "HTHG <= FTHG",
                ),
                (
                    away_goals_ht,
                    away_goals_ft,
                    "HTAG|FTAG",
                    "HTAG <= FTAG",
                ),
            ]

            for (
                half_time_goals,
                full_time_goals,
                columns_label,
                expected_label,
            ) in ht_ft_rules:
                invalid_mask = (
                    half_time_goals.notna()
                    & full_time_goals.notna()
                    & half_time_goals.gt(
                        full_time_goals
                    )
                )

                for row_index in (
                    dataframe.index[
                        invalid_mask
                    ]
                ):
                    register_issue(
                        records=issue_records,
                        dataframe=dataframe,
                        filename=filename,
                        rule=(
                            "GOLES_HT_SUPERAN_FT"
                        ),
                        category=(
                            "COHERENCIA_DEPORTIVA"
                        ),
                        severity=(
                            "ADVERTENCIA"
                        ),
                        description=(
                            "Los goles al descanso "
                            "superan los finales."
                        ),
                        row_index=row_index,
                        column=columns_label,
                        observed_value=(
                            f"{half_time_goals.at[row_index]} "
                            f"| {full_time_goals.at[row_index]}"
                        ),
                        expected_value=(
                            expected_label
                        ),
                    )

        # ----------------------------------------------------
        # Tiros a puerta frente a tiros
        # ----------------------------------------------------

        shot_rules = [
            (
                "HST",
                "HS",
                "TIROS_PUERTA_SUPERAN_TOTALES_LOCAL",
            ),
            (
                "AST",
                "AS",
                "TIROS_PUERTA_SUPERAN_TOTALES_VISITANTE",
            ),
        ]

        for (
            on_target_column,
            total_column,
            rule_name,
        ) in shot_rules:
            if (
                on_target_column
                not in numeric_data
                or total_column
                not in numeric_data
            ):
                continue

            on_target = numeric_data[
                on_target_column
            ]

            total_shots = numeric_data[
                total_column
            ]

            invalid_mask = (
                on_target.notna()
                & total_shots.notna()
                & on_target.gt(
                    total_shots
                )
            )

            for row_index in dataframe.index[
                invalid_mask
            ]:
                register_issue(
                    records=issue_records,
                    dataframe=dataframe,
                    filename=filename,
                    rule=rule_name,
                    category=(
                        "COHERENCIA_DEPORTIVA"
                    ),
                    severity="ADVERTENCIA",
                    description=(
                        "Los tiros a puerta "
                        "superan los tiros totales."
                    ),
                    row_index=row_index,
                    column=(
                        f"{on_target_column}"
                        f"|{total_column}"
                    ),
                    observed_value=(
                        f"{on_target.at[row_index]} "
                        f"| {total_shots.at[row_index]}"
                    ),
                    expected_value=(
                        f"{on_target_column} "
                        f"<= {total_column}"
                    ),
                )

    issues = pd.DataFrame(
        issue_records,
        columns=ISSUE_COLUMNS,
    )

    coverage = pd.DataFrame(
        coverage_records
    )

    quality_summary_records = []

    for filename in dataframes:
        source_issues = issues.loc[
            issues[
                "archivo_origen"
            ].eq(filename)
        ]

        severity_counts = Counter(
            source_issues[
                "severidad"
            ]
        )

        critical_count = int(
            severity_counts[
                "CRITICA"
            ]
        )

        error_count = int(
            severity_counts[
                "ERROR"
            ]
        )

        warning_count = int(
            severity_counts[
                "ADVERTENCIA"
            ]
        )

        information_count = int(
            severity_counts[
                "INFORMACION"
            ]
        )

        if critical_count > 0:
            status = "NO COMPATIBLE"
        elif error_count > 0:
            status = (
                "REQUIERE CORRECCION"
            )
        elif warning_count > 0:
            status = (
                "COMPATIBLE CON ADVERTENCIAS"
            )
        else:
            status = "COMPATIBLE"

        quality_summary_records.append(
            {
                "archivo_origen": filename,
                "incidencias_criticas": (
                    critical_count
                ),
                "errores": error_count,
                "advertencias": (
                    warning_count
                ),
                "informaciones": (
                    information_count
                ),
                "registros_con_incidencias": int(
                    source_issues[
                        "indice_origen"
                    ]
                    .dropna()
                    .nunique()
                ),
                "estado_calidad": status,
            }
        )

    quality_summary = pd.DataFrame(
        quality_summary_records
    )

    return (
        issues,
        coverage,
        quality_summary,
    )


# ============================================================
# 6. ARMONIZACIÓN Y CONSOLIDACIÓN
# ============================================================

def harmonize_dataframes(
    dataframes: dict[
        str,
        pd.DataFrame,
    ],
    inventory: pd.DataFrame,
    issues: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Armoniza únicamente los archivos compatibles."""
    inventory_by_file = (
        inventory
        .set_index("archivo_origen")
        .to_dict(orient="index")
    )

    quality_by_file = (
        quality_summary
        .set_index("archivo_origen")
        .to_dict(orient="index")
    )

    row_issue_lookup = {}

    if not issues.empty:
        row_issues = (
            issues.loc[
                issues[
                    "indice_origen"
                ].notna()
                & issues[
                    "severidad"
                ].isin(
                    [
                        "CRITICA",
                        "ERROR",
                        "ADVERTENCIA",
                    ]
                )
            ]
            .copy()
        )

        if not row_issues.empty:
            row_issues[
                "indice_origen"
            ] = (
                row_issues[
                    "indice_origen"
                ]
                .astype(int)
            )

            grouped_issues = (
                row_issues
                .groupby(
                    [
                        "archivo_origen",
                        "indice_origen",
                    ],
                    sort=False,
                )
                .agg(
                    numero_incidencias=(
                        "regla",
                        "size",
                    ),
                    numero_advertencias=(
                        "severidad",
                        lambda values: int(
                            values.eq(
                                "ADVERTENCIA"
                            ).sum()
                        ),
                    ),
                    reglas_incidencia=(
                        "regla",
                        lambda values: " | ".join(
                            sorted(
                                set(values)
                            )
                        ),
                    ),
                )
                .reset_index()
            )

            row_issue_lookup = {
                (
                    row[
                        "archivo_origen"
                    ],
                    int(
                        row[
                            "indice_origen"
                        ]
                    ),
                ): {
                    "numero_incidencias": int(
                        row[
                            "numero_incidencias"
                        ]
                    ),
                    "numero_advertencias": int(
                        row[
                            "numero_advertencias"
                        ]
                    ),
                    "reglas_incidencia": row[
                        "reglas_incidencia"
                    ],
                }
                for _, row
                in grouped_issues.iterrows()
            }

    harmonized_dataframes = {}
    harmonization_records = []
    rejected_file_records = []

    for filename, raw_dataframe in (
        dataframes.items()
    ):
        quality_record = (
            quality_by_file[
                filename
            ]
        )

        quality_status = (
            quality_record[
                "estado_calidad"
            ]
        )

        if quality_status not in {
            "COMPATIBLE",
            "COMPATIBLE CON ADVERTENCIAS",
        }:
            rejected_file_records.append(
                {
                    "archivo_origen": filename,
                    "estado_calidad": (
                        quality_status
                    ),
                    "motivo": (
                        "El archivo contiene "
                        "errores esenciales."
                    ),
                }
            )
            continue

        dataframe = (
            raw_dataframe.copy(
                deep=True
            )
        )

        input_rows = int(
            len(dataframe)
        )

        for column in OPTIONAL_COLUMNS:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA

        dataframe = dataframe[
            SOURCE_COLUMNS
        ].copy()

        dataframe[
            "indice_origen"
        ] = dataframe.index.astype(
            "int64"
        )

        dataframe[
            "fila_csv_origen"
        ] = (
            dataframe[
                "indice_origen"
            ]
            + 2
        )

        dataframe = dataframe.rename(
            columns=COLUMN_MAPPING
        )

        for column in (
            CANONICAL_TEXT_COLUMNS
        ):
            dataframe[column] = (
                normalize_text(
                    dataframe[column]
                )
            )

        dataframe[
            "resultado_ft"
        ] = (
            dataframe[
                "resultado_ft"
            ]
            .str.upper()
        )

        dataframe[
            "resultado_ht"
        ] = (
            dataframe[
                "resultado_ht"
            ]
            .str.upper()
        )

        dataframe["fecha"] = (
            parse_dates(
                dataframe["fecha"]
            )
        )

        for column in (
            CANONICAL_NUMERIC_COLUMNS
        ):
            dataframe[column] = (
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )
                .astype("Int64")
            )

        competition, season = (
            infer_context_from_filename(
                filename
            )
        )

        dataframe[
            "competicion"
        ] = pd.Series(
            competition,
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "temporada"
        ] = pd.Series(
            season,
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "codigo_fuente"
        ] = pd.Series(
            build_source_code(
                filename
            ),
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "archivo_origen"
        ] = pd.Series(
            filename,
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "sha256_archivo"
        ] = pd.Series(
            str(
                inventory_by_file[
                    filename
                ]["sha256"]
            ),
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "indice_origen"
        ] = (
            dataframe[
                "indice_origen"
            ]
            .astype("Int64")
        )

        dataframe[
            "fila_csv_origen"
        ] = (
            dataframe[
                "fila_csv_origen"
            ]
            .astype("Int64")
        )

        dataframe[
            "id_registro"
        ] = pd.Series(
            [
                build_record_id(
                    filename=filename,
                    original_index=int(
                        dataframe.at[
                            row_index,
                            "indice_origen",
                        ]
                    ),
                    date_value=dataframe.at[
                        row_index,
                        "fecha",
                    ],
                    home_team=dataframe.at[
                        row_index,
                        "equipo_local",
                    ],
                    away_team=dataframe.at[
                        row_index,
                        "equipo_visitante",
                    ],
                )
                for row_index
                in dataframe.index
            ],
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "datos_descanso_disponibles"
        ] = (
            dataframe[
                [
                    "goles_local_ht",
                    "goles_visitante_ht",
                    "resultado_ht",
                ]
            ]
            .notna()
            .all(axis=1)
            .astype("boolean")
        )

        dataframe[
            "estadisticas_partido_disponibles"
        ] = (
            dataframe[
                MATCH_STAT_COLUMNS
            ]
            .notna()
            .all(axis=1)
            .astype("boolean")
        )

        dataframe[
            "arbitro_disponible"
        ] = (
            dataframe["arbitro"]
            .notna()
            .astype("boolean")
        )

        quality_data = []

        for row_index in (
            dataframe.index
        ):
            original_index = int(
                dataframe.at[
                    row_index,
                    "indice_origen",
                ]
            )

            quality_data.append(
                row_issue_lookup.get(
                    (
                        filename,
                        original_index,
                    ),
                    {
                        "numero_incidencias": 0,
                        "numero_advertencias": 0,
                        "reglas_incidencia": "",
                    },
                )
            )

        dataframe[
            "numero_incidencias"
        ] = pd.Series(
            [
                item[
                    "numero_incidencias"
                ]
                for item in quality_data
            ],
            index=dataframe.index,
            dtype="Int64",
        )

        dataframe[
            "numero_advertencias"
        ] = pd.Series(
            [
                item[
                    "numero_advertencias"
                ]
                for item in quality_data
            ],
            index=dataframe.index,
            dtype="Int64",
        )

        dataframe[
            "reglas_incidencia"
        ] = pd.Series(
            [
                item[
                    "reglas_incidencia"
                ]
                for item in quality_data
            ],
            index=dataframe.index,
            dtype="string",
        )

        dataframe[
            "estado_registro"
        ] = pd.Series(
            np.where(
                dataframe[
                    "numero_incidencias"
                ].gt(0),
                (
                    "VALIDADO CON "
                    "ADVERTENCIAS"
                ),
                "VALIDADO",
            ),
            index=dataframe.index,
            dtype="string",
        )

        dataframe = dataframe[
            CANONICAL_COLUMN_ORDER
        ].copy()

        harmonized_dataframes[
            filename
        ] = dataframe

        harmonization_records.append(
            {
                "archivo_origen": filename,
                "filas_entrada": (
                    input_rows
                ),
                "filas_salida": int(
                    len(dataframe)
                ),
                "columnas_salida": int(
                    dataframe.shape[1]
                ),
                "registros_con_advertencias": int(
                    dataframe[
                        "numero_advertencias"
                    ]
                    .gt(0)
                    .sum()
                ),
                "incidencias_vinculadas": int(
                    dataframe[
                        "numero_incidencias"
                    ]
                    .sum()
                ),
                "estado_armonizacion": (
                    "COMPATIBLE CON ADVERTENCIAS"
                    if quality_record[
                        "advertencias"
                    ] > 0
                    else (
                        "COMPATIBLE TRAS "
                        "ARMONIZACIÓN"
                    )
                ),
            }
        )

    if harmonized_dataframes:
        master_dataframe = pd.concat(
            [
                dataframe.copy(
                    deep=True
                )
                for dataframe
                in harmonized_dataframes.values()
            ],
            ignore_index=True,
            sort=False,
        )

        master_dataframe = (
            master_dataframe
            .sort_values(
                [
                    "archivo_origen",
                    "fecha",
                    "equipo_local",
                    "equipo_visitante",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    else:
        master_dataframe = pd.DataFrame(
            columns=(
                CANONICAL_COLUMN_ORDER
            )
        )

    return (
        harmonized_dataframes,
        master_dataframe,
        pd.DataFrame(
            harmonization_records
        ),
        pd.DataFrame(
            rejected_file_records,
            columns=[
                "archivo_origen",
                "estado_calidad",
                "motivo",
            ],
        ),
    )


# ============================================================
# 7. COMPARACIÓN DE MÉTODOS Y ORQUESTACIÓN
# ============================================================

def build_method_comparison(
    raw_dataframes: dict[
        str,
        pd.DataFrame,
    ],
    master_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compara la unión directa con el flujo auditable."""
    if raw_dataframes:
        conventional = pd.concat(
            [
                dataframe.copy(
                    deep=True
                )
                for dataframe
                in raw_dataframes.values()
            ],
            ignore_index=True,
            sort=False,
        )
    else:
        conventional = pd.DataFrame()

    canonical_numeric_count = int(
        sum(
            (
                column
                in master_dataframe.columns
                and str(
                    master_dataframe[
                        column
                    ].dtype
                )
                == "Int64"
            )
            for column
            in CANONICAL_NUMERIC_COLUMNS
        )
    )

    traceability_columns = [
        "archivo_origen",
        "indice_origen",
        "fila_csv_origen",
        "sha256_archivo",
    ]

    traceable_rows = (
        int(
            master_dataframe[
                traceability_columns
            ]
            .notna()
            .all(axis=1)
            .sum()
        )
        if not master_dataframe.empty
        else 0
    )

    linked_issues = (
        int(
            master_dataframe[
                "numero_incidencias"
            ]
            .sum()
        )
        if not master_dataframe.empty
        else 0
    )

    warning_records = (
        int(
            master_dataframe[
                "numero_advertencias"
            ]
            .gt(0)
            .sum()
        )
        if not master_dataframe.empty
        else 0
    )

    return pd.DataFrame(
        [
            {
                "criterio": (
                    "Registros conservados"
                ),
                "union_convencional": int(
                    len(conventional)
                ),
                "flujo_auditable": int(
                    len(master_dataframe)
                ),
            },
            {
                "criterio": (
                    "Columnas generadas"
                ),
                "union_convencional": int(
                    conventional.shape[1]
                ),
                "flujo_auditable": int(
                    master_dataframe.shape[1]
                ),
            },
            {
                "criterio": (
                    "Fecha como datetime"
                ),
                "union_convencional": (
                    "SÍ"
                    if (
                        "Date"
                        in conventional.columns
                        and is_datetime64_any_dtype(
                            conventional[
                                "Date"
                            ]
                        )
                    )
                    else "NO"
                ),
                "flujo_auditable": (
                    "SÍ"
                    if (
                        "fecha"
                        in master_dataframe.columns
                        and is_datetime64_any_dtype(
                            master_dataframe[
                                "fecha"
                            ]
                        )
                    )
                    else "NO"
                ),
            },
            {
                "criterio": (
                    "Columnas estadísticas Int64"
                ),
                "union_convencional": 0,
                "flujo_auditable": (
                    canonical_numeric_count
                ),
            },
            {
                "criterio": (
                    "Filas con trazabilidad"
                ),
                "union_convencional": 0,
                "flujo_auditable": (
                    traceable_rows
                ),
            },
            {
                "criterio": (
                    "Incidencias vinculadas"
                ),
                "union_convencional": 0,
                "flujo_auditable": (
                    linked_issues
                ),
            },
            {
                "criterio": (
                    "Registros para revisión"
                ),
                "union_convencional": 0,
                "flujo_auditable": (
                    warning_records
                ),
            },
        ]
    )


def process_payloads(
    payloads: Iterable[
        tuple[str, bytes]
    ],
) -> dict:
    """Ejecuta el flujo completo sobre archivos cargados."""
    payload_list = list(payloads)

    dataframes, inventory = (
        load_payloads(
            payload_list
        )
    )

    profiles = profile_dataframes(
        dataframes
    )

    (
        issues,
        coverage,
        quality_summary,
    ) = validate_dataframes(
        dataframes
    )

    (
        harmonized_dataframes,
        master_dataframe,
        harmonization_summary,
        rejected_files,
    ) = harmonize_dataframes(
        dataframes=dataframes,
        inventory=inventory,
        issues=issues,
        quality_summary=quality_summary,
    )

    method_comparison = (
        build_method_comparison(
            raw_dataframes=dataframes,
            master_dataframe=(
                master_dataframe
            ),
        )
    )

    valid_files = int(
        inventory[
            "estado_carga"
        ]
        .eq("OK")
        .sum()
    )

    manifest = {
        "nombre_proyecto": (
            "SportData Harmonizer"
        ),
        "archivos_recibidos": int(
            len(payload_list)
        ),
        "archivos_legibles": (
            valid_files
        ),
        "archivos_armonizados": int(
            len(
                harmonized_dataframes
            )
        ),
        "archivos_rechazados": int(
            len(rejected_files)
        ),
        "filas_dataset_maestro": int(
            len(master_dataframe)
        ),
        "columnas_dataset_maestro": int(
            master_dataframe.shape[1]
        ),
        "registros_con_advertencias": (
            int(
                master_dataframe[
                    "numero_advertencias"
                ]
                .gt(0)
                .sum()
            )
            if not master_dataframe.empty
            else 0
        ),
        "incidencias_vinculadas": (
            int(
                master_dataframe[
                    "numero_incidencias"
                ]
                .sum()
            )
            if not master_dataframe.empty
            else 0
        ),
        "archivos": (
            inventory
            .fillna("")
            .to_dict(
                orient="records"
            )
        ),
    }

    return {
        "dataframes_raw": dataframes,
        "inventario": inventory,
        "perfil_archivos": profiles[
            "perfil_archivos"
        ],
        "presencia_columnas": profiles[
            "presencia_columnas"
        ],
        "tipos_inferidos": profiles[
            "tipos_inferidos"
        ],
        "incidencias": issues,
        "cobertura": coverage,
        "resumen_calidad": (
            quality_summary
        ),
        "dataframes_armonizados": (
            harmonized_dataframes
        ),
        "dataset_maestro": (
            master_dataframe
        ),
        "resumen_armonizacion": (
            harmonization_summary
        ),
        "archivos_rechazados": (
            rejected_files
        ),
        "comparacion_metodos": (
            method_comparison
        ),
        "manifiesto": manifest,
    }


__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "COLUMN_MAPPING",
    "CANONICAL_COLUMN_ORDER",
    "calculate_sha256",
    "dataframe_to_csv_bytes",
    "dictionary_to_json_bytes",
    "load_payloads",
    "profile_dataframes",
    "validate_dataframes",
    "harmonize_dataframes",
    "build_method_comparison",
    "process_payloads",
]
