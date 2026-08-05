# SportData Harmonizer

## Descripción

SportData Harmonizer es una solución desarrollada en Python y
Streamlit para validar, armonizar y consolidar múltiples archivos
deportivos sin perder la trazabilidad de los registros originales.

El proyecto corresponde a la Opción 4 de la Actividad Colaborativa
del Módulo 8 del Máster en Python Avanzado Aplicado al Deporte.

## Resultados del caso práctico

- Archivos fuente: 7
- Competiciones: 3
- Partidos consolidados: 2594
- Columnas del dataset académico: 40
- Registros con advertencias: 2
- Incidencias vinculadas: 4
- Registros rechazados: 0

## Estructura del proyecto

~~~text
data/
├── raw/              Archivos originales sin modificar
├── interim/          Archivos armonizados individualmente
└── processed/        Dataset maestro

src/
└── harmonizer_core.py

app/
└── streamlit_app.py

outputs/
├── figures/          Visualizaciones
├── manifests/        Procedencia y hashes
└── quality/          Resultados de validación

docs/                 Documentación complementaria
reports/              Informe breve de la actividad
logs/                 Registros de ejecución
~~~

## Ejecución del notebook

1. Abrir el notebook desde la raíz del proyecto.
2. Seleccionar el intérprete de Python adecuado.
3. Ejecutar las celdas en orden.
4. Verificar que las validaciones finalicen con estado `OK`.

## Ejecución de Streamlit

Desde la raíz del proyecto:

~~~bash
streamlit run app/streamlit_app.py
~~~

También puede utilizarse:

~~~bash
python -m streamlit run app/streamlit_app.py
~~~

La aplicación se abrirá normalmente en:

~~~text
http://127.0.0.1:8501
~~~

## Dependencias

Las dependencias reproducibles se encuentran en
`requirements.txt`.

Instalación:

~~~bash
python -m pip install -r requirements.txt
~~~

## Esquema mínimo de entrada

Los archivos propios deben contener:

- `Date`
- `HomeTeam`
- `AwayTeam`
- `FTHG`
- `FTAG`
- `FTR`

Las demás variables del esquema son opcionales.

## Política de tratamiento

- Los archivos originales no se modifican.
- Los valores ausentes no se imputan.
- Las anomalías no se corrigen silenciosamente.
- Los archivos con errores esenciales no se consolidan.
- Cada fila conserva archivo, índice, fila y hash de procedencia.

## Datos utilizados

Colección: DataHub Football Datasets.

Proveedor original documentado: Football-Data.

Commit utilizado:

~~~text
436a51f15258e7d0c6042c231227202b70a5a0ae
~~~

## Autor

Aitor Cuenca Retuerto
