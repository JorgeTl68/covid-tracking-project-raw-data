# COVID Tracking Project: dataset original

Este repositorio contiene los datos históricos originales del COVID Tracking Project, en bruto(datos sin limpieza).

Incluye 58 archivos CSV con datos diarios nacionales y por estado o territorio. Excel puede abrirlos, aunque su formato original es CSV.

## Fuente

[COVID Tracking Project: descarga de datos](https://covidtracking.com/data/download/)

Los datos llegan hasta el 7 de marzo de 2021.

## U3 Saber Hacer: análisis supervisado

Pregunta: con los datos que un estado reporta hoy, ¿cómo cambiará su número de hospitalizados en los próximos 7 días?

- **Regresión:** cambio % del promedio de 7 días de hospitalizados entre el día t y t + 7.
- **Clasificación:** alerta si ese cambio es mayor a +5 %.

Modelos: línea base (tendencia), regresión lineal Ridge, regresión logística, Random Forest y Gradient Boosting. Entrenamiento: abr-oct 2020 con validación cruzada temporal y GridSearchCV. Prueba: nov 2020-feb 2021 con reentrenamiento mensual.

| Ruta | Contenido |
|---|---|
| `all-states-history.csv` y demás CSV de la raíz | Dataset original |
| `data/clean/covid_states_clean.csv` | Dataset limpio (50 estados + DC, un registro por estado y día) |
| `data/clean/covid_model_dataset.csv` | Variables de entrada y objetivos del modelo |
| `src/u3_supervised_analysis.py` | Código fuente: limpieza, entrenamiento, optimización, evaluación y gráficas |
| `notebooks/u3_supervised_analysis.ipynb` | Notebook del proceso con resultados |
| `outputs/` | Tablas de comparación y `results.json` |

### Ejecutar

```bash
pip install -r requirements.txt
python src/u3_supervised_analysis.py
```
