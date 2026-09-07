# Red ART1 (Carpenter–Grossberg)

## Trabajo Final Integrador — Redes Neuronales

**Materia:** Redes Neuronales
**Institución:** UADER — Facultad de Ingeniería — IDTI Lab
**Fecha:** Marzo 2026
**Estado:** En desarrollo · Alineado con la consigna oficial

---

## 1. Sobre este proyecto

Este Trabajo Final Integrador (TFI) implementa la red neuronal **ART1 (Adaptive Resonance Theory 1)** de Carpenter y Grossberg, según la descripción de Lau (1992) pp. 12–14. La consigna oficial del trabajo se encuentra en `_legacy/CarGross_TP/consignas_TP.md` (Actividad #3 exige generar un módulo `CarGross.py` para la red de Carpenter–Grossberg).

El proyecto se orienta a una **aplicación exploratoria** sobre dos datasets propios:

- `data/dataset1_pacientes.csv` — 55 pacientes simulados (dominio clínico cardiovascular/metabólico).
- `data/dataset2_sensores.csv` — 55 sensores simulados (dominio industrial, mantenimiento predictivo).

La elección del algoritmo, los umbrales de binarización y los criterios de evaluación se documentan en `docs/`.

## 2. Índice de documentación

| Doc | Contenido |
|-----|-----------|
| [`README.md`](README.md) | Este índice. Punto de entrada. |
| [`01_marco_teorico.md`](01_marco_teorico.md) | Qué es ART1, el dilema estabilidad–plasticidad y la posición de ART1 en la taxonomía de Lippmann/Lau. |
| [`02_problema_y_alcance.md`](02_problema_y_alcance.md) | Qué hace y qué NO hace el sistema, motivación y usuarios objetivo. |
| [`03_dataset_y_preprocesamiento.md`](03_dataset_y_preprocesamiento.md) | Datasets canónicos y reglas de binarización basadas en umbrales clínicos y operativos. |
| [`04_algoritmo.md`](04_algoritmo.md) | Transcripción y análisis del Box 3 (Lau 1992) paso a paso. |
| [`05_corridas_y_evaluacion.md`](05_corridas_y_evaluacion.md) | Diseño experimental, métricas no supervisadas y formato de las tablas de resultados. |
| [`06_limitaciones_y_etica.md`](06_limitaciones_y_etica.md) | Limitaciones técnicas de ART1, consideraciones éticas del dominio clínico y disclaimer formal. |
| [`07_iteraciones.md`](07_iteraciones.md) | Bitácora de iteraciones. |
| [`informe_corridas.md`](informe_corridas.md) | **08** — Informe narrativo de las 30 corridas de ART1: interpretación clínica tentativa, análisis estructural del dataset sensores (absorción del sexto vector S027 durante fit) y amenazas a la validez. |
| [`manual_referencia.md`](manual_referencia.md) | **09** — Manual de referencia de `src/CarGross.py` (entregable Actividad #4 del TFI): API, instalación, ejemplos de uso y troubleshooting. |

## 3. Orden de lectura sugerido

1. `README.md` (este archivo).
2. `01_marco_teorico.md` para entender qué es ART1 y por qué se eligió sobre alternativas.
3. `02_problema_y_alcance.md` para entender el alcance del TFI y qué afirmaciones NO se hacen.
4. `03_dataset_y_preprocesamiento.md` para conocer los datos con los que se trabaja y los umbrales usados.
5. `04_algoritmo.md` para entender el algoritmo Box 3 que implementa `src/CarGross.py`.
6. `05_corridas_y_evaluacion.md` para entender cómo se evalúan las corridas (métricas no supervisadas).
7. `06_limitaciones_y_etica.md` antes de cualquier interpretación clínica.
8. `07_iteraciones.md` para seguir la evolución del trabajo.
9. `informe_corridas.md` para leer la interpretación narrativa de las 30 corridas, los clusters obtenidos y el análisis estructural del dataset sensores.
10. `manual_referencia.md` para reproducir las corridas (entregable Actividad #4 del TFI — *Manual de Referencia*): API de `src/CarGross.py`, instalación, ejemplos de uso y troubleshooting.

## 4. Materiales de referencia

- **Consigna oficial**: [`_legacy/CarGross_TP/consignas_TP.md`](../_legacy/CarGross_TP/consignas_TP.md).
- **Paper de Lau transcrito**: [`_legacy/CarGross_TP/lau_contenido.md`](../_legacy/CarGross_TP/lau_contenido.md) (Lau 1992, pp. 5–14).
- **Implementación previa de referencia** (no se modifica, se usa como contraste histórico): `_legacy/CarGross_TP/src/CarGross.py`, `_legacy/CarGross_TP/docs/manual.md`, `_legacy/CarGross_TP/docs/informe.md`.

## 5. Estructura del repositorio

```
ia2026/
├── docs/                      ← documentación de este TFI
├── src/CarGross.py            ← implementación de ART1 (en desarrollo por separado)
├── data/                      ← datasets canónicos
│   ├── dataset1_pacientes.csv
│   └── dataset2_sensores.csv
├── results/                   ← salidas de las corridas
├── _legacy/CarGross_TP/               ← TFI previo (referencia, no se modifica)
├── Lau.pp5.a.11.pdf
├── Lau.pp12.a.14.pdf
└── README.md
```

## 6. Estado del proyecto

| Componente | Estado |
|------------|--------|
| Documentación (`docs/`) | Completa y alineada con la consigna |
| Implementación (`src/CarGross.py`) | Esqueleto presente, completa a cargo de otro flujo |
| Datasets canónicos | Listos |
| Corridas y resultados | Pendientes (ver `05_corridas_y_evaluacion.md`) |

## 7. Nota sobre iteraciones previas

En una iteración previa, esta carpeta `docs/` contuvo documentación orientada a un **perceptrón multicapa (MLP) supervisado** como clasificador de pacientes para triaje. Esa descripción **no coincidía con la consigna oficial** (`_legacy/CarGross_TP/consignas_TP.md` exige ART1 de manera explícita) ni con el comportamiento real de la red documentada en `_legacy/CarGross_TP/lau_contenido.md` (ART1 es no supervisada, no supervisada). Se descartó y se reemplazó por la presente documentación. El detalle de ese descarte queda en `07_iteraciones.md`.

## 8. Cómo citar

> Trabajo Final Integrador — Redes Neuronales. (2026). *Implementación de ART1 (Carpenter–Grossberg) en Python*. UADER — IDTI Lab.
