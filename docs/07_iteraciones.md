# 07 · Bitácora de iteraciones

Este documento registra las iteraciones del proyecto, qué se decidió en cada una y por qué. Es la **memoria** del TFI; complementa a los otros documentos (que describen el estado actual) narrando el camino recorrido.

## Iteración 0 — Realineación a la consigna oficial

**Fecha**: marzo 2026.

**Decisión**: pivotar desde un enfoque de perceptrón multicapa (MLP) supervisado hacia una implementación de **ART1 no supervisado**, alineada con la consigna oficial.

**Motivo**: la consigna `_legacy/CarGross_TP/consignas_TP.md`, en su Actividad #3, exige *"generar una red de Carpenter–Grossberg llamada `CarGross.py`"*. La documentación previa, que también vivía en `docs/` en una versión anterior, describía un clasificador MLP supervisado con métricas tipo accuracy / F1 / precision / recall, lo cual **no corresponde** con la consigna ni con el comportamiento real de la red documentada en `_legacy/CarGross_TP/lau_contenido.md` (ART1 es binaria, no supervisada, comparable al "leader clustering algorithm"). La discrepancia fue detectada al releer la consigna y el material de Lau en preparación de esta iteración.

**Cambios concretos**:

1. Reescritura completa de `docs/` con 7 documentos (teoría, alcance, datos, algoritmo, corridas, ética, iteraciones).
2. Encuadre médico reformulado: el sistema **no** clasifica pacientes para triaje, sino que **explora perfiles latentes** para que un profesional los interprete.
3. Eliminación explícita de toda mención a métricas supervisadas (accuracy, F1, precision, recall, ROC, AUC).
4. Adopción del Box 3 de Lau 1992 como referencia algorítmica no negociable.
5. Diseño experimental (`05_corridas_y_evaluacion.md`) redefinido en términos de $K(\rho)$, ARI, compactness e interpretabilidad cualitativa.
6. Adición de un disclaimer formal y una sección ética explícita, que faltaban en el intento previo.

**Lecciones**:

- Antes de escribir cualquier doc, **leer la consigna** y el material provisto completo. No subestimar el papel del paper de Lau como fuente primaria.
- ART1 ≠ MLP: vocabularios, métricas y modelos mentales son distintos. Mezclar ambos produce documentación que no satisface ninguno de los dos.
- En dominios sensibles como el clínico, documentar **qué NO hace** el sistema es tan importante como documentar qué sí hace. El disclaimer formal no es accesorio: es parte de la entrega.
- La taxonomía de Lippmann es engañosa: el transcription en `lau_contenido.md` lista ART1 bajo "supervisadas", pero el cuerpo del paper y el algoritmo concreto la describen como no supervisada. Hay que leer más allá del cuadro resumen.

## Iteración 0.5 — Validación con el docente

- Decision D0.5.1: `data/metadata.csv` creado con 14 umbrales (7 pacientes + 7 sensores) y 6 columnas (dataset, feature, threshold, rule, unit, justification)
- Decision D0.5.2: Repeticiones para métrica ARI subidas de R=3 a R=5 (justificación: 55 muestras hacen ARI ruidoso con 3 pares, R=5 da estimación más estable sin costo excesivo)
- Decision D0.5.3: caso `out_of_range` de voltaje modelado como dos filas (reglas `lt` 219 + `gt` 221) — documentado en CSV
- Cross-reference: `data/metadata.csv`, `docs/05_corridas_y_evaluacion.md`

## Iteración 1 — Implementación de ART1

- Decision D1.1: `src/CarGross.py` reescrito completamente, reemplazando el esqueleto de 131 líneas con una implementación de 561 líneas de ART1 puro stdlib (Box 3 de Lau 1992).
- Decision D1.2: máximo de clusters con default 1000 (en lugar de `n_inputs = 7`) para permitir que ART1 cree clusters hasta uno por vector de entrada.
- Decision D1.3: clusters con exemplar cero (todos los bits en 0) se reportan por separado en el TXT y no se cuentan entre los clusters "con miembros". Esto evita ruido en la salida y refleja la realidad del dataset (filas degeneradas bajo todos los umbrales).
- Decision D1.4: stability metric = pairwise agreement fraction vs run 0 (en lugar de ARI formal). Más simple para TP de grado, suficiente para N=55.
- Cross-references: `src/CarGross.py`, `data/dataset1_pacientes.csv`, `data/metadata.csv`, `docs/04_algoritmo.md`.

## Iteración 2 — Corridas con barrido de ρ

**Estado**: pendiente.

**Objetivo**: correr el barrido de $\rho$ sobre `dataset1_pacientes` (rango definido en `05_corridas_y_evaluacion.md` §2), registrar $K(\rho)$ y la stability metric pairwise vs run 0 para cada valor, y verificar que el comportamiento coincide con el esperado teóricamente (monotonicidad de $K$ y clustering no trivial).

**Entregables previstos**:

- `results/resultado_dataset1.csv` y `.txt` por cada valor de $\rho$ del barrido.
- Tabla resumen con $K(\rho)$ y stability metric para el informe.

## Iteración 3 — Barrido completo y comparaciones

**Estado**: pendiente.

**Objetivo**: completar el barrido de $\rho$ para ambos datasets (rango definido en `05_corridas_y_evaluacion.md`), analizar estabilidad entre corridas (ARI), contrastar con el `docs/informe.md` de `_legacy/CarGross_TP/`, y escribir el informe de corridas definitivo.

**Entregables previstos**:

- Resultados completos en `results/`.
- Tablas con el formato definido en `05_corridas_y_evaluacion.md` §3.
- Notas de comparación con el TFI previo.

## Iteración 4 — Cierre del TFI

**Estado**: pendiente.

**Objetivo**: consolidar el manual de referencia, las FAQ y la PPT con audio exigidos por la consigna (`_legacy/CarGross_TP/consignas_TP.md` actividades 2 y 4), y verificar que la carátula, las referencias y el disclaimer estén completos.

**Entregables previstos**:

- `docs/manual.md` (alcances, instalación, test demo, FAQ).
- PPT con audio (origen, características, aplicaciones, fortalezas/debilidades, diferencias con otras redes).
- Carátula con logos UADER e IDTI Lab.

## Iteración 1.5 — Aislamiento del intento anterior

- Decision D1.5.1: la carpeta `CarGross_TP/` (intento anterior no entregado) se mueve a `_legacy/CarGross_TP/` para evitar confusión con la entrega final.
- Decision D1.5.2: se crea `_legacy/README.md` documentando el contenido de la carpeta.
- Decision D1.5.3: las menciones a `CarGross_TP/` en los docs se actualizan a `_legacy/CarGross_TP/` con nota explicando el motivo.
- Cross-references: `_legacy/`, `_legacy/CarGross_TP/`, `_legacy/README.md`, `docs/01_marco_teorico.md`.
