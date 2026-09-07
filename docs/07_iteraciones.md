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

## Iteración 1.5 — Aislamiento del intento anterior

- Decision D1.5.1: la carpeta `CarGross_TP/` (intento anterior no entregado) se mueve a `_legacy/CarGross_TP/` para evitar confusión con la entrega final.
- Decision D1.5.2: se crea `_legacy/README.md` documentando el contenido de la carpeta.
- Decision D1.5.3: las menciones a `CarGross_TP/` en los docs se actualizan a `_legacy/CarGross_TP/` con nota explicando el motivo.
- Cross-references: `_legacy/`, `_legacy/CarGross_TP/`, `_legacy/README.md`, `docs/01_marco_teorico.md`.

## Iteración 2 — Corridas con barrido de ρ

**Fecha**: septiembre 2026.
**Estado**: COMPLETADA.

**Decisión**: ejecutar el barrido experimental completo definido en `05_corridas_y_evaluacion.md` §2.

- Decision D2.1: matriz experimental = 3 valores de ρ × 2 datasets × R=5 barajados = 30 corridas totales.
- Decision D2.2: rangos de ρ diferenciados por dataset. Pacientes: {0.40, 0.60, 0.80} (rango amplio para mostrar efecto). Sensores: {0.50, 0.65, 0.80} (rango más estrecho, se anticipó rigidez estructural).
- Decision D2.3: implementación con `--shuffle N` para producir métrica de estabilidad interna (pairwise agreement vs run 0).
- Decision D2.4: outputs en `results/r_<dataset>_r<rho>_s<seed>.{csv,txt}` con naming consistente.

**Por qué estos valores de ρ**: rango amplio en pacientes (0.40–0.80) para capturar regímenes de baja/media/alta vigilancia; rango más estrecho en sensores (0.50–0.80) porque la intuición era que el dataset tiene pocas clases intrínsecas y ρ alto no agregaría información.

**Resultados clave**: ver `docs/informe_corridas.md` §3 y §4.

## Iteración 3 — Informe narrativo de corridas

**Fecha**: septiembre 2026.
**Estado**: COMPLETADA.

**Decisión**: redactar `docs/informe_corridas.md` con interpretación narrativa, no solo tabla.

- Decision D3.1: el informe incluye interpretaciones clínicas tentativas (no triaje automático) para los clusters del dataset pacientes.
- Decision D3.2: el informe documenta honestamente que el dataset sensores es estructuralmente rígido (ρ no influye).
- Decision D3.3: discovery clave — sensores tiene 6 vectores binarios únicos, ART1 produce 5 clusters efectivos (S027 absorbido por cluster 3 durante fit).

**Por qué interpretaciones tentativas y no automáticas**: el modelo agrupa; el médico interpreta. Ver `docs/02_problema_y_alcance.md` §3 sobre el encuadre de "exploración, no triaje".

## Iteración 4 — Cierre del TFI: manuales y materiales auxiliares

**Fecha**: septiembre 2026.
**Estado**: COMPLETADA.

- Decision D4.1: `docs/manual_referencia.md` cubre las 4 secciones exigidas por la consigna (alcances/limitaciones, instalación, test demo, FAQ).
- Decision D4.2: FAQ con 6 preguntas (exigían mínimo 3). Cubre: pocos clusters, exemplares vacíos, features no binarios, sensibilidad al orden, rigidez de sensores, comparación con otros algoritmos.
- Decision D4.3: `docs/ppt_outline.md` con 13 slides + notas del orador. Duración sugerida ~10 min. Decisión: proveer outline, no generar el .pptx (requiere audio embebido, fuera del alcance automatizado).
- Decision D4.4: `docs/caratula_template.md` con datos a completar y layout sugerido. Decisión: proveer template, no generar el PDF/imagen final (requiere logos UADER/IDTI que no están en el repo).

## Iteración 4.5 — Corrección factual (verificación empírica)

**Fecha**: septiembre 2026.
**Estado**: COMPLETADA.

**Decisión**: verificar empíricamente la afirmación de "5 vectores únicos" en sensores.

- Decision D4.5.1: se escribió un script temporal que binariza el dataset con `DataLoader` y cuenta vectores únicos con `set()` de tuplas.
- Decision D4.5.2: resultado empírico = **6 vectores únicos**, no 5. ART1 produce 5 clusters porque el sexto vector (S027, freq=1) se absorbe en el cluster 3 durante fit por un artefacto de orden de presentación + fast learning (AND entre S027 y T_3 produce el vector completo, ratio 1.0, pasa cualquier vigilancia).
- Decision D4.5.3: corrección propagada a `docs/informe_corridas.md` (líneas 111, 145, 151, 198, 234) y `results/resumen_corridas.md` (líneas 71, 74).

**Lección**: las afirmaciones cuantitativas en informes técnicos deben verificarse empíricamente, no por intuición. "5 clusters = 5 vectores únicos" era razonable pero incorrecto.

## Iteración 5 — Limpieza final y publicación

**Fecha**: septiembre 2026.
**Estado**: COMPLETADA.

- Decision D5.1: borrar `src/data_loader.py` y `src/utils.py` (eran placeholders vacíos del esqueleto original, nadie los importa).
- Decision D5.2: agregar `*.pdf` al `.gitignore` (los papers Lau son material de la cátedra, no parte del entregable).
- Decision D5.3: reescribir `README.md` raíz para reflejar el estado real del proyecto + documentar los 5 comandos de prueba.
- Decision D5.4: 7 commits pusheados a GitHub con conventional commits format, sin atribución de IA.

**Por qué borrar los archivos vacíos en vez de dejarlos**: el código está consolidado en `src/CarGross.py` (660 líneas) por la consigna oficial que exige "una red llamada `CarGross.py`". Archivos vacíos sugieren modularidad inexistente.

## Resumen de cobertura

| Iteración | Qué decidió | Estado |
|-----------|--------------|--------|
| 0 | Pivot MLP→ART1 | ✅ documentado |
| 0.5 | metadata.csv + R=5 + voltaje out_of_range | ✅ documentado |
| 1 | Implementación ART1 puro stdlib | ✅ documentado |
| 1.5 | Aislamiento de CarGross_TP/ a _legacy/ | ✅ documentado |
| 2 | 30 corridas con barrido de ρ | ✅ NUEVO |
| 3 | Informe narrativo + corrección 6 vectores | ✅ NUEVO |
| 4 | Manual + PPT outline + Carátula template | ✅ NUEVO |
| 4.5 | Verificación empírica 6 vs 5 vectores | ✅ NUEVO |
| 5 | Limpieza src/ + gitignore + README + push | ✅ NUEVO |