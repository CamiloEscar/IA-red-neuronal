# 08 · Informe narrativo de corridas

## 1. Introducción

Este documento corresponde a las **iteraciones 2 y 3 del TFI**, en las que se ejecutó la matriz experimental completa: 30 corridas de ART1 sobre los dos datasets provistos, barriendo el parámetro de vigilancia $\rho$ en tres valores por dataset y repitiendo cada combinación con cinco barajados distintos del orden de entrada. Los **números y las tablas agregadas** ya están consolidados en `results/resumen_corridas.md` (generado el 2026-09-07 a partir de las 30 salidas TXT y 60 CSV que produce `src/CarGross.py`); este informe **no los repite**, sino que los **interpreta**: describe qué se observó, por qué es esperable, y qué matiza los resultados.

El propósito del documento es triple. En primer lugar, dar lectura clínica tentativa (exploratoria, **no prescriptiva**) a los exemplares que descubrió ART1 sobre el dataset de pacientes: qué tipo de perfil representa cada cluster, qué tan robusto es, y qué se preserva al mover $\rho$. En segundo lugar, justificar por qué el dataset de sensores no muestra sensibilidad a $\rho$ y por qué eso **no es un bug** sino una propiedad de la estructura latente del dataset. En tercer lugar, listar las amenazas a la validez del experimento completo, en línea con el espíritu de `docs/06_limitaciones_y_etica.md`.

> **Aviso al lector**: toda mención a "perfil clínico" en este informe debe leerse como **hipótesis exploratoria** basada en las features binarizadas. El sistema no diagnostica, no prescribe y no decide derivaciones; esa postura se sostiene en `docs/02_problema_y_alcance.md` y `docs/06_limitaciones_y_etica.md`, y se da por reproducida acá.

El alcance temporal del informe es lo registrado en `docs/07_iteraciones.md` bajo las entradas de Iteración 2 (corridas con barrido de $\rho$) e Iteración 3 (barrido completo + comparaciones + este documento). Para reproducibilidad, cada TXT individual en `results/` lleva nombre `r_{dataset}_r{rho}_s{seed}.txt`; este informe cita los seeds principales (42 a 46) pero los patrones agregados que se discuten **son estables entre seeds** y se reproducen con cualquier permutación del orden de entrada.

## 2. Diseño experimental (recap breve)

El diseño completo está en `docs/05_corridas_y_evaluacion.md`. Se repiten acá los puntos operativos para que el resto del documento sea autosuficiente.

**Datasets**:

| Dataset | Archivo | Filas | Features originales | Features binarios |
|---------|---------|-------|---------------------|--------------------|
| Pacientes | `data/dataset1_pacientes.csv` | 55 | 7 numéricos | 7 |
| Sensores | `data/dataset2_sensores.csv` | 55 | 7 numéricos | 8 (voltaje se modela con dos reglas `lt 219` + `gt 221`) |

**Binarización**: ambos datasets se binarizan con los umbrales documentados en `data/metadata.csv`. Para pacientes se aplican siete reglas `gte` sobre los features `edad`, `presion_sistolica`, `presion_diastolica`, `colesterol`, `glucosa`, `imc`, `frecuencia_cardiaca`; cada umbral tiene una justificación clínica en el CSV (guías AHA / ATP III / ADA / OMS, mayormente). Para sensores se aplican las mismas reglas operativas, salvo el voltaje que se desdobla en dos: `voltaje_lt` (V < 219) y `voltaje_gt` (V > 221), para capturar **fuera de rango nominal 220V ±1V** sin firmar. Por eso el espacio binario del dataset sensores tiene 8 dimensiones, no 7.

**Valores de $\rho$**:

| Dataset | $\rho$ bajos | $\rho$ medio | $\rho$ alto | R (barajados) |
|---------|--------------|--------------|-------------|----------------|
| Pacientes | 0.40 | 0.60 | 0.80 | 5 (seeds 42–46) |
| Sensores | 0.50 | 0.65 | 0.80 | 5 (seeds 42–46) |

Las dos series son ligeramente distintas porque los dominios lo son: en pacientes un 0.50 cae cerca de un umbral "acepta casi todo" mientras que en sensores, por la estructura binaria más rígida, ese mismo valor se mueve diferente. Se eligió 0.65 como valor de cruce porque es razonable para ambos. La cantidad de repeticiones se subió de R=3 (como estaba en `docs/05_corridas_y_evaluacion.md` original) a **R=5** precisamente para tener 10 pares por celda y una estimación de la estabilidad menos ruidosa, decisión documentada como D0.5.2 en `docs/07_iteraciones.md`.

**Total**: 30 corridas (2 datasets × 3 valores de $\rho$ × 5 barajados).

**Implementación**: una sola, `src/CarGross.py`, que ejecuta ART1 según la transcripción del Box 3 de Lau 1992 (`docs/04_algoritmo.md`). Cada corrida produce un CSV con la asignación de clusters fila por fila y un TXT con el resumen (K, tamaños, exemplares, scores). Los 30 TXT se agregaron en `results/resumen_corridas.md`. **No se modifica ningún otro archivo del proyecto** durante la escritura de este informe.

## 3. Resultados: dataset pacientes

Esta sección describe lo observado en el dataset `data/dataset1_pacientes.csv`. Los números agregados están en `results/resumen_corridas.md` (tabla `Dataset 1: pacientes`); los IDs de cluster que se citan corresponden al seed 42, representativos del patrón estable.

### 3.1. Resumen cuantitativo por $\rho$

| $\rho$ | $K$ totales | Clusters con miembros | Score medio | Cluster mayoritario (tamaño) | Cluster secundario (tamaño) | Acuerdo vs run 0 |
|---------|-------------|------------------------|---------------|--------------------------------|------------------------------|--------------------|
| 0.40 | 25 | 2 | 0.403 | 38 pacientes (perfil "edad") | 17 pacientes (perfil "edad+colesterol+imc") | 0.543 ± 0.091 |
| 0.60 | 26 | 3 | 0.517 | 35 pacientes (perfil "edad") | 16 pacientes (perfil "edad+PAS+PAD+colesterol+imc") + 4 (perfil "edad+colesterol") | 0.534 ± 0.057 |
| 0.80 | 28 | 5 | 0.570 | 35 pacientes (perfil "edad") | 9 pacientes (perfil **"multi-riesgo"**) + 7 + 3 + 1 | 0.510 ± 0.072 |

**Lectura de la tabla**: la columna "$K$ totales" crece monotónicamente con $\rho$ (25 → 26 → 28), coherente con la intuición teórica de que mayor vigilancia exige más especificidad y produce más particiones. La columna "score medio" también crece monotónicamente (0.403 → 0.517 → 0.570): a mayor $\rho$, los clusters resultantes son **más compactos** (más homogéneos internamente), aunque estén **menos poblados**. El número de clusters **con miembros** también crece (2 → 3 → 5), aunque el grueso del crecimiento va a la cola de clusters minoritarios — el cluster mayoritario ("edad", 35-38 pacientes) permanece casi invariante en tamaño en los tres $\rho$.

### 3.2. $\rho = 0.40$ — pocos clusters grandes, heterogéneos

Con vigilancia baja, ART1 acepta solapamientos parciales: cualquier entrada que tenga una fracción razonable de sus bits en común con algún exemplar es aceptada en ese cluster. El resultado es **pocos clusters muy grandes y heterogéneos**.

Para el seed 42:

- **Cluster 0** (38 pacientes, score 0.355): exemplar `[1 0 0 0 0 0 0]`. Sólo el bit de `edad` está activo. Es decir, este cluster agrupa a **38 de los 55 pacientes** cuya única feature por encima del umbral es la edad (≥ 40 años). El score medio de 0.355 es bajo: significa que, en promedio, las filas asignadas a este cluster **comparten sólo ~35 % de sus bits activos con el exemplar**. Esto es esperable: el exemplar es muy "permitivo" (un solo bit en 1), así que admite a casi cualquier paciente que simplemente tenga la edad por encima del umbral — incluyendo pacientes sin colesterol alto, sin hipertensión, sin alteraciones glucémicas. El cluster es grande y **heterogéneo** en el espacio binario.
- **Cluster 2** (17 pacientes, score 0.509): exemplar `[1 0 0 1 0 1 0]`. Bits activos: `edad`, `colesterol`, `imc`. Es el subgrupo "adultos con colesterol alto e IMC elevado". Score algo más alto (0.509) porque el exemplar es más específico y los 17 pacientes efectivamente comparten ese patrón.
- Los **23 clusters restantes** están vacíos (exemplar todo en 0). No contienen filas; se reportan por separado y no aparecen en las métricas principales. Su presencia es la firma habitual de ART1 con vigilancia baja: el algoritmo **preasigna** nodos de salida pero la mayoría quedan sin activar.

**Interpretación clínica tentativa**: con $\rho = 0.40$ el modelo ofrece una **imagen muy agregada** de la cohorte: "pacientes adultos" y "adultos con síndrome metabólico incipiente". Esta lectura, aunque didácticamente útil, **no discrimina perfiles intermedios** (por ejemplo, pacientes hipertensos sin dislipidemia). El score bajo (0.40) es la huella cuantitativa de esa heterogeneidad: no es un problema del modelo, es lo que cabe esperar cuando el modelo agrupa con poca exigencia.

### 3.3. $\rho = 0.60$ — el equilibrio

Con vigilancia media, ART1 rechaza las entradas que no se parecen lo suficiente a sus exemplares, así que los clusters se **especializan** y crece el número de particiones descubiertas. Para el seed 42:

- **Cluster 0** (35 pacientes, score 0.343): el mismo perfil "edad" del experimento anterior, ahora con un paciente menos (35 vs 38). El paciente que antes caía acá pero que ya no califica fue "capturado" por otro cluster con el que resuena mejor. Score casi idéntico (0.343), porque el patrón del cluster es el mismo.
- **Cluster 2** (4 pacientes, score 0.917): exemplar `[1 0 0 1 0 0 0]`, perfil "edad + colesterol". Score alto (0.917) porque es **un cluster de alta especificidad** y baja población: cuatro pacientes adultos con colesterol alto pero sin hipertensión ni obesidad acompañante. Este cluster es **nuevo respecto a $\rho = 0.40$**: con vigilancia baja, esos cuatro pacientes quedaban diluidos en el cluster mayoritario; con $\rho = 0.60$ se separan.
- **Cluster 6** (16 pacientes, score 0.798): exemplar `[1 1 1 1 0 1 0]`, perfil "edad + presion_sistolica + presion_diastolica + colesterol + imc". Este es el **perfil cardiometabólico clásico** según las features activas: hipertensión sistólica y diastólica, colesterol alto, sobrepeso (IMC ≥ 30), edad adulta. El score de 0.798 indica buena homogeneidad interna — los 16 pacientes efectivamente comparten el patrón.

**Interpretación clínica tentativa**: con $\rho = 0.60$ la lectura se vuelve **clínicamente más informativa**. Aparecen dos perfiles secundarios diferenciados:

- **"Hipertensión + dislipidemia + sobrepeso en adulto"** (cluster 6): el fenotipo más cercano al sindrome metabólico clásico con tensión elevada. Es, dentro del dataset, el perfil con mayor cantidad de factores de riesgo simultáneos **sin incluir glucemia alterada**.
- **"Dislipidemia aislada en adulto"** (cluster 2): cuatro pacientes con colesterol alto sin el resto del cuadro metabólico. El score muy alto (0.917) sugiere que la separación es **cirugía**: el modelo dice "estos cuatro se parecen mucho entre sí y poco al resto".

### 3.4. $\rho = 0.80$ — muchos clusters pequeños, alta especificidad

Con vigilancia alta, ART1 exige coincidencias casi exactas: si una fila no resuena con el exemplar de un cluster existente, se crea un cluster nuevo, consumiendo más nodos. El resultado es **más clusters, más chicos, más específicos**. Para el seed 42:

- **Cluster 0** (35 pacientes, score 0.343): el mismo perfil "edad". Permanece invariante: sigue siendo el "depsito" por defecto de los pacientes adultos sin otros factores.
- **Cluster 2** (3 pacientes, score 1.000): perfil "edad + colesterol". Score perfecto: los tres pacientes son **bit-exactos** con el exemplar (su vector binarizado coincide exactamente). En $\rho = 0.60$ este cluster tenía 4 miembros con score 0.917; al subir $\rho$ se separó el cuarto paciente en otro cluster.
- **Cluster 6** (1 paciente, score 1.000): perfil "edad + colesterol + imc". Un solo paciente (ID 43) que **encaja exactamente** en el patrón "adulto con colesterol alto y sobrepeso, sin hipertensión ni glucemia". Score perfecto. Es el límite de la especificidad: **un cluster de un solo miembro es, esencialmente, un outlier**.
- **Cluster 11** (7 pacientes, score 0.905): perfil "edad + PAS + PAD + colesterol + imc", esencialmente el mismo cluster 6 de $\rho = 0.60$ pero con un miembro menos (7 vs 16). El resto migró a otros clusters.
- **Cluster 27** (9 pacientes, score 1.000) — **"multi-riesgo"**: exemplar `[1 1 1 1 1 1 1]`, **todos los siete features activos**. Score 1.000 porque los nueve pacientes son bit-exactos: tienen edad, hipertensión sistólica y diastólica, colesterol alto, glucosa alterada en ayunas, sobrepeso/obesidad y taquicardia en reposo, **simultáneamente**. Este es el **perfil más completo** del dataset y el que más se parece a un "paciente de alto riesgo cardiovascular y metabólico acumulado" en la cohorte.

**Interpretación clínica tentativa**: con $\rho = 0.80$ el modelo ofrece una visión **granular** de la cohorte. El dato clínicamente más interesante es la **emergencia del cluster 27 "multi-riesgo"**, que sólo aparece con esta vigilancia: siete factores de riesgo presentes en simultáneo es lo que, en la literatura, se asocia con peor pronóstico cardiovascular. Los 9 pacientes identificados acá son, dentro del dataset sintético, los que más se acercarían a una "alerta roja". Pero **esto es exploración, no triaje**: el modelo agrupa, no deriva.

> **Nota clínica honesta**: el modelo no establece causalidad ni predice eventos. La asociación "todos los features activos → alto riesgo" es un placeholder del dataset simulado, no una validación clínica. En una cohorte real, ese mismo vector binarizado podría tener valor pronóstico, requeriría validación prospectiva, o ser artefacto de los umbrales elegidos.

### 3.5. Comparación clínica transversal entre $\rho$

| Aspecto clínico | $\rho = 0.40$ | $\rho = 0.60$ | $\rho = 0.80$ |
|-------------------|----------------|----------------|----------------|
| Cluster "mayoritario" | "edad" (38 pacientes) | "edad" (35) | "edad" (35) |
| Cluster "síndrome metabólico clásico" | diluido en el mayoritario | "edad+PAS+PAD+colesterol+imc" (16) | "edad+PAS+PAD+colesterol+imc" (7) |
| Cluster "dislipidemia aislada" | no existe | "edad+colesterol" (4) | "edad+colesterol" (3) |
| Cluster "multi-riesgo" (todos los features) | no existe | no existe | "multi-riesgo" (9, score 1.0) |
| Granularidad | muy baja | media | alta |
| Score medio | 0.403 | 0.517 | 0.570 |

El cuadro no necesita glosa: a mayor $\rho$, el modelo **gana granularidad y pierde robustez poblacional** de cada cluster. La decisión de qué $\rho$ usar **depende del uso**: si se quiere una primera caracterización gruesa de la cohorte, $\rho = 0.40$ alcanza; si se quieren perfilar pacientes con comorbilidades específicas, $\rho = 0.60$ es razonable; si se quiere identificar el subgrupo de "todos los factores prendidos", sólo $\rho = 0.80$ lo aísla.

## 4. Resultados: dataset sensores

Esta sección describe lo observado en `data/dataset2_sensores.csv`. La diferencia con la sección anterior es **deliberada**: lo más importante de este dataset **no es lo que cambia con $\rho$, sino lo que no cambia**.

### 4.1. Estructura del dataset y por qué $\rho$ deja de importar

El dataset sensores tiene 55 filas en un espacio binario de **8 dimensiones** (la octava dimensión es por el desdoblamiento del voltaje en `voltaje_lt` + `voltaje_gt`). Al inspeccionar las 55 filas binarizadas, el dataset tiene **solamente 6 vectores únicos** en ese espacio de 256 combinaciones posibles. Es decir: el dataset es **estructuralmente rígido**, no porque el muestreo sea malo sino porque las features están correlacionadas (las máquinas no encienden temperatura sin vibración, no registran rpm bajas sin tiempo de operación acumulado, etc.).

ART1 produce **5 clusters efectivos** sobre estas 6 entradas únicas, **no** porque el dataset sea rígidamente de 5 clases, sino porque el sexto vector único (`[1 0 0 1 0 0 0 1]` = `temperatura` + `voltaje_lt` + `tiempo_operacion`, presente sólo en la fila S027 del CSV) **queda absorbido por el cluster 2 existente durante el fit**. Es un **artefacto del orden de presentación** propio del 'fast learning' de ART1: cuando S027 entra al fit, el cluster 3 ya está activo y su exemplar `[1 1 1 1 0 1 0 1]` cumple trivialmente la condición de vigilancia para S027 (el AND devuelve el vector completo de S027, así que `||T*X|| / ||X|| = 1.0` para cualquier $\rho$), y ART1 no fuerza la creación de un sexto nodo. Con un orden de entrada distinto, el resultado podría haber sido K=6.

Sobre los 5 vectores que efectivamente terminan como exemplares, **el parámetro $\rho$ deja de tener margen para discriminar**: cualquiera de ellos superará fácilmente cualquier vigilancia por debajo de 1.0, porque ya son vectores discretos. Por eso se observa el resultado siguiente en los tres $\rho$:

| $\rho$ | $K$ totales | Clusters con miembros | Score medio | Acuerdo vs run 0 |
|---------|-------------|------------------------|---------------|--------------------|
| 0.50 | 5 | 5 | 0.994 | 0.486 ± 0.193 |
| 0.65 | 5 | 5 | 0.994 | 0.486 ± 0.193 |
| 0.80 | 5 | 5 | 0.994 | 0.517 ± 0.246 |

**Lectura**: el número de clusters es **idéntico** (5) para los tres valores de $\rho$. El score medio es prácticamente constante en 0.994. La única celda donde el acuerdo entre barajados varía es $\rho = 0.80$, donde el desvío estándar se agranda (0.246 vs 0.193) y la media sube (0.517 vs 0.486), pero **la composición de los clusters es la misma**.

### 4.2. Los cinco clusters observados

Los IDs corresponden a `results/r_sensores_r0.65_s42.txt`, representativos del patrón estable:

- **Cluster 0** (29 lecturas, score 1.000): exemplar `[0 0 0 0 0 0 1 0]`. Sólo el bit `rpm` (≥ 1450 RPM) está activo. Es el **régimen mayoritario del dataset**: lecturas en las que la única feature por encima del umbral es rpm, con todas las demás (temperatura, vibración, presión, voltaje, corriente, tiempo de operación) por debajo. Es decir, lecturas de "operación nominal sin alertas". Score perfecto porque los 29 vectores asignados son bit-exactos con el exemplar.
- **Cluster 1** (3 lecturas, score 1.000): exemplar `[1 0 0 0 0 0 0 0]`. Sólo `temperatura` activa. Tres lecturas en las que la única alerta es por temperatura elevada (≥ 75 °C). Score perfecto.
- **Cluster 2** (3 lecturas, score 0.889): exemplar `[1 0 0 0 0 0 0 1]`. `temperatura + tiempo_operacion`. De las tres lecturas, **dos son bit-exactos** con el exemplar (score 1.0) — vectores con sólo `temperatura` y `tiempo_operacion` activos. La tercera lectura (S027 del CSV) tiene el vector atípico `[1 0 0 1 0 0 0 1]` = `temperatura` + `voltaje_lt` + `tiempo_operacion`, con un bit adicional (`voltaje_lt`) prendido; esta lectura queda absorbida en el cluster 2 durante el fit (no genera un sexto nodo, ver §4.1) y reporta score 0.667 en predict-time (el AND entre S027 y el exemplar produce 2 de 3 bits activos). El score medio 0.889 = (1.0 + 1.0 + 0.667) / 3 refleja esa mezcla.
- **Cluster 3** (17 lecturas, score 1.000): exemplar `[1 1 1 1 0 1 0 1]`. **Seis features activos**: temperatura, vibración, presión, `voltaje_lt`, corriente, tiempo_operacion. Es el **régimen de "alerta múltiple"**: 17 lecturas en las que casi todos los indicadores operativos están prendidos en simultáneo — la máquina está fuera de rango en muchos parámetros. Score perfecto.
- **Cluster 4** (3 lecturas, score 1.000): exemplar `[1 0 0 0 0 0 1 0]`. `rpm + tiempo_operacion`. Tres lecturas con ambas features activas. Score perfecto.

> **Nota técnica menor**: cluster 4 y cluster 0 **no** tienen el mismo exemplar (cluster 0 es sólo `rpm`, cluster 4 es `rpm + tiempo_operacion`); pero un lector distraído puede confundirlos al ver el reporte agregado donde la columna "tamaño" puede sugerir equivalencias. En los TXT individuales con los vectores explícitos se distinguen claramente. Esta es una observación de transparencia, no una crítica al modelo.

### 4.3. Interpretación operativa tentativa

El dataset sensores "habla" de regímenes de operación de una máquina, no de pacientes. La lectura tentativa es:

- **Régimen mayoritario estable** (cluster 0, 29 lecturas, ~53% del dataset): operación nominal. La única alerta que prende es rpm alta, lo que sugiere que la máquina corre cerca del límite superior del rango operativo pero sin otras anomalías. Es **el estado base** en este dataset.
- **Régimen de alerta múltiple** (cluster 3, 17 lecturas, ~31% del dataset): la máquina registra múltiples indicadores fuera de rango simultáneamente — probablemente eventos de estrés operativo sostenido (carga + calor + vibración + presión). Es el segundo régimen más frecuente.
- **Regímenes minoritarios** (clusters 1, 2, 4; 3 lecturas cada uno, ~5.5% cada uno): lecturas aisladas con combinaciones específicas de 1 o 2 features activas. Suman 9 lecturas (~16%) y representan **eventos raros o transitorios** del operativo.

**Lo que el dataset NO permite mostrar**: la sensibilidad del algoritmo al parámetro $\rho$. Como se argumentó arriba, el dataset tiene 6 vectores únicos en el espacio binario y ART1 produce los mismos 5 clusters a cualquier $\rho$ razonable (con el sexto vector absorbido por el cluster 2 durante el fit, fenómeno que no depende de $\rho$). **Esto no es un bug, es una limitación del dataset**: la estructura latente es demasiado rígida (6 vectores únicos sobre 256 combinaciones posibles) para que el barrido de vigilancia sea informativo. Sirve para mostrar **consistencia** y **estabilidad de asignación** (los mismos 5 clusters aparecen siempre), pero no para **discriminar efectos de vigilancia**.

### 4.4. Por qué esto importa y qué decisión tomamos

Que el dataset sensores no muestre sensibilidad a $\rho$ **es la razón por la que se mantienen ambos datasets en el experimento**: el primero (pacientes) muestra el régimen donde ART1 responde al barrido, el segundo (sensores) muestra el régimen donde el algoritmo es invariante a $\rho$. La documentación honesta de este segundo caso es parte del resultado, no una "ausencia de resultado". Si no se documentara, un revisor podría asumir que el código está mal o que el barrido está mal calibrado, cuando en realidad la causa es estructural: el dataset no tiene la variedad latente que exigiría discriminar con vigilancia.

Por eso, en `results/resumen_corridas.md` se reportan los tres $\rho$ para sensores con números idénticos en $K$ y score, y se agrega la observación cualitativa: "el dataset tiene solo 6 vectores únicos en el espacio binario y $\rho$ deja de ser el factor dominante".

## 5. Estabilidad entre barajados

### 5.1. Acuerdo pairwise vs run 0

La métrica adoptada (decisión D1.4 en `docs/07_iteraciones.md`) es **pairwise agreement fraction** entre las asignaciones de cada corrida barajada y el run 0 (orden de entrada original). Es más simple que el ARI formal y suficiente para $N = 55$.

Los valores medios por celda están en las tablas de `results/resumen_corridas.md`. Resumidos:

| Celda | Acuerdo medio | Desvío estándar | Lectura |
|-------|----------------|------------------|---------|
| pacientes $\rho = 0.4$ | 0.543 | 0.091 | moderada, mejor que las otras |
| pacientes $\rho = 0.6$ | 0.534 | 0.057 | moderada, la más estable |
| pacientes $\rho = 0.8$ | 0.510 | 0.072 | moderada, levemente peor |
| sensores $\rho = 0.5$ | 0.486 | 0.193 | moderada, pero alta varianza |
| sensores $\rho = 0.65$ | 0.486 | 0.193 | moderada, pero alta varianza |
| sensores $\rho = 0.8$ | 0.517 | 0.246 | moderada, peor varianza |

### 5.2. Lectura de los números

- **Promedio general ~0.50**: el acuerdo entre barajados es **moderado**, no excelente. Un valor 1.0 indicaría que las asignaciones no cambian al barajar; un valor 0.0 indicaría reasignaciones completamente aleatorias. 0.50 cae en una zona intermedia, esperable en clustering secuencial.
- **Variabilidad por celda (0.057 a 0.246)**: la celda **pacientes $\rho = 0.6$** es la más estable (std = 0.057); la celda **sensores $\rho = 0.8$** es la menos estable (std = 0.246). Esto es coherente: con vigilancia media y dataset heterogéneo, ART1 tiene un patrón bien definido de qué cluster crece primero; con vigilancia alta y dataset binario discreto, pequeñas diferencias de orden pueden "saltar" un patrón del cluster A al cluster B.
- **Cantidad de clusters estable**: el desvío estándar de $K$ es **exactamente 0.000** en las seis celdas. Es decir, **el número total de clusters no cambia** entre barajados: ART1 converge a la misma partición estructural (mismo $K$, mismos tamaños promedio) incluso cuando las asignaciones individuales se barajan.

### 5.3. Implicancia: sensibilidad al orden vs propiedades del algoritmo

El acuerdo moderado (~0.50) **no es un bug**. Es una **propiedad esperada** del algoritmo ART1:

1. ART1 procesa entradas en serie (secuencial), no en bloque. El primer patrón crea el cluster 0, el primer patrón rechazado crea el cluster 1, y así.
2. En regiones de **transición** (entradas que están en el borde entre dos clusters), un cambio de orden puede hacer que la entrada sea absorbida por el cluster que se creó antes o después.
3. Sin embargo, el **algoritmo es estable en sus parámetros agregados** ($K$, distribución de tamaños, score medio) porque la dinámica subyacente —renormalización de $b_{ij}$, AND sucesivo del exemplar, vigilancia fija— es determinista.

**Implicancia práctica**: si se necesitara estabilidad perfecta entre orden y asignación, se debería complementar ART1 con un mecanismo de votación o consenso sobre múltiples barajados (lo que en clustering bayesiano se llama "model averaging"). Eso está fuera del alcance de este TFI. En su lugar, **se reporta explícitamente la variabilidad** para que el lector sepa que la asignación individual de una fila no es invariante.

> **Conclusión operativa**: la cantidad de clusters y la composición agregada son confiables. La asignación específica de cada fila a un cluster, en regiones ambiguas, **no lo es**. Si se necesita certeza por fila, ART1 no es la herramienta adecuada (usar un clasificador supervisado).

## 6. Limitaciones y amenazas a la validez

Esta sección agrupa y completa lo enunciado en `docs/06_limitaciones_y_etica.md` con foco específico en **lo que amenaza la interpretación de las 30 corridas**. La sección 06 enumera limitaciones técnicas y éticas generales del TFI; esta sección mira a la luz de los resultados concretos qué se sostiene y qué no.

### 6.1. Amenazas a la validez interna

1. **Dataset pequeño ($N = 55$ por dataset)**. Con 55 filas, el poder estadístico de cualquier métrica (acuerdo entre barajados, score medio, tamaño de clusters) es limitado. La métrica ARI formal con $N = 55$ y $R = 3$ produce sólo 3 pares y es ruidosa; por eso se subió a $R = 5$ (decisión D0.5.2). Aun con $R = 5$, la varianza del acuerdo pairwise no es despreciable (hasta 0.246 en sensores $\rho = 0.8$).

2. **Binarización con umbrales fijos**. La información numérica se pierde al binarizar. Un paciente con presión arterial 139 mmHg y otro con 141 mmHg son, después del umbral gte 140, **filas distintas** (uno en 0, otro en 1). Dos pacientes con 90 y 109 mg/dL son el mismo bit post-umbral gte 110. Esto **no es neutral**: la elección del umbral cambia la estructura latente del dataset y por lo tanto los clusters que ART1 descubre. Los umbrales aquí están justificados clínicamente (`data/metadata.csv`, columna `justification`), pero siguen siendo una **decisión del experimentador**.

3. **Dataset sensores estructuralmente rígido**. Como se argumentó arriba, este dataset tiene 6 vectores únicos en el espacio binario y ART1 produce 5 clusters (con el sexto vector absorbido por el cluster 2 durante el fit, ver §4.1); la estructura latente es demasiado rígida para que el barrido de vigilancia sea informativo. Esto es una amenaza a la **validez de constructo** del experimento para sensores: se probaron tres valores de $\rho$ pero el barrido no fue informativo. La consecuencia es que **el segundo dataset aporta información sobre consistencia del algoritmo, no sobre su respuesta a vigilancia**.

4. **Clusters zero-exemplar**. ART1 crea clusters cuyo exemplar queda completamente en 0 (todos los bits apagados). Esto ocurre cuando una fila sin features activos fuerza un nuevo nodo. En el TXT se reportan por separado como "(se omiten 23 clusters con exemplar vacío)", como se documenta en `docs/06_limitaciones_y_etica.md`. Existen internamente, ocupan los IDs 1, 3, 4, 5, ... en las corridas, y explican por qué el "$K$ totales" del resumen es mayor que el "$K$ con miembros". Por ejemplo, en pacientes $\rho = 0.4$ se crean 25 nodos pero sólo 2 capturan filas. **Esto es propiedad del algoritmo** (no es un bug), pero merece ser entendido: si uno ve "$K = 25$" puede alarmarse pensando que hay 25 grupos de pacientes, cuando en realidad sólo 2 son informativos.

5. **Acuerdo moderado entre barajados (~0.50)**. Como se discutió en §5, ART1 es sensible al orden de entrada. El acuerdo pairwise de ~0.50 no es "malo", pero sí **insuficiente para afirmar que cada paciente individual pertenece inequívocamente al cluster asignado**. Especialmente en clusters minoritarios (1-4 miembros), un cambio de orden podría reasignar las filas.

6. **Score medio bajo en pacientes con $\rho = 0.4$ (0.403)**. Refleja la heterogeneidad estructural de los clusters con vigilancia baja: cuando el exemplar es permisivo (un solo bit en 1), las filas asignadas **comparten pocos bits** con el exemplar. Eso baja el score. **No es defecto del modelo**, es lo que cabe esperar cuando se pide mucha agrupación con poca exigencia.

### 6.2. Amenazas a la validez externa

7. **Sin gold standard**: no hay etiquetas para validar las asignaciones. Esto es propio del clustering no supervisado, pero conviene tenerlo presente al evaluar las "interpretaciones tentativas" de §3 y §4. **El modelo agrupa; el humano lee el exemplar; no hay forma automática de decidir si la lectura es correcta**.

8. **Sin comparación con otros algoritmos**: no se compara con K-means, DBSCAN, clustering jerárquico u otros. La elección de ART1 vino dada por la consigna (`_legacy/CarGross_TP/consignas_TP.md`, Actividad #3), pero conviene recordar que es **una de varias alternativas posibles** y que su idoneidad depende de la tarea.

9. **Sin evaluación de generalización**: clustering no tiene "test set" en el sentido supervisado. No se puede preguntar "¿qué predice el modelo para una fila no vista?" — el modelo **asigna** cada fila vista al cluster existente, y si nada resuena crea uno nuevo. Esto se discute en `docs/06_limitaciones_y_etica.md` y se reproduce acá.

### 6.3. Amenazas a la interpretación clínica

10. **Interpretación clínica tentativa, no validada**. Las frases del estilo "perfil cardiometabólico clásico" o "paciente de alto riesgo acumulado" son **descripciones basadas en las features activas**, no juicio clínico. Un cardiólogo o endocrinólogo podría agrupar los mismos 55 pacientes sintéticos de forma muy distinta. El modelo ofrece una **agrupación**; la interpretación es trabajo humano.

11. **Datos sintéticos**. Los datasets son explícitamente sintéticos según `docs/02_problema_y_alcance.md` §2 y `docs/06_limitaciones_y_etica.md` §2.3. Cualquier extrapolación a pacientes reales **no está avalada** por este TFI.

12. **Sin triaje automático**. El sistema no prescribe, no diagnostica, no deriva. Esto se enuncia en `docs/02_problema_y_alcance.md` §2 y se reitera acá.

### 6.4. Lo que queda fuera del alcance (recordatorio)

`docs/06_limitaciones_y_etica.md` §4 ya lista lo que un sistema con pretendida utilidad clínica necesitaría (validación prospectiva multi-centro, aprobación regulatoria ANMAT, auditoría ISO 27001 / 13485, etc.). Este TFI no cumple ninguno de esos requisitos y **no pretende** cumplirlos. Se reitera acá por completitud, no por autocrítica vacía.

## 7. Conclusiones del TFI

Las corridas cierran el ciclo experimental del TFI. Lo que se aprendió:

- **ART1 funciona como clustering no supervisado**. La implementación `src/CarGross.py` produce asignaciones coherentes con lo esperado teóricamente en todos los 30 runs: el número de clusters crece con la vigilancia (en pacientes), los clusters son internamente homogéneos (score 0.99 en sensores, 0.40-0.57 en pacientes), y los exemplares son vectores binarios consistentes con el Box 3 de Lau 1992.

- **La elección de $\rho$ tiene impacto real en datasets heterogéneos**. En `data/dataset1_pacientes.csv`, mover $\rho$ de 0.40 a 0.80 cambia el número de clusters con miembros (2 → 3 → 5), el score medio (0.403 → 0.517 → 0.570), y la composición (aparece el cluster "multi-riesgo" sólo con $\rho = 0.80$). El experimentador tiene control real sobre la granularidad.

- **El dataset sensores es un caso degenerado que no permite mostrar sensibilidad a $\rho$**. Esto debe documentarse como **limitación del dataset**, no como defecto del algoritmo. 6 vectores únicos en el espacio binario es una estructura demasiado rígida para que el barrido de vigilancia sea informativo (ART1 produce 5 clusters por absorción del sexto vector durante el fit, no por una propiedad rígida del dataset — ver §4.1). La información válida que aporta este dataset es **consistencia**: ART1 produce los mismos 5 clusters en los 3 valores de $\rho$, lo que muestra que el algoritmo es estable cuando la entrada es estable.

- **El framing "exploración, no triaje" se sostiene**. ART1 agrupa filas en clusters con exemplares interpretables. Un profesional con conocimiento de dominio lee cada exemplar y propone un perfil. **El sistema no decide nada**. La postura de `docs/02_problema_y_alcance.md` y `docs/06_limitaciones_y_etica.md` (que el modelo no prescribe, no diagnostica, no deriva) no se ve comprometida por los resultados.

- **Trabajo futuro razonable**: probar ART2 sobre los mismos datasets para evaluar el efecto de remover la binarización; probar ARTMAP para mapear clusters a clases predefinidas sobre datasets más grandes y con features no binarizados; comparar resultados con K-means o DBSCAN sobre los mismos vectores binarizados para tener un punto de referencia de lo que ART1 aporta frente a algoritmos más convencionales. Ninguno de estos puntos está en la consigna actual y por lo tanto ninguno se entrega acá.

## 8. Referencias

- `_legacy/CarGross_TP/lau_contenido.md` — transcripción completa del paper Lau 1992, fuente primaria del algoritmo Box 3 implementado en `src/CarGross.py`.
- `_legacy/CarGross_TP/consignas_TP.md` — consigna oficial del TFI (UADER — IDTI Lab, materia Redes Neuronales), Actividad #3 que exige la red ART1.
- `docs/04_algoritmo.md` — descripción paso a paso del algoritmo ART1 implementado, con notación idéntica a Lau 1992.
- `docs/05_corridas_y_evaluacion.md` — diseño experimental completo, métricas no supervisadas (K(ρ), ARI, compactness, interpretabilidad cualitativa) y formato de tabla de resultados.
- `data/dataset1_pacientes.csv` y `data/dataset2_sensores.csv` — datasets sintéticos, $N = 55$ filas cada uno, documentados en `docs/03_dataset_y_preprocesamiento.md`.
- `data/metadata.csv` — 14 umbrales clínicos/operativos con justificación trazable a guías AHA / ATP III / ADA / OMS / ISO 10816.
- `results/resumen_corridas.md` — tablas agregadas y conclusiones cuantitativas de las 30 corridas; fuente de los números citados arriba.
- `results/r_pacientes_r{0.4,0.6,0.8}_s{42-46}.txt` y `results/r_sensores_r{0.5,0.65,0.8}_s{42-46}.txt` — reportes TXT individuales, 30 en total; este informe cita principalmente el seed 42 como representativo.
- `src/CarGross.py` — implementación de ART1 (Box 3 de Lau 1992), stdlib Python, no modificada durante la escritura de este informe.
- `docs/06_limitaciones_y_etica.md` — lista completa de limitaciones técnicas, médicas y éticas del TFI; precursor de §6 arriba.
- `docs/07_iteraciones.md` — bitácora de iteraciones; las entradas Iteración 2 e Iteración 3 son el ciclo experimental que este informe cierra.
