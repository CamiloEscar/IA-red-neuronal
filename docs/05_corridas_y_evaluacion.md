# 05 · Corridas y Evaluación

## Diseño experimental

Las corridas de ART1 se organizan como un **barrido del parámetro de vigilancia** $\rho$ sobre cada dataset, con cinco repeticiones por valor (orden de entrada barajado entre repeticiones). No hay split train/test, no hay etiquetas verdaderas, y por lo tanto **tampoco hay métricas supervisadas** (accuracy, F1, precision, recall). Toda evaluación es no supervisada.

## 1. Configuración por dataset

| Dataset | Vigilancia $\rho$ probada | Repeticiones por $\rho$ | Semilla |
|---------|--------------------------|--------------------------|---------|
| `dataset1_pacientes` | 0.40, 0.60, 0.80 | 5 (orden barajado) | aleatoria, registrada en cada corrida |
| `dataset2_sensores` | 0.50, 0.65, 0.80 | 5 (orden barajado) | aleatoria, registrada en cada corrida |

Los valores de $\rho$ se eligieron para cubrir tres regímenes:

- **Baja vigilancia** ($\rho \approx 0.4$–$0.5$): se esperan pocos clusters grandes.
- **Media** ($\rho \approx 0.6$): balance entre granularidad y generalización.
- **Alta** ($\rho \approx 0.8$): se esperan muchos clusters pequeños.

Los dos datasets usan series de $\rho$ ligeramente distintas porque sus dominios son distintos; un valor "neutro" como 0.65 es razonable para ambos.

## 2. Métricas no supervisadas

### Cantidad de clusters descubiertos

Para cada corrida se reporta $K(\rho)$. La curva $K$ vs. $\rho$ describe la **sensibilidad del modelo al umbral**: idealmente, $K$ es monótona no creciente al bajar $\rho$.

### Tamaño de los clusters

Histograma de $|C_k|$ para $k = 1, \dots, K$. Las métricas resumen son la **media** y **desvío estándar** del tamaño de cluster:

$$
\bar{s} = \frac{1}{K} \sum_{k=1}^{K} |C_k|, \qquad
\sigma_s = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \left(|C_k| - \bar{s}\right)^2}
$$

Un $\sigma_s$ muy alto señala clusters muy desbalanceados (algunos grandes, otros casi vacíos), lo que suele indicar un valor de $\rho$ mal elegido.

### Estabilidad al reordenamiento

Dada la naturaleza secuencial de ART1, distintas presentaciones pueden dar lugar a particiones estructuralmente equivalentes pero con IDs de cluster permutados. La métrica adoptada es el **Adjusted Rand Index (ARI)** entre pares de corridas del mismo $\rho$ con barajados distintos — **no requiere etiquetas**, sólo compara asignaciones.

$$
\text{ARI}_{\rho} = \frac{1}{\binom{R}{2}} \sum_{i<j} \text{ARI}(\mathcal{P}_i, \mathcal{P}_j)
$$

donde $R$ es la cantidad de repeticiones. **Con N=55 filas, R=3 producía sólo 3 pares para promediar ARI, lo que volvía la métrica ruidosa. Adoptamos entonces R=5 (5 repeticiones, 10 pares), un valor que da una estimación más robusta sin inflar el costo computacional.** Lectura:

- ARI $\approx 1$: reasignaciones consistentes (mismos clusters, posiblemente con IDs permutados).
- ARI $\approx 0$: las particiones son esencialmente aleatorias entre corridas.
- ARI negativo: peor que azar.

### Compactness (distancia media al exemplar)

Para cada fila $x \in C_k$, se calcula la **distancia de Hamming** al exemplar $E_k$:

$$
d_H(x, E_k) = \sum_{i=0}^{N-1} \mathbb{1}\{x_i \neq E_{k,i}\}
$$

y se promedia por cluster. Una compactness baja (pocos bits diferentes) indica que el AND sucesivo (Step 7) no "evaporó" el exemplar.

### Interpretabilidad cualitativa

Para cada cluster se reporta:

- el **exemplar binario** en texto (1s y 0s por feature);
- una **propuesta de perfil clínico / operativo** hecha por el equipo, basada en las features activas;
- una nota sobre **coherencia interna**: ¿el perfil es internamente consistente? ¿se parece a un fenotipo conocido?

> Esta métrica no es automática: depende del juicio humano y de las fuentes de dominio. Por su naturaleza cualitativa, debe quedar registrada como una **hipótesis a discutir** y nunca como una asignación definitiva del modelo.

## 3. Formato de la tabla de resultados

Para cada dataset, se completa una tabla con el siguiente formato. Las celdas se rellenan en la etapa de implementación.

| $\rho$ | Corrida | $K$ | $\bar{s}$ | $\sigma_s$ | ARI vs. corrida 1 | Compactness media | Interpretación |
|--------|---------|-----|-----------|------------|--------------------|--------------------|----------------|
| 0.40   | #1      |     |           |            | —                  |                    |                |
| 0.40   | #2      |     |           |            |                    |                    |                |
| 0.40   | #3      |     |           |            |                    |                    |                |
| 0.40   | #4      |     |           |            |                    |                    |                |
| 0.40   | #5      |     |           |            |                    |                    |                |
| 0.60   | #1      |     |           |            | —                  |                    |                |
| 0.60   | #2      |     |           |            |                    |                    |                |
| 0.60   | #3      |     |           |            |                    |                    |                |
| 0.60   | #4      |     |           |            |                    |                    |                |
| 0.60   | #5      |     |           |            |                    |                    |                |
| 0.80   | #1      |     |           |            | —                  |                    |                |
| 0.80   | #2      |     |           |            |                    |                    |                |
| 0.80   | #3      |     |           |            |                    |                    |                |
| 0.80   | #4      |     |           |            |                    |                    |                |
| 0.80   | #5      |     |           |            |                    |                    |                |

Una segunda tabla **compara entre valores de $\rho$** agregando las cinco corridas (media y desvío de $K$, ARI promedio, etc.).

## 4. Salidas esperadas

Tras cada corrida, `src/CarGross.py` produce:

- `results/resultado_dataset{N}.csv` con columnas `id, cluster, matching_score`.
- `results/resultado_dataset{N}.txt` con resumen (K, tamaños, exemplares).
- `results/graficos/histograma_clusters_dataset{N}.png` con distribución de tamaños.

Las semillas y los parámetros de cada corrida se guardan en `results/corridas.log`.

## 5. Comparación con la implementación previa

El TFI previo (`_legacy/CarGross_TP/`) contiene un `docs/informe.md` con corridas análogas pero con datasets ligeramente distintos (rangos de $\rho$ y barajados diferentes). Se recomienda **contrastar** las curvas $K(\rho)$ obtenidas acá con las del informe previo para verificar reproducibilidad metodológica. Cualquier divergencia sistemática se discute como parte del análisis.

*Nota: el material de referencia se preserva en `_legacy/CarGross_TP/` por valor histórico. Es el intento anterior del alumno que no se entregó; se cita aquí como antecedente conceptual.*

## 6. Criterios de "corrida satisfactoria"

El TFI se considera satisfactorio respecto a la etapa experimental si:

- ART1 corre sin errores en ambos datasets y en todos los valores de $\rho$.
- $K(\rho)$ es **monótonamente no creciente** al bajar $\rho$ (propiedad teórica esperada).
- La compactness es **estrictamente positiva** en todos los clusters (de lo contrario hay un cluster duplicado o vacío).
- La estabilidad entre corridas es al menos **moderada** ($\text{ARI} > 0.5$) en al menos un valor de $\rho$.
- Las interpretaciones cualitativas son **coherentes** y se documentan en la tabla.

No se considera satisfactorio (y queda fuera del alcance) que la asignación automática de clusters "resuelva" algo clínicamente correcto, porque el modelo no está diseñado para eso.
