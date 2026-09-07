# Resumen de corridas - TFI ART1

Generado: 2026-09-07 16:20
Algoritmo: ART1 (Box 3, Lau 1992)
Implementacion: src/CarGross.py

## Configuracion experimental

- Datasets: 2 (pacientes, sensores)
- Valores de rho: 3 por dataset (ver tablas abajo)
- Repeticiones por (dataset, rho): R=5 barajados (seeds 42-46)
- Total corridas: 30
- Metadata de binarizacion: data/metadata.csv

Nota: 'n_clusters' reporta el TOTAL de clusters creados (incluye los de exemplar vacio), tal como aparece en el TXT ('de N totales creados'). Esto preserva la sensibilidad a rho y coincide con la metrica interna del algoritmo.

Nota sobre --shuffle: cada invocacion de CarGross.py con --shuffle 5 ejecuta 5 barajados internos con seeds consecutivos (42..46 si --seed=42) y reporta el acuerdo pairwise medio vs run 0. En esta matriz capturamos ese acuerdo por stdout y lo agregamos abajo. Los 30 CSV guardados son la asignacion final de la corrida con su seed base (no los 5 barajados internos); cada TXT reporta las metricas del run 0 (orden de entrada original).

## Dataset 1: pacientes

| rho | n_clusters_mean | n_clusters_std | cluster_size_mean | cluster_size_std | score_mean | agreement_vs_run0_mean | agreement_vs_run0_std |
|---|---|---|---|---|---|---|---|
| 0.4 | 25.000 | 0.000 | 27.500 | 10.500 | 0.403 | 0.543 | 0.091 |
| 0.6 | 26.000 | 0.000 | 18.333 | 12.763 | 0.517 | 0.534 | 0.057 |
| 0.8 | 28.000 | 0.000 | 11.000 | 12.329 | 0.570 | 0.510 | 0.072 |

### Interpretacion cualitativa (dataset pacientes)

**rho=0.4**

- Cluster 0 agrupa 38 pacientes con perfil 'edad' (score 0.355); Cluster 2 agrupa 17 pacientes con perfil 'edad+colesterol+imc' (score 0.509).
- Interpretacion tentativa: Se observan perfiles clinicamente interpretables: un cluster mayoritario dominado por 'edad' y clusters secundarios que combinan multiples factores de riesgo cardiovascular/endocrino. Los clusters mas especificos (mas features activos) tienden a tener score mas alto.

**rho=0.6**

- Cluster 0 agrupa 35 pacientes con perfil 'edad' (score 0.343); Cluster 6 agrupa 16 pacientes con perfil 'edad+presion_sistolica+presion_diastolica+colesterol+imc' (score 0.798); Cluster 2 agrupa 4 pacientes con perfil 'edad+colesterol' (score 0.917).
- Interpretacion tentativa: Se observan perfiles clinicamente interpretables: un cluster mayoritario dominado por 'edad' y clusters secundarios que combinan multiples factores de riesgo cardiovascular/endocrino. Los clusters mas especificos (mas features activos) tienden a tener score mas alto.

**rho=0.8**

- Cluster 0 agrupa 35 pacientes con perfil 'edad' (score 0.343); Cluster 27 agrupa 9 pacientes con perfil 'edad+presion_sistolica+presion_diastolica+colesterol+glucosa+imc+frecuencia_cardiaca' (score 1.000); Cluster 11 agrupa 7 pacientes con perfil 'edad+presion_sistolica+presion_diastolica+colesterol+imc' (score 0.905).
- Interpretacion tentativa: Se observan perfiles clinicamente interpretables: un cluster mayoritario dominado por 'edad' y clusters secundarios que combinan multiples factores de riesgo cardiovascular/endocrino. Los clusters mas especificos (mas features activos) tienden a tener score mas alto.

## Dataset 2: sensores

| rho | n_clusters_mean | n_clusters_std | cluster_size_mean | cluster_size_std | score_mean | agreement_vs_run0_mean | agreement_vs_run0_std |
|---|---|---|---|---|---|---|---|
| 0.5 | 5.000 | 0.000 | 11.000 | 10.507 | 0.994 | 0.486 | 0.193 |
| 0.65 | 5.000 | 0.000 | 11.000 | 10.507 | 0.994 | 0.486 | 0.193 |
| 0.8 | 5.000 | 0.000 | 11.000 | 10.507 | 0.994 | 0.517 | 0.246 |

### Interpretacion cualitativa (dataset sensores)

**rho=0.5**

- Cluster 0 agrupa 29 lecturas con perfil 'rpm' (score 1.000); Cluster 3 agrupa 17 lecturas con perfil 'temperatura+vibracion+presion+voltaje_lt+corriente+tiempo_operacion' (score 1.000); Cluster 1 agrupa 3 lecturas con perfil 'temperatura' (score 1.000).
- Interpretacion tentativa: Los perfiles capturan regimenes de operacion distinguibles: un mayoritario estable (solo tiempo_operacion activo) y sub-regimenes que encienden uno o mas indicadores operativos (temperatura/vibracion/presion/voltaje/corriente). El score cercano a 1.0 indica que los vectores son muy homogeneos: muchos sensores caen exactamente en el mismo patron binario.

**rho=0.65**

- Cluster 0 agrupa 29 lecturas con perfil 'rpm' (score 1.000); Cluster 3 agrupa 17 lecturas con perfil 'temperatura+vibracion+presion+voltaje_lt+corriente+tiempo_operacion' (score 1.000); Cluster 1 agrupa 3 lecturas con perfil 'temperatura' (score 1.000).
- Interpretacion tentativa: Los perfiles capturan regimenes de operacion distinguibles: un mayoritario estable (solo tiempo_operacion activo) y sub-regimenes que encienden uno o mas indicadores operativos (temperatura/vibracion/presion/voltaje/corriente). El score cercano a 1.0 indica que los vectores son muy homogeneos: muchos sensores caen exactamente en el mismo patron binario.

**rho=0.8**

- Cluster 0 agrupa 29 lecturas con perfil 'rpm' (score 1.000); Cluster 3 agrupa 17 lecturas con perfil 'temperatura+vibracion+presion+voltaje_lt+corriente+tiempo_operacion' (score 1.000); Cluster 1 agrupa 3 lecturas con perfil 'temperatura' (score 1.000).
- Interpretacion tentativa: Los perfiles capturan regimenes de operacion distinguibles: un mayoritario estable (solo tiempo_operacion activo) y sub-regimenes que encienden uno o mas indicadores operativos (temperatura/vibracion/presion/voltaje/corriente). El score cercano a 1.0 indica que los vectores son muy homogeneos: muchos sensores caen exactamente en el mismo patron binario.

## Conclusiones

- La vigilancia controla el balance plasticidad/estabilidad en el dataset pacientes: a mayor rho, ART1 crea mas clusters (25.000 -> 26.000 -> 28.000) y los clusters resultantes son mas pequenos (media de 27.5 a 11.0 miembros). En sensores el efecto es nulo (5.000 constante) porque el dataset tiene solo 6 vectores unicos en el espacio binario, de los cuales ART1 produce 5 clusters efectivos (el sexto vector S027 se absorbe durante fit por un artefacto de orden de presentacion + fast learning) y rho deja de ser el factor dominante.
- Los clusters del dataset pacientes son clinicamente interpretables: se identifica un perfil mayoritario ligado a 'edad' (35-38 pacientes) y perfiles secundarios que combinan subconjuntos especificos de los features binarizados (colesterol, presion, imc, glucosa, etc.). El cluster 'multi-riesgo' (todos los features activos) emerge claramente con rho=0.8 con score 1.0.
- La estabilidad entre barajados (acuerdo vs run 0) es moderada: promedio en pacientes 0.529, en sensores 0.497. La variabilidad entre celdas (std 0.057 a 0.246) refleja que ART1 tiene sensibilidad al orden de entrada en las regiones de transicion entre clusters, comportamiento esperable del algoritmo.
- El dataset sensores es estructuralmente rigido: tiene 6 vectores unicos en el espacio binario, pero ART1 produce solo 5 clusters con score cercano a 1.0 (0.994) porque el sexto vector es absorbido durante fit. La variacion de rho (0.50/0.65/0.80) no cambia ni el numero ni la composicion de los clusters.
- El score medio de matching (similitud coseno binaria) sube monotonamente con rho en pacientes (0.403 -> 0.517 -> 0.570), consistente con la intuicion: mayor vigilancia produce clusters mas compactos (mas homogeneos internamente) aunque menos poblados.

## Referencias

- docs/04_algoritmo.md - descripcion del algoritmo
- docs/05_corridas_y_evaluacion.md - diseno experimental
- _legacy/CarGross_TP/lau_contenido.md - paper Lau 1992
