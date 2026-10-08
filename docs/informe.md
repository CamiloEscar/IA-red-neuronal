# Informe de corridas - Red ART1

Materia Redes Neuronales - UADER
Fecha: 10/10/2026

## Datasets usados

| Archivo                               | Registros | Variables                                            |
| ------------------------------------- | --------- | ---------------------------------------------------- |
| `data/dataset_entrenamiento_ART1.csv` | 100       | 5 (Tos, Moco, Fiebre, Dolor abdominal, Presión alta) |
| `data/dataset_validacion_ART1.csv`    | 100       | 5 + columna "Especialidad esperada"                  |

Se usaron los `.csv` porque es el formato de entrada del módulo. Los `.xlsx`
equivalentes dan exactamente el mismo resultado (lo verifica el notebook).

## Cómo se corrió

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.65 -o results/res_train.csv --save-txt results/res_train.txt
python src/CarGross.py data/dataset_validacion_ART1.csv -r 0.65 -o results/res_val.csv --save-txt results/res_val.txt
```

Cada tabla de abajo se hizo repitiendo la corrida con distintos valores de `-r`.

## Corrida con el dataset de entrenamiento (sin etiquetas)

| rho      | clusters | matching ratio | estabilidad |
| -------- | -------- | -------------- | ----------- |
| 0.30     | 6        | 0.7050         | 1.50        |
| 0.50     | 8        | 0.8008         | 1.62        |
| **0.65** | **14**   | **0.9500**     | **1.64**    |
| 0.80     | 17       | 1.0000         | 1.88        |
| 0.90     | 17       | 1.0000         | 1.88        |

Con rho 0.65 se forman 14 clusters y las plantillas se entienden bien: el
cluster de "Tos" queda asignado a Neumonología, "Dolor abdominal" a
Gastroenterología, "Moco + Fiebre" a Infectología, etc.

## Corrida con el dataset de validación (con etiquetas)

| rho      | clusters | matching ratio | estabilidad | pureza  |
| -------- | -------- | -------------- | ----------- | ------- |
| 0.30     | 5        | 0.7500         | 1.00        | 56%     |
| 0.50     | 7        | 0.8150         | 1.43        | 57%     |
| **0.65** | **12**   | **0.9600**     | **1.58**    | **69%** |
| 0.80     | 17       | 1.0000         | 2.00        | 72%     |
| 0.90     | 17       | 1.0000         | 2.00        | 72%     |

La pureza compara el cluster de cada paciente con su "Especialidad esperada"
del dataset.

## Estabilidad ante el barajado de los pacientes

Se corrieron 10 veces cada dataset con el orden de pacientes mezclado
(seed 42, rho 0.65):

| Dataset       | clusters (base) | corridas barajadas                     | media |
| ------------- | --------------- | -------------------------------------- | ----- |
| Entrenamiento | 14              | 14, 14, 15, 15, 13, 13, 14, 14, 14, 14 | 14.0  |
| Validación    | 12              | 12, 14, 13, 13, 13, 13, 13, 12, 13, 13 | 12.9  |

## Evaluación

- Con rho 0.65 el matching ratio queda arriba de 0.95 en los dos datasets: las
  plantillas retienen casi todos los síntomas de los pacientes que las formaron.
- Subir rho a 0.80/0.90 no conviene: los clusters pasan de 12 a 17, el matching
  ratio llega a 1 (los pacientes de un cluster tienen exactamente los mismos
  síntomas) y la pureza solo sube de 69% a 72%.
- Bajar rho a 0.30/0.50 arma pocos clusters (5 a 8) y la pureza cae a 56-57%.
- La variación por barajado es chica (13 a 15 y 12 a 14 clusters). ART1
  depende del orden de entrada, pero con estos datos el resultado se mantiene
  parecido.
- La pureza de 69% con rho 0.65 es el dato principal de la validación. El 31%
  restante son pacientes con síntomas que no alcanzan a separar bien entre
  especialidades.

## Conclusión

Con rho 0.65 el módulo funciona bien sobre los dos datasets: 14 clusters en
entrenamiento, 12 en validación, matching ratio 0.95/0.96 y pureza del 69%.
Esos valores se toman como referencia en el manual (sección 6). La especialidad
asignada es orientativa y debe ser validada por un profesional de la salud.
