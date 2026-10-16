# Informe de corridas - Red ART1

Inteligencia Artificial - UADER
Fecha: 10/10/2026

## Datasets

| Archivo                               | Registros | Variables                                            |
| ------------------------------------- | --------- | ---------------------------------------------------- |
| `data/dataset_entrenamiento_ART1.csv` | 100       | 5 (Tos, Moco, Fiebre, Dolor abdominal, Presión alta) |
| `data/dataset_validacion_ART1.csv`    | 100       | 5 + columna "Especialidad esperada"                  |

Se usaron los `.csv`, que es el formato de entrada del módulo. Los `.xlsx` dan el mismo resultado (se comprueba en el notebook).

## Cómo se corrió

```powershell
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.65 -o results/res_train.csv --save-txt results/res_train.txt
python src/CarGross.py data/dataset_validacion_ART1.csv -r 0.65 -o results/res_val.csv --save-txt results/res_val.txt
```

Las tablas siguientes salen de repetir la corrida con distintos valores de `-r`. La columna "estabilidad" es el promedio de bits en 1 por plantilla.

## Entrenamiento (sin etiquetas)

| rho      | clusters | matching ratio | estabilidad |
| -------- | -------- | -------------- | ----------- |
| 0.30     | 6        | 0.7050         | 1.50        |
| 0.50     | 8        | 0.8008         | 1.62        |
| **0.65** | **14**   | **0.9500**     | **1.64**    |
| 0.80     | 17       | 1.0000         | 1.88        |
| 0.90     | 17       | 1.0000         | 1.88        |

Con rho 0.65 se forman 14 clusters y las plantillas se entienden bien: el cluster de "Tos" queda en Neumonología, el de "Dolor abdominal" en Gastroenterología, el de "Moco + Fiebre" en Infectología, etc.

## Validación (con etiquetas)

| rho      | clusters | matching ratio | estabilidad | pureza  |
| -------- | -------- | -------------- | ----------- | ------- |
| 0.30     | 5        | 0.7500         | 1.00        | 56%     |
| 0.50     | 7        | 0.8150         | 1.43        | 57%     |
| **0.65** | **12**   | **0.9600**     | **1.58**    | **69%** |
| 0.80     | 17       | 1.0000         | 2.00        | 72%     |
| 0.90     | 17       | 1.0000         | 2.00        | 72%     |

La pureza compara el cluster de cada paciente con su "Especialidad esperada": para cada cluster se cuenta la etiqueta más frecuente y se suma sobre los 100 pacientes.

## Orden de los pacientes

Se corrió 10 veces cada dataset con los pacientes mezclados (seed 42, rho 0.65):

| Dataset       | clusters (base) | corridas mezcladas                     | media |
| ------------- | --------------- | -------------------------------------- | ----- |
| Entrenamiento | 14              | 14, 14, 15, 15, 13, 13, 14, 14, 14, 14 | 14.0  |
| Validación    | 12              | 12, 14, 13, 13, 13, 13, 13, 12, 13, 13 | 12.9  |

## Análisis

- Con rho 0.65 el matching ratio es de 0.95 o más en los dos datasets, o sea que las plantillas conservan casi todos los síntomas de los pacientes del cluster.
- Subir rho a 0.80 o 0.90 no conviene: los clusters pasan de 12 a 17, el matching ratio llega a 1 y la pureza solo sube de 69% a 72%.
- Bajar rho a 0.30 o 0.50 deja pocos clusters (5 a 8) y la pureza cae a 56-57%.
- Al mezclar el orden la cantidad de clusters varía poco (13 a 15 en entrenamiento, 12 a 14 en validación). ART1 depende del orden de entrada, pero con estos datos el resultado se mantiene parecido.
- La pureza de 69% con rho 0.65 es el dato principal de la validación. Un 31% de los pacientes quedó en un cluster cuya etiqueta más común no es la suya.

## Conclusión

Con rho 0.65 se obtienen 14 clusters en entrenamiento y 12 en validación, matching ratio 0.95 y 0.96, y pureza de 69%. Esos son los valores de referencia que aparecen en el manual (sección 6). La especialidad asignada es orientativa y la tiene que revisar un profesional de la salud.
