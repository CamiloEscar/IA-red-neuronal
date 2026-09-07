# 02 · Problema y Alcance

## Contexto del TFI

Este Trabajo Final Integrador corresponde a la materia **Redes Neuronales** de la carrera de Ingeniería en la **UADER — Facultad de Ingeniería**, en el marco del **IDTI Lab**. La consigna oficial (`_legacy/CarGross_TP/consignas_TP.md`, Actividad #3) exige implementar el algoritmo ART1 de Carpenter–Grossberg. Este TFI adopta ese algoritmo y lo aplica, en carácter **exploratorio**, sobre dos datasets propios.

*Nota: el material de referencia se preserva en `_legacy/CarGross_TP/` por valor histórico. Es el intento anterior del alumno que no se entregó; se cita aquí como antecedente conceptual.*

## 1. Qué hace el sistema

El sistema toma como entrada un **dataset CSV** (típicamente `data/dataset1_pacientes.csv` o `data/dataset2_sensores.csv`) y, tras binarizar las columnas numéricas según umbrales documentados, ejecuta ART1 con un parámetro de vigilancia $\rho$ fijado por el operador. Su salida es:

- Un **ID de cluster** asignado a cada fila.
- Un **puntaje de coincidencia** (matching score) entre cada fila y el exemplar del cluster asignado.
- El **número total de clusters** descubiertos.
- Los **exemplares** binarios de cada cluster.

Formalmente, el sistema aproxima una **partición** $\mathcal{P} = \{C_1, \dots, C_K\}$ de las filas, donde cada $C_k$ es el conjunto de filas que mejor resuenan con el exemplar $E_k$ bajo el criterio de vigilancia $\rho$. El sistema no entrena contra etiquetas verdaderas: los clusters son **descubiertos**, no impuestos.

## 2. Qué NO hace el sistema

Esta lista de exclusiones es fundamental y se reproduce también en `06_limitaciones_y_etica.md`. El sistema:

- **NO prescribe** tratamientos ni medicaciones.
- **NO decide** a qué especialista debe derivarse un paciente.
- **NO diagnostica** condiciones médicas.
- **NO reemplaza** el juicio clínico de un profesional matriculado.
- **NO generaliza** en el sentido supervisado: ART1 no predice etiquetas para filas no vistas con probabilidad calibrada; lo que hace es asignar cada fila vista al cluster cuyo exemplar más le resuene.
- **NO valida clínicamente** las asociaciones cluster–perfil: esa interpretación es trabajo humano.
- **NO opera** sobre datos en tiempo real ni se integra con historias clínicas electrónicas (EHR).
- **NO compara** con ground-truth usando accuracy/F1/precision/recall, porque no hay etiquetas verdaderas.

## 3. Motivación clínica (con honestidad)

La elección del dataset clínico simulado (`dataset1_pacientes.csv`) tiene una motivación **exploratoria**: queremos ver si ART1 puede revelar **perfiles latentes** en una cohorte de pacientes con factores de riesgo cardiovascular y metabólico.

> **Aclaración fundamental**: clustering no es lo mismo que soporte a la decisión de triaje. ART1 agrupa; un profesional interpreta cada grupo como un posible perfil clínico (por ejemplo: *"este grupo con colesterol alto + glucosa alterada + hipertensión se parece a un perfil de riesgo cardiometabólico"*). El modelo **no** toma la decisión de derivación; el profesional **sí** lo hace.

En este TFI el foco es **didáctico**: entender cómo trabaja ART1, cómo responde al barrido de $\rho$, qué tan estables son los clusters cuando cambia el orden de presentación, y cómo se interpretan cualitativamente los exemplares que aparecen. Cualquier interpretación clínica queda a cargo del lector profesional, no del sistema.

## 4. Fuera de alcance

Quedan explícitamente fuera del alcance de este TFI los siguientes puntos, que a menudo se piden a un sistema "clínico" pero no son parte de este trabajo:

- Despliegue en tiempo real o producción.
- Integración con historias clínicas electrónicas (EHR) o pipelines clínicos.
- Cumplimiento regulatorio (ANMAT en Argentina, FDA, EMA, IEC 62304).
- Validación prospectiva en cohorte real.
- Auditoría de seguridad informática (ISO 27001 / 13485).
- Generalización a datasets mayores o multi-centro.
- Adaptación a entradas continuas (ello requeriría ART2, fuera de la consigna de este TFI).
- Cualquier afirmación de utilidad clínica concreta o de mejora de outcomes.

## 5. Usuarios objetivo

El sistema se orienta a tres perfiles, en orden de cercanía al trabajo:

1. **Estudiantes de Redes Neuronales**: entender el algoritmo ART1 leyéndolo, corriéndolo y modificando parámetros.
2. **Docentes**: contar con un material reproducible para mostrar ART1 en clase y discutir sus propiedades.
3. **Investigadores clínicos**: usar el resultado como **insumo exploratorio** para diseñar hipótesis (por ejemplo: *"¿es razonable que este cluster represente un fenotipo X?"*), **no** para tomar decisiones operativas.

## 6. Mensaje honesto

> Este TFI **no** pretende resolver el triaje clínico. Sí pretende ser un **ejercicio académico riguroso** de aplicación de ART1, con datasets acotados, métricas no supervisadas (ver `05_corridas_y_evaluacion.md`), y resultados discutidos con sus limitaciones explícitas (ver `06_limitaciones_y_etica.md`).
>
> Cualquier extensión hacia uso clínico real requeriría, como mínimo, lo enumerado en §4 — y eso **no es lo que se entrega acá**.
