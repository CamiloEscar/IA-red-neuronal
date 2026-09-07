# 06 · Limitaciones y Ética

Este documento enumera las restricciones técnicas del algoritmo, las restricciones éticas que impone el dominio clínico, y un **disclaimer formal** que debe acompañar cualquier entrega o presentación derivada de este TFI.

## 1. Limitaciones técnicas de ART1

### Sensibilidad al orden de presentación

ART1 procesa las entradas **una a una y en serie**, en el orden en que aparecen. Sin barajado y sin mecanismos de estabilización adicionales, el primer patrón define el cluster 1, el primer patrón que falla la vigilancia define el cluster 2, y así. Dos corridas del mismo dataset con el mismo $\rho$ pueden producir particiones estructuralmente distintas si el orden cambia.

**Mitigación adoptada**: las corridas definidas en `05_corridas_y_evaluacion.md` barajan y reportan ARI entre repeticiones, de modo que la variabilidad sea cuantificable.

### Sólo entradas binarias

ART1 no acepta valores intermedios. Esto obliga a binarizar las features continuas, lo que:

- descarta matices (dos pacientes con colesterol 239 y 241 mg/dL quedan en categorías distintas; dos con 90 y 109, en la misma);
- vuelve al modelo **muy sensible a la elección de umbral**;
- exige justificación clínica u operativa para cada umbral, lo que se hace en `03_dataset_y_preprocesamiento.md`.

Una alternativa sería ART2 (entradas continuas con la misma arquitectura general), pero está fuera de la consigna de este TFI.

### Dilema estabilidad–plasticidad

El parámetro $\rho$ lo fija el humano. No existe un $\rho$ "óptimo" sin una noción externa de qué agrupamiento se quiere obtener:

- $\rho$ muy bajo → clusters gigantes, poca diferenciación.
- $\rho$ muy alto → un cluster por fila, ningún agrupamiento real.

La elección de $\rho$ es una **decisión del experimentador**, no algo aprendido por el modelo. Esto se discute en detalle en `01_marco_teorico.md` y se operacionaliza en `05_corridas_y_evaluacion.md`.

### Sin generalización supervisada

ART1 **no predice** una etiqueta para una fila no vista: **asigna** cada fila al cluster cuyo exemplar más le resuene. Esto es muy distinto de un clasificador MLP o Random Forest: no hay probabilidad calibrada de pertenencia, no hay confidence score Bayesiano, no hay hold-out. Si se necesita generalización con etiquetas verdaderas, ART1 no es la herramienta apropiada.

### Interpretación manual

ART1 no devuelve "este cluster significa X". Devuelve un **vector binario** como exemplar. La interpretación es **trabajo humano** y depende de conocimiento de dominio. Esto se hace explícito en `02_problema_y_alcance.md`.

### Ruido y evaporación del exemplar

El AND sucesivo del Step 7 puede llevar a un exemplar **evaporado** (todo en 0) si las entradas sucesivas son ruidosas y la vigilancia es alta. Lau (1992, p. 13, Fig. 11) describe este fenómeno: con $\rho = 0.9$ y entradas con bits faltantes, el algoritmo puede generar muchos clusters espurios hasta agotar los nodos disponibles. En este TFI se acepta el fenómeno como propio del modelo y se documenta su ocurrencia cuando aparezca.

## 2. Limitaciones médicas y éticas

### ART1 no es una herramienta clínica

Un sistema que produce clusters no prescribe, no diagnostica y no deriva. Aunque el exemplar de un cluster se parezca a un perfil clínico conocido, **el modelo no toma decisiones**: lo hace el profesional que interpreta.

### Validez de cualquier afirmación clínica

Sobre un dataset de **55 filas** (y aún con datasets mucho más grandes), ninguna afirmación del tipo *"ART1 identifica grupos de riesgo cardiovascular"* debe hacerse como verdad clínica. La cohorte es pequeña, los umbrales son aproximaciones, los clusters no son fenotipos clínicos validados, y el modelo no evalúa causalidad.

### Privacidad y datos

Los datasets incluidos (`data/dataset1_pacientes.csv`, `data/dataset2_sensores.csv`) son **simulados** y no corresponden a pacientes ni a máquinas reales. Aun así, se tratan como si lo fueran en términos de manejo:

- no se suben a servicios externos;
- no se commitean con metadatos identificables;
- se documentan explícitamente como sintéticos.

### Regulación

En Argentina, cualquier sistema con pretendida utilidad clínica debe atravesar controles de ANMAT. Este TFI no los atraviesa y **no debe** interpretarse como evidencia de utilidad clínica regulatoria válida.

## 3. Disclaimer formal

El siguiente texto debe acompañar toda entrega o presentación oral derivada de este TFI, en la portada del informe y al inicio de cualquier PPT:

> **DESCARGO DE RESPONSABILIDAD**
>
> Este sistema es una herramienta **educativa** desarrollada como Trabajo Final Integrador de la materia *Redes Neuronales* de la UADER — IDTI Lab. **No** constituye una herramienta de diagnóstico médico, prescripción farmacológica, ni derivación a especialistas. Su uso en contextos clínicos reales **no está avalado** por los autores ni por la institución. Toda decisión clínica debe ser tomada por profesionales médicos matriculados, sobre la base de evidencia clínica validada, y está fuera del alcance de este trabajo.

## 4. Vacíos para producción real

Un sistema con pretendida utilidad clínica debería contar, como mínimo, con los siguientes elementos. Ninguno existe para este TFI, y **no es objetivo del trabajo** establecerlos:

- validación prospectiva en cohorte multi-centro;
- aprobación regulatoria (ANMAT en Argentina, FDA / EMA en el exterior);
- auditoría de seguridad informática (ISO 27001 / ISO 13485);
- plan de monitoreo de drift del modelo;
- pruebas de equidad entre subgrupos demográficos;
- protocolo documentado de re-entrenamiento;
- plan de gestión de incidentes.

Se mencionan únicamente para que el lector entienda la enorme distancia que existe entre un ejercicio académico acotado y una herramienta clínica operativa. Esta sección es **pedagógica**, no una autocrítica vacía: explicita qué se necesitaría para cruzar esa distancia.
