# INFORME DE CORRIDAS — CarGross.py
## Red Neuronal ART1 (Carpenter/Grossberg)

---

**Materia:** Redes Neuronales  
**Institución:** UADER — IDTI Lab  
**Fecha:** Marzo 2026  
**Docente:** [Nombre del Docente]  
**Integrantes del grupo:** [Nombre 1] — [Nombre 2] — [Nombre 3]

---

## 1. OBJETIVO DEL INFORME

Este informe documenta los resultados obtenidos al ejecutar el módulo `CarGross.py` sobre dos datasets de prueba. Se analiza el comportamiento de la red ART1 con distintos valores del parámetro de vigilancia (rho), se evalúa la calidad de los clusters formados y se discuten los resultados en función de la teoría del algoritmo.

---

## 2. DESCRIPCIÓN DEL ENTORNO DE PRUEBA

- **Sistema Operativo:** Ubuntu 24.04 / Windows 11
- **Versión de Python:** 3.11
- **Módulo:** CarGross.py v1.0
- **Dependencias externas:** ninguna (solo librería estándar de Python)

---

## 3. CORRIDAS CON DATASET 1 — PACIENTES MÉDICOS

### 3.1 Descripción del dataset

Archivo: `dataset1_pacientes.csv`  
Registros: 55 pacientes  
Variables numéricas: 8 (id, edad, presión sistólica, presión diastólica, colesterol, glucosa, IMC, frecuencia cardíaca)

### 3.2 Umbrales de binarización calculados automáticamente

El algoritmo binarizó las columnas usando la media de cada variable como umbral. Los valores >= media se convierten en 1, los menores en 0.

| Columna | Umbral (media) |
|---------|---------------|
| edad | 44.44 años |
| presion_sistolica | 130.33 mmHg |
| presion_diastolica | 82.85 mmHg |
| colesterol | 217.55 mg/dL |
| glucosa | 97.80 mg/dL |
| imc | 27.23 kg/m² |
| frecuencia_cardiaca | 73.91 lpm |

### 3.3 Corrida 1: rho = 0.4

**Comando ejecutado:**
```bash
python CarGross.py dataset1_pacientes.csv --vigilance 0.4 --output resultados_ds1_rho04.csv
```

**Resultados:**

| Cluster | Cantidad | % |
|---------|----------|---|
| 0 | 14 | 25.5% |
| 1 | 16 | 29.1% |
| 2 | 15 | 27.3% |
| 3 | 10 | 18.2% |
| **Total clusters formados** | **4** | |

**Análisis:** Con vigilancia baja (0.4), la red formó solo 4 clusters. Los grupos son amplios: cada uno contiene patrones con diferencias relativamente grandes entre sí. Los 4 clusters se distribuyen de forma bastante equilibrada (~25% cada uno), lo que sugiere que la red capturó cuatro perfiles generales en los datos.

### 3.4 Corrida 2: rho = 0.6

**Comando ejecutado:**
```bash
python CarGross.py dataset1_pacientes.csv --vigilance 0.6 --output resultados_dataset1.csv
```

**Resultados:**

| Cluster | Cantidad | % | Interpretación aproximada |
|---------|----------|---|--------------------------|
| 0 | 14 | 25.5% | Pacientes jóvenes / valores normales bajos |
| 1 | 15 | 27.3% | Pacientes mayores / valores elevados |
| 2 | 13 | 23.6% | Pacientes intermedios |
| 3 | 4 | 7.3% | Valores extremos (posible riesgo alto) |
| 4 | 9 | 16.4% | Perfil mixto |
| **Total clusters formados** | **5** | |

**Análisis:** Con vigilancia media (0.6), se formaron 5 clusters. La distribución es más heterogénea que con rho=0.4, lo que indica que la red comenzó a distinguir subgrupos más finos. El cluster 3 (7.3%) agrupa a pacientes con valores muy extremos en múltiples variables, que no encajan en los grupos más amplios.

### 3.5 Corrida 3: rho = 0.8

**Comando ejecutado:**
```bash
python CarGross.py dataset1_pacientes.csv --vigilance 0.8 --output resultados_ds1_rho08.csv
```

**Resultados:**

| Cluster | Cantidad | % |
|---------|----------|---|
| 0 | 14 | 25.5% |
| 1 | 6 | 10.9% |
| 2 | 18 | 32.7% |
| 3 | 13 | 23.6% |
| 4 | 3 | 5.5% |
| 5 | 1 | 1.8% |
| **Total clusters formados** | **6** | |

**Análisis:** Con vigilancia alta (0.8), se formaron 6 clusters. Aparece un cluster con un único registro (cluster 5, 1.8%), lo que indica un patrón que la red considera suficientemente diferente de todos los demás. Esto es un comportamiento esperado en ART1 con vigilancias altas: los patrones "raros" u outliers forman sus propios clusters.

### 3.6 Comparativa Dataset 1

| Vigilancia | Clusters formados | Distribución | Observación |
|-----------|------------------|--------------|-------------|
| 0.4 | 4 | Equilibrada (~25%) | Grupos amplios, poco específicos |
| 0.6 | 5 | Moderada | Balance entre generalización y especificidad |
| 0.8 | 6 | Desigual (un cluster = 1 registro) | Grupos finos, outliers aislados |

**Conclusión Dataset 1:** El valor rho = 0.6 ofrece el mejor balance. Produce clusters interpretables con tamaños razonables y sin aislamiento excesivo de casos individuales.

---

## 4. CORRIDAS CON DATASET 2 — SENSORES INDUSTRIALES

### 4.1 Descripción del dataset

Archivo: `dataset2_sensores.csv`  
Registros: 55 sensores  
Variables numéricas: 7 (temperatura, vibración, presión, voltaje, corriente, RPM, tiempo de operación)  
Nota: la columna `sensor_id` es alfanumérica y fue excluida del clustering.

### 4.2 Umbrales de binarización calculados automáticamente

| Columna | Umbral (media) |
|---------|---------------|
| temperatura | 70.48 °C |
| vibracion | 0.178 mm/s |
| presion | 102.55 bar |
| voltaje | 219.17 V |
| corriente | 18.72 A |
| rpm | 1443.49 RPM |
| tiempo_operacion | 2934.18 horas |

### 4.3 Corrida 1: rho = 0.5

**Comando ejecutado:**
```bash
python CarGross.py dataset2_sensores.csv --vigilance 0.5 --output resultados_ds2_rho05.csv
```

**Resultados:**

| Cluster | Cantidad | % | Interpretación |
|---------|----------|---|----------------|
| 0 | 6 | 10.9% | Alta temperatura, alta vibración → posible falla |
| 1 | 27 | 49.1% | Operación normal de carga media |
| 2 | 22 | 40.0% | Operación liviana / baja carga |
| **Total clusters formados** | **3** | |

**Análisis:** Con rho=0.5 se identificaron claramente 3 patrones operativos: sensores en condición crítica (cluster 0, 10.9%), sensores en operación normal de carga media (cluster 1, 49.1%) y sensores en operación liviana (cluster 2, 40%). Esta segmentación es altamente útil para mantenimiento predictivo: el cluster 0 debería ser el primero en recibir inspección.

### 4.4 Corrida 2: rho = 0.65

**Comando ejecutado:**
```bash
python CarGross.py dataset2_sensores.csv --vigilance 0.65 --output resultados_dataset2.csv
```

**Resultados:**

| Cluster | Cantidad | % |
|---------|----------|---|
| 0 | 6 | 10.9% |
| 1 | 27 | 49.1% |
| 2 | 22 | 40.0% |
| **Total clusters formados** | **3** | |

**Análisis:** Con rho=0.65 la estructura de clusters es idéntica a rho=0.5. Esto indica que los tres grupos del dataset son suficientemente distintos entre sí para mantenerse separados incluso con vigilancia más exigente. Confirma que los clusters identificados son robustos.

### 4.5 Corrida 3: rho = 0.8

**Comando ejecutado:**
```bash
python CarGross.py dataset2_sensores.csv --vigilance 0.8 --output resultados_ds2_rho08.csv
```

**Resultados:**

| Cluster | Cantidad | % |
|---------|----------|---|
| 0 | 2 | 3.6% |
| 1 | 27 | 49.1% |
| 2 | 22 | 40.0% |
| 3 | 1 | 1.8% |
| 4 | 3 | 5.5% |
| **Total clusters formados** | **5** | |

**Análisis:** Con vigilancia alta (0.8), el cluster original 0 (sensores críticos) se fragmentó en tres grupos más pequeños (clusters 0, 3 y 4), con casos individuales aislados. Los clusters 1 y 2 permanecen estables, lo que confirma su homogeneidad interna. La fragmentación del cluster crítico puede ser útil si se busca identificar distintos tipos de falla.

### 4.6 Comparativa Dataset 2

| Vigilancia | Clusters formados | Observación |
|-----------|-----------------|-------------|
| 0.5 | 3 | Estructura clara: normal, liviano, crítico |
| 0.65 | 3 | Misma estructura. Grupos robustos. |
| 0.8 | 5 | Fragmentación del grupo crítico |

**Conclusión Dataset 2:** La estructura subyacente del dataset es clara y se detecta robustamente con valores de vigilancia entre 0.5 y 0.65. El agrupamiento en "normal", "liviano" y "crítico" es consistente y tiene interpretación práctica directa para mantenimiento industrial.

---

## 5. CONCLUSIONES GENERALES

1. **El algoritmo ART1 funciona correctamente** sobre ambos datasets, produciendo clusters coherentes sin requerir especificación previa del número de grupos.

2. **El parámetro de vigilancia es el control clave.** Valores entre 0.5 y 0.7 producen resultados balanceados en ambos datasets. Vigilancias demasiado altas (> 0.8) fragmentan los grupos y aíslan outliers en clusters propios.

3. **La binarización automática por media es suficiente** para datasets bien distribuidos. Para datos con distribuciones sesgadas, se recomienda usar el archivo de metadatos para ajustar umbrales manualmente.

4. **Los clusters tienen interpretación semántica clara:** en el dataset médico separó perfiles de riesgo cardiovascular; en el industrial identificó estados operativos de maquinaria. Esto valida la utilidad práctica del algoritmo.

5. **Diferencia clave con K-Means:** ART1 determinó automáticamente 3-6 clusters según la vigilancia, sin que el usuario indicara el número. K-Means hubiera requerido especificarlo. Esta es la ventaja central de la arquitectura ART.

---

## 6. REFERENCIAS

[1] Carpenter, G.A. & Grossberg, S. (1987). *A massively parallel architecture for a self-organizing neural pattern recognition machine.* Computer Vision, Graphics, and Image Processing, 37, 54-115.

[2] Lau, C. (Ed.) (1992). *Artificial Neural Networks.* IEEE Press. pp. 12-14.

[3] Lippmann, R.P. (1987). *An Introduction to Computing with Neural Nets.* IEEE ASSP Magazine, April 1987, pp. 4-22.
