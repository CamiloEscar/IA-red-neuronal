# MANUAL DE REFERENCIA — CarGross.py
## Red Neuronal ART1 (Carpenter/Grossberg)

**Materia:** Redes Neuronales  
**Institución:** UADER — IDTI Lab  
**Versión:** 1.0  
**Fecha:** Marzo 2026

---

## TABLA DE CONTENIDOS

1. [Introducción y Alcances](#1-introducción-y-alcances)
2. [Limitaciones](#2-limitaciones)
3. [Instalación](#3-instalación)
4. [Modo de Uso](#4-modo-de-uso)
5. [Descripción de Parámetros](#5-descripción-de-parámetros)
6. [Formato de Archivos](#6-formato-de-archivos)
7. [Test Demo](#7-test-demo)
8. [Datasets incluidos](#8-datasets-incluidos)
9. [FAQ — Preguntas Frecuentes](#9-faq--preguntas-frecuentes)
10. [Referencias](#10-referencias)

---

## 1. INTRODUCCIÓN Y ALCANCES

`CarGross.py` es una implementación en Python del algoritmo **ART1 (Adaptive Resonance Theory 1)** desarrollado por Gail Carpenter y Stephen Grossberg, tal como está descripto en Lau (1992), pp. 12-14.

### ¿Qué hace este programa?

El programa realiza **clustering no supervisado** de datos. Dado un conjunto de datos en formato CSV, agrupa automáticamente las filas en clusters (grupos) según su similitud. A diferencia de algoritmos como K-Means, ART1 **no requiere que se especifique de antemano cuántos clusters habrá**: los va creando dinámicamente a medida que procesa los datos.

### Característica central: el parámetro de vigilancia

El parámetro `--vigilance` (rho, entre 0.0 y 1.0) controla cuán parecidos deben ser dos patrones para pertenecer al mismo cluster:

- **rho cercano a 1.0**: exige alta similitud → muchos clusters pequeños y específicos.
- **rho cercano a 0.0**: acepta baja similitud → pocos clusters grandes y generales.

### Flujo del algoritmo (Box 3, Lau 1992)

```
Step 1: Inicialización de pesos t_ij = 1, b_ij = 1/(1+N)
Step 2: Aplicar nueva entrada X
Step 3: Calcular matching scores: mu_j = sum(b_ij * x_i)
Step 4: Seleccionar mejor match: mu_j* = max(mu_j)
Step 5: Test de vigilancia: ||T·X|| / ||X|| > rho?
           NO → Step 6 (deshabilitar cluster, buscar otro)
           SÍ → Step 7 (adaptar pesos)
Step 6: Deshabilitar cluster actual. Repetir Step 3.
Step 7: Actualizar pesos: t_ij*(t+1) = t_ij(t) AND x_i
                          b_ij*(t+1) = t_ij*(t) AND x_i / (0.5 + sum(t_ij*(t) AND x_i))
Step 8: Repetir desde Step 2 con la siguiente entrada.
```

### Alcances

- Soporta cualquier dataset CSV con columnas numéricas.
- Binarización automática de datos continuos (por media o por umbrales custom vía metadata).
- Salida en formato CSV con cluster asignado por fila.
- Funciona en Python 3.6+ sin dependencias externas.
- Compatible con Windows, Linux y macOS.
- Maneja columnas mixtas (numéricas y no numéricas): usa solo las numéricas para clustering y preserva todas en la salida.

---

## 2. LIMITACIONES

- **ART1 trabaja con datos binarios**: los datos continuos son binarizados usando la media de cada columna como umbral. Esta simplificación puede perder matices en datos con distribuciones complejas.

- **Sensibilidad al orden de presentación**: el resultado puede variar si el orden de las filas en el CSV cambia, ya que ART1 aprende de forma incremental (online learning).

- **No es determinístico entre ejecuciones con datos reordenados**: dos corridas del mismo dataset con filas en diferente orden pueden producir una asignación de cluster distinta (aunque la estructura de grupos es equivalente).

- **La columna `id` o columnas identificadoras son tratadas como numéricas** si contienen valores numéricos. Se recomienda usar columnas con texto como ID (ej. `S001`, `pac_01`) para que sean ignoradas en el clustering.

- **No implementa la versión ART2** (para entradas continuas) ni ART-MAP (clasificación supervisada).

- **El número máximo de clusters** está limitado por el parámetro `--max-clusters` (default: 50). Si el dataset es muy heterogéneo y la vigilancia muy alta, puede requerirse aumentar este límite.

- **No soporta valores faltantes explícitos** (NaN, vacío): los trata como 0 en la binarización.

---

## 3. INSTALACIÓN

### Requisitos

- Python 3.6 o superior
- Sin dependencias externas (solo librería estándar de Python)

### Verificar versión de Python

```bash
python --version
# o
python3 --version
```

Se requiere Python 3.6+. Si el sistema tiene Python 2, usar `python3` en lugar de `python`.

### Instalación en Windows

1. Descargar Python desde https://www.python.org/downloads/
2. Durante la instalación, marcar "Add Python to PATH"
3. Abrir una terminal (CMD o PowerShell)
4. Verificar: `python --version`
5. Copiar `CarGross.py` y los datasets a una carpeta de trabajo

### Instalación en Linux / macOS

```bash
# Verificar Python
python3 --version

# Si no está instalado (Ubuntu/Debian):
sudo apt-get install python3

# Si no está instalado (macOS con Homebrew):
brew install python3
```

### No se requieren librerías adicionales

El script usa únicamente módulos de la librería estándar:
`sys`, `csv`, `os`, `argparse`, `math`

---

## 4. MODO DE USO

### Sintaxis básica

```bash
python CarGross.py <archivo_entrada.csv> [opciones]
```

### Ejemplos

```bash
# Uso básico con valores por defecto (vigilancia = 0.5)
python CarGross.py mis_datos.csv

# Con vigilancia personalizada
python CarGross.py mis_datos.csv --vigilance 0.7

# Con archivo de salida específico
python CarGross.py mis_datos.csv --output clusters_resultado.csv

# Con metadatos de umbral custom
python CarGross.py mis_datos.csv --metadata umbrales.csv

# Modo detallado (verbose) para ver cada paso del algoritmo
python CarGross.py mis_datos.csv --vigilance 0.6 --verbose

# Ver ayuda completa
python CarGross.py --help
```

---

## 5. DESCRIPCIÓN DE PARÁMETROS

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `archivo_entrada.csv` | obligatorio | — | Ruta al CSV de entrada |
| `--vigilance` | float [0.0–1.0] | 0.5 | Umbral de vigilancia rho |
| `--max-clusters` | int | 50 | Máximo número de clusters |
| `--output` | string | `resultados_cargross.csv` | Ruta del CSV de salida |
| `--metadata` | string | None | Ruta a CSV de metadatos |
| `--verbose` | flag | False | Activa salida detallada |
| `--help` | flag | — | Muestra el manual |

---

## 6. FORMATO DE ARCHIVOS

### 6.1 Archivo de entrada (CSV)

- Primera fila: encabezados de columna.
- Separador: coma (`,`).
- Columnas numéricas: valores enteros o decimales (punto como separador decimal).
- Columnas no numéricas: preservadas en la salida pero no usadas para clustering.

**Ejemplo:**
```
id,edad,temperatura,glucosa,imc
1,45,36.8,95,27.5
2,32,36.5,88,23.1
```

### 6.2 Archivo de metadatos (CSV, opcional)

Permite definir umbrales de binarización personalizados en lugar de usar la media automática.

**Columnas obligatorias:**
- `columna`: nombre exacto de la columna en el CSV de entrada.
- `umbral_binarizacion`: valor numérico. Valores >= umbral → 1; valores < umbral → 0.

**Ejemplo:**
```
columna,umbral_binarizacion
edad,40
temperatura,37.0
glucosa,100
imc,25
```

### 6.3 Archivo de salida (CSV)

Igual al archivo de entrada más una columna `cluster` al final con el número de cluster asignado (entero, empieza en 0).

**Ejemplo:**
```
id,edad,temperatura,glucosa,imc,cluster
1,45,36.8,95,27.5,1
2,32,36.5,88,23.1,0
```

---

## 7. TEST DEMO

Para verificar que la instalación funciona correctamente, ejecutar:

### Demo con Dataset 1 (pacientes)

```bash
python CarGross.py dataset1_pacientes.csv --vigilance 0.6 --output resultado_demo1.csv
```

**Salida esperada en consola:**
```
CarGross.py — Red ART1 Carpenter/Grossberg
==================================================
Archivo de entrada:  dataset1_pacientes.csv
Vigilancia (rho):    0.6
Máx. clusters:       50
Archivo de salida:   resultado_demo1.csv
Registros cargados:  55
Columnas numéricas:  id, edad, presion_sistolica, ...

Entrenando red ART1 (N=8, rho=0.6)...

Resultados guardados en: resultado_demo1.csv

============================================================
  RESUMEN DE RESULTADOS
============================================================
  Vigilancia (rho):      0.6
  Columnas procesadas:   8
  Registros procesados:  55
  Clusters formados:     5
  ...
============================================================
```

Si la ejecución termina con código 0 y se genera el archivo CSV de salida, la instalación es correcta.

### Demo con Dataset 2 (sensores)

```bash
python CarGross.py dataset2_sensores.csv --vigilance 0.65 --output resultado_demo2.csv
```

**Salida esperada:** 3 clusters formados a partir de 55 registros.

---

## 8. DATASETS INCLUIDOS

### Dataset 1: `dataset1_pacientes.csv`

**Descripción:** Datos médicos simulados de 55 pacientes con variables clínicas. Útil para identificar perfiles de riesgo cardiovascular.

| Columna | Descripción | Unidad |
|---------|-------------|--------|
| `id` | Identificador numérico del paciente | — |
| `edad` | Edad del paciente | años |
| `presion_sistolica` | Presión arterial sistólica | mmHg |
| `presion_diastolica` | Presión arterial diastólica | mmHg |
| `colesterol` | Nivel de colesterol total | mg/dL |
| `glucosa` | Nivel de glucosa en sangre | mg/dL |
| `imc` | Índice de Masa Corporal | kg/m² |
| `frecuencia_cardiaca` | Frecuencia cardíaca en reposo | lpm |

**Cómo correrlo:**
```bash
python CarGross.py dataset1_pacientes.csv --vigilance 0.6 --output resultados_dataset1.csv
```

**Salida esperada con rho=0.6:** 5 clusters con la siguiente distribución aproximada:
- Cluster 0 (25.5%): pacientes jóvenes con valores normales (bajo riesgo)
- Cluster 1 (27.3%): pacientes maduros con valores elevados (alto riesgo)
- Cluster 2 (23.6%): pacientes intermedios
- Cluster 3 (7.3%): pacientes con valores muy extremos (crítico)
- Cluster 4 (16.4%): perfil mixto

### Dataset 2: `dataset2_sensores.csv`

**Descripción:** Datos de sensores industriales simulados de 55 unidades de maquinaria. Útil para detectar patrones de operación y posibles fallas.

| Columna | Descripción | Unidad |
|---------|-------------|--------|
| `sensor_id` | Identificador alfanumérico del sensor | — |
| `temperatura` | Temperatura operativa | °C |
| `vibracion` | Nivel de vibración | mm/s |
| `presion` | Presión del sistema | bar |
| `voltaje` | Voltaje de alimentación | V |
| `corriente` | Corriente eléctrica | A |
| `rpm` | Velocidad de rotación | RPM |
| `tiempo_operacion` | Tiempo acumulado de operación | horas |

**Cómo correrlo:**
```bash
python CarGross.py dataset2_sensores.csv --vigilance 0.65 --output resultados_dataset2.csv
```

**Salida esperada con rho=0.65:** 3 clusters:
- Cluster 0 (10.9%): sensores en condición crítica (alta temperatura, alta vibración, muchas horas)
- Cluster 1 (49.1%): sensores en operación normal
- Cluster 2 (40.0%): sensores en operación liviana (baja temperatura, poca corriente)

---

## 9. FAQ — PREGUNTAS FRECUENTES

### P1: ¿Por qué el programa produce clusters diferentes cada vez que lo corro?

**R:** Si el CSV de entrada tiene las mismas filas en el mismo orden, los resultados son **completamente reproducibles**. ART1 es determinístico dado un orden fijo de presentación. Si los resultados varían, verificar que el archivo no fue reordenado entre corridas. ART1 aprende en línea (online): el orden importa, porque el primer patrón siempre forma el primer cluster.

---

### P2: ¿Qué valor de vigilancia debo usar?

**R:** Depende del nivel de granularidad deseado:

- **rho = 0.3 a 0.5**: pocos clusters, agrupamiento amplio. Bueno para encontrar grupos generales.
- **rho = 0.5 a 0.7**: clusters de granularidad media. Recomendado como punto de partida.
- **rho = 0.7 a 0.9**: muchos clusters finos. Útil cuando los datos tienen subgrupos muy específicos.
- **rho > 0.9**: puede generar un cluster por cada patrón único. Raramente útil.

Se recomienda experimentar con valores entre 0.5 y 0.7 y evaluar la distribución de los clusters resultantes.

---

### P3: ¿Qué significa el error "Se alcanzó el límite máximo de clusters"?

**R:** El parámetro `--max-clusters` limita cuántos clusters puede crear la red. Si la vigilancia es muy alta y los datos son muy variados, la red intentará crear más clusters de los permitidos. Soluciones:

1. **Reducir la vigilancia**: `--vigilance 0.4` generará menos clusters.
2. **Aumentar el límite**: `--max-clusters 100` permite más clusters.
3. **Revisar el dataset**: puede haber valores atípicos (outliers) que la red intenta clasificar en clusters propios.

---

### P4: ¿Puedo usar datos con texto?

**R:** Sí, con limitaciones. Las columnas con texto (ej. nombres, categorías) son detectadas automáticamente como no numéricas y **se excluyen del proceso de clustering**, pero se preservan en el archivo de salida. Solo las columnas con valores numéricos participan en el algoritmo. Si necesita incluir variables categóricas, debe codificarlas numéricamente antes de usar el programa (ej. 0/1 para binarias, o valores enteros para ordinales).

---

### P5: ¿Qué diferencia hay entre ART1 y K-Means?

**R:** Las principales diferencias son:

| Característica | ART1 | K-Means |
|----------------|------|---------|
| Número de clusters | Determinado automáticamente | Debe especificarse (K) |
| Tipo de aprendizaje | Online (una pasada, incremental) | Iterativo (múltiples pasadas) |
| Estabilidad-Plasticidad | Resuelve el dilema con la vigilancia | Sin mecanismo equivalente |
| Tipo de datos | Binario (continuo binarizado) | Continuo |
| Nuevo dato | Se adapta sin reentrenar todo | Requiere reentrenamiento completo |

---

## 10. REFERENCIAS

[1] Carpenter, G.A. & Grossberg, S. (1987). *A massively parallel architecture for a self-organizing neural pattern recognition machine.* Computer Vision, Graphics, and Image Processing, 37, 54-115.

[2] Lau, C. (Ed.) (1992). *Artificial Neural Networks.* IEEE Press. (Capítulo: "An Introduction to Computing with Neural Nets", R.P. Lippmann, pp. 5-14. Descripción del algoritmo Carpenter/Grossberg: pp. 12-14.)

[3] Carpenter, G.A. & Grossberg, S. (1987). *ART 2: Self-organization of stable category recognition codes for analog input patterns.* Applied Optics, 26(23), 4919-4930.

[4] Grossberg, S. (1976). *Adaptive pattern classification and universal recoding: Part I. Parallel development and coding of neural feature detectors.* Biological Cybernetics, 23, 121-134.

[5] Lippmann, R.P. (1987). *An Introduction to Computing with Neural Nets.* IEEE ASSP Magazine, April 1987, pp. 4-22.

[6] Heaton, J. (2008). *Introduction to Neural Networks with Java (2nd ed.).* Heaton Research.

[7] Haykin, S. (1994). *Neural Networks: A Comprehensive Foundation.* Macmillan College Publishing.
