# Guía de estudio para la defensa del TFI: red ART1 (CarGross.py)

Esta guía es para prepararnos antes de defender frente al profesor, que pregunta de todo, también del código. Todo lo que hay acá se verificó contra `src/CarGross.py`, los datasets de `data/`, el notebook, el visualizador y las páginas de Lau (1992) que dio la cátedra (`Lau.pp12.a.14.pdf` y `Lau.pp5.a.11.pdf`). Los números de las tablas salen de correr el código real, no de memoria.

## Índice

1. [Cómo usar esta guía](#1-cómo-usar-esta-guía)
2. [El proyecto en 1 minuto y en 5 minutos](#2-el-proyecto-en-1-minuto-y-en-5-minutos)
3. [Teoría de ART1](#3-teoría-de-art1)
4. [Un ejemplo resuelto a mano](#4-un-ejemplo-resuelto-a-mano)
5. [El dataset](#5-el-dataset)
6. [Recorrido del código](#6-recorrido-del-código)
7. [Resultados reales](#7-resultados-reales)
8. [El visualizador y el notebook en la defensa](#8-el-visualizador-y-el-notebook-en-la-defensa)
9. [Banco de preguntas con respuestas modelo](#9-banco-de-preguntas-con-respuestas-modelo)
10. [Ejercicios para hacer a mano](#10-ejercicios-para-hacer-a-mano)
11. [Glosario, trampas y puntos débiles](#11-glosario-trampas-y-puntos-débiles)
12. [Hoja de trucos](#12-hoja-de-trucos)

---

## 1. Cómo usar esta guía

Orden sugerido: leer la sección 2 en voz alta hasta que salga sin mirar, después la 3 y la 4 con lápiz y papel (el ejemplo a mano es lo que más suele pedir un profesor), después la 6 con el código abierto al lado. Las preguntas de la sección 9 se practican tapando la respuesta.

Tres avisos antes de arrancar.

Los números de línea que se citan en la sección 6 son aproximados: el archivo se sigue tocando (textos, mensajes) y las líneas se pueden correr unas pocas posiciones. Lo estable son los nombres de las funciones. Si el profesor pide "mostrame dónde se hace el test de vigilancia", se busca `match_ratio >= self.rho` en `train_pattern`.

No hay que depender de la redacción exacta de los textos que imprime el programa (`--man`, la ficha del paciente, los reportes). Esa redacción puede cambiar. Lo que sí es estable son los números y la lógica.

La terminología sigue a la cátedra: en el paper de Lau (Box 3) las capas no se llaman F1/F2, hablan de nodos de entrada y nodos de salida, pesos bottom-up `b` y top-down `t`, puntajes de coincidencia `μ` y umbral de vigilancia `ρ`. F1/F2 es la terminología original de Carpenter y Grossberg y es la que usamos en el notebook y el visualizador. Conviene decir las dos cosas: "capa de comparación F1 (los nodos de entrada de Lau) y capa de reconocimiento F2 (los nodos de salida de Lau)".

---

## 2. El proyecto en 1 minuto y en 5 minutos

### 2.1 Versión de 1 minuto (para decir en voz alta)

"Implementamos en Python, sin librerías externas para los CSV, la red neuronal de Carpenter y Grossberg, ART1, tal como la describe el Box 3 del paper de Lau. Es una red de clustering no supervisado para vectores binarios: no se le dice cuáles son las categorías, las va creando ella a medida que le llegan los datos.

La usamos sobre 100 pacientes sintéticos, cada uno descripto por cinco síntomas con un 1 si lo tiene y un 0 si no: tos, moco, fiebre, dolor abdominal y presión alta. La red procesa a los pacientes de a uno. Para cada paciente busca la categoría más parecida, y si se parece lo suficiente según un umbral que se llama vigilancia, el paciente entra y la plantilla de esa categoría se actualiza con un AND lógico. Si ninguna se parece lo suficiente, se crea una categoría nueva.

Con vigilancia 0.65 la red arma 14 categorías en el dataset de entrenamiento y 12 en el de validación. En validación, que trae la especialidad esperada de cada paciente, comparamos después los grupos contra esas etiquetas y da una pureza del 69 por ciento. Además hay un notebook con el algoritmo paso a paso y un visualizador web para ver cómo la red decide."

Eso se dice en menos de un minuto. Si hay que acortar más: qué es (clustering no supervisado binario), cómo decide (vigilancia) y qué dio (14 y 12 categorías, 69 por ciento de pureza).

### 2.2 Versión de 5 minutos

Minuto 1, el problema. Queremos agrupar perfiles de síntomas sin tener etiquetas. Un algoritmo como k-means necesita que le digamos cuántos grupos queremos de antemano. ART1 no: el número de categorías crece según los datos y según un parámetro de exigencia, la vigilancia ρ. Además aprende de forma incremental, de a un patrón, sin reentrenar todo, y eso resuelve el dilema estabilidad-plasticidad: puede aprender cosas nuevas sin borrar lo que ya aprendió.

Minuto 2, la arquitectura. Dos capas. F1 recibe el vector binario de 5 bits. F2 tiene hasta 50 nodos de categoría (el parámetro `--max_cat`). Hay dos juegos de pesos: bottom-up `b` (de F1 a F2, sirven para elegir quién compite mejor) y top-down `t` (de F2 a F1, guardan la plantilla de cada categoría). Al inicio `t = 1` para todo y `b = 1/(1+N) = 1/6`.

Minuto 3, el ciclo. Entra un paciente. Cada categoría activa calcula un puntaje `μ_j = Σ b_ij x_i` y gana la mayor (si hay empate, la de índice menor). Se compara la plantilla ganadora con el paciente: `T = t AND x`, y se calcula el matching ratio `‖T‖/‖x‖`, o sea qué fracción de los síntomas del paciente está en la plantilla. Si ese ratio es mayor o igual a ρ hay resonancia: el paciente entra y la plantilla pasa a ser `T`, o sea solo puede perder síntomas. Si no, reset: esa categoría se descarta solo para este paciente y compite la siguiente. Si todas las existentes fallan, gana una categoría libre, que tiene `t` todo en 1 y por eso siempre acepta, y nace una categoría nueva.

Minuto 4, qué hicimos. `src/CarGross.py` tiene la clase `ART1Network`, el cargador de datasets (CSV y XLSX), una capa de reglas aparte que sugiere una especialidad, el CLI con manejo de errores, el manual con `--man` y un autotest con `--test`. También hay un notebook que reimplementa el algoritmo fórmula por fórmula y comprueba que da exactamente los mismos clusters que el módulo, y un visualizador en HTML que anima el ciclo.

Minuto 5, resultados y límites. Con ρ = 0.65: entrenamiento 14 categorías, matching ratio promedio 0.95; validación 12 categorías, matching ratio 0.96 y pureza 69 por ciento. Subir ρ a 0.8 o 0.9 da 17 categorías y pureza 72, o sea casi nada de mejora a cambio de más fragmentación. Bajar ρ a 0.3 o 0.5 mezcla especialidades. Los límites: es sensible al orden de presentación (barajando los datos da entre 13 y 15 categorías en entrenamiento), solo trabaja con entradas binarias, y la pureza de 69 por ciento tiene un techo de 73 por ciento con estos cinco síntomas, porque hay pacientes con exactamente los mismos síntomas y distinta especialidad esperada (más adelante se explica). No diagnostica: agrupa y sugiere, y un profesional tiene que validar.

---

## 3. Teoría de ART1

### 3.1 Qué problema resuelve

Clustering incremental de patrones binarios, sin etiquetas y sin fijar de antemano la cantidad de grupos. Según Lau (Fig. 3 del paper, Lippmann 1987), el clasificador Carpenter/Grossberg está en la rama de entradas binarias, no supervisado, y el algoritmo clásico equivalente es el algoritmo "leader" de clustering secuencial (Hartigan, 1975). Eso es oro para la defensa: el algoritmo leader toma la primera entrada como ejemplar del primer cluster, y cada entrada siguiente "sigue al líder" si está lo bastante cerca, o inaugura un cluster nuevo. ART1 es eso más una regla de actualización (el AND) y una forma de elegir candidato (los puntajes `μ`).

La taxonomía del paper, para ubicar a ART1 frente a otras redes:

| Red                        | Entrada  | Aprendizaje    | Algoritmo clásico parecido (Fig. 3)      |
| -------------------------- | -------- | -------------- | ---------------------------------------- |
| Hopfield                   | binaria  | supervisado    | clasificador óptimo / memoria asociativa |
| Hamming                    | binaria  | supervisado    | clasificador óptimo                      |
| Carpenter/Grossberg (ART1) | binaria  | no supervisado | leader clustering                        |
| Perceptrón                 | continua | supervisado    | clasificador gaussiano                   |
| Perceptrón multicapa       | continua | supervisado    | k vecinos más cercanos / mezcla          |
| Mapas de Kohonen (SOM)     | continua | no supervisado | k-means                                  |

En el PDF original (Fig. 3, página 7), Hopfield y Hamming están bajo "supervised" dentro de la rama binaria. Ojo: la transcripción `_legacy/CarGross_TP/lau_contenido.md` tiene una tabla propia en "Notas adicionales" que dice "No" para Hopfield y Hamming. Esa tabla no coincide con la figura del PDF. Ante la duda, vale el PDF.

### 3.2 El dilema estabilidad-plasticidad

Una red demasiado plástica aprende todo lo nuevo y por eso destruye lo anterior (el "olvido catastrófico" de las redes entrenadas con retropropagación cuando se les presentan datos nuevos). Una red demasiado estable no incorpora nada nuevo. ART1 lo resuelve con la vigilancia: un patrón solo modifica una categoría existente si se parece lo suficiente; si no, no la toca y se crea otra. Lo aprendido se conserva (la plantilla solo se afina por intersección) y se pueden agregar categorías nuevas sin límite hasta agotar los nodos disponibles.

Cuidado con la frase "ρ alto favorece la plasticidad y ρ bajo la estabilidad" que aparece en las notas adicionales de `lau_contenido.md`. No está en las páginas del paper que tenemos. Es una forma posible de leerlo (ρ alto crea categorías nuevas fácilmente, o sea aprende más cosas nuevas), pero conviene no atribuírsela a Lau. Lo que sí dice Lau literal: "un valor cercano a 1 requiere una coincidencia cercana, valores más chicos aceptan una coincidencia más pobre".

### 3.3 Entradas binarias y aprendizaje no supervisado

Cada paciente es un vector `x ∈ {0,1}^5`. Una posición en 1 significa "tiene el síntoma". ART1 solo entiende ceros y unos (las variantes para valores continuos son ART2 y siguientes, que no implementamos). El código lo valida: `train_pattern` levanta `ValueError` si el vector tiene algún valor distinto de 0 o 1, y `load_dataset` corta con un mensaje que indica fila y columna.

No supervisado quiere decir que durante el entrenamiento la red nunca ve etiquetas. En el dataset de validación hay una columna "Especialidad esperada", pero `load_dataset` la separa y solo se usa al final para calcular la pureza.

### 3.4 Capas, pesos y notación

| Elemento            | Dimensión          | Qué es                                                                     | Valor inicial       |
| ------------------- | ------------------ | -------------------------------------------------------------------------- | ------------------- |
| F1 (comparación)    | N = 5              | Un nodo por síntoma. Recibe `x`.                                           |                     |
| F2 (reconocimiento) | M = 50 por defecto | Un nodo por categoría posible. Compiten por inhibición lateral.            |                     |
| `b[i][j]` bottom-up | N x M              | Peso de la entrada `i` a la categoría `j`. Sirve para calcular el puntaje. | `1/(1+N) = 1/6`     |
| `t[j][i]` top-down  | M x N              | Plantilla (prototipo) de la categoría `j`.                                 | `1` (todo prendido) |
| `ρ` vigilancia      | escalar            | Qué tan parecido tiene que ser el paciente a la plantilla.                 | 0.65 por defecto    |

Una categoría "libre" (nunca usada) tiene `t` todo en 1: es una plantilla que acepta cualquier cosa. Lau lo dice de otra forma: "la red se inicializa poniendo efectivamente todos los ejemplares en cero", el mismo estado visto desde el otro lado. Por eso una categoría libre siempre pasa el test de vigilancia: su `T` es igual a `x` y el ratio es 1.

En el código, `self.t` es una lista de M listas de N flotantes en 1.0 y `self.b` una lista de N listas de M flotantes en `1/(1+N)`. Fijarse que los índices están transpuestos: `t[j][i]` (categoría, entrada) y `b[i][j]` (entrada, categoría).

### 3.5 El ciclo completo, paso a paso

Es el Box 3 de Lau, ocho pasos. Entre paréntesis dónde vive cada uno en el código.

| Paso Lau                    | Qué hace                                                    | En el código                                                         |
| --------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| 1. Inicialización           | `t = 1`, `b = 1/(1+N)`, se fija ρ                           | `ART1Network.__init__`                                               |
| 2. Aplicar nueva entrada    | llega el vector `x`                                         | `train_pattern(x)`                                                   |
| 3. Puntajes de coincidencia | `μ_j = Σ b_ij x_i`                                          | `_compute_matching_scores`                                           |
| 4. Elegir el mejor          | `j* = argmax μ_j`                                           | `scores.index(max_score)`                                            |
| 5. Test de vigilancia       | `‖T·X‖/‖X‖` contra ρ                                        | `match_ratio >= self.rho`                                            |
| 6. Deshabilitar el mejor    | se apaga `j*` solo para este patrón y se vuelve al paso 3   | `active_mask[j_star] = False`                                        |
| 7. Adaptar                  | `t_new = t AND x`, `b_new = t_new/(0.5 + Σ t_new)`          | `_adapt_weights`                                                     |
| 8. Repetir                  | se rehabilitan los nodos apagados y entra el próximo patrón | el bucle de `fit` (el `active_mask` se arma de nuevo en cada patrón) |

Con los nombres de la capa F:

1. Elección. Los puntajes `μ_j` de las categorías todavía activas compiten y gana la mayor.
2. Comparación. La ganadora manda su plantilla `t` de vuelta a F1 y se hace `T = t AND x`.
3. Test de vigilancia. `‖T‖/‖x‖` se compara con ρ.
4. Resonancia o reset. Si pasa, hay resonancia (el patrón y la categoría "resuenan" y la red se estabiliza). Si no pasa, reset: se apaga esa categoría para este patrón y se vuelve al paso 1 con las que quedan.
5. Aprendizaje rápido. En resonancia, `t_j* ← T` y `b` se recalcula. Se llama "fast learning" porque la plantilla salta al AND en un solo paso, sin tasa de aprendizaje gradual.

Si todas las categorías comprometidas fallan, gana una libre y pasa siempre. No hay una rama explícita de "crear categoría" en el código: se crea sola porque la libre gana, pasa el test (ratio 1) y entonces `_adapt_weights` la deja igual a `x`. Eso se ve en `if j_star >= self.num_committed_categories: self.num_committed_categories = j_star + 1`.

### 3.6 Las fórmulas, una por una

Norma de un vector binario: `‖x‖ = Σ x_i`, o sea cuántos unos tiene. En el código, `norm_x = sum(x)`.

Puntaje: `μ_j = Σ_i b_ij x_i`. Como `x` es binario, es la suma de los pesos bottom-up de los síntomas presentes.

Coincidencia (AND): `T_i = t_{j*,i} · x_i`. Con valores 0 y 1 el producto es un AND lógico.

Matching ratio: `‖T‖/‖x‖ = Σ t_{j*,i} x_i / Σ x_i`. Fracción de los síntomas del paciente que están en la plantilla. Es un número entre 0 y 1.

Test de vigilancia: pasa si `‖T‖/‖x‖ ≥ ρ` (igual que en Lau y en el código; ver 3.8 por qué importa el `≥`).

Adaptación: `t_new = T` y `b_ij* = t_new_i / (0.5 + Σ_k t_new_k)`.

De dónde sale el `0.5`: es la constante de la formulación de Lippmann/Lau. Qué hace, verificado con números: hace que los pesos `b` de una plantilla con más unos sean más chicos (por ejemplo, con 1 uno cada peso vale `1/1.5 = 0.667`, con 2 unos vale `1/2.5 = 0.4`, con 3 unos `1/3.5 = 0.286`). Entonces, ante un paciente que coincide entero con dos plantillas de distinto tamaño, el puntaje favorece a la plantilla más específica (la de menos unos): paciente `[1,0,0,0,0]` contra plantillas `[1,0,0,0,0]` y `[1,1,0,0,0]` da `0.667` contra `0.4`. Sin el 0.5 empatarían en 1. Y además el denominador `0.5 + ‖t‖` nunca se anula aunque la plantilla quede vacía. Si preguntan por una justificación más profunda, se contesta con honestidad: "es la normalización de la formulación de Lippmann que sigue el paper de Lau; lo que verificamos es que desempata a favor de la plantilla más específica".

### 3.7 El caso ρ alto y ρ bajo

ρ cercano a 1: el paciente tiene que estar casi entero dentro de la plantilla. Se crean más categorías, más chicas y más homogéneas. En nuestro dataset, con ρ ≥ 0.8 hay 17 categorías (una por cada combinación distinta de síntomas, más o menos) y el matching ratio promedio es 1.

ρ cercano a 0: casi todo pasa. Pocas categorías, grandes y mezcladas. Cuidado: ρ = 0 no da un cluster único. Probado con el código: con ρ = 0 salen 5 categorías, todas de un solo síntoma (Tos 45 pacientes, Fiebre 22, Dolor abdominal 15, Moco 4, Presión alta 14). La razón es que los puntajes `μ` ordenan los candidatos antes de la vigilancia: una categoría comprometida que no tiene ningún síntoma en común con el paciente tiene puntaje 0, y una libre tiene puntaje `‖x‖/6 > 0`, así que la libre gana y abre una categoría nueva aunque ρ sea 0. Es decir que ρ no es lo único que decide cuántas categorías hay: la competencia por puntaje viene primero.

ρ = 1 en el código (con `>=`): solo entra a una categoría un paciente cuyos síntomas estén todos en la plantilla. Con nuestros datos da 17 categorías.

El ratio solo puede tomar fracciones chicas (`k/‖x‖` con `‖x‖` entre 1 y 5), por eso los resultados se mueven a saltos: ver la tabla de la sección 7.

### 3.8 El test de vigilancia: por qué `>=`

Lau (Box 3, paso 5) y nuestro código usan el mismo operador, `>=`: `if match_ratio >= self.rho:`. Un ratio igual al umbral pasa el test. Vale la pena entender por qué importa. Con `>=` el rango completo de ρ de 0 a 1 es utilizable, y ρ = 1 significa "coincidencia total". Con `>` y ρ = 1 nada podría resonar nunca (un ratio no puede ser mayor que 1), y la red se llenaría de categorías repetidas hasta agotar los nodos.

En este dataset la igualdad además se nota, porque los ratios son fracciones exactas (1/2, 2/3): el ratio de P021 contra la categoría 0 es exactamente `0.5`, y con ρ = 0.5 pasa y P021 se une a la categoría 0. Dato de sensibilidad, probado con el módulo: si alguien cambiara el operador a `>`, con ρ = 0.5 saldrían 14 categorías en vez de 8, y con ρ = 2/3 exacto saldrían 17 en vez de 14.

### 3.9 Límites y desventajas

Sensibilidad al orden. El primer paciente que llega define la primera plantilla, y los siguientes se acomodan alrededor. Con el mismo dataset en distinto orden salen distintas particiones. Medido: barajando 10 veces el entrenamiento (semilla 42, ρ = 0.65) salen entre 13 y 15 categorías (base: 14), y en validación entre 12 y 14 (base: 12). Ordenado por cantidad de síntomas (primero los más cargados) salen 9 categorías en entrenamiento, y con los menos cargados primero salen 15. O sea, el orden puede mover el resultado bastante más que un barajado al azar.

Solo binario. Si el síntoma fuera "temperatura 38.5" habría que discretizar a mano (fiebre sí/no), y se pierde información. Para valores continuos existe ART2.

La plantilla solo pierde bits. Una plantilla es la intersección acumulada. Si entra temprano un paciente con un síntoma de menos, la plantilla lo pierde para siempre. Ejemplo real del dataset en la sección 4.

Sin ruido tolerado. El paper lo dice en la página 13: con patrones perfectos funciona bien pero incluso una pizca de ruido (un pixel faltante en la "F" de la Fig. 11) puede generar categorías nuevas, y con vigilancia alta el número de ejemplares puede crecer hasta agotar los nodos disponibles. Propone adaptar más lento y variar ρ durante el entrenamiento, ideas que no implementamos.

Capacidad limitada. M es un tope duro (`--max_cat`, 50). Si se agotan, `train_pattern` levanta `RuntimeError`.

La vigilancia se elige a mano. No hay un valor óptimo demostrado. Elegimos 0.65 porque da categorías interpretables, no porque sea el óptimo.

No es un algoritmo "optimizador": no minimiza ninguna función de costo. No hay noción de convergencia a una partición única.

### 3.10 Comparación breve con otras técnicas

| Aspecto               | ART1                                        | k-means                                                   | SOM (Kohonen)                                     |
| --------------------- | ------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------- |
| Entrada               | binaria                                     | numérica continua                                         | numérica continua                                 |
| Cantidad de grupos    | crece sola, controlada por ρ                | se fija de antemano (k)                                   | se fija la grilla de neuronas                     |
| Aprendizaje           | incremental, un patrón por vez              | por lotes, iterativo hasta converger                      | incremental, muchas épocas                        |
| Plantilla/prototipo   | intersección (AND) de sus miembros          | media de sus miembros                                     | pesos que se acercan a las entradas, con vecindad |
| Orden de los datos    | importa (un solo pasaje)                    | casi no importa si converge, depende de la inicialización | importa menos, hay muchas épocas                  |
| Qué conserva          | categorías viejas (estabilidad-plasticidad) | reentrena todo                                            | tiende a reescribir con datos nuevos              |
| Estructura topológica | no                                          | no                                                        | sí, mapa ordenado                                 |

En el paper: k-means es el algoritmo clásico parecido a Kohonen, y leader clustering el de Carpenter/Grossberg. Frente a Hopfield y Hamming (también binarios) la diferencia es que esas son supervisadas, necesitan ejemplares conocidos de antemano. Frente al perceptrón multicapa, que también necesita etiquetas y trabaja con valores continuos, ART1 crea sus categorías solo.

Una respuesta honesta a "¿por qué no k-means?": con vectores de 5 bits y sin saber cuántos grupos hay, ART1 permite explorar la granularidad moviendo ρ, y el resultado (las plantillas) se lee directamente como combinaciones de síntomas, lo cual es fácil de explicar. k-means sobre ceros y unos funcionaría pero sus centroides serían fracciones (por ejemplo 0.37 de tos), menos interpretables.

---

## 4. Un ejemplo resuelto a mano

Es el mismo caso del notebook (celdas de la sección 5.10), con los números reales del dataset de entrenamiento, ρ = 0.65, N = 5, M = 50. Síntomas por posición: 0 Tos, 1 Moco, 2 Fiebre, 3 Dolor abdominal, 4 Presión alta.

### 4.1 Estado inicial antes de P021

Los dos primeros pacientes del archivo son P015 `[1,0,0,0,0]` y P075 `[0,0,1,0,0]`.

P015, primer paciente: no hay categorías comprometidas, gana la categoría 0 (libre, `μ = 1/6 = 0.1667`), `T = [1,0,0,0,0]`, ratio `1/1 = 1`. Resonancia. La plantilla queda `[1,0,0,0,0]` y los pesos de la categoría 0: `b = t/(0.5 + 1) = 1/1.5 = 0.6667` en el bit de tos y 0 en el resto.

P075: la categoría 0 tiene `μ = b_Tos · x_Tos + ... = 0.6667·0 + ... = 0` porque P075 no tiene tos. Una libre tiene `μ = 1/6 = 0.1667`. Gana la categoría 1 (primera libre). Ratio 1. Plantilla `[0,0,1,0,0]`.

Estado: `t[0] = [1,0,0,0,0]`, `t[1] = [0,0,1,0,0]`, `b[:,0] = [0.6667, 0, 0, 0, 0]`, `b[:,1] = [0, 0, 0.6667, 0, 0]`, resto libre con `b = 0.1667`.

### 4.2 P021: `x = [1,0,0,0,1]` (tos y presión alta)

Paso 1, norma. `‖x‖ = 1 + 0 + 0 + 0 + 1 = 2`.

Paso 2, puntajes (ciclo 1).

- `μ_0 = 0.6667·1 + 0·0 + 0·0 + 0·0 + 0·1 = 0.6667`
- `μ_1 = 0·1 + 0 + 0.6667·0 + 0 + 0·1 = 0`
- `μ_2 = (1/6)(1) + (1/6)(1) = 0.3333` (categoría libre: suma los dos síntomas presentes)
- `μ_3, ..., μ_49` igual, 0.3333

Gana `j* = 0` con 0.6667.

Paso 3, AND. `t[0] = [1,0,0,0,0]`, `x = [1,0,0,0,1]`, `T = [1·1, 0·0, 0·0, 0·0, 0·1] = [1,0,0,0,0]`, `‖T‖ = 1`. Matching ratio `1/2 = 0.5`.

Paso 4, vigilancia. `0.5 ≥ 0.65`: falso. Reset. La categoría 0 se apaga solo para P021.

Paso 5, puntajes (ciclo 2), sin la categoría 0. `μ_0 = -1` (el código usa -1 como marca de "apagada"), `μ_1 = 0`, `μ_2 = 0.3333`. Gana `j* = 2`, la primera libre (empate entre libres, `index` devuelve la primera).

AND: `t[2] = [1,1,1,1,1]`, `T = [1,0,0,0,1]`, `‖T‖ = 2`, ratio `2/2 = 1`. `1 ≥ 0.65`: sí. Resonancia, categoría nueva.

Paso 6, adaptación. `t[2] ← [1,0,0,0,1]`. Denominador `0.5 + 2 = 2.5`. `b[:,2] = [1/2.5, 0, 0, 0, 1/2.5] = [0.4, 0, 0, 0, 0.4]`.

Resultado: P021 va al cluster 2 con matching ratio 1.0. Es lo que imprime el notebook y lo que da el módulo real. (En el notebook se chequea con `assert` que el rastreo manual y `ART1Network` dan lo mismo.)

### 4.3 Qué le pasó después a ese cluster 2

Esto vale la pena entenderlo porque aparece en la ficha del paciente. Los pacientes que fueron al cluster 2 (posición en el archivo, 0-based):

| Posición | Paciente | `x`           | Plantilla del cluster 2 después |
| -------- | -------- | ------------- | ------------------------------- |
| 2        | P021     | `[1,0,0,0,1]` | `[1,0,0,0,1]`                   |
| 9        | P094     | `[1,0,0,0,1]` | `[1,0,0,0,1]`                   |
| 13       | P035     | `[0,0,0,0,1]` | `[0,0,0,0,1]`                   |

En la posición 13 entra P035, que tiene solo presión alta. Contra la plantilla `[1,0,0,0,1]` el AND da `[0,0,0,0,1]`, ratio `1/1 = 1`, pasa, y la plantilla se achica a `[0,0,0,0,1]`. A partir de ahí, un paciente con tos y presión alta contra esa plantilla da `1/2 = 0.5 < 0.65` y ya no entra: va a otra categoría (así nace el cluster 12, `[1,0,0,0,1]`, con otros 5 pacientes). Moraleja, que sirve para responder preguntas:

- La plantilla final de un cluster no es la que tenía cuando entró cada paciente. P021 y P094 quedaron en un cluster cuya plantilla final es solo "Presión alta".
- El matching ratio de cada paciente se calcula contra la plantilla en el momento de la asignación, no contra la final.
- Por eso la ficha individual de P021 (`--patient-id P021`) muestra como "patrón representante" `[0,0,0,0,1]`, aunque P021 tiene tos y presión alta. Y la especialidad recomendada de la ficha sale de los síntomas del propio paciente (tos manda a Neumonología), no de la plantilla.

Verificado corriendo `python src/CarGross.py data/dataset_entrenamiento_ART1.csv --patient-id P021`.

### 4.4 Un caso con ratio menor a 1 que igual pasa: P001

P001 `[1,1,1,0,0]` (posición 14) llega cuando ya hay 8 categorías: `[1,0,0,0,0]`, `[0,0,1,0,0]`, `[0,0,0,0,1]`, `[0,1,0,0,0]`, `[0,1,1,0,0]`, `[0,0,0,1,0]`, `[0,0,1,1,0]`, `[0,0,1,0,1]`.

Puntajes: categoría 4 (`[0,1,1,0,0]`, `b = 1/2.5 = 0.4` en moco y fiebre) da `0.4 + 0.4 = 0.8`. Las categorías 0, 1 y 3 (un solo bit, `b = 0.6667`) dan `0.6667`. Libre: `3/6 = 0.5`. Gana la 4 con 0.8.

AND: `T = [0,1,1,0,0]`, `‖T‖ = 2`, `‖x‖ = 3`, ratio `2/3 = 0.6667`. `0.6667 ≥ 0.65`: pasa por poquito. La plantilla no cambia (T es igual a la plantilla). P001 entra al cluster 4 aunque la plantilla no tiene tos: ese 1/3 de su perfil queda sin explicar. Con ρ = 0.7 ese mismo paciente no hubiese pasado (`0.6667 < 0.7`) y habría creado categoría nueva. Por eso el salto de 14 a 17 categorías ocurre entre 0.65 y 0.7.

Observación útil: en el entrenamiento a ρ = 0.65, los matching ratio son solo dos valores: 1.0 (85 pacientes) y 0.6667 (15 pacientes). El promedio `(85·1 + 15·0.6667)/100 = 0.95`. Es exactamente el 0.9500 que se informa.

---

## 5. El dataset

### 5.1 Archivos y columnas

| Archivo                                                       | Filas | Columnas                                                   |
| ------------------------------------------------------------- | ----- | ---------------------------------------------------------- |
| `data/dataset_entrenamiento_ART1.csv` (y `.xlsx` equivalente) | 100   | Paciente, Tos, Moco, Fiebre, Dolor abdominal, Presión alta |
| `data/dataset_validacion_ART1.csv` (y `.xlsx`)                | 100   | las anteriores + Especialidad esperada                     |

La columna "Paciente" es el ID (P001 a P100 en entrenamiento, P101 a P200 en validación). Los cinco síntomas son 1 (presente) o 0 (ausente). Los `.xlsx` dan exactamente los mismos vectores que los `.csv` (el notebook lo comprueba y se verificó que la salida CSV de la corrida con el `.xlsx` es idéntica a la del `.csv`).

### 5.2 Qué significa cada síntoma

Son síntomas clínicos simples, representados como presente o ausente:

| Bit | Síntoma         | Qué es                                              |
| --- | --------------- | --------------------------------------------------- |
| 0   | Tos             | tos presente                                        |
| 1   | Moco            | expectoración o mucosidad                           |
| 2   | Fiebre          | fiebre                                              |
| 3   | Dolor abdominal | dolor en la zona abdominal                          |
| 4   | Presión alta    | hipertensión (como indicador sí o no, sin el valor) |

El dataset es sintético: no son pacientes reales. La validación tiene exactamente 20 pacientes por cada una de las 5 especialidades esperadas (Neumología, Infectología, Gastroenterología, Cardiología, Medicina general).

### 5.3 Entrenamiento y validación: son distintos

Verificado con código:

- 0 IDs en común entre los dos archivos.
- Mirando fila por fila en la misma posición, solo 9 de 100 filas tienen el mismo vector de síntomas (coincidencia casual: hay solo 32 vectores posibles de 5 bits).
- 21 patrones distintos en entrenamiento y 20 en validación.
- La validación agrega la columna "Especialidad esperada".
- Los dos tienen 9 pacientes sin ningún síntoma.

Cantidad de síntomas por paciente en entrenamiento: 0 síntomas 9 pacientes, 1 síntoma 36, 2 síntomas 39, 3 síntomas 15, 4 síntomas 1. Frecuencia total de cada síntoma en entrenamiento: tos 36, moco 28, fiebre 38, dolor abdominal 27, presión alta 34. En validación: 37, 20, 39, 27 y 30.

Patrones más frecuentes en entrenamiento: solo presión alta (14 pacientes), solo dolor abdominal (11), sin síntomas (9), tos+moco+fiebre (8), tos+presión alta (6), tos+moco (6), fiebre+dolor (6).

### 5.4 Los patrones sin síntomas

Hay 9 en cada dataset. Un vector todo en ceros tiene `‖x‖ = 0` y el matching ratio sería `0/0`. El código lo resuelve con una convención en `train_pattern`:

```python
norm_x = sum(x)
if norm_x == 0:
    return 0, 1.0
```

Lo asigna a la categoría 0 con ratio 1.0, sin tocar ningún peso. Consecuencias reales que conviene tener claras:

- Esos 9 pacientes cuentan en el promedio del matching ratio con 1.0 (si se los excluye, el promedio del entrenamiento baja de 0.9500 a 0.9451).
- Cuentan como miembros del cluster 0 en la pureza. En entrenamiento el cluster 0 tiene 13 pacientes: 4 con tos sola y 9 sin síntomas. En validación los 9 sin síntomas tienen etiquetas variadas: 5 Medicina general, 2 Infectología, 1 Cardiología, 1 Gastroenterología.
- Si el primer paciente del archivo estuviera vacío, se devuelve el cluster 0 pero `num_committed_categories` queda en 0 (porque no se compromete ninguna categoría). No pasa con nuestros datos, es un borde del código.
- Un paciente sin síntomas no tiene un "cluster" propio porque no se puede decir a cuál se parece. Es una decisión de diseño, no una propiedad de ART1.

Excluyendo esos 9 pacientes de la pureza de validación (ρ = 0.65) da 70.33% sobre los 91 restantes, apenas distinta del 69%.

### 5.5 Carga de datos: qué detecta `load_dataset`

Detecta el formato por extensión (`.xlsx` o cualquier otra cosa se trata como CSV). La primera columna se toma como ID si el encabezado contiene "id" o "paciente". La última se toma como etiqueta si su encabezado contiene "especialidad", "label" o "diag". Todo lo demás son síntomas. Si no hay encabezado de etiqueta, igual mira la primera fila de datos: si la última celda no es un número, la trata como etiqueta. Si no hay columna ID, genera `PAC_001`, `PAC_002`, etc. Cada celda de síntoma se convierte con `int(float(valor))` y si no es 0 ni 1 corta con un `ValueError` que dice fila y columna.

---

## 6. Recorrido del código

Archivo: `src/CarGross.py` (unas 900 líneas). Biblioteca estándar nada más (`csv`, `argparse`, `random`, `datetime`, `os`, `sys`, `typing`, `unicodedata`); `openpyxl` se usa solo si está instalado para leer `.xlsx`, y si no hay un lector propio con `zipfile` y `xml.etree`. Las líneas son aproximadas.

### 6.1 Mapa del archivo

| Zona                                                                                                                         | Líneas aprox.   | Qué hay                             |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------- | ----------------------------------- |
| `class ART1Network`                                                                                                          | 13 a 147        | el algoritmo y las métricas         |
| `infer_specialty_and_recommendations`                                                                                        | 150 a 192       | reglas de especialidad (no es ART1) |
| lectores (`_normalize_numeric_str`, `_cell_to_str`, `_read_csv_rows`, `_xlsx_*`, `_read_xlsx_rows`, `_drop_phantom_columns`) | 195 a 350       | utilitarios de lectura              |
| `load_dataset`                                                                                                               | 352 a 420       | carga y validación del dataset      |
| `generate_individual_patient_report`                                                                                         | 423 a 483       | ficha individual                    |
| `run_interactive_mode`                                                                                                       | 486 a 524       | consulta por consola                |
| `print_manual`                                                                                                               | 527 a ~680      | el texto de `--man`                 |
| `run_smoke_test`                                                                                                             | 683 a ~723      | el `--test`                         |
| `main` y `_run`                                                                                                              | ~726 a el final | CLI, flujo y salida                 |

### 6.2 `ART1Network.__init__` (líneas 14 a 34)

```python
def __init__(self, num_inputs: int, max_categories: int = 50, rho: float = 0.65):
    if not (0.0 <= rho <= 1.0):
        raise ValueError("El parámetro de vigilancia (rho) debe estar entre 0.0 y 1.0.")
    if num_inputs <= 0:
        raise ValueError(...)
    self.N = num_inputs
    self.M = max_categories
    self.rho = rho
    self.num_committed_categories = 0
    self.t = [[1.0 for _ in range(self.N)] for _ in range(self.M)]   # top-down, M x N
    initial_b = 1.0 / (1.0 + self.N)
    self.b = [[initial_b for _ in range(self.M)] for _ in range(self.N)]  # bottom-up, N x M
    self.history_assignments = []
    self.history_ratios = []
```

Parámetros: `num_inputs` (N), `max_categories` (M, por defecto 50), `rho` (por defecto 0.65). Valida que ρ esté en [0,1] y que N sea positivo. `num_committed_categories` es la cantidad de categorías usadas (se calcula como el mayor índice usado más 1). Las dos historias guardan, para cada paciente en orden de entrada, a qué cluster fue y con qué ratio; de ahí salen las métricas y la pureza.

Por qué así: es literalmente el paso 1 del Box 3. Por qué listas de Python y no NumPy: para que el módulo no dependa de nada y sea fácil de instalar (la consigna lo pide: "debe ser sencillo de instalar"). Con 5 entradas y 50 categorías no hace falta nada más rápido.

### 6.3 `_compute_matching_scores` (líneas 36 a 44)

```python
for j in range(self.M):
    if not active_mask[j]:
        scores.append(-1.0)
    else:
        score = sum(self.b[i][j] * x[i] for i in range(self.N))
        scores.append(score)
```

Es `μ_j = Σ b_ij x_i`. Las categorías apagadas por reset reciben `-1.0`, que es menor que cualquier puntaje real (siempre mayor o igual a 0) y por eso nunca vuelven a ganar en ese patrón. Calcula las M categorías, no solo las comprometidas: las libres tienen `b = 1/6` en todas las entradas y puntúan `‖x‖/6`.

### 6.4 `train_pattern` (líneas 46 a 81)

Es el corazón. Se lee en bloques.

```python
if len(x) != self.N: raise ValueError(...)
if any(bit not in (0, 1) for bit in x): raise ValueError(...)
norm_x = sum(x)
if norm_x == 0:
    return 0, 1.0
active_mask = [True] * self.M
while True:
    scores = self._compute_matching_scores(x, active_mask)
    max_score = max(scores)
    if max_score < 0:
        raise RuntimeError("Capacidad de la red alcanzada: ...")
    j_star = scores.index(max_score)
    T = [self.t[j_star][i] * x[i] for i in range(self.N)]
    norm_T = sum(T)
    match_ratio = norm_T / float(norm_x)
    if match_ratio >= self.rho:
        self._adapt_weights(j_star, T)
        if j_star >= self.num_committed_categories:
            self.num_committed_categories = j_star + 1
        return j_star, match_ratio
    else:
        active_mask[j_star] = False
```

- Validaciones de entrada: dimensión y binario.
- Patrón vacío: convención (ver 5.4).
- `active_mask`: se reinicia en cada patrón. Es el "rehabilitar los nodos deshabilitados" del paso 8.
- El `while True` es el ciclo reset/reintento. Termina porque siempre hay una categoría que pasa (la libre) o se agotan los nodos.
- `max_score < 0` solo ocurre si todas las M categorías quedaron apagadas, o sea, se agotó la capacidad. Es un `RuntimeError`.
- `j_star = scores.index(max_score)`: ver pregunta "por qué desempata así" en la sección 9. En breve: `index` devuelve la primera posición del máximo, entonces en empates gana el índice menor, y como las libres están al final y puntúan igual, siempre se abre la primera libre.
- `T` es el AND (producto de 0/1). `T` queda como lista de flotantes (1.0 y 0.0) porque `t` es de flotantes.
- `match_ratio = norm_T / float(norm_x)`: nunca divide por cero porque el caso `norm_x == 0` salió antes.
- `>=`: ver 3.8.
- Si pasa, adapta, actualiza el contador de categorías y devuelve `(cluster, ratio)`.
- Si no pasa, apaga esa categoría y el `while` vuelve a empezar con las restantes.

Por qué no hay una rama "crear categoría nueva": se crea sola. La categoría libre tiene `t` todo en 1, entonces `T = x`, el ratio es 1 y `1 ≥ ρ` siempre (ρ ≤ 1). Eso garantiza que el bucle termina mientras queden nodos.

### 6.5 `_adapt_weights` (líneas 83 a 92)

```python
for i in range(self.N):
    self.t[j_star][i] = T[i]
sum_t = sum(self.t[j_star][k] for k in range(self.N))
denominator = 0.5 + sum_t
for i in range(self.N):
    self.b[i][j_star] = self.t[j_star][i] / denominator
```

Paso 7 de Lau. `t ← T` (la plantilla pasa a la intersección) y `b ← t/(0.5 + ‖t‖)`. Solo se tocan los pesos de la ganadora. Las demás categorías no cambian: eso es lo que da estabilidad.

### 6.6 `fit`, `get_cluster_templates` (líneas 94 a 113)

`fit(dataset)` recorre la lista en el orden recibido, llama a `train_pattern` por cada uno, guarda en las historias y devuelve una lista de diccionarios con `patient_index`, `cluster_assigned` y `matching_ratio`. Es una sola pasada (una época). No hay reentrenamiento ni segunda vuelta.

`get_cluster_templates()` devuelve `{j: [0/1,...]}` para las categorías `0 .. num_committed_categories-1`, convirtiendo a enteros. Son las plantillas finales (después del último patrón).

### 6.7 `compute_metrics` y `_calculate_purity` (líneas 115 a 147)

Definiciones exactas como las calcula el código:

| Clave                      | Cálculo                                                                                                                                              | Cómo se imprime                                         |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `num_clusters`             | `num_committed_categories`                                                                                                                           | "Categorías / Clusters Formados (M)"                    |
| `avg_matching_ratio`       | `sum(history_ratios) / len(history_ratios)`: promedio de los ratios del momento de asignación de cada paciente (incluye los patrones vacíos con 1.0) | "Matching Ratio Promedio"                               |
| `avg_template_active_bits` | suma de los unos de todas las plantillas finales dividida por la cantidad de plantillas                                                              | "Estabilidad (bits '1' activos promedio por plantilla)" |
| `cluster_purity`           | ver abajo, solo si hay etiquetas y su cantidad coincide con la de asignaciones                                                                       | "Pureza de Clusters (Validación Externa)"               |

Pureza:

```python
correct_count = sum(max(counts.values()) for counts in cluster_counts.values())
return correct_count / float(len(labels))
```

Para cada cluster se cuenta cuántos pacientes hay de cada etiqueta, se toma el conteo de la etiqueta más frecuente, se suman esos máximos y se divide por el total de pacientes. Es "qué porcentaje de pacientes habría acertado si cada cluster respondiera con su etiqueta mayoritaria".

Dos cosas que hay que saber: la "estabilidad" es un nombre desafortunado, es solo el promedio de unos por plantilla, no mide estabilidad en ningún sentido de series o de reproducibilidad (mide qué tan "anchos" son los prototipos). Y la pureza solo se calcula si el archivo trae la columna de etiquetas; en el dataset de entrenamiento no aparece.

La pureza tiene una propiedad que hay que conocer: con más clusters sube sola. Con tantos clusters como pacientes la pureza sería 100% sin que eso sea una buena agrupación. Por eso hay que mirarla junto con la cantidad de clusters.

### 6.8 `infer_specialty_and_recommendations` (líneas 150 a 192)

No es parte de ART1: es una cadena de `if` por prioridad sobre los síntomas activos.

| Prioridad | Condición (sin tildes, en minúsculas) | Especialidad                  |
| --------- | ------------------------------------- | ----------------------------- |
| 1         | `disnea` o `tos`                      | NEUMONOLOGÍA                  |
| 2         | `presion_alta` o `presion alta`       | CARDIOLOGÍA                   |
| 3         | `dolor_abdominal` o `dolor abdominal` | GASTROENTEROLOGÍA             |
| 4         | `fiebre` o `moco`                     | INFECTOLOGÍA / CLÍNICA MÉDICA |
| resto     |                                       | CLÍNICA MÉDICA GENERAL        |

Detalle de normalización: el encabezado de la columna es `Presión alta`, con tilde. Antes la regla comparaba contra `presion alta` sin tilde y nunca se activaba (la presión alta sola caía en CLÍNICA MÉDICA GENERAL). Ahora los síntomas se pasan a minúsculas y se les sacan las tildes (`unicodedata.normalize("NFD", ...)`) antes de comparar, así `infer_specialty_and_recommendations(["Presión alta"])` devuelve CARDIOLOGÍA. Efecto medido: cambia la especialidad sugerida de 22 pacientes en validación y de 24 en entrenamiento; la coincidencia entre la sugerencia de las reglas y la especialidad esperada pasó de 49 a 60 de 100. Eso es de la capa de reglas, no de ART1: los clusters, los ratios y la pureza no cambian (la pureza se calcula contra la columna "Especialidad esperada" del CSV, no contra las reglas).

También: la regla toma el primer síntoma que coincide, entonces tos más presión alta da Neumonología aunque presión alta sola da Cardiología.

### 6.9 Lectura de datos (líneas 195 a 420)

`_normalize_numeric_str("1.0")` devuelve `"1"` para que CSV y XLSX den lo mismo; no toca IDs como `"007"`. `_cell_to_str` pasa una celda a texto (None a vacío, True a `"1"`, `1.0` a `"1"`). `_xlsx_col_index("B3")` devuelve 1 (columna base 0). `_xlsx_cell_value` y `_read_xlsx_rows_stdlib` leen un `.xlsx` como lo que es, un ZIP con XML adentro: `sharedStrings.xml`, `workbook.xml`, la primera hoja. `_read_xlsx_rows` usa `openpyxl` si está instalado y si no el lector propio.

`load_dataset` lee los `.csv` con `_read_csv_rows`, que prueba `utf-8-sig` (tolera el BOM de Excel), después `cp1252` y después `latin-1`, salta comentarios (`#`) y filas vacías y normaliza las celdas con `_cell_to_str`. Después `_drop_phantom_columns` saca las columnas fantasma de pandas (`Unnamed: 0` o encabezado vacío). Los `.xlsx` pasan por `_read_xlsx_rows`. Con los archivos de `data/` el resultado es el mismo que antes (100 filas y 5 síntomas en los cuatro archivos, mismas asignaciones y métricas), y un CSV guardado en `cp1252` con tildes ahora carga bien.

### 6.10 `load_dataset` (líneas 352 a 420)

Devuelve una tupla de 4: `(patrones, nombres_de_síntomas, etiquetas_o_None, ids)`. Pasos: verifica que el archivo exista (`FileNotFoundError`), lee filas, identifica columnas (ID, etiqueta, síntomas) mirando los encabezados, convierte cada celda de síntoma a `int(float(...))` y valida 0/1 con mensajes que dicen fila (contando el encabezado como fila 1) y columna. Filas con menos celdas que el encabezado se saltan en silencio. Si el archivo no tiene datos, `ValueError("El archivo está vacío...")`. Un archivo con solo encabezado y sin filas levanta `ValueError("El archivo tiene encabezado pero no tiene filas de datos válidas.")`, que `main` muestra como `[ERROR]`.

### 6.11 `generate_individual_patient_report` y `run_interactive_mode`

La ficha individual muestra: fecha, ID, ρ, síntomas presentes y ausentes, el vector, el cluster asignado, el porcentaje de coincidencia, el "estado de vigilancia" (siempre dice pasado, porque si el paciente está asignado es porque pasó), el patrón representante (plantilla final del cluster, ver 4.3), el total de clusters, la especialidad (con la etiqueta de referencia si existe) y recomendaciones según la especialidad, más una nota de que debe validarlo un profesional.

`run_interactive_mode` pide nombre o ID y luego un 1 o 0 por cada síntoma (repite si no es 1 ni 0). Importante: ese paciente nuevo se pasa por `art1.train_pattern(vector)`, o sea que se clasifica y también se aprende (modifica la red que ya se entrenó con el dataset). La red del dataset ya está entrenada cuando se llega a este modo. El modo interactivo no escribe el CSV de resultados.

### 6.12 `main`, `_run` y el CLI

`main()` llama a `_run()` y atrapa `FileNotFoundError, FileExistsError, ValueError, ImportError, OSError, RuntimeError`: imprime `[ERROR] motivo` y una pista (`--man` o `--help`) por stderr y sale con código 1. Probado: `--max_cat 3` sobre el dataset de entrenamiento imprime `[ERROR] Capacidad de la red alcanzada: ...` y un CSV con solo encabezado imprime `[ERROR] El archivo tiene encabezado pero no tiene filas de datos válidas.`, los dos con código 1.

Argumentos:

| Argumento                                   | Defecto                   | Qué hace                                                                                 |
| ------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| `pos_input` (posicional) o `--input` / `-i` |                           | ruta del dataset (.csv o .xlsx); el posicional tiene prioridad                           |
| `--rho` / `-r`                              | 0.65                      | vigilancia entre 0 y 1; fuera de rango corta con `[ERROR]`                               |
| `--output` / `-o`                           | `resultados_cargross.csv` | CSV de salida (una fila por paciente)                                                    |
| `--save-txt`                                | ninguno                   | reporte de texto                                                                         |
| `--max_cat`                                 | 50                        | máximo de categorías M (con guion bajo, no con guion medio)                              |
| `--verbose` / `-v`                          | apagado                   | agrega la lista de síntomas detectados (casi no cambia nada más)                         |
| `--interactive`                             | apagado                   | modo consulta de paciente en vivo                                                        |
| `--patient-id`                              | ninguno                   | ficha de un paciente por ID (case-insensitive) o por número de orden ("1" es el primero) |
| `--shuffle`                                 | 0                         | cantidad de corridas con orden barajado                                                  |
| `--seed`                                    | 42                        | semilla del barajado                                                                     |
| `--test`                                    | apagado                   | autotest                                                                                 |
| `--man`                                     | apagado                   | imprime el manual                                                                        |

Orden de decisiones en `_run`: primero `--man`, luego `--test`, luego exige un archivo de entrada. Imprime el encabezado con archivo y ρ (antes de cargar, por eso en un error de archivo igual se ve el encabezado). Carga, crea la red con `N = cantidad de síntomas` y entrena. Calcula métricas. Después, según el modo:

- `--interactive`: pide un paciente y termina.
- `--patient-id`: imprime la ficha y termina. Si el ID no existe avisa y muestra el primer paciente. No escribe el CSV general.
- sin lo anterior (modo general): imprime el resumen, las plantillas con su especialidad, el test de barajado si se pidió, y escribe el CSV y, si hay `--save-txt`, el TXT.

Detalle que sorprende: como `--output` tiene valor por defecto, el CSV siempre se escribe en el modo general, aunque no se pase `-o` (en ese caso queda como `resultados_cargross.csv` en la carpeta desde donde se corrió; `.gitignore` lo ignora). Las carpetas de salida se crean solas con `os.makedirs`.

Archivos de salida:

- CSV: encabezado `ID_Paciente, Cluster_Asignado, Matching_Ratio, Especialidad_Sugerida`, una fila por paciente, ratio con 4 decimales. La especialidad de cada fila sale de los síntomas del propio paciente, no de la plantilla de su cluster. Ejemplo verificado: en validación, P188 (`[1,0,0,0,1]`) tiene Especialidad_Sugerida NEUMONOLOGÍA.
- TXT (`--save-txt`): cabecera, resumen de métricas, plantillas, resultado del barajado si corresponde, y un pie con fecha.

### 6.13 El test de barajado

```python
rng = random.Random(args.seed)
for run_idx in range(1, args.shuffle + 1):
    shuffled_indices = list(range(len(patterns)))
    rng.shuffle(shuffled_indices)
    shuffled_patterns = [patterns[i] for i in shuffled_indices]
    art1_shuffled = ART1Network(...)
    art1_shuffled.fit(shuffled_patterns)
```

Con una semilla fija el resultado es reproducible. Cada corrida arma una red nueva desde cero, la entrena con el orden barajado y registra clusters y matching ratio. Al final imprime el promedio contra la base original. No calcula pureza en las corridas barajadas.

### 6.14 `run_smoke_test` (`--test`)

Corre cinco chequeos chicos sobre un dataset mínimo en memoria (3 patrones de 4 bits): con ρ = 0.3 se forman 2 categorías, con ρ = 0.9 se forman 3, un patrón vacío va a la categoría 0 con ratio 1.0, el aprendizaje achica la plantilla con el AND, y "Presión alta" (con tilde) deriva a CARDIOLOGÍA. Imprime `ok` o `FALLO` por chequeo; `TEST PASSED` solo aparece si pasan todos, y si falla alguno imprime `[ERROR]` por stderr y sale con código 1. Sigue siendo una prueba chica: no reemplaza la verificación cruzada del notebook, que compara el módulo contra la implementación pedagógica con `assert`.

---

## 7. Resultados reales

Todo corrido con el módulo real (`ART1Network`), orden original de los archivos, M = 50.

### 7.1 Barrido de ρ en entrenamiento (sin etiquetas)

| ρ         | Categorías | Matching ratio | "Estabilidad" (bits por plantilla) |
| --------- | ---------- | -------------- | ---------------------------------- |
| 0.0 a 0.2 | 5          | 0.6975         | 1.00                               |
| 0.3       | 6          | 0.7050         | 1.50                               |
| 0.4       | 8          | 0.8008         | 1.62                               |
| 0.5       | 8          | 0.8008         | 1.62                               |
| 0.6       | 14         | 0.9500         | 1.64                               |
| 0.65      | 14         | 0.9500         | 1.64                               |
| 0.7       | 17         | 0.9975         | 1.88                               |
| 0.8       | 17         | 1.0000         | 1.88                               |
| 0.9       | 17         | 1.0000         | 1.88                               |
| 1.0       | 17         | 1.0000         | 1.88                               |

### 7.2 Barrido de ρ en validación (con etiquetas)

| ρ         | Categorías | Matching ratio | "Estabilidad" | Pureza |
| --------- | ---------- | -------------- | ------------- | ------ |
| 0.0 a 0.2 | 5          | 0.7425         | 1.00          | 56%    |
| 0.3       | 5          | 0.7500         | 1.00          | 56%    |
| 0.4       | 7          | 0.8150         | 1.43          | 57%    |
| 0.5       | 7          | 0.8150         | 1.43          | 57%    |
| 0.6       | 12         | 0.9600         | 1.58          | 69%    |
| 0.65      | 12         | 0.9600         | 1.58          | 69%    |
| 0.7       | 17         | 0.9975         | 2.00          | 73%    |
| 0.8       | 17         | 1.0000         | 2.00          | 72%    |
| 0.9       | 17         | 1.0000         | 2.00          | 72%    |
| 1.0       | 17         | 1.0000         | 2.00          | 72%    |

Estos números coinciden con `docs/informe.md` (que usa ρ = 0.3, 0.5, 0.65, 0.8 y 0.9) y con el barrido del notebook (que usa 0.4, 0.5, 0.65 y 0.8). Lo único que no está en esos documentos es la fila de ρ = 0.7 y la de ρ = 0 a 0.2.

### 7.3 Por qué los números se mueven a saltos

El ratio de un paciente con `k` síntomas solo puede valer `m/k` con `m ≤ k`. Con 5 síntomas y pacientes de 1 a 4 síntomas, los valores posibles son 1/4, 1/3, 1/2, 2/3, 3/4 y 1. Entonces los cortes de ρ son:

- Entre 0.5 y 0.6: el ratio `0.5` (un paciente de 2 síntomas con 1 en la plantilla) deja de pasar. Por eso 0.4 y 0.5 dan el mismo resultado (8), y 0.6 y 0.65 dan el mismo (14). El salto de 8 a 14 ocurre cuando ρ supera 0.5.
- Entre 0.65 y 0.7: el ratio `0.6667` (2 de 3 síntomas) deja de pasar. Por eso 0.65 da 14 y 0.7 da 17.
- Con ρ ≥ 0.8 solo pasa el ratio 1.0 (el paciente está completo en la plantilla). Por eso 0.8, 0.9 y 1.0 son idénticos.

Por eso 0.65 es un valor razonable por defecto: cae dentro del escalón que va de 0.5 a 0.6667 (da lo mismo que 0.6 o que ρ = 2/3 exacto), pero no es un óptimo demostrado.

### 7.4 Resultados con ρ = 0.65

Entrenamiento: 14 categorías, matching ratio 0.9500 (85 pacientes con 1.0 y 15 con 0.6667), 1.64 bits por plantilla (23 bits en total sobre 14 plantillas).

| Cluster | Plantilla     | Síntomas                      | Pacientes                 | Especialidad sugerida por las reglas (sobre la plantilla) |
| ------- | ------------- | ----------------------------- | ------------------------- | --------------------------------------------------------- |
| 0       | `[1,0,0,0,0]` | Tos                           | 13 (4 con tos + 9 vacíos) | NEUMONOLOGÍA                                              |
| 1       | `[0,0,1,0,0]` | Fiebre                        | 4                         | INFECTOLOGÍA / CLÍNICA MÉDICA                             |
| 2       | `[0,0,0,0,1]` | Presión alta                  | 16                        | CARDIOLOGÍA                                               |
| 3       | `[0,1,0,0,0]` | Moco                          | 4                         | INFECTOLOGÍA / CLÍNICA MÉDICA                             |
| 4       | `[0,1,1,0,0]` | Moco, Fiebre                  | 15                        | INFECTOLOGÍA / CLÍNICA MÉDICA                             |
| 5       | `[0,0,0,1,0]` | Dolor abdominal               | 11                        | GASTROENTEROLOGÍA                                         |
| 6       | `[0,0,1,1,0]` | Fiebre, Dolor abdominal       | 7                         | GASTROENTEROLOGÍA                                         |
| 7       | `[0,0,1,0,1]` | Fiebre, Presión alta          | 6                         | CARDIOLOGÍA                                               |
| 8       | `[1,1,0,0,0]` | Tos, Moco                     | 7                         | NEUMONOLOGÍA                                              |
| 9       | `[0,1,0,0,1]` | Moco, Presión alta            | 1                         | CARDIOLOGÍA                                               |
| 10      | `[1,0,1,0,0]` | Tos, Fiebre                   | 5                         | NEUMONOLOGÍA                                              |
| 11      | `[1,0,0,1,0]` | Tos, Dolor abdominal          | 2                         | NEUMONOLOGÍA                                              |
| 12      | `[1,0,0,0,1]` | Tos, Presión alta             | 5                         | NEUMONOLOGÍA                                              |
| 13      | `[0,0,0,1,1]` | Dolor abdominal, Presión alta | 4                         | CARDIOLOGÍA                                               |

Los 15 pacientes con ratio 0.6667 son los que entraron a una categoría que cubre 2 de sus 3 síntomas (por ejemplo, los 8 pacientes con tos, moco y fiebre que cayeron en el cluster 4, Moco+Fiebre; los 3 con moco, fiebre y dolor, también al cluster 4; P072 `[1,0,1,1,0]` en el cluster 6; P019 y P004 `[1,1,0,0,1]` en el 8; P002 `[1,0,1,0,1]` en el 7). Verificable en el notebook con el conteo "casos con ratio menor a 1.0: 15".

Validación, ρ = 0.65: 12 categorías, matching ratio 0.9600 (88 con 1.0, 12 con 0.6667), 1.58 bits por plantilla, pureza 69.00%. Matriz de contingencia (cluster contra especialidad esperada):

| Cluster | Plantilla       | Cardio | Gastro | Infecto | Med. general | Neumo | Total | Pureza del cluster |
| ------- | --------------- | ------ | ------ | ------- | ------------ | ----- | ----- | ------------------ |
| 0       | Tos             | 1      | 1      | 2       | 8            | 2     | 14    | 57.1%              |
| 1       | Moco            | 0      | 0      | 0       | 0            | 12    | 12    | 100%               |
| 2       | Presión alta    | 12     | 0      | 0       | 2            | 0     | 14    | 85.7%              |
| 3       | Fiebre          | 0      | 2      | 7       | 1            | 0     | 10    | 70.0%              |
| 4       | Moco+Fiebre     | 0      | 0      | 3       | 1            | 3     | 7     | 42.9%              |
| 5       | Tos+Fiebre      | 1      | 0      | 6       | 2            | 2     | 11    | 54.5%              |
| 6       | Dolor abdominal | 0      | 12     | 0       | 3            | 0     | 15    | 80.0%              |
| 7       | Tos+Presión     | 3      | 0      | 0       | 1            | 0     | 4     | 75.0%              |
| 8       | Fiebre+Presión  | 1      | 2      | 1       | 1            | 0     | 5     | 40.0%              |
| 9       | Dolor+Presión   | 2      | 1      | 0       | 1            | 0     | 4     | 50.0%              |
| 10      | Fiebre+Dolor    | 0      | 1      | 1       | 0            | 0     | 2     | 50.0%              |
| 11      | Tos+Dolor       | 0      | 1      | 0       | 0            | 1     | 2     | 50.0%              |

La pureza es `(8 + 12 + 12 + 7 + 3 + 6 + 12 + 3 + 2 + 2 + 1 + 1)/100 = 69/100`, tomando el máximo de cada fila.

### 7.5 Estabilidad ante el barajado (semilla 42, ρ = 0.65)

Entrenamiento, 10 corridas: clusters 14, 14, 15, 15, 13, 13, 14, 14, 14, 14 (media 14.0); matching ratio entre 0.9500 y 0.9600. Las primeras 5 son las que reporta el CLI con `--shuffle 5` (promedio 14.20, ratio promedio 0.9573).

Validación, 10 corridas: 12, 14, 13, 13, 13, 13, 13, 12, 13, 13 (media 12.9); matching ratio entre 0.9600 y 0.9667. Las primeras 5 (media 13.00, ratio promedio 0.9607).

Órdenes sistemáticos (ρ = 0.65), para mostrar la sensibilidad al orden en serio:

| Orden de presentación           | Entrenamiento | Validación |
| ------------------------------- | ------------- | ---------- |
| original del archivo            | 14            | 12         |
| ordenado por vector ascendente  | 15            | 14         |
| ordenado por vector descendente | 9             | 8          |
| más síntomas primero            | 9             | 8          |
| menos síntomas primero          | 15            | 14         |

Es decir, el orden puede llevar el resultado de 8 a 15. Los barajados al azar quedan cerca del orden original porque el archivo original ya está en orden aleatorio.

### 7.6 Interpretación

Por qué sube la cantidad de clusters con ρ: exigir más parecido separa a los pacientes que antes se toleraban (por ejemplo tos+moco+fiebre ya no entra a Moco+Fiebre con ρ > 0.667).

Por qué sube el matching ratio: es una consecuencia casi mecánica, porque a mayor ρ solo pasan pacientes con ratio alto. Con ρ = 0.8 el ratio es 1.0 por construcción. No es un indicador de calidad independiente: de hecho llega al máximo con 17 categorías, justo cuando el agrupamiento está más fragmentado.

Por qué la pureza sube poco: de 69% a 72-73% al pasar de 12 a 17 clusters, y baja a 56-57% con 5 a 7. La razón de fondo es que los síntomas no determinan la especialidad esperada. Hay pacientes con exactamente los mismos síntomas y distinta etiqueta. Calculado: si se agrupa por vector de síntomas exacto (el mejor agrupamiento posible usando solo los 5 síntomas), la pureza máxima es 73%. Ejemplos de la validación: vector `[0,0,0,0,0]` (sin síntomas) tiene 5 Medicina general, 2 Infectología, 1 Cardiología y 1 Gastroenterología; vector `[1,0,0,0,0]` (solo tos) tiene 3 Medicina general y 2 Neumología; vector `[0,0,1,0,0]` (solo fiebre) tiene 6 Infectología, 2 Gastroenterología y 1 Medicina general. El 69% de ART1 está a 4 puntos de ese techo del 73%, y el azar con la clase mayoritaria da 20%. Esa es la forma correcta de leer el 69%: no es "falla el 31%", es "con estos cinco síntomas el techo es 73%".

La pureza de ρ = 0.7 (73%) es mayor que la de ρ = 0.8 (72%): no es monótona. A ρ alto un paciente que es subconjunto de una plantilla anterior entra y le achica la plantilla; el orden y esas absorciones cambian un poco las etiquetas mayoritarias.

---

## 8. El visualizador y el notebook en la defensa

### 8.1 El visualizador (`visualizador/index.html`)

Es un único archivo HTML, sin servidor ni instalación: se abre con doble clic en el navegador. Implementa la misma lógica que `ART1Network` en JavaScript (el propio archivo lo aclara en un comentario: "misma lógica que ART1Network en CarGross.py"), con los 100 pacientes de entrenamiento embebidos en el código. El desempate también es por índice menor (`a.j - c.j`), el patrón vacío se manda a la categoría 0 sin cambiar la red y la vigilancia usa `>=`, igual que el módulo.

Tiene tres secciones.

1. Paso a paso. Muestra las capas F1 (síntomas del paciente) y F2 (categorías) con las plantillas, el puntaje de cada categoría y el ciclo en cinco fases: Elección, Comparación, Vigilancia, Resonancia o Reset, Aprendizaje. Un medidor muestra el ratio contra la marca de ρ. Una tabla muestra `x`, `w` (la plantilla) y `x AND w`. Hay barra de progreso, un mapa de colores de los 100 pacientes por categoría y un control de ρ (deslizador de 0 a 1, por defecto 0.65). Cada vez que se mueve ρ la corrida se reinicia.
2. Comparar valores de ρ. Tres recuadros con ρ editables (por defecto 0.3, 0.6 y 0.9), mismos pacientes y mismo orden. "Ver cómo se forman" anima la creación de categorías, "Mostrar el final" salta al resultado. Cada recuadro muestra cantidad de categorías, plantillas y los pacientes de cada una, y similitud media al final.
3. Jugá con la red. Se arma un paciente prendiendo y apagando síntomas, se elige ρ, y muestra en qué categoría entraría la red entrenada con los 100 pacientes, con la tabla de intentos (orden, categoría, puntaje, `|x AND w|`, similitud, pasa o reset). "Aprender este patrón" lo incorpora a la red, "Volver a entrenar" la reinicia. Clasificar no cambia la red; solo cambia si se aprieta "Aprender".

Atajos de teclado (en la sección paso a paso):

| Tecla                    | Acción                              |
| ------------------------ | ----------------------------------- |
| Espacio o flecha derecha | siguiente paso                      |
| Flecha izquierda         | paso atrás                          |
| Flecha abajo             | siguiente patrón                    |
| P                        | auto-play / pausa                   |
| R                        | reiniciar                           |
| + y -                    | velocidad                           |
| E (o F)                  | modo exposición (pantalla completa) |
| Esc                      | salir del modo exposición           |

Hay un botón de tema claro/oscuro y un deslizador de velocidad.

Qué mostrar, en este orden (unos 3 minutos):

1. Abrirlo con ρ = 0.65 y apretar Espacio para entrar al primer paciente (P015). Decir: "entra x, F2 todavía no tiene categorías, compite sola la categoría libre, nace la categoría 0".
2. Apretar flecha abajo dos veces (la red aprende P015 y P075) y después Espacio para recorrer a P021 paso a paso. Mostrar el reset de la categoría 0 (0.5 contra 0.65) y el nacimiento de la categoría 2. Es el mismo ejemplo de la sección 4.
3. Poner ρ en 0.3, 0.6 y 0.9 en la sección 2 y apretar "Mostrar el final": 6, 14 y 17 categorías (los 0.3 y 0.9 son los valores por defecto de los recuadros; el 0.6 da lo mismo que 0.65).
4. En la sección 3, prender Tos y Presión alta: con ρ = 0.65 entra a la categoría 12 (Tos, Presión alta) con similitud 1.00 (la categoría 2, que quedó solo con Presión alta, daría 0.5 y no pasa); con ρ = 0.9 entra a la categoría 14, también con 1.00. Prender los cinco síntomas: con ρ = 0.65 crea una categoría nueva.
5. Modo exposición con E si hay proyector.

Si el visualizador se traba o no abre: no depende de nada externo, se abre desde el explorador de archivos; y el notebook cubre lo mismo.

Una trampa: el visualizador carga una copia del dataset dentro del HTML (no lee el CSV). Si alguien cambia el CSV, el visualizador no se entera.

### 8.2 El notebook (`notebooks/01_demo_art1.ipynb`)

Es la demostración ejecutable. Estructura (las secciones del propio notebook):

| Sección                   | Qué hace                                                                                                                                                                                                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2. Marco teórico          | F1/F2, pesos, fórmulas (en markdown con LaTeX)                                                                                                                                                                                                                         |
| 3. Carga de datos         | usa `cg.load_dataset`; comprueba que CSV y XLSX dan lo mismo; muestra los helpers de normalización                                                                                                                                                                     |
| 4. Preprocesamiento       | tabla de bits, conteo de síntomas por paciente (0 síntomas: 9, 1: 36, 2: 39, 3: 15, 4: 1), 21 patrones distintos                                                                                                                                                       |
| 5. ART1 paso a paso       | reimplementa cada fórmula en funciones chicas (`art1_inicializar`, `art1_puntajes`, `art1_ganador`, `art1_coincidencia_T`, `art1_matching_ratio`, `art1_test_vigilancia`, `art1_adaptar_pesos`, `art1_entrenar_paso_a_paso`) y sigue a P021 con salida línea por línea |
| 6. Entrenamiento completo | 14 clusters, ratio 0.9500; verificación cruzada contra `ART1Network` (asignaciones, plantillas y métricas idénticas, con `assert`)                                                                                                                                     |
| 7. Resultados             | tabla de plantillas, métricas, nota sobre la capa de especialidades y el problema de la tilde                                                                                                                                                                          |
| 8. Visualizaciones        | heatmap de plantillas, distribución por cluster, evolución del ratio, barrido de ρ                                                                                                                                                                                     |
| 9. Validación externa     | pureza 69%, matriz de contingencia                                                                                                                                                                                                                                     |
| 10. Discusión             | estabilidad-plasticidad, ρ, orden, ruido, sesgos, ética; test de barajado                                                                                                                                                                                              |
| 11. Reproducir el CLI     | ejecuta el CLI con `subprocess` y compara las métricas contra las del notebook                                                                                                                                                                                         |

Cómo usarlo en la defensa: no hace falta ejecutarlo en vivo (tarda poco, pero depende de `matplotlib`). Conviene tenerlo ya ejecutado con las salidas visibles y mostrar tres cosas: la celda del rastreo de P021 (sección 5.10), la verificación cruzada con `assert` ("COINCIDENCIA TOTAL") y la matriz de contingencia de la sección 9. Si preguntan "¿cómo saben que su código está bien?", la respuesta es la verificación cruzada: dos implementaciones independientes (una fórmula por fórmula y otra el módulo) dan los mismos clusters, las mismas plantillas y las mismas métricas.

Cosas del notebook que conviene saber antes de que las vea el profesor:

- Las salidas guardadas muestran rutas de otra computadora (`C:\Users\claud\IA-red-neuronal`), de quien lo ejecutó. No afecta, pero puede preguntar.
- El notebook usa los `.xlsx` como entrada principal (`RUTA_TRAIN` apunta al `.xlsx`) y verifica que los `.csv` den lo mismo.
- Las salidas guardadas del notebook se generaron antes de arreglar la regla de Cardiología: las reglas y fichas que muestran todavía dan la especialidad vieja para los pacientes con presión alta (ver 6.8). Los clusters, ratios y pureza son los mismos.
- El notebook corre el barrido con 0.4, 0.5, 0.65 y 0.8; el informe con 0.3, 0.5, 0.65, 0.8 y 0.9. Ambos son correctos.

---

## 9. Banco de preguntas con respuestas modelo

Las respuestas están escritas para decirlas en voz alta, cortas. Si el profesor insiste, se amplía con lo de las secciones anteriores. Donde hay un número, es el que salió de correr el código.

### 9.1 Teoría

**1. ¿Qué es ART1?**
Es una red neuronal de Carpenter y Grossberg para agrupar vectores binarios sin supervisión. Va creando categorías a medida que le llegan los datos, y un parámetro, la vigilancia, controla qué tan parecido tiene que ser un dato a una categoría para entrar. Si no se parece a ninguna, crea una nueva.

**2. ¿Qué es el dilema estabilidad-plasticidad?**
Es el problema de que una red tiene que poder aprender cosas nuevas, la plasticidad, sin olvidar lo que ya aprendió, la estabilidad. ART1 lo resuelve con el test de vigilancia: un patrón solo modifica una categoría existente si se parece lo suficiente, y lo que se modifica es una intersección, que es conservadora. Si no se parece, no toca nada y se crea una categoría nueva. Así las categorías viejas se conservan y igual se aprenden cosas nuevas.

**3. ¿Por qué es no supervisado?**
Porque durante el entrenamiento nunca ve la especialidad ni ninguna etiqueta. Recibe solo los vectores de síntomas y arma los grupos por parecido. En el dataset de validación hay una columna con la especialidad esperada, pero el código la separa al cargar y recién la usa al final para calcular la pureza.

**4. ¿Por qué solo trabaja con entradas binarias? ¿Y si tengo datos continuos?**
Porque toda la matemática se apoya en que cada entrada es 0 o 1: la norma es contar unos, la coincidencia es un AND, la plantilla es una intersección. Para datos continuos existe ART2, que no implementamos. Con ART1 habría que discretizar, por ejemplo fiebre sí o no según un umbral, y se pierde información.

**5. ¿Qué son las capas F1 y F2?**
F1 es la capa de comparación o entrada: tiene un nodo por síntoma y recibe el vector. F2 es la capa de reconocimiento: tiene un nodo por categoría, hasta 50 en nuestro caso, y sus nodos compiten entre sí por inhibición lateral, gana el de mayor puntaje. En el paper de Lau son los nodos de entrada y de salida.

**6. ¿Qué diferencia hay entre los pesos bottom-up y top-down?**
Los bottom-up, `b`, van de F1 a F2 y sirven para elegir qué categoría compite mejor con un patrón: con ellos se calculan los puntajes. Los top-down, `t`, van de F2 a F1 y guardan la plantilla de cada categoría: con ellos se hace la comparación AND y el test de vigilancia. Son dos matrices, `b` de N por M y `t` de M por N.

**7. ¿Qué es la vigilancia?**
Es un número entre 0 y 1 que dice qué fracción de los síntomas del paciente tiene que estar en la plantilla para aceptarlo. Con 0.65, al menos el 65 por ciento de sus síntomas tienen que estar en la plantilla. Es la perilla que controla cuántos clusters se forman.

**8. ¿Qué es resonancia y qué es reset?**
Resonancia es cuando la categoría ganadora pasa el test de vigilancia: el patrón y la plantilla concuerdan, la red se estabiliza y se aprende. Reset es cuando no pasa: esa categoría se descarta solo para ese paciente y se prueba con la siguiente mejor. No es un borrado permanente, en el paciente siguiente vuelven a estar todas disponibles.

**9. ¿Qué pasa con ρ alto y con ρ bajo?**
Con ρ alto la red es exigente: hace más categorías, más chicas y más puras. Con ρ bajo tolera diferencias: pocas categorías, grandes y mezcladas. En nuestro caso, entrenamiento: ρ = 0.3 da 6 categorías, 0.65 da 14 y 0.8 da 17.

**10. ¿Qué es el aprendizaje rápido (fast learning)?**
Que la plantilla pasa a su valor final en un solo paso: en cuanto hay resonancia, `t` se reemplaza directamente por el AND con el patrón, sin una tasa de aprendizaje gradual. El paper de Lau aclara que el algoritmo del Box 3 asume fast learning y valores binarios.

**11. ¿Qué diferencia hay con k-means?**
K-means necesita saber la cantidad de grupos de antemano, trabaja con valores continuos, usa el promedio como centro y suele iterar sobre todos los datos hasta converger. ART1 no necesita k, trabaja con binarios, la plantilla es una intersección y aprende en una sola pasada, de a un dato. A cambio, ART1 depende del orden de los datos.

**12. ¿Y con un mapa de Kohonen, SOM?**
Según el paper, el SOM es el equivalente neuronal de k-means, para entradas continuas, y mantiene un orden topológico en el mapa. ART1 es el equivalente del algoritmo leader, para binarios, y crea categorías nuevas cuando hace falta en lugar de tener un número fijo de neuronas.

**13. ¿A qué algoritmo clásico se parece ART1?**
Al algoritmo leader de clustering secuencial, que menciona el paper (Hartigan). El primer dato es el ejemplar del primer cluster, cada dato siguiente se agrupa con un cluster si está dentro de un umbral y, si no, funda uno nuevo. ART1 hace eso, con la diferencia de que el ejemplar se va afinando con el AND.

**14. ¿Qué diferencia hay con Hopfield y Hamming, que también son binarias?**
En la taxonomía del paper, Hopfield y Hamming son supervisadas: necesitan ejemplares conocidos de las clases y clasifican un patrón entre esos. ART1 forma sus propios ejemplares sin supervisión y su cantidad crece con los datos.

**15. ¿Qué ventajas tiene ART1?**
Aprende de forma incremental sin reentrenar todo, no necesita fijar el número de clusters, no olvida las categorías viejas, y los resultados son interpretables porque cada plantilla es una combinación de síntomas que se puede leer.

**16. ¿Y qué desventajas?**
Depende del orden de presentación, solo acepta binarios, una plantilla solo puede perder bits, es sensible al ruido (el paper lo marca) y el valor de vigilancia se elige a mano.

**17. ¿Qué aplicaciones tiene?**
El paper lo ubica en clasificación y clustering de patrones, como reconocimiento de imágenes y de voz, y en cuantización vectorial. En nuestro caso, agrupar perfiles de síntomas. Con la salvedad de que la versión binaria es sensible al ruido.

### 9.2 Algoritmo

**18. ¿Cómo elige la red la categoría ganadora?**
Calcula para cada categoría activa un puntaje, la suma de los pesos `b` de los síntomas que tiene el paciente, y gana la de mayor puntaje. Todavía no mira si se parece lo suficiente: eso viene después con la vigilancia.

**19. El puntaje y el matching ratio, ¿no son lo mismo?**
No, y es una pregunta típica. El puntaje `μ` sirve para ordenar a los candidatos (quién se prueba primero) y se calcula con `b`. El matching ratio mide qué fracción del paciente está en la plantilla de ese candidato y se calcula con `t`. Se decide con el ratio, no con el puntaje. Un ejemplo del dataset: P021 contra la categoría 0 da puntaje 0.667, el más alto, pero ratio 0.5, y no pasa.

**20. ¿Qué es `T` y por qué un AND?**
`T = t AND x` es lo que tienen en común la plantilla y el paciente. Con ceros y unos, el AND es el producto. La norma de `T` cuenta cuántos síntomas del paciente están en la plantilla. Y como después la plantilla pasa a ser `T`, el AND es también la regla de aprendizaje: la plantilla es la intersección de todos los pacientes del cluster.

**21. ¿Qué es el matching ratio?**
`‖T‖/‖x‖`: de los síntomas que tiene el paciente, qué fracción está en la plantilla. Si el paciente tiene 2 síntomas y la plantilla contiene 1, es 0.5. Es la cantidad que se compara con ρ.

**22. ¿Por qué `b = t / (0.5 + ‖t‖)`?**
Es la regla de Lippmann que sigue el paper, el paso 7 del Box 3. Lo que hace, verificado con números: reparte un peso total menor que 1 entre los bits de la plantilla, y como el denominador crece con la cantidad de unos, las plantillas más chicas tienen pesos más grandes. Eso hace que, ante un paciente que coincide entero con dos plantillas, gane la más específica: `[1,0,0,0,0]` da 0.667 y `[1,1,0,0,0]` da 0.4, y sin el 0.5 empatarían. Además evita dividir por cero con una plantilla vacía.

**23. ¿Cómo se crea una categoría nueva?**
Cuando todas las comprometidas fallan el test de vigilancia, la mejor candidata que queda es una categoría libre, con `t` todo en 1. Su `T` es igual al paciente y el ratio da 1, que siempre pasa. La adaptación la deja igual al paciente. No hay una instrucción explícita de crear categoría: nace sola.

**24. ¿Una plantilla puede ganar síntomas?**
No. Solo pierde: `t_new = t AND x` nunca tiene más unos que `t`. Si un paciente con un síntoma de más entra a la categoría, el síntoma de más no entra a la plantilla. Lo que sí puede pasar es que se cree una categoría nueva con ese síntoma.

**25. ¿Qué pasa con un paciente sin síntomas?**
Tiene `‖x‖ = 0` y el ratio sería 0/0. El código lo resuelve por convención: lo manda a la categoría 0 con ratio 1.0 y no toca ningún peso. Es una decisión nuestra, no de ART1. En nuestros datasets hay 9 en cada uno.

**26. Una vez asignado, ¿un paciente puede cambiar de cluster?**
No. ART1 hace una sola pasada y la asignación queda fija, aunque después la plantilla cambie. Por eso el ratio de cada paciente es contra la plantilla del momento, y la plantilla final de un cluster puede ser más chica que la que tenía cuando entraron los primeros pacientes.

**27. ¿Cuántas categorías puede tener?**
Hasta M, que es 50 por defecto y se cambia con `--max_cat`. Si se agotan, el código levanta un `RuntimeError`, que `main` muestra como `[ERROR]`. Con nuestros datos el máximo que usa es 17.

### 9.3 Código

**28. ¿Por qué `scores.index(max_score)` desempata así?**
Porque `list.index` devuelve la primera posición donde aparece el valor, entonces en un empate gana el índice menor. Tiene dos efectos útiles. Uno, es determinista: dos corridas con el mismo orden dan lo mismo, que es lo que permite verificar el módulo contra el notebook y contra el visualizador. Dos, como las categorías libres puntúan todas igual y están al final de la lista, siempre se usa la primera libre, y por eso las categorías se numeran en orden de creación, sin huecos. El paper no dice cómo desempatar, es una decisión de implementación.

**29. ¿Qué pasa con `>=` frente a `>` en el test de vigilancia?**
Lau (Box 3) y el código usan `>=`: un ratio igual al umbral pasa. El operador importa en este dataset porque los ratios son fracciones exactas como 1/2 o 2/3. Si alguien lo cambiara a `>` (prueba hecha con el módulo), con ρ = 0.5 saldrían 14 categorías en vez de 8, y con ρ = 2/3 saldrían 17 en vez de 14. Además, con `>` el valor ρ = 1 no serviría nunca: ningún ratio puede ser mayor que 1, así que nada resonaría y se crearía una categoría por paciente hasta agotar los nodos. Con `>=`, ρ = 1 significa "coincidencia total".

**30. ¿Por qué las categorías apagadas tienen puntaje -1?**
Para sacarlas de la competencia sin cambiar el tamaño de la lista. Como los puntajes reales son siempre mayores o iguales a 0, el -1 nunca gana el `max`. Y si el máximo es negativo es que no queda ninguna disponible: ahí se levanta el error de capacidad.

**31. ¿Por qué se recrea `active_mask` en cada patrón?**
Porque el reset es temporal, solo vale para el paciente actual. En el paper es el "rehabilitar los nodos deshabilitados" del paso 8. Si no se reiniciara, una categoría descartada para un paciente quedaría descartada para siempre.

**32. ¿Por qué listas de Python y no NumPy?**
Porque la consigna pide que sea sencillo de instalar y funcione sin errores. Con 5 entradas y 50 categorías la velocidad no importa, y sin NumPy los CSV se procesan solo con la biblioteca estándar. El costo es que no escala a miles de dimensiones.

**33. ¿Qué pasa si ρ está fuera de rango, por ejemplo 1.5?**
El constructor de `ART1Network` levanta `ValueError`, y `main` lo atrapa y muestra `[ERROR]` con el motivo (ρ tiene que estar entre 0.0 y 1.0) y código de salida 1. Probado.

**34. ¿Qué pasa si ρ = 0? ¿Y si ρ = 1?**
Con ρ = 0 todo pasa la vigilancia, pero no sale un cluster único: salen 5, uno por síntoma, porque los puntajes ordenan los candidatos primero y una categoría sin ningún síntoma en común tiene puntaje 0 mientras que una libre tiene puntaje positivo y gana. Con ρ = 1 solo entra un paciente cuyos síntomas estén todos en la plantilla: salen 17 categorías.

**35. ¿Cómo maneja errores el programa?**
`main` atrapa `FileNotFoundError`, `FileExistsError`, `ValueError`, `ImportError`, `OSError` y `RuntimeError` y muestra `[ERROR] motivo` más una pista para usar `--man` o `--help`, con código de salida 1. Cubre archivo inexistente, valores no binarios (indica fila y columna), ρ fuera de rango, falta de archivo de entrada, encoding inválido, capacidad agotada (`--max_cat` chico) y un archivo con encabezado pero sin filas.

**36. ¿Cómo lee un `.xlsx` sin librerías?**
Un `.xlsx` es un ZIP con XML adentro. Si `openpyxl` está instalado se usa. Si no, el lector propio abre el ZIP con `zipfile`, lee `sharedStrings.xml`, ubica la primera hoja a través de `workbook.xml` y sus relaciones, y recorre las filas con `xml.etree`, rellenando con vacío las celdas que faltan. Los `.xlsx` de este TFI dan exactamente los mismos vectores que los `.csv`.

**37. ¿Qué hace `--shuffle`?**
Hace N corridas nuevas, cada una con una red desde cero y los pacientes en orden barajado con una semilla fija (`--seed`, 42 por defecto), e imprime cuántos clusters y qué matching ratio da cada una, más el promedio contra la corrida original. Sirve para medir la sensibilidad al orden. Con `--shuffle 5 --seed 42` en entrenamiento: 14, 14, 15, 15, 13 clusters.

**38. ¿Cómo prueban que el código funciona?**
Tres capas. El autotest `--test` corre cinco chequeos sobre un dataset mínimo (cantidad de categorías con ρ alto y bajo, patrón vacío, AND de la plantilla, regla de Cardiología) e imprime TEST PASSED solo si pasan todos; no verifica los resultados de los datasets reales. El notebook reimplementa el algoritmo fórmula por fórmula y compara con el módulo real con un `assert`: mismas asignaciones, mismas plantillas, mismas métricas en los 100 pacientes. Y el visualizador tiene la misma lógica en JavaScript y da los mismos números (6, 14 y 17 categorías con 0.3, 0.6 y 0.9). El notebook además ejecuta el CLI y compara los números.

**39. ¿Qué hace `--patient-id`?**
Entrena la red con el dataset completo y después imprime la ficha de un paciente: busca por ID sin distinguir mayúsculas, o por número de orden si es un número. Si no lo encuentra, avisa y muestra el primero. La ficha muestra el cluster, el porcentaje de coincidencia, la plantilla final de ese cluster y la especialidad que sugieren las reglas para sus síntomas. No escribe el CSV general.

**40. En el modo interactivo, ¿la red aprende al paciente nuevo?**
Sí: usa `train_pattern`, que clasifica y a la vez adapta los pesos. El modo interactivo primero entrena con el dataset y después pasa al paciente por la red entrenada. Si la red lo mete en una categoría, la plantilla puede achicarse; si no, se crea una nueva. Para solo clasificar sin aprender, el visualizador tiene esa opción en "Jugá con la red".

**41. El ratio del CSV de un paciente, ¿se recalcula contra la plantilla final?**
No. Es el ratio del momento en que se asignó. Si se recalculara contra la plantilla final, podría dar distinto (por ejemplo para P021, la plantilla final de su cluster es solo Presión alta). Se guarda el del momento porque es el que justificó la asignación.

**42. ¿Para qué sirve `_normalize_numeric_str`?**
Para que `1.0` (como lo guarda Excel) y `1` den lo mismo, así un `.xlsx` y un `.csv` con los mismos datos producen los mismos vectores. No toca IDs como `007`.

### 9.4 Datos

**43. ¿De dónde salen los datos?**
Son sintéticos: se generaron para el trabajo. El dataset de validación tiene 20 pacientes por cada una de las 5 especialidades esperadas, y los dos tienen 100 pacientes y 5 síntomas, que cumple con la consigna de al menos 50 datos y 5 variables. No son pacientes reales, y por eso el 69 por ciento de pureza no se puede extrapolar a un contexto clínico.

**44. ¿La validación es de verdad validación?**
Hay que contestar con precisión. No es una validación en el sentido supervisado de separar entrenamiento y test: ART1 no usa las etiquetas para aprender, entonces no hay "generalización a datos no vistos" que medir. Lo que hacemos es una validación externa del agrupamiento: corremos la red sobre otro conjunto de pacientes (los dos datasets son disjuntos, 0 IDs en común) y comparamos los clusters con una etiqueta que la red no vio, usando la pureza. Es válido para responder "¿los grupos que arma tienen sentido clínico?", no para responder "¿qué tan bien predice la especialidad de un paciente nuevo?". Además, el dataset de validación se entrena de cero, no se usa la red del entrenamiento para clasificarlo.

**45. ¿Por qué hay pacientes sin síntomas y cómo los trata?**
Son datos del dataset (9 en cada uno), representan a personas sin ninguno de los cinco síntomas. ART1 no puede compararlos con nada, porque su norma es cero. El código los manda a la categoría 0 con ratio 1.0 y no modifica la red. Eso infla un poco el matching ratio promedio (sin esos 9, el del entrenamiento baja de 0.9500 a 0.9451) y los mete en el cluster 0 para la pureza.

**46. Si es no supervisado, ¿cómo saben que agrupa bien?**
Con tres criterios. Uno, interno: el matching ratio, que mide qué parte de cada paciente explica su plantilla (0.95 y 0.96). Dos, externo: la pureza contra la especialidad esperada en validación (69%), con la aclaración de que el techo con estos síntomas es 73%. Tres, de interpretación: las plantillas se leen como combinaciones de síntomas coherentes, por ejemplo Moco más Fiebre. Ninguno de los tres es una prueba definitiva, y lo decimos.

**47. ¿Qué es la pureza y por qué sube con más clusters?**
Es la fracción de pacientes que coincide con la etiqueta mayoritaria de su cluster. Con más clusters, cada uno es más chico y más fácil de que sea homogéneo, hasta llegar a 100% con un cluster por paciente, que no sería una buena agrupación. Por eso se interpreta junto con la cantidad de clusters. Y es una métrica externa: depende de que la etiqueta sea una buena referencia.

**48. ¿Por qué esos 5 síntomas?**
La consigna pide al menos 5 variables. Son síntomas que se asocian a especialidades distintas (respiratorios, infecciosos, digestivo, cardiovascular), lo que permite interpretar las plantillas. Con tan pocas variables hay solo 32 vectores posibles, y de hecho aparecen 21 distintos en entrenamiento.

### 9.5 Resultados

**49. ¿Por qué 14 clusters en entrenamiento y 12 en validación?**
Porque los datasets son distintos: en validación hay 20 vectores distintos contra 21, con otras frecuencias, y además el orden de llegada de cada archivo cambia qué plantillas se forman. Con ρ = 0.65 los dos quedan en el mismo escalón de vigilancia, pero no tienen por qué dar el mismo número de categorías.

**50. ¿Por qué ρ = 0.65?**
Es un compromiso, no un óptimo demostrado. Con ρ por debajo de 0.5 las plantillas mezclan demasiado (pureza 56-57%), y con ρ de 0.7 para arriba se llega a 17 categorías, varias muy chicas, y la pureza solo sube a 72-73%. 0.65 da categorías interpretables. Da lo mismo que 0.6.

**51. ¿Por qué ρ = 0.4 y ρ = 0.5 dan el mismo resultado?**
Porque el ratio está discretizado: con hasta 4 síntomas, los valores posibles son 1/4, 1/3, 1/2, 2/3, 3/4 y 1. Entre 0.4 y 0.5 no hay ningún ratio posible nuevo, entonces la decisión de cada paciente es idéntica. El primer cambio llega cuando ρ supera 0.5 y el ratio 1/2 deja de pasar (0.6 da 14 categorías en lugar de 8).

**52. ¿Por qué el matching ratio llega a 1 con ρ alto? ¿Es bueno?**
Con ρ ≥ 0.8 solo pasa un ratio de 1, entonces el promedio es 1 por construcción. No es un buen indicador de calidad por sí solo: llega a 1 justo cuando el agrupamiento es más fragmentado (17 clusters). Hay que leerlo junto con la cantidad de clusters.

**53. ¿69% de pureza es bueno o malo?**
Hay tres referencias. El azar con la clase más frecuente daría 20%. El techo, agrupando por el vector de síntomas exacto (el mejor agrupamiento posible con estos cinco síntomas), da 73%, porque hay pacientes con los mismos síntomas y distinta especialidad esperada. Y ART1 con ρ = 0.65 da 69%. Entonces está a 4 puntos del techo. Las confusiones son las esperables: la etiqueta "Medicina general" no describe un síntoma sino un nivel de atención, y las combinaciones como tos o moco con fiebre se reparten entre Infectología y Neumología.

**54. ¿Por qué la pureza a ρ = 0.7 es 73% y a ρ = 0.8 es 72%? ¿No debería subir siempre?**
No tiene por qué ser monótona. La diferencia es de un solo paciente. Con ρ alto un paciente que es subconjunto de una plantilla previa sigue entrando con ratio 1 y le achica la plantilla, y el orden de llegada define en qué cluster cae cada combinación; son efectos del orden más que de la vigilancia.

**55. ¿Qué muestra el test de barajado?**
Que la cantidad de clusters depende del orden de los datos pero la calidad casi no. En 10 barajados con semilla 42 hay entre 13 y 15 clusters en entrenamiento (media 14.0) y entre 12 y 14 en validación (media 12.9), con matching ratio entre 0.95 y 0.97. Es decir que cambia la granularidad, no la calidad. Pero con órdenes sistemáticos el efecto es mayor: con los pacientes más cargados de síntomas primero salen 9 clusters, y con los menos cargados primero, 15.

**56. En validación, el cluster 1 tiene plantilla "Moco" pero 12 de 12 pacientes son de Neumología. ¿No es contradictorio?**
Es un efecto del AND. Ese cluster tiene 8 pacientes con tos y moco, 3 con tos, moco y fiebre y uno con solo moco. La plantilla se fue achicando a medida que entraban: cuando entró el de solo moco, la intersección eliminó la tos y quedó Moco. Por eso la plantilla final no cuenta toda la historia del cluster, y la regla de especialidad aplicada a la plantilla sugeriría Infectología cuando los pacientes eran de Neumología. Es una limitación de usar la plantilla final como nombre del cluster.

### 9.6 Decisiones de diseño

**57. ¿Por qué separaron la especialidad de la red?**
Porque ART1 solo agrupa, y la traducción a especialidad es una regla clínica aparte, una cadena de `if` sobre los síntomas. Mantenerlas separadas deja claro qué es aprendido y qué es una regla escrita a mano, y se puede cambiar una sin tocar la otra. Y la pureza se calcula contra las etiquetas del dataset, no contra esa regla.

**58. ¿Por qué hay una regla de Cardiología que nunca se activa?**
Es un error nuestro que hay que reconocer: la regla busca `presion alta` sin tilde y la columna se llama `Presión alta` con tilde, y al pasar a minúsculas no coincide, entonces la presión alta sola cae en Clínica Médica General. No afecta a ART1 ni a la pureza (que usa las etiquetas del dataset), pero sí a la columna de especialidad sugerida del CSV de salida y de las fichas. Es una corrección simple: normalizar tildes antes de comparar.

**59. ¿Por qué aceptan CSV y XLSX?**
La consigna pide entrada en CSV, y el XLSX se agregó como comodidad porque hay planillas de datos en ese formato. Los dos caminos dan exactamente los mismos vectores. El lector de XLSX sin dependencias se hizo para poder leerlo sin instalar nada.

**60. ¿Por qué hay una versión del algoritmo en el notebook y otra en el módulo?**
La del notebook es pedagógica: una función por fórmula, para poder seguir el ciclo paso a paso y mostrar la traza de P021. La del módulo es la de producción, que usa el CLI. El notebook las compara con un `assert` para demostrar que son lo mismo, que es una forma de verificación cruzada.

**61. ¿Por qué un visualizador, si ya hay notebook?**
Porque el notebook muestra números y el visualizador muestra el proceso: se ve la elección, la comparación, el reset y el nacimiento de una categoría, y se puede mover ρ en vivo. Es un archivo HTML que se abre sin instalar nada. Es complementario, no reemplaza al módulo.

### 9.7 Limitaciones y mejoras

**62. ¿Qué mejorarían?**
En orden de impacto: uno, pensar si la especialidad sugerida debería salir de la plantilla o del paciente, y si el orden de prioridad de las reglas es el clínicamente correcto. Dos, para la sensibilidad al orden: correr varios órdenes barajados y quedarse con el más representativo, o hacer varias pasadas. Tres, probar ρ adaptativo o aprendizaje más lento, como sugiere el paper para tolerar ruido. Cuatro, validar con datos clínicos reales y con más variables.

**63. ¿Se puede usar esto para diagnosticar?**
No. Agrupa perfiles de síntomas y sugiere una derivación. Los datos son sintéticos, los 5 síntomas son pocos, y en medicina un falso negativo pesa mucho más que un falso positivo, y ρ debería elegirse con criterio clínico. Cualquier uso real necesitaría datos reales, consentimiento, anonimización y un profesional validando.

**64. Si el profesor pide probar un paciente en el momento, ¿cómo lo hacen?**
Con el modo interactivo del CLI (pide los 0 y 1 por consola) o con la sección "Jugá con la red" del visualizador, que además permite clasificar sin que la red aprenda. Si quiere cambiar un dato del CSV, se edita y se vuelve a correr: el entrenamiento es una pasada y tarda milisegundos.

**65. ¿Qué pasaría con muchos más síntomas?**
Funcionaría, pero con listas de Python sería más lento (el cálculo de puntajes cuesta N por M por patrón) y habría que pasar a NumPy. Y con datos ruidosos es de esperar más categorías con vigilancia alta, que es lo que advierte el paper.

---

## 10. Ejercicios para hacer a mano

Todos verificados con el módulo (`ART1Network`, M chico). Orden de bits: `[Tos, Moco, Fiebre, Dolor abdominal, Presión alta]`. Hacerlos tapando la solución.

### Ejercicio 1: pesos iniciales y primer paciente

N = 5. ¿Cuánto vale cada `b` inicial? Entra el paciente `[1,1,0,0,0]` a una red vacía con ρ = 0.65. ¿Qué categoría gana, con qué puntaje y qué ratio, y cómo quedan `t` y `b` de esa categoría?

Solución. `b = 1/(1+5) = 1/6 = 0.1667`. `‖x‖ = 2`. Todas las categorías son libres: puntaje `2 · (1/6) = 0.3333`; gana la categoría 0 (empate, índice menor). `T = [1,1,1,1,1] AND [1,1,0,0,0] = [1,1,0,0,0]`, `‖T‖ = 2`, ratio `2/2 = 1`, pasa. Plantilla `t[0] = [1,1,0,0,0]`. Denominador `0.5 + 2 = 2.5`, así que `b[:,0] = [0.4, 0.4, 0, 0, 0]`.

### Ejercicio 2: el efecto del orden (dos pacientes)

Pacientes A = `[1,1,0,0,0]` y B = `[1,0,0,0,0]`, ρ = 0.8. Calcular cuántos clusters salen con el orden A, B y con el orden B, A.

Solución. Orden A, B: A crea la categoría 0 con plantilla `[1,1,0,0,0]` y `b = [0.4, 0.4, 0, 0, 0]`. B: puntaje de la categoría 0 `0.4`, de una libre `1/6 = 0.1667`. Gana la 0. `T = [1,0,0,0,0]`, `‖T‖ = 1`, `‖B‖ = 1`, ratio 1 ≥ 0.8, pasa. La plantilla se achica a `[1,0,0,0,0]`. Resultado: 1 cluster.

Orden B, A: B crea la categoría 0 con `[1,0,0,0,0]` y `b = 1/1.5 = 0.6667`. A: puntaje de la 0 `0.6667`, libre `2/6 = 0.3333`. Gana la 0. `T = [1,0,0,0,0]`, `‖T‖ = 1`, `‖A‖ = 2`, ratio 0.5 < 0.8, reset. Gana una libre, ratio 1. Resultado: 2 clusters, con plantillas `[1,0,0,0,0]` y `[1,1,0,0,0]`.

Con ρ = 0.5 y orden B, A: ratio 0.5 ≥ 0.5 pasa (por el `>=`), 1 cluster. Es un ejemplo de la sensibilidad al orden y de por qué importa `>=`.

### Ejercicio 3: vigilancia y reset con cuatro pacientes

Pacientes en orden: `[1,1,0,0,0]`, `[1,1,1,0,0]`, `[0,0,1,1,0]`, `[1,0,0,0,0]`. Calcular las categorías y las plantillas finales con ρ = 0.5 y con ρ = 0.7.

Solución con ρ = 0.5. El primero crea la categoría 0, `[1,1,0,0,0]`. El segundo: puntaje de la 0 `0.4 + 0.4 = 0.8`, libre `3/6 = 0.5`; gana la 0; `T = [1,1,0,0,0]`, ratio 2/3 = 0.667 ≥ 0.5, pasa; la plantilla no cambia. El tercero: puntaje de la 0 es 0, libre `2/6`; gana la libre (categoría 1), `[0,0,1,1,0]`. El cuarto: puntaje de la 0 `0.4`, de la 1 `0`, libre `0.1667`; gana la 0; `T = [1,0,0,0,0]`, ratio 1; la plantilla se achica a `[1,0,0,0,0]`. Resultado: 2 clusters, plantillas `[1,0,0,0,0]` y `[0,0,1,1,0]`; asignaciones 0, 0, 1, 0; ratios 1, 0.667, 1, 1.

Solución con ρ = 0.7. El primero igual. El segundo: ratio 0.667 < 0.7, reset de la 0; gana la libre (categoría 1), ratio 1, plantilla `[1,1,1,0,0]`. El tercero: la 0 puntúa 0; la 1 puntúa `1/3.5 = 0.2857` (el tercer bit, fiebre, es el único en común); la libre `2/6 = 0.3333`. Gana la libre (categoría 2), ratio 1, `[0,0,1,1,0]`. El cuarto: la 0 puntúa 0.4, la 1 puntúa 0.2857, la libre 0.1667; gana la 0; ratio 1; la plantilla de la 0 se achica a `[1,0,0,0,0]`. Resultado: 3 clusters, plantillas `[1,0,0,0,0]`, `[1,1,1,0,0]`, `[0,0,1,1,0]`; asignaciones 0, 1, 2, 0.

### Ejercicio 4: pureza

Una corrida deja estos clusters con estas etiquetas esperadas: cluster 0: Neumología, Neumología, Neumología, Medicina general; cluster 1: Cardiología, Cardiología, Infectología; cluster 2: Gastroenterología, Infectología, Medicina general. Calcular la pureza.

Solución. Cluster 0: la más frecuente es Neumología con 3. Cluster 1: Cardiología con 2. Cluster 2: todas empatan con 1 (el máximo vale 1, no importa cuál se elija). Suma de máximos `3 + 2 + 1 = 6`, total de pacientes 10, pureza `6/10 = 60%`. El código usa `max(counts.values())`, que no depende de cuál de las empatadas es la elegida.

### Ejercicio 5: P021 con otro ρ

Con el estado de la sección 4 (categoría 0 `[1,0,0,0,0]`, categoría 1 `[0,0,1,0,0]`), ¿qué pasa con P021 `[1,0,0,0,1]` si ρ = 0.5? ¿Y si ρ = 0.4?

Solución. El puntaje de la categoría 0 es 0.6667 y gana en el primer ciclo. `T = [1,0,0,0,0]`, ratio 0.5. Con ρ = 0.5, `0.5 ≥ 0.5` pasa: P021 entra a la categoría 0 con ratio 0.5 y la plantilla se queda `[1,0,0,0,0]` (la presión alta no se aprende). Con ρ = 0.4 también pasa (0.5 ≥ 0.4), mismo resultado. En ambos casos no se crea la categoría para tos con presión alta. Probado con el módulo para ρ = 0.5: devuelve `(0, 0.5)`. Comparar con ρ = 0.65, donde hay reset y nace la categoría 2.

---

## 11. Glosario, trampas y puntos débiles

### 11.1 Glosario

| Término                             | Qué es                                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| ART                                 | Adaptive Resonance Theory (teoría de la resonancia adaptativa), de Carpenter y Grossberg. ART1 es la versión para entradas binarias. |
| Categoría, cluster, ejemplar        | Un grupo de pacientes representado por una plantilla. Lau lo llama "exemplar".                                                       |
| Plantilla, prototipo                | El vector `t` de una categoría: la intersección (AND) de todos los pacientes que entraron.                                           |
| F1                                  | Capa de comparación (entrada). Un nodo por síntoma.                                                                                  |
| F2                                  | Capa de reconocimiento (categorías). Sus nodos compiten por inhibición lateral.                                                      |
| Bottom-up (`b`)                     | Pesos de F1 a F2. Calculan el puntaje.                                                                                               |
| Top-down (`t`)                      | Pesos de F2 a F1. Son la plantilla.                                                                                                  |
| Puntaje de coincidencia (`μ_j`)     | `Σ b_ij x_i`. Ordena los candidatos.                                                                                                 |
| Norma `‖x‖`                         | Cantidad de unos del vector.                                                                                                         |
| `T`                                 | `t AND x`: lo que comparten plantilla y paciente.                                                                                    |
| Matching ratio                      | `‖T‖/‖x‖`. Qué fracción del paciente está en la plantilla.                                                                           |
| Vigilancia (ρ)                      | Umbral de parecido mínimo, entre 0 y 1.                                                                                              |
| Resonancia                          | La ganadora pasa la vigilancia: se acepta y se aprende.                                                                              |
| Reset                               | La ganadora no pasa: se descarta para ese paciente.                                                                                  |
| Categoría comprometida, libre       | Comprometida: ya usada. Libre: todavía tiene `t` todo en 1.                                                                          |
| Aprendizaje rápido                  | La plantilla salta de un paso al AND, sin tasa gradual.                                                                              |
| Inhibición lateral, winner-take-all | Mecanismo por el cual solo el nodo con más puntaje queda activo.                                                                     |
| Estabilidad-plasticidad             | Aprender lo nuevo sin olvidar lo viejo.                                                                                              |
| Pureza                              | Fracción de pacientes que tienen la etiqueta mayoritaria de su cluster.                                                              |
| Validación externa                  | Evaluar el agrupamiento contra una etiqueta que el algoritmo no usó.                                                                 |
| Algoritmo leader                    | Clustering secuencial al que se parece ART1 (Hartigan, citado por Lau).                                                              |
| Patrón vacío                        | Paciente sin ningún síntoma (`‖x‖ = 0`).                                                                                             |

### 11.2 Trampas: cosas que no hay que decir o que se confunden

- No decir "la red diagnostica". Agrupa y sugiere una derivación orientativa que valida un profesional. La propia ficha lo aclara.
- No decir "entrenamos con el dataset de entrenamiento y validamos con el de validación" como si fuera un esquema train/test supervisado. ART1 no usa etiquetas. Decir "corrimos la red sobre dos datasets y en el segundo comparamos contra las etiquetas con la pureza".
- No decir que hay "épocas" de entrenamiento ni "convergencia". Es una pasada, y no converge a una partición única.
- No decir "ρ alto es más estable" o "ρ alto es más plástico" sin aclarar qué se quiere decir. Decir lo que sí está en el paper: ρ cerca de 1 exige coincidencia cercana, ρ bajo acepta coincidencia pobre.
- No decir que ρ = 0 da un solo cluster. Da 5 en nuestro dataset (ver pregunta 34).
- Ojo, el operador del test de vigilancia importa por las igualdades: los ratios son fracciones exactas y con `>` en lugar de `>=` cambia el número de categorías (ver 3.8).
- No decir que el matching ratio mide calidad por sí solo: llega a 1 con 17 clusters.
- No confundir puntaje `μ` (elige candidato, usa `b`) con matching ratio (decide, usa `t`).
- No confundir la plantilla con "el síntoma más común": es la intersección de todos los miembros, nunca una moda ni un promedio.
- No confundir `b` y `t`: son dos matrices con dimensiones transpuestas (`b[i][j]`, `t[j][i]`).
- No decir "Estabilidad 1.64" como si midiera estabilidad temporal. Es el promedio de bits en 1 por plantilla.
- No decir que la especialidad la calcula ART1. La calcula una regla escrita a mano, aparte.
- No decir que 69% es "mala" ni "buena" sin el techo de 73% y el azar de 20%.
- No decir que el visualizador lee el CSV: tiene los 100 pacientes pegados dentro.
- No decir que "el autotest verifica que el resultado es correcto". Verifica unos pocos casos chicos, no los datasets reales.
- No atribuir a Lau la frase de que ρ alto favorece la plasticidad: está en las notas agregadas de `lau_contenido.md`, no en las páginas del paper.
- Ojo con la materia: la consigna oficial dice "Redes Neuronales" (Museo de Redes Neuronales), el informe y el repositorio dicen "Inteligencia Artificial - UADER". Si el profesor nombra la materia, seguirle el nombre a él.

### 11.3 Puntos débiles del proyecto y cómo responder

Estos son los flancos reales. Es mejor decirlos nosotros antes de que los encuentre el profesor, con la respuesta preparada.

**Pureza de 69%.**
Qué puede decir el profesor: "el 31% está mal agrupado". Cómo responder: la pureza se compara contra una etiqueta que no es función de los síntomas. Con exactamente los mismos síntomas hay pacientes con distinta especialidad esperada, y agrupar por vector exacto da un techo de 73%. Además "Medicina general" no es un síntoma sino un nivel de atención. Entonces 69% está a 4 puntos del techo y casi 50 puntos arriba del azar. Mostrar la matriz de contingencia del notebook: hay clusters con 100% (cluster 1, 12 de 12) y los que fallan son los que mezclan dominios.

**La validación no es supervisada.**
Qué puede decir: "esto no es una validación". Cómo responder: correcto, no es train/test, porque ART1 no usa etiquetas ni predice. Es una validación externa del agrupamiento: comparar los clusters contra una referencia independiente. Lo que mide es la coherencia clínica de los grupos, no la capacidad predictiva. Si se quisiera predecir especialidad habría que usar un clasificador supervisado y separar entrenamiento y test.

**Sensibilidad al orden.**
Qué puede decir: "entonces el resultado es arbitrario". Cómo responder: es una propiedad conocida del algoritmo, de una sola pasada, y la medimos: entre 13 y 15 clusters al barajar en entrenamiento, con matching ratio casi constante (0.95 a 0.97). Pero hay que admitir que con órdenes sistemáticos puede variar de 8 a 15. Mitigaciones posibles: fijar el orden, repetir con varios órdenes y elegir, o usar otro algoritmo si se necesita una partición única.

**Datos sintéticos y pocos síntomas.**
Qué puede decir: "no demuestra nada clínico". Cómo responder: de acuerdo, el objetivo del TFI es demostrar el algoritmo, no hacer investigación médica; con 5 bits hay solo 32 vectores posibles, y con datos reales haría falta reentrenar y revalidar.

**La regla de Cardiología (corregida).**
Era un error de normalización de tildes: la regla buscaba `presion alta` y la columna se llama `Presión alta`. Ya se compara sin tildes y sin mayúsculas. Cambió la especialidad sugerida de 24 pacientes de entrenamiento y 22 de validación (la coincidencia con la especialidad esperada pasó de 49 a 60 de 100); no cambió ART1 ni la pureza.

**El autotest.**
Antes solo imprimía TEST PASSED si no explotaba. Ahora tiene cinco chequeos sobre un dataset mínimo y sale con código 1 si falla alguno. Sigue siendo una prueba chica; la verificación fuerte es la del notebook (módulo contra implementación pedagógica, con `assert`), más el visualizador y las corridas reproducibles.

**Manejo de errores.**
La capacidad agotada (`RuntimeError`) y un archivo con encabezado pero sin filas (ahora `ValueError`) salen como `[ERROR]` con código 1, igual que el resto de los errores previsibles.

**Lectura de CSV.**
`load_dataset` usa `_read_csv_rows` (utf-8-sig, con fallback a cp1252 y latin-1) y `_drop_phantom_columns`, así que un CSV guardado desde Excel con tildes carga bien.

**Plantilla final contra plantilla de asignación.**
Un cluster se nombra por su plantilla final, que puede ser más estrecha que la que tenía cuando entraron los pacientes (cluster 2 de entrenamiento con P021, o cluster 1 de validación: plantilla Moco y pacientes de tos y moco). Lo correcto es decir que la plantilla es la intersección de todos sus miembros, no su descripción completa.

**El patrón vacío.**
Se asigna al cluster 0 por convención y cuenta con ratio 1.0, lo que infla un poco el matching ratio y mezcla pacientes sin síntomas en el cluster 0. No es una propiedad de ART1.

**El nombre "estabilidad".**
La métrica del informe llamada estabilidad es el promedio de unos por plantilla. Mejor llamarla "ancho promedio de plantilla".

---

## 12. Hoja de trucos

### Fórmulas

```
Norma            ||x|| = suma de x_i               (cantidad de unos)
Inicio           b_ij = 1/(1+N) = 1/6     t_ji = 1
Puntaje          mu_j = suma_i b_ij * x_i
Ganadora         j* = argmax mu_j          (empate: índice menor)
Coincidencia     T_i = t_j*i AND x_i
Matching ratio   ||T|| / ||x||
Vigilancia       ||T||/||x|| >= rho  -> resonancia   |  < rho -> reset
Aprendizaje      t_j* = T      b_ij* = t_j*i / (0.5 + suma_k t_j*k)
Categoría nueva  una libre (t = 1) siempre pasa: ratio = 1
Vacío            ||x|| = 0 -> cluster 0, ratio 1.0, sin tocar pesos
Pureza           suma_clusters max(conteo por etiqueta) / total
```

Ejemplo mínimo para recordar: P021 `[1,0,0,0,1]` contra la categoría 0 `[1,0,0,0,0]`: `μ = 0.667`, `T = [1,0,0,0,0]`, ratio `1/2 = 0.5 < 0.65`, reset; gana una libre: ratio `2/2 = 1`, nace la categoría 2 con `b = [0.4,0,0,0,0.4]`.

### Números clave

| Dato                                        | Valor                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Pacientes por dataset                       | 100 (5 síntomas)                                                                           |
| IDs                                         | entrenamiento P001 a P100, validación P101 a P200, 0 en común                              |
| Filas idénticas por posición                | 9 de 100                                                                                   |
| Patrones distintos                          | 21 (entrenamiento), 20 (validación)                                                        |
| Pacientes sin síntomas                      | 9 en cada uno                                                                              |
| Etiquetas de validación                     | 5 especialidades, 20 pacientes cada una                                                    |
| ρ por defecto                               | 0.65                                                                                       |
| M por defecto                               | 50                                                                                         |
| Entrenamiento, ρ = 0.65                     | 14 clusters, ratio 0.9500, 1.64 bits por plantilla                                         |
| Validación, ρ = 0.65                        | 12 clusters, ratio 0.9600, 1.58 bits, pureza 69.00%                                        |
| Techo de pureza (vector exacto)             | 73% (azar de la clase mayor: 20%)                                                          |
| Barrido entrenamiento (clusters)            | 0.3 a 6, 0.5 a 8, 0.6 a 14, 0.65 a 14, 0.8 a 17, 0.9 a 17                                  |
| Barrido validación (clusters / pureza)      | 0.3 a 5 / 56%, 0.5 a 7 / 57%, 0.65 a 12 / 69%, 0.8 a 17 / 72%, 0.9 a 17 / 72%              |
| Saltos de ρ                                 | cambian al pasar 0.5 (ratio 1/2) y 0.667 (ratio 2/3); de 0.8 en adelante solo pasa ratio 1 |
| Barajado, semilla 42, ρ = 0.65, 10 corridas | entrenamiento 13 a 15 (media 14.0); validación 12 a 14 (media 12.9)                        |
| Con `>` en lugar de `>=`                    | ρ = 0.5: 14 clusters (en vez de 8); ρ = 2/3: 17 (en vez de 14)                             |
| ρ = 0                                       | 5 clusters (uno por síntoma)                                                               |
| Ratios en entrenamiento a 0.65              | 85 pacientes con 1.0 y 15 con 0.6667                                                       |

### Comandos (desde la raíz del repo)

```powershell
python src/CarGross.py --test
python src/CarGross.py --man
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -o results/res_train.csv --save-txt results/res_train.txt
python src/CarGross.py data/dataset_validacion_ART1.csv  -o results/res_val.csv   --save-txt results/res_val.txt
python src/CarGross.py data/dataset_entrenamiento_ART1.csv -r 0.8 -o results/res_train_080.csv
python src/CarGross.py data/dataset_entrenamiento_ART1.csv --shuffle 5 --seed 42 -o results/estable.csv
python src/CarGross.py data/dataset_entrenamiento_ART1.csv --patient-id P021 -o results/p021.csv
python src/CarGross.py data/dataset_entrenamiento_ART1.csv --interactive
```

Argumentos: `-r/--rho`, `-o/--output`, `--save-txt`, `--max_cat`, `-v/--verbose`, `--interactive`, `--patient-id`, `--shuffle`, `--seed`, `--test`, `--man`, `-i/--input` (o el archivo como argumento posicional). Salida: CSV con `ID_Paciente, Cluster_Asignado, Matching_Ratio, Especialidad_Sugerida`; el CSV se escribe siempre en el modo general (por defecto `resultados_cargross.csv`).

### Frases para tener a mano

- "ART1 agrupa sin etiquetas y crea sus propias categorías; la vigilancia decide cuándo crear una nueva."
- "La plantilla es la intersección de todos sus miembros: solo puede perder síntomas."
- "El puntaje elige a quién probar; el matching ratio decide si pasa."
- "Con estos cinco síntomas el techo de pureza es 73%; sacamos 69%."
- "El orden cambia la granularidad, no la calidad, pero con órdenes extremos cambia bastante."
- "No diagnostica: agrupa y sugiere, lo valida un profesional."

---

Material de referencia usado para armar la guía: Lau (1992), pp. 5 a 14, tomado de Lippmann (1987), "An Introduction to Computing with Neural Nets", IEEE ASSP Magazine, y Carpenter y Grossberg (1987).
