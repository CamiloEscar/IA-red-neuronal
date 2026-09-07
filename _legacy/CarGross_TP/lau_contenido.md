# Contenido de Lau — Artificial Neural Networks (IEEE Press, 1992)
## Transcripción de pp. 5–14

Fuente: Lippmann, R.P. (1987). *An Introduction to Computing with Neural Nets.*  
IEEE ASSP Magazine, April 1987, pp. 4–22.  
Reproducido en: Lau, C. (Ed.) (1992). *Artificial Neural Networks.* IEEE Press.

---

## PÁGINAS 5–11 (`Lau.pp5.a.11.pdf`)

### An Introduction to Computing with Neural Nets
**Richard P. Lippmann**

#### Abstract

Los modelos de redes neuronales artificiales han sido estudiados durante muchos años con la esperanza de lograr un rendimiento similar al humano en los campos del reconocimiento de voz e imagen. Estos modelos están compuestos de muchos elementos computacionales no lineales que operan en paralelo y organizados en patrones que recuerdan a las redes neuronales biológicas. Los elementos computacionales o nodos están conectados mediante pesos que típicamente se adaptan durante el uso para mejorar el rendimiento.

Este artículo provee una introducción al campo de las redes neuronales artificiales revisando seis modelos importantes que pueden usarse para clasificación de patrones. Estas redes son bloques de construcción altamente paralelos que ilustran los componentes y principios de diseño de las redes neuronales, y pueden usarse para construir sistemas más complejos. Además de describir estas redes, se hace énfasis en explorar cómo algunos algoritmos de clasificación y clustering existentes pueden realizarse usando componentes simples similares a neuronas.

- Las redes de una capa pueden implementar los algoritmos requeridos por los clasificadores gaussianos de máxima verosimilitud y los clasificadores de mínimo error óptimo para patrones binarios corrompidos por ruido.
- En general, las regiones de decisión requeridas por cualquier algoritmo de clasificación pueden generarse de forma directa mediante redes de tres capas feed-forward.

---

#### INTRODUCCIÓN

Los modelos de redes neuronales artificiales, o simplemente "redes neuronales", reciben muchos nombres: modelos conexionistas, modelos de procesamiento distribuido paralelo, sistemas neuromórficos. Cualquiera sea el nombre, todos estos modelos intentan lograr buen rendimiento mediante la interconexión densa de elementos computacionales simples. En este sentido, la estructura de las redes neuronales artificiales se basa en nuestra comprensión actual de los sistemas nerviosos biológicos.

Los modelos de redes neuronales tienen mayor potencial en áreas como el reconocimiento de voz e imagen, donde se persiguen simultáneamente muchas hipótesis en paralelo, se requieren altas tasas de cómputo, y los mejores sistemas están lejos de igualar el rendimiento humano.

En lugar de ejecutar un programa de instrucciones secuencialmente como en una computadora von Neumann, los modelos de redes neuronales exploran simultáneamente muchas hipótesis competidoras usando redes masivamente paralelas compuestas de muchos elementos computacionales conectados por enlaces con pesos variables.

Los elementos computacionales o nodos usados en los modelos de redes neuronales son no lineales, típicamente analógicos, y pueden ser lentos comparados con los circuitos digitales modernos. El nodo más simple suma N entradas ponderadas y pasa el resultado a través de una no linealidad (Fig. 1). El nodo se caracteriza por un umbral interno u offset θ y por el tipo de no linealidad. La Figura 1 ilustra tres tipos comunes: hard limiters, elementos de lógica de umbral, y no linealidades sigmoidales.

Los modelos de redes neuronales se especifican por la topología de la red, las características del nodo, y las reglas de entrenamiento o aprendizaje. Estas reglas especifican un conjunto inicial de pesos e indican cómo deben adaptarse los pesos durante el uso para mejorar el rendimiento.

Los beneficios potenciales de las redes neuronales van más allá de las altas tasas de cómputo proporcionadas por el paralelismo masivo. Las redes neuronales típicamente proveen un mayor grado de robustez o tolerancia a fallas que las computadoras secuenciales von Neumann, porque hay muchos más nodos de procesamiento, cada uno con conexiones principalmente locales. El daño a unos pocos nodos o enlaces no necesita deteriorar significativamente el rendimiento global.

La mayoría de los algoritmos de redes neuronales también adaptan los pesos de conexión en el tiempo para mejorar el rendimiento basándose en los resultados actuales. La adaptación o aprendizaje es un foco principal de la investigación en redes neuronales. La capacidad de adaptarse y continuar aprendiendo es esencial en áreas como el reconocimiento de voz, donde los datos de entrenamiento son limitados y continuamente se encuentran nuevos hablantes, nuevas palabras, nuevos dialectos, nuevas frases y nuevos entornos.

---

#### CLASIFICADORES NEURALES Y TRADICIONALES

Los diagramas de bloque de los clasificadores tradicionales y de redes neuronales se presentan en la Fig. 2. Ambos tipos de clasificadores determinan cuál de M clases es más representativa de un patrón de entrada estático que contiene N elementos de entrada.

El clasificador tradicional (parte superior de Fig. 2) contiene dos etapas:
1. La primera computa los puntajes de coincidencia para cada clase.
2. La segunda selecciona la clase con el puntaje máximo.

El clasificador de red neuronal adaptativo (parte inferior de Fig. 2) tiene entradas analógicas que se alimentan en paralelo a la primera etapa mediante N conexiones de entrada. La primera etapa computa puntajes de coincidencia y los saca en paralelo a la siguiente etapa. Solo la salida correspondiente a la clase más probable estará en "alto"; las otras salidas estarán en "bajo".

---

#### UNA TAXONOMÍA DE REDES NEURONALES

Se presenta una taxonomía de seis redes neuronales importantes que pueden usarse para clasificación de patrones estáticos (Fig. 3). Esta taxonomía se divide primero entre redes con entradas binarias y con valores continuos. Por debajo, las redes se dividen entre las entrenadas con y sin supervisión.

**Redes con entradas binarias:**
- **Supervisadas:** Red de Hopfield, Red de Hamming, Clasificador Carpenter/Grossberg
- **No supervisadas:** (mismas redes con variantes)

**Redes con entradas continuas:**
- **Supervisadas:** Perceptrón, Perceptrón Multicapa
- **No supervisadas:** Mapas de características de Kohonen

Los algoritmos clásicos más similares a los modelos de redes neuronales se listan al fondo de la Fig. 3.

---

#### LA RED DE HOPFIELD

La red de Hopfield y otras dos redes en la Fig. 3 se usan normalmente con entradas binarias. Estas redes son más apropiadas cuando las representaciones binarias exactas son posibles, como en imágenes en blanco y negro donde los elementos de entrada son valores de píxeles, o con texto ASCII donde los bits de entrada podrían representar los 8 bits de la representación ASCII de cada carácter.

Hopfield reactivó el interés en las redes neuronales con su extenso trabajo en diferentes versiones de la red de Hopfield. Esta red puede usarse como memoria asociativa o para resolver problemas de optimización. Una versión de la red original puede usarse como memoria de direccionamiento por contenido.

**Algoritmo de la Red de Hopfield (Box 1):**

**Step 1. Asignar pesos de conexión:**

$$t_{ij} = \sum_{s=0}^{M-1} x_i^s x_j^s, \quad i \neq j$$
$$t_{ij} = 0, \quad i = j, \quad 0 \leq i, j \leq M-1$$

donde $t_{ij}$ es el peso de conexión del nodo i al nodo j, y $x_i^s$ puede ser +1 o −1 (elemento i del exemplar de la clase s).

**Step 2. Inicializar con patrón de entrada desconocido:**

$$\mu_i(0) = x_i, \quad 0 \leq i \leq N-1$$

**Step 3. Iterar hasta convergencia:**

$$\mu_j(t+1) = f_h\left[\sum_{i=0}^{N-1} t_{ij}\mu_i(t)\right], \quad 0 \leq j \leq M-1$$

La función $f_h$ es la no linealidad hard limiting de la Fig. 1. El proceso se repite hasta que las salidas de los nodos no cambian con nuevas iteraciones.

**Step 4. Repetir desde Step 2.**

**Limitaciones de la red de Hopfield:**
1. El número de patrones que pueden almacenarse y recuperarse con precisión es severamente limitado. Si se almacenan demasiados patrones, la red puede converger a un patrón espurio diferente de todos los patrones exemplares.
2. Un patrón exemplar será inestable si comparte muchos bits en común con otro patrón exemplar.

---

#### LA RED DE HAMMING

La red de Hamming se prueba frecuentemente en problemas donde las entradas se generan seleccionando un exemplar y revirtiendo bits aleatoria e independientemente con una probabilidad dada. Este es un problema clásico en la teoría de comunicaciones que ocurre cuando señales binarias de longitud fija se envían a través de un canal binario simétrico sin memoria.

El clasificador óptimo de mínimo error en este caso calcula la distancia de Hamming al exemplar de cada clase y selecciona la clase con la mínima distancia de Hamming. La distancia de Hamming es el número de bits en la entrada que no coinciden con los bits del exemplar correspondiente.

**Algoritmo de la Red de Hamming (Box 2):**

**Step 1. Asignar pesos de conexión y offsets:**

En la subred inferior:
$$w_{ij} = \frac{x_j^i}{2}, \quad \theta_j = \frac{N}{2}, \quad 0 \leq i \leq N-1, \quad 0 \leq j \leq M-1$$

En la subred superior (MAXNET):
$$t_{kl} = \begin{cases} 1, & k = l \\ -\varepsilon, & k \neq l \end{cases}, \quad \varepsilon < \frac{1}{M}, \quad 0 \leq k, l \leq M-1$$

**Step 2. Inicializar con patrón de entrada desconocido:**

$$\mu_j(0) = f_t\left(\sum_{i=0}^{N-1} w_{ij}x_i - \theta_j\right), \quad 0 \leq j \leq M-1$$

**Step 3. Iterar hasta convergencia:**

$$\mu_j(t+1) = f_t\left(\mu_j(t) - \varepsilon\sum_{k \neq j}\mu_k(t)\right), \quad 0 \leq j, k \leq M-1$$

Este proceso se repite hasta la convergencia, después de la cual la salida de solo un nodo permanece positiva.

**Step 4. Repetir desde Step 2.**

**Ventajas sobre la red de Hopfield:**
- Implementa el clasificador óptimo de mínimo error cuando los errores de bits son aleatorios e independientes.
- Requiere muchas menos conexiones (con 100 entradas y 10 clases, la red de Hamming requiere solo 1.100 conexiones, mientras que la red de Hopfield requiere casi 10.000).
- No sufre de patrones de salida espurios.

---

#### SELECCIONAR O MEJORAR LA ENTRADA MÁXIMA

La necesidad de seleccionar o mejorar la entrada con valor máximo ocurre frecuentemente en problemas de clasificación. El MAXNET descrito anteriormente usa inhibición lateral pesada similar a la usada en otros diseños de redes neuronales. Estos diseños crean un tipo de red "winner-take-all" cuyo diseño imita el uso intensivo de inhibición lateral evidente en las redes neuronales biológicas del cerebro humano.

---

## PÁGINAS 12–14 (`Lau.pp12.a.14.pdf`)

### EL CLASIFICADOR CARPENTER/GROSSBERG

Carpenter y Grossberg, en el desarrollo de su Teoría de Resonancia Adaptativa (ART), diseñaron una red que forma clusters y es entrenada sin supervisión. Esta red implementa un algoritmo de clustering muy similar al simple algoritmo de clustering secuencial "leader" descripto en la referencia [16].

El algoritmo leader selecciona la primera entrada como exemplar del primer cluster. La siguiente entrada se compara con el primer cluster exemplar. "Sigue al líder" y es agrupada con el primero si la distancia al primero es menor que un umbral. De lo contrario, es el exemplar de un nuevo cluster. Este proceso se repite para todas las entradas. El número de clusters crece así con el tiempo y depende tanto del umbral como de la métrica de distancia usada para comparar entradas con cluster exemplares.

Los componentes principales de una red de clasificación Carpenter/Grossberg con tres entradas y dos nodos de salida se presentan en la Fig. 10. La estructura de esta red es similar a la de la red de Hamming. Los puntajes de coincidencia se computan usando conexiones feed-forward y el valor máximo se mejora usando inhibición lateral entre los nodos de salida. Esta red difiere de la red de Hamming en que se proveen conexiones de retroalimentación desde los nodos de salida hacia los nodos de entrada. También se proveen mecanismos para desactivar el nodo de salida con valor máximo, y para comparar los exemplares con la entrada para el test de umbral requerido por el algoritmo leader.

Esta red está completamente descripta usando ecuaciones diferenciales no lineales, incluye retroalimentación extensa, y ha demostrado ser estable. En la operación típica, las ecuaciones diferenciales pueden mostrarse para implementar el algoritmo de clustering presentado en el Box 3.

---

### ALGORITMO CARPENTER/GROSSBERG (Box 3)

**Step 1. Inicialización:**

$$t_{ij}(0) = 1$$
$$b_{ij}(0) = \frac{1}{1+N}$$
$$0 \leq i \leq N-1, \quad 0 \leq j \leq M-1$$
$$\text{Set } \rho, \quad 0 \leq \rho \leq 1$$

donde $b_{ij}(t)$ es el peso de conexión bottom-up y $t_{ij}(t)$ es el peso de conexión top-down entre el nodo de entrada i y el nodo de salida j en el tiempo t. Estos pesos definen el exemplar especificado por el nodo de salida j. La fracción ρ es el umbral de vigilancia (vigilance threshold) que indica cuán cercana debe estar una entrada a un exemplar almacenado para coincidir.

**Step 2. Aplicar nueva entrada**

**Step 3. Computar puntajes de coincidencia (Matching Scores):**

$$\mu_j = \sum_{i=0}^{N-1} b_{ij}(t) x_i, \quad 0 \leq j \leq M-1$$

donde $\mu_j$ es la salida del nodo de salida j y $x_i$ es el elemento i de la entrada que puede ser 0 o 1.

**Step 4. Seleccionar el mejor exemplar coincidente:**

$$\mu_{j^*} = \max_j(\mu_j)$$

Esto se realiza usando inhibición lateral extensa como en el MAXNET.

**Step 5. Test de Vigilancia (Vigilance Test):**

$$\|X\| = \sum_{i=0}^{N-1} x_i$$

$$\|T \cdot X\| = \sum_{i=0}^{N-1} t_{ij^*} x_i$$

$$\text{¿Es } \frac{\|T \cdot X\|}{\|X\|} > \rho \text{ ?}$$

- **NO** → IR AL STEP 6
- **SÍ** → IR AL STEP 7

**Step 6. Deshabilitar el mejor exemplar coincidente:**

La salida del nodo de mejor coincidencia seleccionado en el Step 4 se establece temporalmente en cero y ya no participa en la maximización del Step 4. Luego ir al Step 3.

**Step 7. Adaptar el mejor exemplar coincidente:**

$$t_{ij^*}(t+1) = t_{ij^*}(t) \cdot x_i$$

$$b_{ij^*}(t+1) = \frac{t_{ij^*}(t) \cdot x_i}{0.5 + \sum_{i=0}^{N-1} t_{ij^*}(t) \cdot x_i}$$

**Step 8. Repetir desde Step 2.**

*(Primero rehabilitar cualquier nodo deshabilitado en el Step 6)*

---

### DESCRIPCIÓN DEL ALGORITMO

El algoritmo presentado en el Box 3 asume que se usa "fast learning" como en las simulaciones presentadas en la referencia [3], y que los elementos tanto de las entradas como de los exemplares almacenados toman solo los valores 0 y 1.

La red se inicializa estableciendo efectivamente todos los exemplares representados por pesos de conexión en cero. Además, un umbral de coincidencia llamado **vigilance** que va entre 0.0 y 1.0 debe establecerse. Este umbral determina cuán cercano debe estar un nuevo patrón de entrada a un exemplar almacenado para ser considerado similar:

- Un valor cercano a 1 requiere una coincidencia cercana.
- Valores más pequeños aceptan una coincidencia más pobre.

Las nuevas entradas se presentan secuencialmente al fondo de la red como en la red de Hamming. Luego de la presentación, la entrada se compara con todos los exemplares almacenados en paralelo como en la red de Hamming para producir puntajes de coincidencia. El exemplar con el puntaje de coincidencia más alto se selecciona usando inhibición lateral. Luego se compara con la entrada computando el ratio del producto punto de la entrada y el mejor exemplar coincidente (número de bits 1 en común) dividido por el número de bits 1 en la entrada.

- Si este ratio es mayor que el umbral de vigilancia, la entrada se considera similar al mejor exemplar coincidente, y ese exemplar se actualiza realizando una operación lógica AND entre sus bits y los de la entrada.
- Si el ratio es menor que el umbral de vigilancia, la entrada se considera diferente de todos los exemplares y se agrega como nuevo exemplar.

Cada nuevo exemplar adicional requiere un nodo y 2N conexiones para computar puntajes de coincidencia.

---

### COMPORTAMIENTO DEL CLASIFICADOR CARPENTER/GROSSBERG

El comportamiento de la red Carpenter/Grossberg se ilustra en la Fig. 11. Aquí se asume que los patrones a reconocer son los tres patrones de las letras "C", "E" y "F" mostrados en el lado izquierdo de esta figura. Estos patrones tienen 64 píxeles cada uno que toman el valor 1 cuando son negros y 0 cuando son blancos. Los resultados se presentan cuando el umbral de vigilancia se estableció en 0.9. Esto fuerza la creación de patrones exemplares separados para cada letra.

El lado izquierdo de la Fig. 11 muestra la entrada a la red en ensayos sucesivos. El lado derecho presenta los patrones exemplares formados después de que cada patrón ha sido aplicado:

- "C" se presenta primero: los pesos de conexión internos se alteran para formar un exemplar interno idéntico a la "C".
- Luego se aplica "E": se agrega un nuevo exemplar "E".
- Luego se aplica "F": se agrega un nuevo exemplar "F", llevando a tres exemplares almacenados.
- Las "F" ruidosas (con un pixel negro faltante en el borde superior) son aceptadas como similares al exemplar "F" y lo degradan debido a la operación AND realizada durante la actualización.
- Cuando se aplica otra "F" ruidosa con solo un pixel negro faltante diferente, se considera diferente de los exemplares existentes y se agrega una nueva "F" ruidosa como exemplar.

Estos resultados ilustran que el algoritmo Carpenter/Grossberg puede funcionar bien con patrones de entrada perfectos, pero que incluso una pequeña cantidad de ruido puede causar problemas. Sin ruido, el umbral de vigilancia puede establecerse de modo que los dos patrones más similares se consideren diferentes. Con ruido, sin embargo, este nivel puede ser demasiado alto y el número de exemplares almacenados puede crecer rápidamente hasta que todos los nodos disponibles se agoten.

Las modificaciones son necesarias para mejorar el rendimiento de este algoritmo con ruido. Estas podrían incluir adaptar los pesos más lentamente y cambiar el umbral de vigilancia durante el entrenamiento y las pruebas, como se sugiere en [3].

---

### EL PERCEPTRÓN DE UNA SOLA CAPA

El perceptrón de una sola capa es la primera de tres redes en la Fig. 3 que puede usarse con entradas tanto de valores continuos como binarios. Esta red simple generó mucho interés cuando se desarrolló inicialmente por su capacidad para aprender a reconocer patrones simples. Un perceptrón que decide si una entrada pertenece a una de dos clases (denotadas A o B) se muestra en la parte superior de la Fig. 12.

El nodo único computa una suma ponderada de los elementos de entrada, resta un umbral (θ) y pasa el resultado a través de una no linealidad hard limiting tal que la salida y es +1 o −1. La regla de decisión es responder clase A si la salida es +1 y clase B si la salida es −1.

**Procedimiento de Convergencia del Perceptrón (Box 4):**

**Step 1. Inicializar pesos y umbral:**

Establecer $w_i(0)$ ($0 \leq i \leq N-1$) y θ en valores aleatorios no nulos pequeños.

**Step 2. Presentar nueva entrada y salida deseada:**

Presentar nueva entrada continua $x_0, x_1, \ldots, x_{N-1}$ junto con la salida deseada $d(t)$.

**Step 3. Calcular la salida actual:**

$$y(t) = f_h\left(\sum_{i=0}^{N-1} w_i(t)x_i(t) - \theta\right)$$

**Step 4. Adaptar pesos:**

$$w_i(t+1) = w_i(t) + \eta[d(t) - y(t)]x_i(t), \quad 0 \leq i \leq N-1$$

$$d(t) = \begin{cases} +1 & \text{si la entrada es de la clase A} \\ -1 & \text{si la entrada es de la clase B} \end{cases}$$

donde η es una fracción de ganancia positiva menor que 1, y d(t) es la salida correcta deseada para la entrada actual. Los pesos no cambian si se toma la decisión correcta.

**Step 5. Repetir desde Step 2.**

Los pesos de conexión y el umbral en un perceptrón pueden fijarse o adaptarse usando varios algoritmos diferentes. El procedimiento de convergencia del perceptrón original fue desarrollado por Rosenblatt para ajustar pesos.

---

## NOTAS ADICIONALES DEL TEXTO

### Sobre el dilema estabilidad-plasticidad

La red de Carpenter/Grossberg resuelve el **dilema estabilidad-plasticidad** que es central en el aprendizaje de redes neuronales:

- **Plasticidad**: la red debe poder aprender nuevos patrones.
- **Estabilidad**: la red no debe olvidar patrones ya aprendidos cuando incorpora nuevos.

El parámetro de vigilancia ρ es el mecanismo que regula este balance: un ρ alto favorece la plasticidad (crea clusters nuevos fácilmente), un ρ bajo favorece la estabilidad (agrupa más patrones en los clusters existentes).

### Sobre la taxonomía (Fig. 3)

| Red | Entrada | Supervisión | Algoritmo clásico equivalente |
|-----|---------|-------------|-------------------------------|
| Hopfield | Binaria | No | Memoria asociativa |
| Hamming | Binaria | No | Clasificador óptimo (mín. error) |
| Carpenter/Grossberg | Binaria | No | Leader clustering algorithm |
| Perceptrón | Continua/Binaria | Sí | Clasificador gaussiano |
| Perceptrón Multicapa | Continua/Binaria | Sí | Clasificador de frontera arbitraria |
| Kohonen (SOM) | Continua | No | K-Means clustering |

### Sobre las conexiones entre redes

- La red de **Hamming** implementa exactamente el clasificador de mínimo error para patrones binarios con ruido aleatorio e independiente.
- La red de **Hopfield** puede usarse como memoria asociativa o para optimización, pero sufre de patrones espurios y capacidad limitada (~0.15N patrones).
- La red de **Carpenter/Grossberg** es única en la taxonomía porque su número de clusters es dinámico: crece según los datos, sin necesidad de especificarlo previamente.
- El **Perceptrón** crea fronteras de decisión lineales (hiperplanos); para fronteras no lineales se necesita el Perceptrón Multicapa.
- **Kohonen** organiza mapas de características topológicamente ordenados, siendo el equivalente neural del K-Means.

---

## REFERENCIAS CITADAS EN EL TEXTO

[3] Carpenter, G.A. & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37, 54–115.

[7] Duda, R.O. & Hart, P.E. (1973). *Pattern Classification and Scene Analysis.* Wiley.

[16] Hartigan, J.A. (1975). *Clustering Algorithms.* Wiley.

[18] Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79, 2554–2558.

[22] Kohonen, T. (1984). *Self-Organization and Associative Memory.* Springer-Verlag.

[25] Lippmann, R.P. & Gold, B. (1987). Neural classifiers useful for speech recognition. *Proceedings of the IEEE First International Conference on Neural Networks*, IV, 417–426.

[39] Rosenblatt, F. (1962). *Principles of Neurodynamics.* Spartan Books.
