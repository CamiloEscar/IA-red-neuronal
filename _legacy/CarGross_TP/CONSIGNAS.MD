Trabajo Final Integrador
[Opción promoción]
Redes Neuronales
Museo de Redes Neuronales

Objetivo
Este TFI tiene la principal finalidad de introducir al alumno en el mundo de las redes neuronales y generar una experiencia pragmática que le permita tener que explicar de manera simple para un tercero, el comportamiento y características de uno de los heurísticos más paradigmáticos de la IA: las redes neuronales. A la vez estará contribuyendo a  la creación de un museo digital sobre la temática.
Mecánica de desarrollo
Para cumplir el objetivo el alumno debe llevar a cabo mínimamente los pasos detallados en el texto, sirviéndose para ello de las referencias y materiales provistos junto con este documento. Este trabajo constituye asimismo la alternativa para quienes deseen aprobar por promoción la materia. Cualquier omisión en la entrega se considerará como no entregado y el alumno deberá cumplir con los requisitos propios de la cursada sin promoción (entrega de todos los TP conforme el plan de la materia y examen final)

Actividad obligatoria del TFI

1) Emplear el material adjunto, Lau.pp12.a.14.pdf, que describe la red solicitada
2) generar un PPT con sonido donde se explique: origen, justificación y características de las redes de Carpenter-Grossberg. También aplicaciones de este tipo de redes, fortalezas y debilidades. Se calificará la originalidad y didáctica. Debe explicar claramente la diferencia con otras redes que puedan resultar de interés.
3) Generar una red de Carpenter-Grossberg llamado CarGross.py con código Python autodocumentado, que tenga las características siguientes:
-todos los elementos deben estar presentes en la entrega (códigos e instructivos de instalación)
-debe ser sencillo de instalar
-debe funcionar sin errores
-su entrada deben ser csv
-se pueden requerir archivos de metadatos pero deben ser documentados y en CSV
-no se exigen interfase gráfica
-debe implementar un man ante corridas infructuosas
-debe manejar errores
4) debe haber un manual de referencia, con alcances y limitaciones, proceso de instalación, modo de correr al menos un test demo, y un FAQ con al menos 3 preguntas.
5) Debe entregar al menos 2 datasets de uso con el módulo, y describir en el manual la manera de correrlo, y la salida esperada. Los datasets debe tener al menos 50 datos y 5 variables.
6) Debe entregar un informe con las corridas con ambos datasets, evaluando los resultados.
NOTA: puede usar más referencias además de las entregadas y en todos los casos debe incluirlas en el manual como una sección al final bajo el título REFERENCIAS.
NOTA: la entrega del informe debe contener carátula con nombre de los integrantes, fecha, título, materia, logo de UADER y del IDTI Lab, docente. No se considera entregado el informe sin esta carátula.
Referencias
[1]Artificial Neural Networks. Edgar Sánchez-Sinencio and Clifford Lau. IEEE Press. 1992.
(este material se provee como Lau.pp5.a.11.pdf)








Anexo I. Instalación de Python

El lenguaje Python es uno de los más difundidos para trabajar de manera avanzada con datos, especialmente en cuestiones de aprendizaje automático, sistemas inteligentes y procesamiento de datos de alta complejidad. En nuestro curso lo usaremos como una de las herramientas a fin de que el alumno tenga al menos un acercamiento al mismo si es que nunca antes lo ha emplado
Alternativas de uso
Se puede instalar Python siguiendo los pasos mencionados a continuación, específicos del sistema operativo con el que se esté trabajando. Hay quienes lo instalan con un entorno llamado Anaconda (incluye por defecto muchas bibliotecas útiles y una interfaz amigable para el desarrollo). El problema de esta opción es que baja muchas cosas que no se usan y es bastante pesado (y puede complicarse un poco configurarlo en algunas ocasiones). En los próximos pasos la instalación es sin Anaconda, pero luego habrá que instalar un IDE (interfase para los comandos) como puede ser PyCharm, Jupyter u otros.

Paso a Paso según el sistema operativo
Windows
Descargar la versión 3.8.5 de la distribución oficial de Python desde el sitio web oficial: https://www.python.org/downloads/. Después de descargar el archivo *.exe,  ejecutarlo y seguir las instrucciones.

Linux
Suele tener ya instalado Python. Para verificar si es así, y qué versión es, hay que abrir una consola o terminal y tipear el siguiente comando: python3 --version
Si no está Python o si se desea instalar una versión diferente, se puede hacer desde la terminal, tipeando en ella lo siguiente: sudo apt-get install python3.8

MacOS
Descargar la versión 3.8.5 de la distribución oficial de Python desde el sitio web oficial: https://www.python.org/downloads/. Después de descargar el archivo *.pkg, ejecutarlo y seguir las instrucciones.

Versión para teléfono inteligente
Existen aplicaciones gratuitas que instalan una consola de Python en el teléfono celular. Permiten correr todos los comandos de Python, aunque tienen limitaciones para el uso de bibliotecas externas.
Solo hay que buscar en la tienda ‘Python’ y descargar la app. Según el sistema operativo que uses:
- Android: QPython 3L, Pydroid 3
- iOS y Windows Phone: Python 3

Uso sin instalación
Existen consolas en línea, que te permiten correr Python como si estuviese instalado en una PC, pero están online. Son completamente gratis y su único requisito es tener conectividad a internet.
•	Una de ellas es la plataforma Colaboratory de Google, que pone a disposición varias bibliotecas interesantes en un entorno Jupyter online muy cómodo y accesible. Se puede acceder con una cuenta de Google registrándose a través de https://colab.research.google.com/
•	Otras opciones más sencillas y útiles para pruebas rápidas:
◦	repl.it (http://repl.it/languages/python3)
◦	CodingGround (https://www.tutorialspoint.com/execute_python3_online.php)


Anexo II. Instalación del fuzzy C-means



Para instalarlo simplemente debe tipearse el siguiente comando en la consola una vez instalado el Python:
pip install scikit-fuzzy


