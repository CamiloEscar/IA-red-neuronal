# IA - Red Neuronal

## Descripción

Este repositorio contiene el desarrollo de un trabajo práctico de la materia Inteligencia Artificial

## Tecnologías utilizadas

- Python

## Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/CamiloEscar/IA-red-neuronal.git
cd IA-red-neuronal
```

### 2. Crear un entorno virtual

```bash
python -m venv venv
```

### 3. Activar el entorno virtual

```bash
venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar el programa

```bash
python src/CarGross.py data/dataset1.csv
```

## Integrantes

- Escar Camilo
- Gonzalez Claudio
- Laballeja Sofia
- Meriano Patricia

```bash
IA-red-neuronal/
│
├── src/                          # Código fuente
│   ├── CarGross.py               # Red de Carpenter-Grossberg (principal)
│   ├── utils.py                  # Funciones auxiliares
│   ├── data_loader.py            
│   └── __init__.py
│
├── data/                         # Datasets 
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── metadata/                 # (archivos extra)
│       └── metadata.csv
│
├── tests/                        # Pruebas
│   └── test_demo.py
│
├── results/                      # Resultados
│   ├── resultado_dataset1.txt
│   ├── resultado_dataset2.txt
│   └── graficos/                 
│
├── docs/                         # Documentación
│   ├── manual.md                 # Manual completo
│   ├── informe.pdf               # Informe con carátula
│   └── presentacion.pptx         # PPT con audio
│
├── README.md                     # Descripción general
├── requirements.txt              # Dependencias
└── .gitignore
```bash
