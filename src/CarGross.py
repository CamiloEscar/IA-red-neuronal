"""
CarGross.py
Implementación básica de una red de Carpenter-Grossberg

Autor: [Tu nombre]
Materia: Inteligencia Artificial

Descripción:
Este módulo implementa una estructura base para una red neuronal
tipo Carpenter-Grossberg con entrada desde archivos CSV.
"""

import numpy as np
import pandas as pd
import sys
import os


class CarGrossNetwork:
    def __init__(self, learning_rate=0.1, epochs=100):
        """
        Inicializa la red

        :param learning_rate: tasa de aprendizaje
        :param epochs: cantidad de iteraciones
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def initialize_weights(self, input_size):
        """Inicializa los pesos de la red"""
        self.weights = np.random.rand(input_size)
        print(f"[INFO] Pesos inicializados: {self.weights}")

    def train(self, X):
        """
        Entrena la red

        :param X: matriz de entrada
        """
        if self.weights is None:
            self.initialize_weights(X.shape[1])

        print("[INFO] Iniciando entrenamiento...")

        for epoch in range(self.epochs):
            for x in X:
                self.update_weights(x)

        print("[INFO] Entrenamiento finalizado")

    def update_weights(self, x):
        """
        Regla de actualización (simplificada)

        :param x: vector de entrada
        """
        # Placeholder de lógica Carpenter-Grossberg
        self.weights += self.learning_rate * (x - self.weights)

    def predict(self, X):
        """
        Genera una salida

        :param X: datos de entrada
        :return: salida de la red
        """
        if self.weights is None:
            raise ValueError("La red no fue entrenada")

        return np.dot(X, self.weights)


def load_csv(file_path):
    """
    Carga un archivo CSV

    :param file_path: ruta al archivo
    :return: datos como numpy array
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        data = pd.read_csv(file_path)

        if data.empty:
            raise ValueError("El archivo CSV está vacío")

        print(f"[INFO] Dataset cargado: {file_path}")
        return data.values

    except Exception as e:
        print(f"[ERROR] Error al cargar CSV: {e}")
        sys.exit(1)


def main():
    """
    Punto de entrada principal
    """
    if len(sys.argv) < 2:
        print("Uso: python CarGross.py <archivo.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Cargar datos
    X = load_csv(file_path)

    # Crear red
    network = CarGrossNetwork(learning_rate=0.1, epochs=50)

    try:
        # Entrenar
        network.train(X)

        # Predecir
        output = network.predict(X)

        print("[RESULTADO]")
        print(output[:5])  # muestra parcial

    except Exception as e:
        print(f"[ERROR] Fallo durante ejecución: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()