# Guía de Contribución

Este documento define cómo trabajar en equipo dentro del repositorio.
Todos los integrantes deben seguir estas reglas para mantener el orden y evitar errores.

---

## RESUMEN

Issue = qué voy a hacer  
PR = ya lo hice, revisalo  

PASOS:

1. Crear un Issue
2. Crear una rama
3. Hacer cambios y commits
4. Subir la rama
5. Crear Pull Request (vinculado al Issue)

---

## Flujo de trabajo

### 1. Crear un Issue (SIEMPRE)

Antes de empezar cualquier tarea:

1. Crear un Issue en GitHub
2. Describir qué se va a hacer
3. Asignarse

Ejemplos:

- "Agregar lectura de dataset"
- "Corregir error en entrenamiento"
- "Agregar gráficos"

---

### 2. No trabajar sobre main

La rama `main` es solo para versiones estables del proyecto.  
Todo desarrollo se hace en ramas nuevas.

---

### 3. Crear una rama

git checkout -b nombre-rama

Ejemplos:
git checkout -b feature/carga-datos
git checkout -b fix/error-entrenamiento

---

### 4. Realizar cambios y commits

git add .
git commit -m "mensaje descriptivo"

Ejemplos:

- "Agrega función de carga de datos"
- "Corrige error en entrenamiento"
- "Mejora resultados del modelo"

---

### 5. Subir cambios al repositorio

git push origin nombre-rama

---

### 6. Crear Pull Request (PR)

En GitHub:

1. Ir al repositorio
2. Click en "Compare & pull request"
3. Explicar qué se hizo

IMPORTANTE:
En la descripción del PR escribir:

Resuelve #numero_del_issue

Ejemplo:
Resuelve #3

Esto conecta automáticamente el PR con el Issue.

---

### 7. Revisión y merge

- NO hacer merge directo a main
- El administrador revisa los cambios
- Se aprueba o se piden correcciones
- Luego se hace merge

---

## Uso de Issues

Los Issues se usan para organizar el trabajo.

✔ Siempre crear un Issue antes de empezar  
✔ Cada tarea debe tener su Issue asociado  

---

## Reglas importantes

✔ Usar ramas para trabajar  
✔ Hacer commits claros  
✔ Usar Issues para organizar tareas  

❌ No subir código que rompa el proyecto  
❌ No trabajar directamente en main  

---

## Convenciones

### Nombres de ramas

feature/... → nuevas funcionalidades  
fix/... → corrección de errores  
docs/... → documentación  

---

### Mensajes de commit

Formato recomendado:

tipo: descripción

Ejemplos:
feat: agrega entrenamiento de red  
fix: corrige división por cero  
docs: actualiza README  

---

## Recomendaciones

- Hacer commits pequeños y frecuentes
- Probar el código antes de subirlo
- Escribir mensajes claros

---
