# Carátula — Template para el TFI

> Este archivo contiene el **contenido textual** de la carátula requerida por la consigna oficial (Actividad obligatoria #6 + Nota sobre carátula de `_legacy/CarGross_TP/consignas_TP.md`). **NO** es la carátula en sí — es lo que tenés que poner en ella.
>
> **Advertencia textual de la consigna**: *"No se considera entregado el informe sin esta carátula."*

---

## 1. Datos requeridos por la consigna

Completá los siguientes campos con tu información real:

| Campo | Valor |
|-------|-------|
| **Título** | Redes de Carpenter-Grossberg (ART1): clustering no supervisado aplicado a datos tabulares clínicos y operativos |
| **Subtítulo** (opcional) | Trabajo Final Integrador — Materia Redes Neuronales |
| **Materia** | Redes Neuronales |
| **Institución** | UADER — Facultad de Ingeniería — IDTI Lab |
| **Docente** | [nombre del docente — completar] |
| **Fecha** | [fecha de entrega — completar] |
| **Integrante 1** | Escar, Camilo |
| **Integrante 2** | Gonzalez, Claudio |
| **Integrante 3** | Laballeja, Sofia |
| **Integrante 4** | Meriano, Patricia |

> **Sobre el equipo**: los nombres de arriba son los que figuran en la documentación del proyecto (`_legacy/CarGross_TP/` y la cadena de docs). Si el equipo real es otro, reemplazá los renglones sin tocar el resto del template.

---

## 2. Layout sugerido

```
+----------------------------------------------------------+
|                                                          |
|   [LOGO UADER]                       [LOGO IDTI Lab]     |
|                                                          |
|                                                          |
|              Trabajo Final Integrador                    |
|                                                          |
|                                                          |
|         Redes de Carpenter-Grossberg (ART1):             |
|         clustering no supervisado aplicado              |
|         a datos tabulares clinicos y operativos          |
|                                                          |
|                                                          |
|   Materia:    Redes Neuronales                           |
|   Docente:    [nombre del docente]                       |
|   Fecha:      [fecha de entrega]                         |
|                                                          |
|   Integrantes:                                           |
|     - Escar, Camilo                                      |
|     - Gonzalez, Claudio                                  |
|     - Laballeja, Sofia                                   |
|     - Meriano, Patricia                                  |
|                                                          |
|                                                          |
|                UADER — Facultad de Ingenieria            |
|                IDTI Lab                                  |
|                [ciudad], [fecha]                         |
|                                                          |
+----------------------------------------------------------+
```

### Notas de layout

- **Logos**: arriba a izquierda y derecha, del mismo tamaño visual. Si uno es más chico o más grande, alinearlos al borde superior común.
- **Título**: tipografía serif (ej. Computer Modern, EB Garamond) o sans-serif institucional (ej. Montserrat, Inter), tamaño 28-36 pt. **Una sola línea** si entra; si no, dos líneas balanceadas.
- **Subtítulo "Trabajo Final Integrador"**: arriba del título, más chico (14-16 pt), en versalitas o en mayúsculas con tracking abierto.
- **Cuerpo (materia / docente / fecha / integrantes)**: tipografía sans-serif, 12-14 pt, alineado a la izquierda.
- **Pie**: ciudad + fecha + logos de la institución. Si va a imprimirse en A4, dejar 2.5 cm de margen en todos los lados.
- **Sin imágenes decorativas** salvo los logos institucionales. Carátula limpia, sin emojis, sin fondos con texturas.

---

## 3. Herramientas sugeridas para armar la carátula

| Opción | Dificultad | Resultado | Cuándo usar |
|--------|------------|-----------|--------------|
| **Pandoc + plantilla LaTeX** | Media | PDF profesional | Si vas a entregar todo el informe en PDF. Incluí la carátula como página 1 con `\titlepage` o con el paquete `geometry`. |
| **LaTeX con paquete `titlepage`** | Media-alta | PDF profesional, tipografía impecable | Si ya estás escribiendo el informe en LaTeX. |
| **PowerPoint / Google Slides** | Baja | PDF o PPTX | Si vas a entregar el informe en PPTX o querés máxima flexibilidad visual. |
| **Google Docs** | Baja | PDF | Si necesitás algo rápido sin instalar nada. Insertar tabla o text-boxes con los datos y exportar como PDF. |
| **Canva** | Baja | Imagen PNG/JPG o PDF | Si querés un diseño más gráfico. Hay plantillas de "carátula de tesis/TFI" listas para personalizar. |
| **Markdown → Pandoc → PDF** | Baja-media | PDF limpio | Si el informe entero va en Markdown. Pandoc soporta una carátula como página 1 con un header YAML. |

### Snippet mínimo en LaTeX (referencia rápida)

```latex
\begin{titlepage}
    \centering
    \includegraphics[height=2cm]{logos/uader.png}\hfill
    \includegraphics[height=2cm]{logos/idti.png}\\[1.5cm]
    {\Large Trabajo Final Integrador\par}\vspace{1cm}
    {\Huge\bfseries Redes de Carpenter-Grossberg (ART1)\par}
    \vspace{0.5cm}
    {\Large Clustering no supervisado aplicado a datos tabulares\par}
    \vfill
    \begin{flushleft}
        \textbf{Materia:} Redes Neuronales \\
        \textbf{Docente:} [nombre del docente] \\
        \textbf{Fecha:} [fecha de entrega] \\[0.5cm]
        \textbf{Integrantes:} \\
        Escar, Camilo \\
        Gonzalez, Claudio \\
        Laballeja, Sofia \\
        Meriano, Patricia
    \end{flushleft}
\end{titlepage}
```

### Snippet mínimo en Markdown (Pandoc)

```yaml
---
title: "Redes de Carpenter-Grossberg (ART1)"
subtitle: "Clustering no supervisado aplicado a datos tabulares clinicos y operativos"
author:
  - Escar, Camilo
  - Gonzalez, Claudio
  - Laballeja, Sofia
  - Meriano, Patricia
date: "[fecha de entrega]"
geometry: margin=2.5cm
---
```

Pandoc con `--pdf-engine=xelatex` y la plantilla por defecto produce una página de título a partir de esos metadatos. Para los logos hay que inyectarlos con un header include (`\includegraphics` en el preámbulo).

---

## 4. Logos institucionales

**Los logos de UADER y del IDTI Lab NO están versionados en este repositorio.** Necesitás obtenerlos:

- **UADER**: descargalo del sitio oficial [uader.edu.ar](https://uader.edu.ar) o pedíselo al área de comunicación de la facultad. Versiones usuales: PNG con fondo transparente (ideal para impresión) y JPG con fondo blanco.
- **IDTI Lab**: pedíselo al docente o al laboratorio. Si no lo conseguís a tiempo, contactá al IDTI Lab directamente — suelen tener un kit de identidad listo para usar en trabajos académicos.

### Una vez que tengas los logos

1. Guardalos en una carpeta `logos/` (en la raíz del proyecto o en `docs/`).
2. Insertalos en la carátula según el layout sugerido.
3. **No los deformes**: mantener la proporción original. Si necesitás que ocupen menos espacio, achicalos de a ambos por igual, no por separado.
4. **No los pongas sobre fondos de colores** que rompan la legibilidad: usá fondo blanco o el color institucional de la facultad.

---

## 5. Verificación final antes de imprimir/entregar

Checklist de la carátula (todos deben estar en OK antes de dar el trabajo por entregado):

- [ ] Título completo, **sin typos**, tal como aparece en el cuerpo del informe
- [ ] Materia correcta ("Redes Neuronales")
- [ ] Nombres y apellidos de **TODOS** los integrantes (chequear tildes: Claudio, no Klaudio)
- [ ] Fecha de entrega en formato claro (ej. "Septiembre 2026" o "07/09/2026")
- [ ] Nombre del docente (con título si corresponde: "Dr.", "Mg.", "Ing.", etc.)
- [ ] Logo de UADER presente y legible
- [ ] Logo del IDTI Lab presente y legible
- [ ] Sin emojis
- [ ] Sin imágenes rotas (logos que no se ven = igual a no tenerlos)
- [ ] Sin texto de borrador tipo "Borrador", "Versión preliminar", "v0.1"
- [ ] Apariencia profesional: tipografía consistente, márgenes parejos, alineación prolija
- [ ] Si va a imprimirse: verificar que la versión en PDF renderiza los logos correctamente

Si la consigna dice que **sin carátula no se considera entregado**, asumí que el docente revisa primero esta página. Vale la pena dedicarle 30 minutos extra.
