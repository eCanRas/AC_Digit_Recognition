# Digit Recognition - MNIST Classification

Proyecto de clasificación de dígitos manuscritos utilizando el dataset MNIST, implementando y comparando múltiples algoritmos de aprendizaje automático con técnicas de reducción de complejidad.

## 📋 Descripción del Proyecto

Este proyecto tiene como objetivo identificar correctamente los dígitos (0-9) de un conjunto de datos de decenas de miles de imágenes manuscritas del dataset **MNIST** (Modified National Institute of Standards and Technology).

**Objetivos principales:**
- Implementar y comparar múltiples modelos de clasificación
- Seleccionar el modelo con mayor **Accuracy** y menor consumo de recursos
- Aplicar técnicas de reducción de dimensionalidad (**PCA** e **Importance**)
- Evaluar el rendimiento mediante métricas: Accuracy, Precision, Recall, F1-Score

## 📊 Dataset

- **Origen:** MNIST (Modified National Institute of Standards and Technology)
- **Tamaño:** 42,000 imágenes de entrenamiento
- **Características:** 28×28 píxeles (784 características) en escala de grises (0-255)
- **Clases:** 10 (dígitos 0-9)
- **Formato:** Archivo CSV con columna `label` (etiqueta) y 784 columnas de píxeles

## 🤖 Modelos Implementados

### Árbol de Decisión (rpart)
Versiones no podadas y podadas. Línea base interpretable y rápida.

### Random Forest 🌲
Ensemble de árboles que reduce varianza. Muy robusto en todas las configuraciones.

### SVM con Kernel Radial ⭐ **MEJOR MODELO**
Captura fronteras no lineales. Excelente con variables seleccionadas.

### Red Neuronal Multicapa (MLP)
Red neuronal tuneada. Captura relaciones complejas.

### Ensemble (Votación Mayoritaria)
Combina Random Forest + KNN + Naive Bayes. Mejora significativamente con PCA e Importance.

### Boosting
Mejora iterativa mediante énfasis en muestras mal clasificadas.

## 🔧 Técnicas de Reducción de Complejidad

### PCA (Análisis de Componentes Principales)
Reduce las 784 características a componentes principales ortogonales. Acelera entrenamiento y mejora SVM.

### Importance (Selección de Variables)
Selecciona píxeles más relevantes. Mantiene interpretabilidad y reduce dimensionalidad.


## 📁 Descripción de Archivos

### Carpeta `digit-recognizer/`
- **train.csv** - Dataset de entrenamiento original (42,000 muestras, 784 características)
- **test.csv** - Dataset de prueba original
- **sample_submission.csv** - Ejemplo de formato de envío

### Subcarpeta `Procesado/Importance/`
- **train_importance.csv** - Dataset de entrenamiento procesado con selección de variables por importancia
- **test_importance.csv** - Dataset de prueba procesado con selección de variables

### Subcarpeta `Procesado/PCA/`
- **train_pca.csv** - Dataset de entrenamiento transformado con PCA
- **test_pca.csv** - Dataset de prueba transformado con PCA

### Carpeta `MODELOS/`
- **rpart_model.R** - Entrenamiento de Árbol de Decisión (rpart) con validación cruzada y podado
- **rf_model.R** - Entrenamiento de Random Forest con sintonización de hiperparámetros
- **SVM_model.R** - Entrenamiento de SVM con kernel radial, escalado de datos y validación cruzada
- **mlp_model.R** - Entrenamiento de Red Neuronal Multicapa (nnet) con tuning de capas ocultas
- **ensemble_model.R** - Entrenamiento de Ensemble por votación mayoritaria (RF+KNN+NB)
- **boosting_model.R** - Entrenamiento de modelo Boosting

### Carpeta `PCA/`
- **PCA.R** - Script para aplicar Análisis de Componentes Principales a los datos originales y generar datasets procesados

### Carpeta `Importance/`
- **importance.R** - Script para seleccionar variables por importancia usando Random Forest y generar datasets procesados

### Carpeta `Resultados/`
- **metrics_summary.csv** - Métricas de todos los modelos entrenados con datos originales
- **pca_metrics_summary.csv** - Métricas de todos los modelos entrenados con datos procesados por PCA
- **importance_metrics_summary.csv** - Métricas de todos los modelos entrenados con variables seleccionadas por importancia
- **generar_gráficas.R** - Script para generar visualizaciones comparativas de rendimiento
- **PNG files** - Gráficas comparativas por métrica y perfiles de rendimiento para cada configuración (datos originales, PCA, Importance)

