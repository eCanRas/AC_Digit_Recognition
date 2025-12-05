# ----------------------------------------------------------------------
# Script: mlp_model.R
# Objetivo: Entrenamiento y ajuste de un Perceptrón Multicapa (MLP) 
#           mediante búsqueda de rejilla (Grid Search) para el dataset MNIST.
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------
# Nota: Asegúrate de tener instalados los paquetes: 
# install.packages(c("data.table", "tidymodels", "caret", "nnet"))
library(dplyr)
library(data.table)
library(tidymodels) 
library(tidyverse)
library(caret)
library(nnet) # Paquete que implementa el método 'nnet' (MLP/Red Neuronal)

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset (Asegúrate de que la ruta sea correcta)
#data <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/digit-recognizer/train.csv")
data <- read.csv("~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train_pca.csv")

# Crea subset
data <- data %>%
  group_by(label) %>%
  slice_sample(prop = 10000 / nrow(data)) %>%
  ungroup()

# ----------------------------------------------------
# 2. Preparación de Datos, División y Preprocesamiento
# ----------------------------------------------------

# Convertir la columna 'label' a factor (CRUCIAL)
data$label <- as.factor(data$label)

# División en entrenamiento (80%) y prueba (20%)
data_split <- initial_split(
  data, 
  prop = 0.8, 
  strata = label 
)

train_data <- training(data_split)
test_data <- testing(data_split)

# === PREPROCESAMIENTO (CRUCIAL para Redes Neuronales) ===

# 1. Definir el preprocesamiento usando caret
pre_process_params <- preProcess(
  train_data[, -1],           # Todas las columnas menos 'label'
  method = c("center", "scale", "zv") # Centrado, Escalamiento (estandarización) y Varianza Cero
)

# 2. Aplicar la transformación a los datos de entrenamiento y prueba
train_data_processed <- predict(pre_process_params, train_data)
test_data_processed <- predict(pre_process_params, test_data)

# ----------------------------------------------------
# 3. Entrenamiento del MLP y Ajuste de Hiperparámetros (Grid Search)
# ----------------------------------------------------

message("Iniciando ajuste de hiperparámetros para el MLP (Red Neuronal). Esto tomará tiempo...")

# --- Definición de la Búsqueda de Rejilla ---
# 'size' (número de neuronas en la capa oculta) y 'decay' (regularización de peso)
mlp_grid <- expand.grid(
  size = c(5, 10),       # Número de neuronas a probar
  decay = c(0.01, 0.1) # Regularización L2 (peso) a probar
)

# --- Definición del Control de Entrenamiento (Cross-Validation) ---
fit_control <- trainControl(
  method = "cv", # Cross-Validation
  number = 5,    # 5-fold CV
  verboseIter = TRUE # Muestra el progreso de la búsqueda de rejilla
)

# --- Entrenamiento con Búsqueda de Rejilla ---
# Usamos el método 'nnet' (Red Neuronal de una sola capa oculta)
mlp_model <- train(
  label ~ .,
  data = train_data_processed, # <--- Usar datos PROCESADOS
  method = "nnet",
  trControl = fit_control,
  tuneGrid = mlp_grid,          # Usar la rejilla definida
  linout = FALSE,               # Falso para clasificación
  trace = FALSE,                 # Suprime mensajes de nnet
  MaxNWts = 20000   # <<--- aumentar límite de pesos
  
)

message("Ajuste de hiperparámetros finalizado.")
print(mlp_model)
message("Mejores hiperparámetros encontrados:")
print(mlp_model$bestTune)




# ----------------------------------------------------
# 4. Evaluación y Extracción de Métricas
# ----------------------------------------------------

# 1. Realizar predicciones sobre el conjunto de prueba
test_predictions <- predict(
  mlp_model, 
  newdata = test_data_processed, # <--- Usar datos PROCESADOS
  type = "raw" # Devuelve la clase predicha
)

# 2. Crear la matriz de confusión
cm <- confusionMatrix(
  data = test_predictions, 
  reference = test_data_processed$label, 
  mode = "everything"
)

# 3. Extracción de Métricas
model_results <- data.frame(
  Model = "MLP (nnet) - Tuneado",
  
  # Extracción de la exactitud global
  Accuracy = cm$overall['Accuracy'], 
  
  # Extracción de las métricas promedio (Macro) por clase
  Precision = mean(cm$byClass[, "Precision"], na.rm = TRUE),
  Recall = mean(cm$byClass[, "Recall"], na.rm = TRUE),
  F1_Score = mean(cm$byClass[, "F1"], na.rm = TRUE)
)
print(model_results)

# ----------------------------------------------------
# 5. Guardar Resultados
# ----------------------------------------------------

results_path <- "C:/Users/maria/Documents/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/Resultados/pca_metrics_summary.csv"

# Leer, adjuntar y escribir los resultados
if (file.exists(results_path)) {
  existing_results <- fread(results_path)
  combined_results <- rbind(existing_results, model_results)
} else {
  combined_results <- model_results
}

# Escribir el archivo CSV
fwrite(combined_results, results_path)

message("Resultados guardados en: ", results_path)