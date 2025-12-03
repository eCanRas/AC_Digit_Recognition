# ----------------------------------------------------------------------
# Script: ensemble_model.R
# Objetivo: Entrenamiento y evaluación de un ensamble por Votación Mayoritaria
# para el dataset Digit Recognition (MNIST), usando múltiples modelos de caret.
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------
# Nota: Asegúrate de tener instalados los paquetes: 
# install.packages(c("data.table", "tidymodels", "caret", "randomForest", "e1071", "kernlab"))
library(dplyr)
library(data.table)
library(tidymodels)
library(tidyverse) 
library(caret)
library(randomForest) # Necesario para el método 'rf'
library(e1071)      # Necesario para el método 'nb'
library(kernlab)    # Necesario para el método 'knn'

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset (Asegúrate de que la ruta sea correcta en tu entorno)
data <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/digit-recognizer/train.csv")

# Crea subset (Manteniendo el subsampling para agilizar el proceso)
data <- data %>%
  group_by(label) %>%
  slice_sample(prop = 10000 / nrow(data)) %>%
  ungroup()

# ----------------------------------------------------
# 2. Preparación de Datos y División
# ----------------------------------------------------

# Convertir la columna 'label' a factor (CRUCIAL para clasificación en R)
data$label <- as.factor(data$label)

# División en entrenamiento (80%) y prueba (20%)
data_split <- initial_split(
  data, 
  prop = 0.8, 
  strata = label 
)

train_data <- training(data_split)
test_data <- testing(data_split)

# Limpiar nombres de columnas para evitar problemas con algunos modelos de caret
names(train_data) <- make.names(names(train_data))
names(test_data) <- make.names(names(test_data))

# ----------------------------------------------------
# 3. Entrenamiento de Modelos Individuales con caret
# ----------------------------------------------------

message("Iniciando entrenamiento de los modelos para el ensamble. Esto puede tardar varios minutos...")

# Definir la estrategia de control de entrenamiento (p.ej., 3-fold Cross-Validation)
fit_control <- trainControl(
  method = "cv", # Cross-Validation
  number = 3,    # Número de folds
  classProbs = FALSE # No necesitamos probabilidades, solo la clase
)

# --- Modelo 1: Random Forest (rf) ---
rf_model <- train(
  label ~ .,
  data = train_data,
  method = "rf",
  trControl = fit_control,
  tuneGrid = data.frame(mtry = 28) # Usamos el mtry anterior para consistencia
)

message("Modelo 1 (Random Forest) entrenado.")

# --- Modelo 2: K-Nearest Neighbors (knn) ---
knn_model <- train(
  label ~ .,
  data = train_data,
  method = "knn",
  trControl = fit_control,
  tuneGrid = data.frame(k = 5) # Un valor simple para K
)

message("Modelo 2 (KNN) entrenado.")

# --- Modelo 3: Naive Bayes (nb) ---
nb_model <- train(
  label ~ .,
  data = train_data,
  method = "nb",
  trControl = fit_control
)

message("Modelo 3 (Naive Bayes) entrenado.")

message("Todos los modelos del ensamble han sido entrenados con éxito.")

# ----------------------------------------------------
# 4. Ensamble por Votación Mayoritaria y Evaluación
# ----------------------------------------------------

# Función para encontrar la moda (la clase más votada)
get_majority_vote <- function(x) {
  # x es un vector de predicciones de clase
  tbl <- table(x)
  # Devuelve el nombre de la clase con la mayor frecuencia
  return(names(which.max(tbl)))
}

# 1. Realizar predicciones de clase sobre el conjunto de prueba para cada modelo
pred_rf <- predict(rf_model, newdata = test_data)
pred_knn <- predict(knn_model, newdata = test_data)
pred_nb <- predict(nb_model, newdata = test_data)

# 2. Combinar las predicciones en un solo dataframe
all_predictions <- data.frame(
  RF = pred_rf,
  KNN = pred_knn,
  NB = pred_nb
)

# 3. Aplicar la votación mayoritaria (moda) fila por fila
ensemble_predictions <- as.factor(apply(
  all_predictions, 
  1, 
  get_majority_vote
))

# 4. Asegurar que los niveles de las predicciones del ensamble coincidan con los datos reales
levels(ensemble_predictions) <- levels(test_data$label)

# 5. Crear la matriz de confusión para el Ensamble
cm_ensemble <- confusionMatrix(
  data = ensemble_predictions,          # Predicciones del ensamble
  reference = test_data$label,          # Etiquetas reales
  mode = "everything"
)

# 6. Extracción de Métricas del Ensamble
model_results <- data.frame(
  Model = "Ensamble (Votación Mayoritaria - RF+KNN+NB)",
  
  # Extracción de la exactitud global
  Accuracy = cm_ensemble$overall['Accuracy'],
  
  # Extracción de las métricas promedio por clase
  # Nota: 'caret' calcula estas métricas por clase y promedia
  Precision = mean(cm_ensemble$byClass[, "Precision"], na.rm = TRUE),
  Recall = mean(cm_ensemble$byClass[, "Recall"], na.rm = TRUE),
  F1_Score = mean(cm_ensemble$byClass[, "F1"], na.rm = TRUE)
)
print(model_results)

# ----------------------------------------------------
# 5. Guardar Resultados
# ----------------------------------------------------

results_path <- "C:/Users/USUARIO/Desktop/AC_Digit_Recognition/Resultados/metrics_summary.csv"

# Leer, adjuntar y escribir los resultados
if (file.exists(results_path)) {
  existing_results <- fread(results_path)
  combined_results <- rbind(existing_results, model_results)
} else {
  combined_results <- model_results
}

# Escribir el archivo CSV
fwrite(combined_results, results_path)

message("Resultados del Ensamble guardados en: ", results_path)
