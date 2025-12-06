# ----------------------------------------------------------------------
# Script: svm_model.R
# Objetivo: Entrenamiento y evaluación de un SVM
# para el dataset Digit Recognition (MNIST).
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------
library(dplyr)
library(data.table)
library(tidymodels)
library(tidyverse) 
library(e1071)

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset
#data <- read.csv("digit-recognizer/train.csv")
data <- read.csv("~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train_importance.csv")

# Crea subset
data <- data %>%
  group_by(label) %>%
  slice_sample(prop = 10000 / nrow(data)) %>%
  ungroup()

# ----------------------------------------------------
# 2. Preparación de Datos y División (MODIFICADO para ESCALADO)
# ----------------------------------------------------

# Convertir la columna 'label' a factor
data$label <- as.factor(data$label)

# División en entrenamiento (80%) y prueba (20%)
data_split <- initial_split(
  data, 
  prop = 0.8, 
  strata = label 
)

train_data <- training(data_split)
test_data <- testing(data_split)

# === ESCALADO DE DATOS (CRUCIAL para SVM) ===

# Identificar las columnas de características (todas menos 'label')
feature_cols <- setdiff(names(train_data), "label")

# 1. Crear el objeto de escalado/centrado solo en los datos de entrenamiento
scaler <- preProcess(train_data[, feature_cols], method = c("center", "scale"))

# 2. Aplicar la transformación a los datos de entrenamiento
train_data_scaled <- predict(scaler, train_data)

# 3. Aplicar la misma transformación a los datos de prueba
test_data_scaled <- predict(scaler, test_data)

# Usaremos train_data_scaled y test_data_scaled para el entrenamiento y la predicción.


# ----------------------------------------------------
# 3. Entrenamiento del Modelo
# ----------------------------------------------------

message("Iniciando entrenamiento de SVM: ")

# Entrena el modelo SVM con la llamada directa (usando los datos ESCALADOS)
svm_model <- svm(
  formula = label ~ .,
  data = train_data_scaled, # <-- USAR DATOS ESCALADOS
  kernel = "radial",       # El kernel radial es la mejor opción para este tipo de datos
  cost = 10,               # Parámetro de penalización (ajustable)
  gamma = 0.01             # Parámetro del kernel (ajustable)
)

message("Modelo SVM entrenado con éxito.")

# ----------------------------------------------------
# 4. Evaluación y Extracción de Métricas
# ----------------------------------------------------

# Realizar predicciones sobre el conjunto de prueba (usando los datos ESCALADOS)
test_predictions <- predict(
  svm_model, 
  newdata = test_data_scaled, # <-- USAR DATOS ESCALADOS
  type = "class"
)

# ------------------------------------>
# Cálculo de Métricas
# ------------------------------------>

# Crear la matriz de confusión
cm <- confusionMatrix(
  data = test_predictions, 
  reference = test_data_scaled$label, # <-- USAR LA ETIQUETA DEL DATO ESCALADO
  mode = "everything"                 # Forzar el cálculo de todas las estadísticas
)

# -------------------------------------------------------------------
# Extracción de Métricas
# -------------------------------------------------------------------


model_results <- data.frame(
  Model = "SVM (radial)",
  
  # Extracción de la exactitud global
  Accuracy = cm$overall['Accuracy'], 
  
  # Extracción de las métricas promedio por clase (Mean_Precision, Mean_Recall, Mean_F1)
  Precision = mean(cm$byClass[, "Precision"], na.rm = TRUE),
  Recall = mean(cm$byClass[, "Recall"], na.rm = TRUE),
  F1_Score = mean(cm$byClass[, "F1"], na.rm = TRUE)
)

print(model_results)


# ----------------------------------------------------
# 6. Guardar Resultados
# ----------------------------------------------------

#results_path <- "C:/Users/USUARIO/Desktop/AC_Digit_Recognition/Resultados/metrics_summary.csv"
results_path <- "C:/Users/maria/Documents/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/Resultados/importance_metrics_summary.csv"

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

