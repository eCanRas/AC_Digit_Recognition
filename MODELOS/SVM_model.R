# ----------------------------------------------------------------------
# Script: svm_model.R
# Objetivo: Entrenamiento y evaluación de un SVM
# para el dataset Digit Recognition (MNIST).
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------
# Nota: Asegúrate de tener instalados los paquetes: install.packages(c("data.table", "tidymodels"))
library(data.table)
library(tidymodels)
library(tidyverse) 
library(e1071)

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset
data <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/digit-recognizer/train.csv")

# ----------------------------------------------------
# 2. Preparación de Datos y División (MODIFICADO para ESCALADO)
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
# 3. Entrenamiento del Modelo (Llamada Directa a SVM)
# ----------------------------------------------------

message("Iniciando entrenamiento de SVM. Esto será más lento que Random Forest...")

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

# ------------------------------------
# Cálculo de Métricas (usando yardstick)
# ------------------------------------

# Crear la matriz de confusión
conf_mat_result <- confusionMatrix(
  data = test_predictions, 
  reference = test_data_scaled$label # <-- USAR LA ETIQUETA DEL DATO ESCALADO
)

# Cálculo de la Precisión (Accuracy)
acc <- test_predictions %>% 
  accuracy(truth = label, estimate = .pred_class) %>% 
  pull(.estimate)

# Cálculo de Métricas Multiclase (Precision, Recall, F1-Score)
class_metrics <- conf_mat_result %>% 
  summary() %>%
  filter(.metric %in% c("precision", "recall", "f_meas")) %>%
  filter(.estimator == "macro") # Usar macro para promediar las métricas por clase

# Consolidar los resultados en un data frame
model_results <- data.frame(
  Model = "Decision Tree (rpart)",
  Accuracy = acc,
  Precision = class_metrics %>% filter(.metric == "precision") %>% pull(.estimate),
  Recall = class_metrics %>% filter(.metric == "recall") %>% pull(.estimate),
  F1_Score = class_metrics %>% filter(.metric == "f_meas") %>% pull(.estimate)
)
print(model_results)


# ----------------------------------------------------
# 6. Guardar Resultados
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

message("Resultados guardados en: ", results_path)

