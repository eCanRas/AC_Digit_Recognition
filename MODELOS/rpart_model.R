# ----------------------------------------------------------------------
# Script: rpart_model.R
# Objetivo: Entrenamiento y evaluación de un Árbol de Decisión (rpart)
# para el dataset Digit Recognition (MNIST).
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------
# Nota: Asegúrate de tener instalados los paquetes: install.packages(c("data.table", "tidymodels"))
library(dplyr)
library(data.table)
library(tidymodels)
library(tidyverse) 
library(rpart.plot)

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset
data <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/digit-recognizer/train.csv")

# Crea subset
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


# ----------------------------------------------------
# 3. Definición de la Receta (Preprocesamiento)
# ----------------------------------------------------
# Para un Árbol de Decisión básico, no es estrictamente necesaria la normalización
# o estandarización, pero la incluimos para consistencia en el flujo de trabajo 
# si decides usar un modelo sensible al escalado después.

digit_recipe <- 
  recipe(label ~ ., data = train_data) %>%
  # Se estandarizan las variables numéricas
  step_normalize(all_numeric_predictors()) %>%
  # Eliminar variables que tienen varianza cero (ruido)
  step_zv(all_predictors())


# ----------------------------------------------------
# 4. Definición, Especificación y Entrenamiento del Modelo (PARTE A CAMBIAR)
# ----------------------------------------------------

# Definición del Modelo (Árbol de Decisión)
# set_engine("rpart") usa el paquete rpart, que es el motor de árbol por defecto
model_spec <- 
  decision_tree(
    cost_complexity = 0.001, # Parámetro de poda (ajustable)
    tree_depth = 10,         # Profundidad máxima del árbol (ajustable)
    mode = "classification"
  ) %>%
  set_engine("rpart") 
# [Image of a simple decision tree diagram]


# Creación del Workflow (Combina Receta y Modelo)
digit_wflow <- 
  workflow() %>%
  add_recipe(digit_recipe) %>%
  add_model(model_spec)

# Entrenamiento del Modelo
# Esto aplica la receta (paso 3) y entrena el modelo (paso 4)
model_fit <- fit(digit_wflow, data = train_data)

# ----------------------------------------------------
# 5. Evaluación y Extracción de Métricas
# ----------------------------------------------------

# Generar predicciones sobre el conjunto de prueba
test_predictions <- 
  predict(model_fit, new_data = test_data) %>% 
  bind_cols(test_data %>% select(label)) # Añadir las etiquetas reales

# ------------------------------------
# Cálculo de Métricas (usando yardstick)
# ------------------------------------

# Crear la matriz de confusión
conf_mat_result <- conf_mat(test_predictions, truth = label, estimate = .pred_class)
print(conf_mat_result)

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

