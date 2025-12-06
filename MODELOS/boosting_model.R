# ----------------------------------------------------------------------
# Script: boosting_model.R
# Objetivo: Entrenamiento y evaluación de un modelo de Boosting (AdaBoost)
# utilizando el paquete 'adabag' para el dataset Digit Recognition (MNIST) con PCA.
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------


library(dplyr)
library(data.table)
library(tidymodels)
library(tidyverse)
library(caret)      # Para confusionMatrix
library(rpart)      # Árbol base necesario para 'adabag'
library(adabag)     # Implementa AdaBoost.M1 / SAMME

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

data <- read.csv(
  "~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train_pca.csv"
)

# Crea subset
data <- data %>%
  group_by(label) %>%
  slice_sample(prop = 5000 / nrow(data)) %>%
  ungroup()

# ----------------------------------------------------
# 2. Preparación de Datos y División
# ----------------------------------------------------

# Convertir la columna 'label' a factor
data$label <- as.factor(data$label)

# División en entrenamiento (80%) y prueba (20%) con estratificación
data_split <- initial_split(
  data,
  prop = 0.8,
  strata = label
)

train_data <- training(data_split)
test_data  <- testing(data_split)

# ----------------------------------------------------
# 3. Entrenamiento del Modelo Boosting (adabag)
# ----------------------------------------------------

message("Iniciando entrenamiento de Boosting (adabag). Esto puede tardar varios minutos...")

# Entrenamiento de AdaBoost.M1 / SAMME con árboles rpart como clasificadores débiles
# mfinal = número de iteraciones / árboles del ensamble
# boos = TRUE permite re-muestreo con pesos actualizados
# coeflearn = "Breiman" o "Zhu" (regla de actualización de pesos)
boost_model <- boosting(
  formula = label ~ .,
  data    = train_data,
  boos    = TRUE,
  mfinal  = 10,
  coeflearn = "Breiman",
  control = rpart.control(
    maxdepth = 5,
    minsplit = 20,
    cp = 0.01
  )
)

message("Modelo Boosting (adabag) entrenado con éxito.")

# ----------------------------------------------------
# 4. Evaluación y Extracción de Métricas
# ----------------------------------------------------

# 1. Realizar predicciones sobre el conjunto de prueba
boost_pred <- predict(
  boost_model,
  newdata = test_data
)

# 'boost_pred$class' contiene la clase predicha para cada observación
test_predictions <- as.factor(boost_pred$class)

# 2. Asegurar niveles idénticos entre predicciones y etiquetas reales
levels(test_predictions) <- levels(test_data$label)

# 3. Crear la matriz de confusión con todas las estadísticas
cm <- confusionMatrix(
  data      = test_predictions,
  reference = test_data$label,
  mode      = "everything"
)

# 4. Extracción de Métricas (macro-promedio por clase)

# Matriz de métricas por clase
by_class <- cm$byClass

# Precision (Pos Pred Value) y Recall (Sensitivity) por clase
precision_class <- by_class[, "Pos Pred Value"]
recall_class    <- by_class[, "Sensitivity"]

# F1 por clase
f1_class <- 2 * precision_class * recall_class / (precision_class + recall_class)

# Promedios macro (ignorando posibles NA)
mean_precision <- mean(precision_class, na.rm = TRUE)
mean_recall    <- mean(recall_class,    na.rm = TRUE)
mean_f1        <- mean(f1_class,        na.rm = TRUE)

model_results <- data.frame(
  Model     = "Boosting (adabag)",
  Accuracy  = cm$overall["Accuracy"],  # exactitud global
  Precision = mean_precision,          # macro-precision
  Recall    = mean_recall,             # macro-recall
  F1_Score  = mean_f1                  # macro-F1
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

message("Resultados de Boosting (adabag) guardados en: ", results_path)
