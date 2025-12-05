# ----------------------------------------------------------------------
# Script: rf_model.R
# Objetivo: Entrenamiento y evaluación de un Random Forest (rf)
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
library(caret)
library(randomForest)

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset
#data <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/digit-recognizer/train.csv")
data <- read.csv("~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train_importance.csv")

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
# Nota: Ahora debes usar 'trainData' en la Sección 3 para entrenar 
# y 'testData' en la Sección 4 para obtener las métricas de prueba.
# ----------------------------------------------------

# ----------------------------------------------------
# 3. Entrenamiento del Modelo (Llamada Directa a randomForest)
# ----------------------------------------------------

message("Iniciando entrenamiento de Random Forest. Esto puede tardar varios minutos...")

# Entrena el modelo Random Forest con la llamada directa
rf_model <- randomForest(
  formula = label ~ .,
  data = train_data,
  ntree = 50,     # Número de árboles a construir (ajustable)
  mtry = 28,       # Número de variables muestreadas aleatoriamente en cada split (p/3 para clasificación)
  importance = TRUE # Calcular la importancia de las variables (opcional)
)

message("Modelo Random Forest entrenado con éxito.")

# ----------------------------------------------------
# 4. Evaluación y Extracción de Métricas (SOLUCIÓN DEFINITIVA)
# ----------------------------------------------------

# 1. Realizar predicciones sobre el conjunto de prueba
test_predictions <- predict(
  rf_model, 
  newdata = test_data, 
  type = "class" 
)

# 2. ASEGURAR NIVELES IDÉNTICOS (Corrección del error NA)
# Forzamos que los niveles del factor predicho sean los mismos que los reales.
levels(test_predictions) <- levels(test_data$label)


# 3. Crear la matriz de confusión, forzando el cálculo de todas las estadísticas
cm <- confusionMatrix(
  data = test_predictions,          # Predicciones con niveles corregidos
  reference = test_data$label,       # Etiquetas reales
  mode = "everything" 
)

# 4. Extracción de Métricas (macro-promedio por clase)

# Matriz de métricas por clase
by_class <- cm$byClass

# Precision (Pos Pred Value) y Recall (Sensitivity) por clase
precision_class <- by_class[ , "Pos Pred Value"]
recall_class    <- by_class[ , "Sensitivity"]

# F1 por clase
f1_class <- 2 * precision_class * recall_class / (precision_class + recall_class)

# Promedios macro (ignorando posibles NA)
mean_precision <- mean(precision_class, na.rm = TRUE)
mean_recall    <- mean(recall_class,    na.rm = TRUE)
mean_f1        <- mean(f1_class,        na.rm = TRUE)

model_results <- data.frame(
  Model     = "Random Forest (randomForest)",
  Accuracy  = cm$overall["Accuracy"],  # exactitud global
  Precision = mean_precision,          # macro-precision
  Recall    = mean_recall,             # macro-recall
  F1_Score  = mean_f1                  # macro-F1
)

print(model_results)



# ----------------------------------------------------
# 5. Guardar Resultados
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
