# ----------------------------------------------------------------------
# Script: rpart_model.R
# Objetivo: Entrenamiento y evaluación de un Árbol de Decisión (rpart)
# para el dataset Digit Recognition (MNIST).
# ----------------------------------------------------------------------

# ----------------------------------------------------
# 1. Carga de Librerías y Carga de Datos
# ----------------------------------------------------
library(dplyr)
library(data.table)
library(tidymodels)
library(tidyverse) 
library(rpart.plot)

# Definir la semilla aleatoria para reproducibilidad
set.seed(42)

# Cargar el dataset
#data <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/digit-recognizer/train.csv")
#data <- read.csv("~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train.csv")
#data <- read.csv("~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train_importance.csv")
data <- read.csv("~/UNIVERSIDAD/CUARTO CURSO/Aprendizaje computacional/Practicas/Digit recognition/AC_Digit_Recognition/digit-recognizer/train_importance.csv")

# Crea subset
data <- data %>%
  group_by(label) %>%
  slice_sample(prop = 10000 / nrow(data)) %>%
  ungroup()

# ----------------------------------------------------
# 2. Preparación de Datos y División
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



# ----------------------------------------------------
# 3. Entrenamiento del Modelo rpart
# ----------------------------------------------------

control <- rpart.control(
  minsplit = 20,
  minbucket = 7,
  cp = 0.001,
  maxdepth = 15,
  xval = 5         
)


arbol <- rpart(
  label ~ .,
  data = train_data,
  method = "class",
  control = control
)
# ----------------------------------------------------
# 4.1. Evaluación y Métricas - Árbol sin podar
# ----------------------------------------------------

predicciones_no_podado <- predict(arbol, newdata = test_data, type = "class")

test_predictions_no_podado <-
  tibble(.pred_class = predicciones_no_podado) %>%
  bind_cols(test_data %>% select(label))

conf_mat_result_no_podado <- conf_mat(test_predictions_no_podado,
                                      truth = label,
                                      estimate = .pred_class)
print(conf_mat_result_no_podado)

acc_no_podado <- test_predictions_no_podado %>%
  accuracy(truth = label, estimate = .pred_class) %>%
  pull(.estimate)

class_metrics_no_podado <- conf_mat_result_no_podado %>%
  summary() %>%
  filter(.metric %in% c("precision", "recall", "f_meas"),
         .estimator == "macro")

precision_no_podado <- class_metrics_no_podado %>%
  filter(.metric == "precision") %>%
  pull(.estimate)

recall_no_podado <- class_metrics_no_podado %>%
  filter(.metric == "recall") %>%
  pull(.estimate)

f1_no_podado <- class_metrics_no_podado %>%
  filter(.metric == "f_meas") %>%
  pull(.estimate)

# ----------------------------------------------------
# 4.2. Poda por mínimo xerror y métricas - Árbol podado
# ----------------------------------------------------

# Índice del mínimo xerror
min_row <- which.min(arbol$cptable[, "xerror"])

# Valor mínimo de xerror y su desviación estándar
min_xerror <- arbol$cptable[min_row, "xerror"]
min_xstd   <- arbol$cptable[min_row, "xstd"]

# Umbral 1-SE: mínimo xerror + 1 * xstd
one_se_limit <- min_xerror + min_xstd

# Filas cuyo xerror está por debajo del umbral
cpt <- arbol$cptable
candidates <- which(cpt[, "xerror"] <= one_se_limit)

# Elegimos el árbol MÁS SIMPLE entre los que cumplen (nsplit más pequeño)
# suele ser el primero en 'candidates'
best_row <- tail(candidates, 1)

cp_opt <- cpt[best_row, "CP"]

arbol_podado <- prune(arbol, cp = cp_opt)


predicciones_podado <- predict(arbol_podado, newdata = test_data, type = "class")

test_predictions_podado <-
  tibble(.pred_class = predicciones_podado) %>%
  bind_cols(test_data %>% select(label))

conf_mat_result_podado <- conf_mat(test_predictions_podado,
                                   truth = label,
                                   estimate = .pred_class)
print(conf_mat_result_podado)

acc_podado <- test_predictions_podado %>%
  accuracy(truth = label, estimate = .pred_class) %>%
  pull(.estimate)

class_metrics_podado <- conf_mat_result_podado %>%
  summary() %>%
  filter(.metric %in% c("precision", "recall", "f_meas"),
         .estimator == "macro")

precision_podado <- class_metrics_podado %>%
  filter(.metric == "precision") %>%
  pull(.estimate)

recall_podado <- class_metrics_podado %>%
  filter(.metric == "recall") %>%
  pull(.estimate)

f1_podado <- class_metrics_podado %>%
  filter(.metric == "f_meas") %>%
  pull(.estimate)


# ----------------------------------------------------
# 5c. Consolidar los resultados en un data frame
# ----------------------------------------------------

model_results <- rbind(
  data.frame(
    Model     = "Decision Tree (rpart) - no podado",
    Accuracy  = acc_no_podado,
    Precision = precision_no_podado,
    Recall    = recall_no_podado,
    F1_Score  = f1_no_podado
  ),
  data.frame(
    Model     = "Decision Tree (rpart) - (min xerror)",
    Accuracy  = acc_podado,
    Precision = precision_podado,
    Recall    = recall_podado,
    F1_Score  = f1_podado
  )
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

