library(randomForest)
library(caret)


train <- read.csv("digit-recognizer/train.csv")
test <- read.csv("digit-recognizer/test.csv")


# ==============================================================================
# PASO 1: CALCULAR LA IMPORTANCIA
# ==============================================================================
# Para cumplir el objetivo de "bajo consumo de recursos", no usamos todo el dataset
# para calcular la importancia, sino una muestra representativa (ej. 10.000 filas).
set.seed(123)
muestra_filas <- sample(1:nrow(train), 5000)
train_sample <- train[muestra_filas, ]

# Aseguramos que la etiqueta es un factor (Solución a tu error anterior)
# La columna 1 es "label" [cite: 19]
train_sample[[1]] <- as.factor(train_sample[[1]])

cat("Calculando importancia de variables con Random Forest...\n")

# Entrenamos un bosque pequeño (ntree=50) solo para ver qué variables pesan más
# importance = TRUE es obligatorio para obtener las métricas
rf_importance <- randomForest(label ~ ., 
                              data = train_sample, 
                              ntree = 50, 
                              importance = TRUE) #obliga al algoritmo a guardar estadísticas sobre qué variables fueron más útiles durante el entrenamiento.


# Extraemos la tabla de importancia
# MeanDecreaseGini nos dice cuánto limpia la clasificación cada píxel
imp_data <- importance(rf_importance) #tabla de estadísticas del modelo entrenado
var_imp_df <- data.frame(Variable = rownames(imp_data), 
                         Importance = imp_data[, "MeanDecreaseGini"]) #cuánto más puro/claro se vuelve el resultado gracias a este píxel

# Ordenamos de mayor a menor importancia
var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]


###################################3
# Asumimos que ya tienes 'rf_rapido' o 'var_imp_df' del paso anterior
# Si no, ejecuta el bloque de cálculo de importancia de mi mensaje anterior primero.

# Graficar la importancia de las 784 variables
plot(var_imp_df$Importance, 
     type = "l", 
     main = "Caída de Importancia de Variables", 
     xlab = "Número de Variables (Píxeles)", 
     ylab = "Importancia (Impurity)",
     col = "blue", lwd = 2)

# Dibujamos una línea en 250 para ver dónde cae
abline(v = 300, col = "red", lty = 2)

###################################



# ==============================================================================
# PASO 2: SELECCIÓN Y REDUCCIÓN
# ==============================================================================
# Decidimos quedarnos con las mejores N variables.
# Nos quedamos con 300 variables, que explicarían aproximadamente el 85-90% del dataset
top_n <- 300 
mejores_variables <- as.character(var_imp_df$Variable[1:top_n])

cat("Seleccionadas las", top_n, "variables más importantes.\n")

# Comprobación visual
print((mejores_variables)[1:10])

# Creamos los nuevos datasets reducidos
# IMPORTANTE: No olvides incluir la columna "label" (variable objetivo) en el train
train_reducido_imp <- train[, c("label", mejores_variables)]
test_reducido_imp  <- test[, mejores_variables]



# ==============================================================================
# PASO 3: GUARDAR Y FINALIZAR
# ==============================================================================
write.csv(train_reducido_imp, "train_importance.csv", row.names = FALSE)
write.csv(test_reducido_imp, "test_importance.csv", row.names = FALSE)

cat("Datos reducidos guardados. Ahora entrena tu modelo final con 'train_reducido_imp'.")