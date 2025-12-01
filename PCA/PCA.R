library(ggplot2)
library(factoextra)
# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Separamos la etiqueta del train (columna 1) para no meterla en el cálculo matemático
train_label <- train[, 1]
train_pixels <- train[, -1]

# ==============================================================================
# 2. LIMPIEZA (Basada en Train, aplicada a ambos)
# ==============================================================================
# Detectamos columnas que son siempre 0 en el train (bordes negros)
cols_sin_info <- apply(train_pixels, 2, var) == 0

# Eliminamos esas columnas de los dos dataframes para que coincidan
train_pixels <- train_pixels[, !cols_sin_info]
test_pixels <- test[, !cols_sin_info] 

# ==============================================================================
# 3. CÁLCULO DEL PCA
# ==============================================================================
pca_model <- prcomp(train_pixels, center = TRUE, scale. = TRUE)
pca_model
summary(pca_model)

# Definimos cuántos componentes guardar
fviz_eig(pca_model)
# La información en estas imágenes está muy dispersa. Los primeros componentes ayudan, pero por sí solos no bastan para explicar gran cosa. 
#Vamos a necesitar muchos más componentes para tener una buena representación

#Contribución de las variables
fviz_contrib(pca_model, choice="var", axes=1)

fviz_contrib(pca_model, choice="var", axes=2)

n_comp <- 150 
#Solo queremos las 150 primeras, que son las que tienen el 95% de la información importante.
#Reducimos apróximadamente el 80,8% el tamaño del dataset
x_pca <- pca_model$x[, 1:n_comp]
# ==============================================================================
# 4. CREACIÓN DE LOS DATASETS
# ==============================================================================
train_pca_final <- data.frame(label = train_label, x_pca)

# Proyectamos los datos de test usando el modelo calculado arriba
test_proyectado <- predict(pca_model, newdata = test_pixels)
test_pca_final <- data.frame(test_proyectado[, 1:n_comp])

# ==============================================================================
# 5. GUARDADO EN DISCO (Dos archivos separados)
# ==============================================================================
write.csv(train_pca_final, "train_pca.csv", row.names = FALSE)
write.csv(test_pca_final, "test_pca.csv", row.names = FALSE)

