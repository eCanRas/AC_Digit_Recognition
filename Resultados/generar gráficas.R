# Cargar los paquetes necesarios
library(tidyverse)
library(tidyr)
library(ggplot2)

# --- Carga de Datos ---

datos_modelos <- read.csv("C:/Users/USUARIO/Desktop/AC_Digit_Recognition/Resultados/metrics_summary.csv")

# Opcional: Mostrar la estructura de los datos cargados
print(head(datos_modelos))

# Convertir los datos a formato largo
datos_largo <- datos_modelos %>%
  # Asegurar que la columna Model sea de tipo factor para la gráfica
  mutate(Model = factor(Model)) %>% 
  # Transformar a formato largo, excepto por la columna 'Model'
  pivot_longer(
    cols = c(Accuracy, Precision, Recall, F1_Score),
    names_to = "Metrica",
    values_to = "Valor"
  )

print(head(datos_largo))

# ------------- Generación del Gráfico de Barras MODIFICADO -------------

# Gráfico de Barras Agrupadas sin etiquetas en el eje X
grafico_barras <- datos_largo %>%
  # El eje X ya no usa 'Model'. Usamos 'Metrica' en el eje X
  # y los modelos se diferencian solo por el color (fill)
  ggplot(aes(x = Metrica, y = Valor, fill = Model)) +
  
  # Usar position="dodge" para agrupar los modelos por métrica
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  
  # Usamos 'Metrica' en el eje X para que haya una barra por métrica,
  # y eliminamos el 'facet_wrap' que agrupaba antes
  # NO usamos facet_wrap, solo las agrupamos por Metrica en el eje X
  
  labs(
    title = "Comparación de Modelos por Métrica de Rendimiento - Entrenados con datos en crudo",
    x = "Métrica de Rendimiento", # Cambiamos la etiqueta del eje X
    y = "Valor de la Métrica",
    fill = "Modelo" # Mantenemos la leyenda por colores
  ) +
  
  # Añadir etiquetas de valor en las barras (ajustado para position_dodge)
  geom_text(aes(label = round(Valor, 3)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  
  theme_minimal() +
  # Asegurar que el eje X no muestre las etiquetas de las sub-barras (Model)
  # ya que ahora el eje X es 'Metrica' y los modelos están agrupados dentro de cada métrica
  theme(legend.position = "bottom")

# Mostrar el gráfico
print(grafico_barras)

# Exportar el Gráfico de Barras Agrupadas a un archivo PNG
ggsave(
  filename = "Comparación de Modelos por Métrica de Rendimiento - Entrenados con datos en crudo.png", # Nombre del archivo de salida
  plot = grafico_barras,            # Objeto de gráfico a exportar
  device = "png",                              # Formato
  width = 10,                                  # Ancho en pulgadas (ajusta si es necesario)
  height = 6,                                  # Alto en pulgadas
  dpi = 300                                    # Resolución (300dpi es un buen estándar)
)

# ------------- Gráfico de Coordenadas Paralelas -------------
grafico_paralelas <- datos_largo %>%
  ggplot(aes(x = Metrica, y = Valor, group = Model, color = Model)) +
  geom_point(size = 3) +
  geom_line(linewidth = 1) +
  labs(
    title = "Perfiles de Rendimiento de los Modelos - Entrenados con datos en crudo",
    x = "Métrica de Rendimiento",
    y = "Valor Normalizado de la Métrica",
    color = "Modelo"
  ) +
  # Añadir etiquetas de valor junto a los puntos
  geom_text(aes(label = round(Valor, 3)), 
            hjust = 1.5, size = 3) + 
  theme_light() +
  theme(legend.position = "right")

# Mostrar el gráfico
print(grafico_paralelas)

# Exportar el Gráfico a un archivo PNG
ggsave(
  filename = "Perfiles de Rendimiento de los Modelos - Entrenados con datos en crudo.png", # Nombre del archivo de salida
  plot = grafico_paralelas,            # Objeto de gráfico a exportar
  device = "png",                              # Formato
  width = 10,                                  # Ancho en pulgadas (ajusta si es necesario)
  height = 6,                                  # Alto en pulgadas
  dpi = 300                                    # Resolución (300dpi es un buen estándar)
)
