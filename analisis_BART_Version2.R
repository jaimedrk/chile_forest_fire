# Cargar las librerías necesarias
library(sf)
library(terra)
library(dbarts)
library(caret)
library(dplyr)
library(ggplot2)

# 1. CARGAR LOS DATOS
# Puntos de ignición y no ignición
ign <- st_read("C:/Users/scray/Documentos/ignicion_1.gpkg")
no_ign <- st_read("C:/Users/scray/Documentos/ignicion_0.gpkg")

# Raster multibanda de covariables
covariables <- rast("C:/Users/scray/Documentos/multi_band_raster.tif")

# 2. EXTRAER VALORES DE COVARIABLES PARA CADA PUNTO
# Añadir una columna de clase (1=ignición, 0=no ignición)
ign$clase <- 1
no_ign$clase <- 0

# Combinar ambos conjuntos de datos
puntos_todos <- rbind(ign, no_ign)

# Extraer valores de covariables para cada punto
valores_covariables <- terra::extract(covariables, vect(puntos_todos))

# Eliminar filas con NA si existen
valores_completos <- na.omit(valores_covariables)
puntos_completos <- puntos_todos[valores_completos$ID,]

# Crear un dataframe con todas las variables para el análisis
datos_analisis <- cbind(
  data.frame(clase = puntos_completos$clase),
  valores_completos[, -1]  # Excluir columna ID
)

# 3. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA (70% - 30%)
set.seed(123)  # Para reproducibilidad
indices <- createDataPartition(datos_analisis$clase, p = 0.7, list = FALSE)
datos_train <- datos_analisis[indices, ]
datos_test <- datos_analisis[-indices, ]

# 4. AJUSTAR EL MODELO BART - CORREGIDO
# Preparar los datos en el formato correcto para dbarts
x.train <- as.matrix(datos_train[, !(names(datos_train) == "clase")])
y.train <- datos_train$clase

x.test <- as.matrix(datos_test[, !(names(datos_test) == "clase")])
y.test <- datos_test$clase

# Configurar el modelo BART con la sintaxis correcta
modelo_bart <- dbarts::bart(
  x.train = x.train,
  y.train = y.train,
  x.test = x.test,
  ntree = 50,        # Número de árboles
  ndpost = 1000,     # Número de muestras posteriores
  nskip = 500,       # Burn-in
  verbose = TRUE,     # Mostrar progreso
  keeptrees = TRUE   # IMPORTANTE: necesario para usar predict()
)

# 5. EVALUAR EL MODELO - ACTUALIZADO PARA TRABAJAR CON LA SALIDA CORRECTA DE BART
# Las predicciones ya están disponibles en modelo_bart$yhat.test
prob_pred <- colMeans(pnorm(modelo_bart$yhat.test))
pred_clases <- ifelse(prob_pred > 0.5, 1, 0)

# Matriz de confusión
conf_matrix <- table(Actual = y.test, Predicho = pred_clases)
print("Matriz de Confusión:")
print(conf_matrix)

# Calcular métricas
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
f1_score <- 2 * precision * recall / (precision + recall)

# Mostrar métricas
cat("\nMétricas de Evaluación:\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")

# 6. ANÁLISIS DE IMPORTANCIA DE VARIABLES - ACTUALIZADO
# Obtener nombres de variables
var_names <- colnames(x.train)

# Calcular la importancia de las variables (un enfoque simple)
var_counts <- modelo_bart$varcount
var_importance <- colMeans(var_counts) / sum(colMeans(var_counts)) * 100

# Crear dataframe para graficar
imp_df <- data.frame(
  Variable = var_names,
  Importancia = var_importance
)
imp_df <- imp_df[order(imp_df$Importancia, decreasing = TRUE), ]

# Graficar importancia de variables
ggplot(imp_df, aes(x = reorder(Variable, Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Importancia de Variables en el Modelo BART",
    x = "Variables",
    y = "Importancia Relativa (%)"
  )
ggsave("importancia_variables.png", width = 8, height = 6)

# 7. EFECTOS PARCIALES DE LAS VARIABLES MÁS IMPORTANTES - ACTUALIZADO
# Tomar las 2 variables más importantes
top_vars <- as.character(imp_df$Variable[1:2])

for (var_idx in 1:length(top_vars)) {
  var <- top_vars[var_idx]
  var_col <- which(colnames(x.train) == var)
  
  # Crear grid para la variable seleccionada
  pd_grid <- x.train[rep(1, 100), ]
  var_seq <- seq(min(x.train[, var_col]), max(x.train[, var_col]), length.out = 100)
  pd_grid[, var_col] <- var_seq
  
  # Predecir con el modelo BART
  pd_pred <- predict(modelo_bart, pd_grid)
  pd_prob <- colMeans(pnorm(pd_pred))
  
  # Dataframe para graficar
  plot_df <- data.frame(
    x = var_seq,
    y = pd_prob
  )
  
  # Graficar efecto parcial
  ggplot(plot_df, aes(x = x, y = y)) +
    geom_line() +
    theme_minimal() +
    labs(
      title = paste("Efecto Parcial de", var),
      x = var,
      y = "Probabilidad de Ignición"
    )
  ggsave(paste0("efecto_parcial_", var, ".png"), width = 8, height = 6)
}

# 8. PREDICCIÓN ESPACIAL (opcional - puede tomar tiempo)
# Esta parte es opcional ya que puede consumir mucha memoria para rasters grandes

# Convertir raster a dataframe para predicción
pred_raster <- TRUE  # Cambiar a FALSE si no quieres la predicción espacial
if (pred_raster) {
  # Crear un dataframe a partir del raster
  raster_df <- as.data.frame(covariables, xy = TRUE)
  names(raster_df)[1:2] <- c("x", "y")
  
  # Eliminar filas con NA
  raster_df <- na.omit(raster_df)
  
  # Si el conjunto de datos es muy grande, tomar una muestra para ahorrar tiempo
  if (nrow(raster_df) > 100000) {
    set.seed(123)
    raster_df <- raster_df[sample(1:nrow(raster_df), 100000), ]
  }
  
  # Preparar matriz para predicción
  raster_matrix <- as.matrix(raster_df[, !(names(raster_df) %in% c("x", "y"))])
  
  # Verificar que las columnas coincidan con las del conjunto de entrenamiento
  # y reorganizar si es necesario
  raster_matrix <- raster_matrix[, colnames(x.train)]
  
  # Predecir con el modelo BART
  pred_spatial <- predict(modelo_bart, raster_matrix)
  prob_spatial <- colMeans(pnorm(pred_spatial))
  
  # Añadir probabilidades al dataframe
  raster_df$prob_ignicion <- prob_spatial
  
  # Crear un raster de probabilidad
  prob_rast <- rast(covariables[[1]])  # Usar el primer raster como plantilla
  
  # Asignar valores
  cells <- cellFromXY(prob_rast, raster_df[, c("x", "y")])
  prob_rast[cells] <- raster_df$prob_ignicion
  
  # Guardar el raster de probabilidad
  writeRaster(prob_rast, "probabilidad_ignicion.tif", overwrite = TRUE)
  
  # Crear un mapa de la probabilidad de incendio
  png("mapa_probabilidad_ignicion.png", width = 800, height = 600)
  plot(prob_rast, main = "Probabilidad de Ignición", col = hcl.colors(100, "Heat"))
  dev.off()
}

# Guardar el modelo para uso futuro
saveRDS(modelo_bart, "modelo_bart_incendios.rds")

cat("\nAnálisis BART completado. Se han guardado gráficos y el modelo.\n")
