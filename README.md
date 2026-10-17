# Clasificación de Iris con PyTorch

## Objetivo

Implementar un pipeline reproducible de entrenamiento y validación
para clasificar flores Iris mediante una red neuronal sencilla.
El proyecto incluye preparación de datos, entrenamiento con Adam
y seguimiento de la pérdida y la exactitud.

## Dataset

Se utiliza el archivo `data/bezdekIris.data`, con 150 registros
distribuidos en tres especies: setosa, versicolor y virginica,
con 50 registros por especie.

Las variables predictoras son:
- Largo del sépalo.
- Ancho del sépalo.
- Largo del pétalo.
- Ancho del pétalo.

Todas las medidas están expresadas en centímetros. No se encontraron
valores faltantes.

Se conservan los cuatro archivos originales en `data/`.
`iris.names` contiene la documentación y describe las correcciones
presentes en `bezdekIris.data`. No se combina este archivo con
`iris.data`, ya que son versiones del mismo dataset.

## Organización del repositorio

- `data/`: datos originales y documentación del dataset.
- `notebooks/01_pipeline_base_iris.ipynb`: código, resultados y gráficos.
- `requirements.txt`: dependencias del proyecto.
- `README.md`: descripción e instrucciones de ejecución.

## Preparación de los datos

Se realiza una separación estratificada con semilla 42:
- Entrenamiento: 105 registros (70 %), con 35 por especie.
- Validación: 45 registros (30 %), con 15 por especie.

Las variables se estandarizan con `StandardScaler`, ajustado
únicamente sobre entrenamiento. Esa misma transformación se aplica
a validación.

Las etiquetas se codifican como 0 = setosa, 1 = versicolor
y 2 = virginica.

## Modelo y entrenamiento

Se utiliza un MLP definido con `nn.Module`:
- Cuatro entradas.
- Una capa oculta de ocho neuronas con activación ReLU.
- Tres salidas, una por especie.
- 67 parámetros entrenables.

Configuración:
- Función de pérdida: `CrossEntropyLoss`.
- Optimizador: Adam.
- Tasa de aprendizaje: 0.01.
- Épocas: 100.
- Un lote completo de entrenamiento por época.
- Semilla: 42.

El ciclo de entrenamiento incluye forward, cálculo de pérdida,
limpieza de gradientes, backward y actualización de parámetros.
Los gradientes se calculan mediante autograd con `loss.backward()`.

La evaluación utiliza `model.eval()` y `torch.no_grad()`.
Se registran la pérdida y la exactitud de ambos conjuntos después
de cada actualización, sin entrenar sobre validación.

El dispositivo se selecciona automáticamente entre CUDA, MPS y CPU.

## Resultados

Ejecución realizada en CPU con PyTorch 2.11.0+cpu.

| Métrica | Entrenamiento | Validación |
|---|---:|---:|
| Pérdida en la primera época | 1.1173 | 1.1130 |
| Pérdida en la última época | 0.0852 | 0.1514 |
| Exactitud final | 97,14 % | 93,33 % |

El modelo clasificó correctamente 42 de los 45 registros de validación.

La pérdida disminuyó en ambos conjuntos durante el entrenamiento.
No se observó un aumento sostenido de la pérdida de validación,
por lo que esta ejecución no presenta señales claras de sobreajuste
marcado.

Los resultados corresponden a una única partición de un dataset
pequeño. La validación no constituye una evaluación independiente
sobre un conjunto de prueba adicional.

## Ejecución en Google Colab

1. Abrir Google Colab.
2. Seleccionar la opción para abrir un notebook desde GitHub.
3. Introducir la URL del repositorio:
   https://github.com/martingnegro/iris-classification-pytorch
4. Seleccionar `notebooks/01_pipeline_base_iris.ipynb`.
5. Conectar el entorno y ejecutar todas las celdas en orden.

El notebook carga el dataset directamente desde el repositorio
público. No requiere montar Google Drive ni subir datos manualmente.
Se necesita conexión a Internet.

La CPU es suficiente para ejecutar este proyecto.

## Dependencias y reproducibilidad

Las versiones de las dependencias se encuentran en `requirements.txt`.
Para instalarlas en un entorno Python compatible:

    python -m pip install -r requirements.txt

Para ejecutar el notebook localmente también se necesita un entorno
como Jupyter o un editor compatible con archivos `.ipynb`.

Se fijan semillas de aleatoriedad para favorecer la reproducibilidad.
Pueden existir pequeñas diferencias entre dispositivos y versiones
de las librerías.

Para repetir el experimento desde cero, ejecutar todas las celdas
en orden, incluyendo la inicialización del modelo y del optimizador.
