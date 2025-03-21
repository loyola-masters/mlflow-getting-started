# Getting Started with MLFlow
```
Modelo de ejemplo:
MNIST dataset entrenado con Pytorch
(inicialmente 2 capas densas de 256 by 128 neuronas respectivamente)
```
Este repositorio contiene un paso a paso que aborda las principales características de MLFlow y se basa en un [tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html) de la documentación oficial de la herramienta.

___
## Paso a paso
La carpeta `scripts` contiene los siguientes scripts:

* [dataset.py](scripts/dataset.py): Contiene la función que carga los dataloaders que se utilizarán para entrenar y probar el modelo.
* [model.py](scripts/model.py): Contiene la función que crea el modelo.
* [train.py](scripts/train.py): Script principal con la función de entrenamiento del modelo. Dentro de este script registramos las métricas y los parámetros con MLFlow tracking.

Setup del entorno:

```bash
conda create -n mlflow python=3.9
conda activate mlflow
pip install -r requirements.txt
```

## MLFlow Tracking
MLFlow tracking es un componente que permite al usuario crear y gestionar experimentos. Proporciona una API y una interfaz de usuario que permite guardar y visualizar métricas, parámetros, modelos y artefactos.

Los resultados de los experimentos realizados con MLFlow se almacenan localmente o en un servidor remoto. Por defecto, los resultados se almacenan localmente en archivos dentro del directorio `mlruns`.

Para más información, consulta la [documentación](https://mlflow.org/docs/latest/tracking.html).

#### 1.1. Entrenando un modelo

Primero, entrenaremos una red neuronal simple con PyTorch. Para ello, ejecuta el siguiente comando:
```bash
$ python scripts/train.py --epochs <epochs> --learning_rate <learning_rate> <batch_size>
```
Los valores por defecto son:
- batch_size = 64
- learning_rate = 0.01
- epochs = 25
Como ejemplo:
```
python scripts/train.py --epochs 25 --learning_rate 0.01 --batch_size 64
```

Dentro de `train.py` encontrarás el siguiente fragmento de código:

```python
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("device", device)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test_acc, test_loss = test(test_dataloader, model, loss_fn)

            mlflow.log_metric("test_acc", test_acc, step=t)
            mlflow.log_metric("test_loss", test_loss, step=t)

            if test_acc > best_acc:
                best_acc = test_acc
                mlflow.log_metric("best_acc", best_acc, step=t)
                mlflow.pytorch.log_model(model.cpu(), "model", signature=signature, code_paths=["scripts/model.py"])
                model.to(device)
```

En este fragmento de código, puedes observar que los parámetros se registran con el método `log_param`, las métricas con `log_metric` y el modelo se guarda con `log_model`. Además de estos, existen otros métodos útiles, como `log_artifact` para guardar archivos en el directorio `artifacts` y `log_image` para guardar imágenes. Para más información, consulta la [documentación](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html).

#### 1.2. Visualizando los resultados

Después del entrenamiento, el script creará una carpeta llamada `mlruns` en el directorio actual. Dentro de esta carpeta se almacenarán los archivos de registro del experimento. MLFlow proporciona una interfaz de usuario para vis
ualizar los resultados del experimento. Para acceder a ella, abre el terminal y ejecuta:

```bash
$ mlflow ui
```

Ahora, en un navegador, puedes acceder a la interfaz de usuario de MLFlow en la dirección [http://127.0.0.1:5000](http://127.0.0.1:5000).

Verás una tabla con los resultados del experimento similar a esta:

Al seleccionar el experimento deseado, tendrás acceso a los gráficos y artefactos generados durante el entrenamiento del modelo.

### 2. MLFlow Projects

Ahora que tenemos nuestro código de entrenamiento, podemos crear un proyecto en un formato que sea reproducible en cualquier plataforma usando MLFlow Projects. Este tipo de proyecto es útil si deseas entrenar un modelo en la nube, por ejemplo, en Databricks.

#### 2.1. Preparando el MLproject

Para crear un nuevo proyecto, debemos agregar un archivo llamado `MLproject`, que contendrá las especificaciones del proyecto.

```python
name: MNIST Tutorial Project

conda_env: conda.yaml

# O si usas pyenv
# python_env: conda.yaml

entry_points:
  main:
    parameters:
      epochs: {type: int, default: 3}
      learning_rate: {type: float, default: 0.01}
      batch_size: {type: int, default: 64}
    command: "python scripts/train.py {epochs} {learning_rate} {batch_size}"
```

Para más información sobre las especificaciones del archivo `MLproject`, consulta [aquí](https://mlflow.org/docs/latest/projects.html).

#### 2.2. Ejecutando nuestro proyecto

Podemos ejecutar nuestro proyecto en el entorno local o dejar que MLFlow prepare un entorno para nuestro proyecto usando `pyenv`. Si deseas usar `pyenv`, consulta [esta guía](https://dev.to/womakerscode/instalando-o-python-com-o-pyenv-2dc7) para instalarlo.

Para ejecutar el proyecto:
```bash
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

mlflow run . --env-manager=local --experiment-name "MNIST_experiment" --run-name "run_$timestamp" -P epochs=5 -P learning_rate=0.01 -P batch_size=64
```

> Nota: Los parámetros son opcionales. Para pasarlos, usa el argumento `-P` seguido del nombre del parámetro y el valor deseado.

# ANEXO 1: Instalación manual de Pytorch con soporte para GPU

Fichero `requirements.txt`:
```
pip==22.1.2
setuptools==62.6.0
wheel==0.37.1
mlflow[extras]
cloudpickle==2.1.0
```
Instala Pytorch con soporte para CUDA manualmente:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

# ANEXO 2: Almacenamiento de modelos en MLflow
MLflow guarda los modelos entrenados en **artifacts**, que pueden estar en un almacenamiento **local** o en una ubicación **remota** dependiendo de la configuración del **Tracking URI**.

---

### **Ubicación donde MLflow guarda el modelo**
#### 🔹 **Por defecto (Local)**
Si **no has configurado un servidor remoto**, MLflow guarda los modelos en la carpeta `mlruns` dentro del directorio donde ejecutaste el script.  

- **Ruta estándar en local**:
  ```
  ./mlruns/{experiment_id}/{run_id}/artifacts/model/
  ```
  Donde:
  - **`experiment_id`**: ID del experimento en MLflow.
  - **`run_id`**: ID único de la ejecución.
  - **`artifacts/model/`**: Carpeta donde se almacena el modelo.

- Puedes verificarlo ejecutando:
  ```bash
  ls mlruns/0/<run_id>/artifacts/model/
  ```

#### 🔹 **Si usas una base de datos o almacenamiento remoto**
Si has configurado un **servidor remoto (MLflow Tracking Server)** y has cambiado el `MLFLOW_TRACKING_URI`, el modelo se guarda en la ubicación configurada, como:
- **S3 (Amazon Web Services)**
- **Azure Blob Storage**
- **Google Cloud Storage**
- **Base de datos SQLite, MySQL, PostgreSQL**
- **Carpeta personalizada en otro servidor**

Puedes ver la ubicación configurada ejecutando:
```python
import mlflow
print(mlflow.get_tracking_uri())  # Muestra la ubicación de almacenamiento
```

---

### **Cómo verificar dónde se ha guardado un modelo**
Si ya has registrado un modelo en MLflow, puedes recuperar su ruta con:

```python
import mlflow

# Obtener la última ejecución registrada
last_run = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]

# Obtener la ruta del modelo
model_uri = f"mlruns/{last_run.experiment_id}/{last_run.run_id}/artifacts/model"
print(f"El modelo se encuentra en: {model_uri}")
```

Si usas un **MLflow Tracking Server remoto**, puedes obtener la URI del modelo con:
```python
import mlflow
logged_model = "runs:/<run_id>/model"
print(mlflow.artifacts.download_artifacts(logged_model))
```

---

### **Cómo cargar un modelo guardado por MLflow**
#### 🔹 **Desde almacenamiento local**
Si el modelo está en `mlruns/`, puedes cargarlo con:
```python
import mlflow.pytorch

model = mlflow.pytorch.load_model("mlruns/0/<run_id>/artifacts/model")
```

#### 🔹 **Desde MLflow Server**
Si el modelo está en un servidor remoto o en la nube:
```python
model = mlflow.pytorch.load_model("runs:/<run_id>/model")
```

#### 🔹 **Desde Model Registry**
Si has registrado el modelo en el **Model Registry**, puedes cargarlo directamente con:
```python
model = mlflow.pytorch.load_model("models:/my_model/latest")
```
Donde `"my_model/latest"` es el nombre del modelo en el **Model Registry**.

---

### **Resumen**
✔ Por defecto, MLflow guarda los modelos en `./mlruns/{experiment_id}/{run_id}/artifacts/model/`
✔ Si usas un servidor remoto, el modelo se almacena en la ubicación configurada (`S3, GCS, Azure, MySQL, etc.`)
✔ Puedes recuperar la ruta con `mlflow.get_tracking_uri()` y cargar el modelo con `mlflow.pytorch.load_model()`

# ANEXO 3: Opciones para ejecutar el Proyecto de MLflow
Para ejecutar el proyecto con **Conda** en lugar de `pyenv`, usa el siguiente comando:

```bash
mlflow run git@github.com:loyola-masters/mlflow-getting-started.git --env-manager=conda -P epochs=5 -P learning_rate=0.01 -P batch_size=64
```

- **`--env-manager=conda`** → Especifica que MLflow debe usar **Conda** 
- **`-P epochs=5`** → Pasa el argumento `epochs` con el valor `5`.
- **`-P learning_rate=0.01`** → Pasa el argumento `learning_rate` con el valor `0.01`.
- **`-P batch_size=64`** → Pasa el argumento `batch_size` con el valor `64`.

### **Alternativa: Ejecutar en un entorno local ya configurado**
Si ya creaste y activaste el entorno Conda manualmente, puedes ejecutar directamente:

```bash
conda activate mlflow-project-env  # Activar el entorno Conda

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
mlflow run . --env-manager=local --experiment-name "MNIST_experiment" --run-name "run_$timestamp" -P epochs=5 -P learning_rate=0.01 -P batch_size=64
```

---

### **📌 Resumen**
| Método | Comando |
|--------|---------|
| **Ejecutar con Conda** | `mlflow run <repo> --env-manager=conda -P epochs=5 -P learning_rate=0.01 -P batch_size=64` |
| **Ejecutar en un entorno local** | `mlflow run . --env-manager=local -P epochs=5 -P learning_rate=0.01 -P batch_size=64` |
