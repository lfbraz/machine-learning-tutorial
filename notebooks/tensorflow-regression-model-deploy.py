# Databricks notebook source
# MAGIC %md
# MAGIC #Deployment do modelo de regressão de previsão de consumo de combustível (Tensorflow).
# MAGIC Utilizaremos o modelo criado através do [tutorial](https://www.tensorflow.org/tutorials/keras/regression?hl=pt-br) e persistido com a classe `save_model` (instruções baseadas no tutorial podem ser obtidas no [notebook](https://github.com/lfbraz/azure-databricks/blob/master/notebooks/tensorflow-regression-model.ipynb) *baseado no tutorial*)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Criar/Utilizar Workspace do Azure Machine Learning
# MAGIC Utilizamos o Azure Machine Learning para disponibilização dos endpoints das APIs que irão consumir os modelos de Machine Learning. Para interação com ele, vamos utilizar o `Azure Machine Learning SDK` em Python, em que é possível criar novas Workspaces (ou utilizar Workspaces existentes) para facilitar o processo de deployment.
# MAGIC 
# MAGIC As variáveis `workspace-name`, `resource-group` e `subscription-id` serão coletadas com o uso de [`secrets`](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/secrets) integrados e gerenciados através do serviço de gereciamento de chaves de segurança [`Azure Key Vault`](https://azure.microsoft.com/pt-br/services/key-vault/). Mais detalhes sobre a integração de segurança entre o Azure Databricks e o Azure Key Vault podem ser obtidas no [link](https://docs.microsoft.com/en-us/azure/databricks/security/secrets/secrets).
# MAGIC 
# MAGIC Utilizaremos também o modo de autenticação com [`Service Principals`](https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals), fazendo com que toda a integração de segurança aconteça sem exposição de chaves de segurança, etc. e possibilitando que os scripts possam ser executados posteriormente em um **Pipeline de MLOps**. Maiores detalhes dos diferentes tipos de autenticação pode ser obtido no [link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication).

# COMMAND ----------

import azureml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

workspace_location = "Central US"
workspace_name = dbutils.secrets.get(scope = "azure-key-vault", key = "workspace-name")
resource_group = dbutils.secrets.get(scope = "azure-key-vault", key = "resource-group")
subscription_id = dbutils.secrets.get(scope = "azure-key-vault", key = "subscription-id")

svc_pr = ServicePrincipalAuthentication(
    tenant_id = dbutils.secrets.get(scope = "azure-key-vault", key = "tenant-id"),
    service_principal_id = dbutils.secrets.get(scope = "azure-key-vault", key = "client-id"),
    service_principal_password = dbutils.secrets.get(scope = "azure-key-vault", key = "client-secret"))

workspace = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             auth=svc_pr,
                             exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Criar um entry point
# MAGIC 
# MAGIC O `entry script` somente possui duas funções obrigatórias, o `init()` e o `run()`. Estas funções são utilizadas para iniciar o serviço e executar o modelo utilizando os dados requisitados pelo cliente. Outras funções que podem ser adicionadas são relacionadas ao `loading` ou aplicação de tratamentos necessários para o `input`.

# COMMAND ----------

# MAGIC %%writefile /dbfs/models/score.py
# MAGIC 
# MAGIC import tensorflow as tf
# MAGIC import json
# MAGIC import pandas as pd
# MAGIC import os
# MAGIC 
# MAGIC # Called when the deployed service starts
# MAGIC def init():
# MAGIC     global model
# MAGIC     global train_stats
# MAGIC 
# MAGIC     # Get the path where the deployed model can be found.
# MAGIC     model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './models/')
# MAGIC     
# MAGIC     # Load keras
# MAGIC     model = tf.keras.models.load_model(model_path + 'model-regressao-tensorflow.h5')
# MAGIC     
# MAGIC     # Load train_stats
# MAGIC     train_stats = pd.read_pickle(model_path + "train_stats.pkl")
# MAGIC 
# MAGIC def norm(x):
# MAGIC   return (x - train_stats['mean']) / train_stats['std']
# MAGIC 
# MAGIC # Handle requests to the service
# MAGIC def run(data):
# MAGIC   # JSON request.
# MAGIC   # {"Cylinders":0, "Displacement":0.0, "Horsepower":0.0, "Weight":0.0, "Acceleration":0.5, "Model Year":0, "USA":0.0, "Europe":0.0, "Japan":0.0}
# MAGIC   data = pd.DataFrame([json.loads(data)])
# MAGIC 
# MAGIC   # Apply norm function
# MAGIC   data = norm(data)
# MAGIC 
# MAGIC   # Return the prediction
# MAGIC   prediction = predict(data)
# MAGIC   
# MAGIC   return prediction
# MAGIC 
# MAGIC def predict(data):
# MAGIC   score = model.predict(data)[0][0]
# MAGIC   return {"MPG_PREDICAO": float(score)}

# COMMAND ----------

# MAGIC %md
# MAGIC ##Definir configurações para deploy
# MAGIC Aqui temos que adicionar todos os pacotes necessários para predição do modelo. No caso deste exemplo precisamos do `tensorflow`, `pandas` e `azureml-sdk`.

# COMMAND ----------

from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create the environment
env = Environment(name="tensorflow_env")

conda_dep = CondaDependencies()

# Define the packages needed by the model and scripts
conda_dep.add_conda_package("tensorflow")

# You must list azureml-defaults as a pip dependency
conda_dep.add_pip_package("azureml-defaults")
conda_dep.add_pip_package("keras")
conda_dep.add_pip_package("pandas")

# Adds dependencies to PythonSection of myenv
env.python.conda_dependencies=conda_dep

inference_config = InferenceConfig(entry_script="/dbfs/models/score.py",
                                   environment=env)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Registrar uma imagem com o modelo
# MAGIC Com as configurações definidas, podemos agora registrar o modelo no **Azure Machine Learning**. Será gerado uma nova versão do modelo a cada novo registro, em que através da interface do AML podemos ver os artefatos (modelos, etc.) atrelados ao registro de modelo gerado.

# COMMAND ----------

from azureml.core.model import Model

model_name = 'model-regressao-tensorflow'
model_path = '/dbfs/models'
model_description = 'Modelo de regressão utilizando tensorflow (keras)'

model_azure = Model.register(model_path = model_path,
                             model_name = model_name,
                             description = model_description,
                             workspace = workspace,
                             tags={'Framework': "Tensorflow", 'Tipo': "Regressão"}
                             )

# COMMAND ----------

model_azure

# COMMAND ----------

# MAGIC %md
# MAGIC #Deploy
# MAGIC Agora com a imagem criada, podemos escolher dois tipos de deployment, utilizando ACI (Azure Container Instance) ou AKS (Azure Kubernetes Service).
# MAGIC 
# MAGIC Para cenários de desenvolvimento é indicado o uso do ACI, já para cenários produtivos AKS terá melhores opções quanto a segurança e escalabilidade.
# MAGIC 
# MAGIC Neste exemplo mostraremos como realizar o `deployment` utilizando [ACI](https://azure.microsoft.com/en-us/services/container-instances/) com um container com 1 CPU core e 1GB de memória RAM.

# COMMAND ----------

# MAGIC %md
# MAGIC ##ACI - Azure Container Instance
# MAGIC Abaixo será demonstrado como criar um endpoint utilizando o ACI. Lembrando que estaremos utilizando o Workspace instanciado no passo anterior.

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice
from azureml.exceptions import WebserviceException
from azureml.core.model import Model

ENDPOINT_NAME = 'car-regression-service-dev'

# Remove any existing service under the same name.
try:
    Webservice(workspace, ENDPOINT_NAME).delete()
except WebserviceException:
    pass

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service = Model.deploy(workspace, ENDPOINT_NAME, [model_azure], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)

print('A API {} foi gerada no estado {}'.format(service.scoring_uri, service.state))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Chamada da API
# MAGIC Finalmente faremos o request da API utilizando a variável query_input. A URL da API pode ser obtida através do dev_webservice.scoring_uri que foi gerado no deploy do endpoint.

# COMMAND ----------

import requests
import json
import numpy as np
import pandas as pd

scoring_uri_dev = service.scoring_uri
headers = {'Content-Type':'application/json'}

json_payload = {"Cylinders":8, "Displacement":500, "Horsepower":300, "Weight":3850, "Acceleration":8, "Model Year":70, "USA":1, "Europe":0, "Japan":0}
json_payload = json.dumps(json_payload)

response = requests.post(scoring_uri_dev, data=json_payload, headers=headers)

print(response.status_code)
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Azure Kubernetes Services (AKS)
# MAGIC Para cenários produtivos, uma melhor opção de deploy é utilizando um **AKS (Azure Kubernetes Services)** que traz maiores benefícios quanto a segurança e escalabilidade.
# MAGIC 
# MAGIC Neste cenário com AKS é possível seguirmos com o Deploy de duas formas: Criando um novo cluster AKS ou realizando o deploy em um cluster existente. Neste tutorial demonstraremos a primeira opção.

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget

# Use the default configuration (you can also provide parameters to customize this)
prov_config = AksCompute.provisioning_configuration()

aks_cluster_name = "aks-regression"

# Create the cluster
aks_target = ComputeTarget.create(workspace = workspace, 
                                  name = aks_cluster_name, 
                                  provisioning_configuration = prov_config)

# Wait for the create process to complete
aks_target.wait_for_completion(show_output = True)
print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)

# COMMAND ----------

# MAGIC %md
# MAGIC Agora podemos fazer o deploy do webservice utilizando o novo cluster AKS

# COMMAND ----------

from azureml.core.webservice import AksWebservice

aks_service_name ='car-regression-service-prod'
aks_config = AksWebservice.deploy_configuration()

aks_service = Model.deploy(workspace=workspace,
                           name=aks_service_name,
                           models=[model_azure],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)

# COMMAND ----------

# MAGIC %md
# MAGIC E agora realizar uma chamada da API para testarmos seu funcionamento. Neste caso será necessário utilizar um key para autenticação da API que pode ser obtida através do método `get_keys()`.

# COMMAND ----------

import requests
import json

# Get the keys
scoring_uri_prod = aks_service.scoring_uri
service_key = aks_service.get_keys()[0] if len(aks_service.get_keys()) > 0 else None

#Payload
json_payload = {"Cylinders":8, "Displacement":500, "Horsepower":200, "Weight":3850, "Acceleration":8, "Model Year":70, "USA":1, "Europe":0, "Japan":0}
json_payload = json.dumps(json_payload)

def query_endpoint_example(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
  
  print('URI: {}'.format(scoring_uri))
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=json.loads(json.dumps(inputs)), headers=headers)
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds

query_endpoint_example(scoring_uri=scoring_uri_prod, service_key=service_key, inputs=json_payload)