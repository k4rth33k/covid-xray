# Pipeline

## Kubeflow
The Kubeflow project is dedicated to making deployments of machine learning (ML) workflows on Kubernetes simple, portable and scalable. Our goal is not to recreate other services, but to provide a straightforward way to deploy best-of-breed open-source systems for ML to diverse infrastructures. Anywhere you are running Kubernetes, you should be able to run Kubeflow. [More](kubeflow.org)

##  Operations
This project uses two operations/tasks in the pipeline

 - Task for fetching new images from the [COVID-chest-xray dataset](https://github.com/ieee8023/covid-chestxray-dataset).
 - Task for training new model and deploying/updating the API.

![Pipeline](https://i.imgur.com/e8gtO8W.png)
