# covid-xray
#### A fully automated, deep learning powered REST API to detect COVID-19 from X-ray images with automated retraining.

**Automation:The model will be re-trained every three days with the new data and the API will be refreshed**

## Usage
You can use curl to test the API or any other API testing tool like postman

    curl -i \
        -X POST \
        -H "Content-Type: multipart/form-data" \
        -F "image=@your_image_name" 
        http://api-covid.eastus2.azurecontainer.io/api
 

## Architecture

 - The REST API is a flask app which runs on an Azure Container Instance
 - The Automation Pipeline is a Kubeflow pipeline which runs on Azure Kubernetes Cluster
 - Both the entities communicate via a Azure file share which is attached as a volume to the API container.
 
![Our architecture](https://i.imgur.com/zfjxm3x.jpg)



