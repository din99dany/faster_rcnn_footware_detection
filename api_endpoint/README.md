# Content

## Scoring script
A scoring script is a script that performs the prediction on new, unseen data based on a pre-trained machine learning model. The goal of the scoring script is to take in new data as input, perform any necessary preprocessing, and then use the pre-trained model to generate predictions.

## Scoring script
A python script that can be used to test the api. In order to use, please update `<THE URL OF THE ENDPOINT GOES HERE>`, `<THE API KEY GOES HERE>` and `<PATH TO IMAGE>` with values specific for your case

# How to deploy to Azure ML
Here is a step-by-step guide to deploy a scoring script on Azure ML:

1. Create a new Azure ML workspace: You can do this through the Azure portal, the Azure ML Studio, or by using the Azure ML SDK.

2. Train a machine learning model: Train a machine learning model using one of the many available algorithms in Azure ML or by using a custom script.

3. Register the model: Register the model with the Azure ML workspace so that it can be accessed and used for deployment.

4. Write the scoring script: The scoring script should take in the new data and preprocess it if necessary, and then use the pre-trained model to generate predictions.

5. Deploy the scoring script: You can deploy the scoring script as a web service using Azure ML's deployment tools. You can choose from several deployment options, such as Azure Container Instances, Azure Kubernetes Service, or Azure Functions.

6. Test the deployed web service: Once the web service is deployed, you can test it by sending requests to the endpoint and verifying the predictions.

Deploying a scoring script on Azure ML involves training a machine learning model, registering it with the workspace, writing a scoring script to perform predictions on new data, and deploying the scoring script as a web service.