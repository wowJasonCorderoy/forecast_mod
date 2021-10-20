# Build and deploy

Command to build the application. PLease remeber to change the project name and application name
```
gcloud builds submit --tag gcr.io/gcp-wow-finance-de-lab-dev/forecast_model  --project=gcp-wow-finance-de-lab-dev
```

Command to deploy the application
```
gcloud run deploy --image gcr.io/gcp-wow-finance-de-lab-dev/forecast_model --platform managed  --project=gcp-wow-finance-de-lab-dev --allow-unauthenticated
```