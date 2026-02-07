# QAR Pipeline Infra (Terraform)

This mirrors the SAM stack and deploys the Lambdas, Step Functions, EventBridge rule, and CloudWatch alarms.

## Prereqs

- Zip the Lambda files:

```bash
cd /Users/samrat.kar/git/prsnl/samratkar.github.io/qar_pipeline/lambdas
zip parse_key.zip parse_key.py
zip update_manifest.zip update_manifest.py
zip load_manifest.zip load_manifest.py
```

## Deploy

```bash
cd /Users/samrat.kar/git/prsnl/samratkar.github.io/qar_pipeline/infra_terraform
make zips
terraform init
terraform apply \
  -var="airline_bucket_name=qar-aa-data" \
  -var="permissions_boundary_arn=arn:aws:iam::ACCOUNT_ID:policy/YourBoundary"
```

## Per-Airline State Isolation

Use Terraform workspaces to isolate per-airline deployments:

```bash
terraform workspace new aa
terraform apply -var="airline_bucket_name=qar-aa-data"

terraform workspace new bb
terraform apply -var="airline_bucket_name=qar-bb-data"
```

## Manual Triggers

```bash
aws stepfunctions start-execution \
  --state-machine-arn <training_state_machine_arn> \
  --input '{"airline":"aa","batch_id":"2026-02-07"}'

aws stepfunctions start-execution \
  --state-machine-arn <savings_state_machine_arn> \
  --input '{"airline":"aa","batch_id":"2026-02-07"}'
```
