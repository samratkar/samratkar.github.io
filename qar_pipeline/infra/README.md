# QAR Pipeline Infra (SAM)

Deploy per airline bucket to keep IAM least-privilege.

## Deploy

```bash
cd /Users/samrat.kar/git/prsnl/samratkar.github.io/qar_pipeline/infra
sam build
sam deploy --guided \
  --region us-east-1 \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides AirlineBucketName=qar-aa-data PermissionsBoundaryArn=arn:aws:iam::ACCOUNT_ID:policy/YourBoundary
```

## Manual Triggers

```bash
aws stepfunctions start-execution \
  --state-machine-arn <TrainingStateMachineArn> \
  --input '{"airline":"aa","batch_id":"2026-02-07"}'

aws stepfunctions start-execution \
  --state-machine-arn <SavingsStateMachineArn> \
  --input '{"airline":"aa","batch_id":"2026-02-07"}'
```

## Notes

- Each airline gets its own stack (bucket parameter changes).
- Update job queues/definitions in `template.yaml` if you use different names.
- `MaxConcurrency` in `statemachines/training.asl.json` controls tail parallelism.
- If the fa-ei-cio account uses IAM permissions boundaries or SCPs, attach the boundary to the SAM-created roles after deploy (or add it to the template).
