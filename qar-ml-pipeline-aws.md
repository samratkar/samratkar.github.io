# QAR ML Pipeline (AWS + Linux)

This document defines a batch-oriented pipeline for QAR CSV ingestion via AWS Transfer Family, preprocessing, manual model training, and manual savings report generation. It is optimized for large CSVs (6–10 GB per airline batch) and parallel processing across airlines and tails.

## Goals

- Ingest QAR CSVs over SFTP into S3.
- Auto-run a pre-processor on each upload and produce Markdown/PDF reports.
- Allow a manual trigger to run model training per airline batch.
- Run training in parallel per tail.
- Allow a manual trigger to generate savings report per airline batch after all tails are trained.
- Keep outputs in S3 with stable prefixes.

## S3 Prefix Layout (Per-Airline Buckets)

- Input CSV: `s3://qar-{airline}-data/incoming/{tail}/{batch_id}/file.csv`
- Preprocess outputs:
- `s3://qar-{airline}-data/preprocess/{batch_id}/report.md`
- `s3://qar-{airline}-data/preprocess/{batch_id}/report.pdf`
- `s3://qar-{airline}-data/preprocess/{batch_id}/manifest.json`
- Training outputs:
- `s3://qar-{airline}-data/processed/{tail}/{batch_id}/model-output.json`
- `s3://qar-{airline}-data/processed/{tail}/{batch_id}/model-report.pdf`
- Savings output:
- `s3://qar-{airline}-data/savings/{batch_id}/savings-report.pdf`

## Batch ID

`batch_id` identifies an airline’s batch and is included in all paths. It can be a date-based value like `2026-02-07` or a UUID.

## Compute Choice (Cost + Throughput + Low Ops)

Use **AWS Batch on EC2 Spot** with an **On-Demand fallback**.

- Best $/throughput for large batch files (6–10 GB).
- Scales horizontally with minimal ops (managed compute environment).
- Lower cost than Fargate for large, long-running jobs.
- Use `SPOT_CAPACITY_OPTIMIZED` allocation and a small On-Demand baseline for stability.

## Preprocess Workflow (Auto Trigger)

Trigger: S3 `PutObject` events for `incoming/{tail}/{batch_id}/*.csv`

High-level steps:
- Validate the CSV key, parse `tail`, `batch_id` and resolve `airline` from the bucket name.
- Run pre-processor job (AWS Batch or ECS).
- Write report files.
- Update `manifest.json` for `{airline}/{batch_id}`.

## Training Workflow (Manual Trigger)

Trigger: Manual StartExecution of the Step Functions state machine with `{airline, batch_id}` input.

High-level steps:
- Read `manifest.json` for the airline batch.
- Map over tails in parallel.
- Run training job per tail.
- Write per-tail JSON/PDF.
- Optionally update batch status.

## Savings Workflow (Manual Trigger)

Trigger: Manual StartExecution with `{airline, batch_id}` input.

High-level steps:
- Load all tail JSON outputs for airline batch.
- Load corresponding input datasets if needed.
- Compute savings and write a single airline PDF report.

## Step Functions State Machines (ASL)

### Preprocess

```json
{
  "Comment": "QAR preprocess on S3 PutObject",
  "StartAt": "ExtractKey",
  "States": {
    "ExtractKey": {
      "Type": "Pass",
      "Parameters": {
        "bucket.$": "$.detail.bucket.name",
        "key.$": "$.detail.object.key"
      },
      "Next": "ParseKey"
    },
    "ParseKey": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:qar-parse-key",
      "Next": "RunPreprocess"
    },
    "RunPreprocess": {
      "Type": "Task",
      "Resource": "arn:aws:states:::batch:submitJob.sync",
      "Parameters": {
        "JobName.$": "States.Format('qar-preprocess-{}', $.parsed.tail)",
        "JobQueue": "qar-preprocess-queue",
        "JobDefinition": "qar-preprocess-job:1",
        "ContainerOverrides": {
          "Environment": [
            {"Name": "INPUT_S3", "Value.$": "$.parsed.s3_uri"},
            {"Name": "AIRLINE", "Value.$": "$.parsed.airline"},
            {"Name": "TAIL", "Value.$": "$.parsed.tail"},
            {"Name": "BATCH_ID", "Value.$": "$.parsed.batch_id"}
          ]
        }
      },
      "Next": "UpdateManifest"
    },
    "UpdateManifest": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:qar-update-manifest",
      "End": true
    }
  }
}
```

### Training (Manual)

```json
{
  "Comment": "QAR training per tail (manual trigger)",
  "StartAt": "LoadManifest",
  "States": {
    "LoadManifest": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:qar-load-manifest",
      "Next": "TrainTails"
    },
    "TrainTails": {
      "Type": "Map",
      "ItemsPath": "$.tails",
      "MaxConcurrency": 10,
      "Parameters": {
        "airline.$": "$.airline",
        "batch_id.$": "$.batch_id",
        "tail.$": "$$.Map.Item.Value.tail",
        "input_s3.$": "$$.Map.Item.Value.input_s3"
      },
      "Iterator": {
        "StartAt": "RunTraining",
        "States": {
          "RunTraining": {
            "Type": "Task",
            "Resource": "arn:aws:states:::batch:submitJob.sync",
            "Parameters": {
              "JobName.$": "States.Format('qar-train-{}', $.tail)",
              "JobQueue": "qar-train-queue",
              "JobDefinition": "qar-train-job:1",
              "ContainerOverrides": {
                "Environment": [
                  {"Name": "INPUT_S3", "Value.$": "$.input_s3"},
                  {"Name": "AIRLINE", "Value.$": "$.airline"},
                  {"Name": "TAIL", "Value.$": "$.tail"},
                  {"Name": "BATCH_ID", "Value.$": "$.batch_id"}
                ]
              }
            },
            "End": true
          }
        }
      },
      "End": true
    }
  }
}
```

### Savings (Manual)

```json
{
  "Comment": "QAR savings per airline batch (manual trigger)",
  "StartAt": "RunSavings",
  "States": {
    "RunSavings": {
      "Type": "Task",
      "Resource": "arn:aws:states:::batch:submitJob.sync",
      "Parameters": {
        "JobName.$": "States.Format('qar-savings-{}', $.airline)",
        "JobQueue": "qar-savings-queue",
        "JobDefinition": "qar-savings-job:1",
        "ContainerOverrides": {
          "Environment": [
            {"Name": "AIRLINE", "Value.$": "$.airline"},
            {"Name": "BATCH_ID", "Value.$": "$.batch_id"}
          ]
        }
      },
      "End": true
    }
  }
}
```

## AWS Batch Job Definitions (Examples)

### Preprocess Job

```json
{
  "jobDefinitionName": "qar-preprocess-job",
  "type": "container",
  "containerProperties": {
    "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/qar-preprocess:latest",
    "vcpus": 2,
    "memory": 4096,
    "command": ["bash", "-lc", "python /app/preprocess.py"],
    "environment": [
      {"name": "INPUT_S3", "value": ""},
      {"name": "AIRLINE", "value": ""},
      {"name": "TAIL", "value": ""},
      {"name": "BATCH_ID", "value": ""}
    ]
  }
}
```

### Training Job

```json
{
  "jobDefinitionName": "qar-train-job",
  "type": "container",
  "containerProperties": {
    "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/qar-train:latest",
    "vcpus": 4,
    "memory": 16384,
    "command": ["bash", "-lc", "python /app/train.py"],
    "environment": [
      {"name": "INPUT_S3", "value": ""},
      {"name": "AIRLINE", "value": ""},
      {"name": "TAIL", "value": ""},
      {"name": "BATCH_ID", "value": ""}
    ]
  }
}
```

### Savings Job

```json
{
  "jobDefinitionName": "qar-savings-job",
  "type": "container",
  "containerProperties": {
    "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/qar-savings:latest",
    "vcpus": 4,
    "memory": 16384,
    "command": ["bash", "-lc", "python /app/savings.py"],
    "environment": [
      {"name": "AIRLINE", "value": ""},
      {"name": "BATCH_ID", "value": ""}
    ]
  }
}
```

## IAM Policies (Minimal Examples)

### Batch Job Role (Read input, write output)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadInput",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::qar-*-data",
        "arn:aws:s3:::qar-*-data/incoming/*"
      ]
    },
    {
      "Sid": "WriteOutputs",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": [
        "arn:aws:s3:::qar-*-data/preprocess/*",
        "arn:aws:s3:::qar-*-data/processed/*",
        "arn:aws:s3:::qar-*-data/savings/*"
      ]
    }
  ]
}
```

### Step Functions Role (Run Batch, Invoke Lambda)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BatchSubmit",
      "Effect": "Allow",
      "Action": [
        "batch:SubmitJob",
        "batch:DescribeJobs",
        "batch:TerminateJob"
      ],
      "Resource": "*"
    },
    {
      "Sid": "LambdaInvoke",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:qar-*"
    }
  ]
}
```

## S3 Event Rule (EventBridge)

```json
{
  "source": ["aws.s3"],
  "detail-type": ["Object Created"],
  "detail": {
    "bucket": { "name": ["qar-aa-data"] },
    "object": { "key": [{ "prefix": "incoming/" }] }
  }
}
```

## AWS Batch Compute Environment (Spot + On-Demand Fallback)

Create two compute environments and a single job queue that prefers Spot first.

### Compute Environment: Spot

```json
{
  "computeEnvironmentName": "qar-spot-ce",
  "type": "MANAGED",
  "state": "ENABLED",
  "computeResources": {
    "type": "SPOT",
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 256,
    "desiredvCpus": 0,
    "instanceTypes": ["m6i", "m5", "c6i", "c5"],
    "subnets": ["subnet-aaaa", "subnet-bbbb"],
    "securityGroupIds": ["sg-aaaa"],
    "instanceRole": "ecsInstanceRole",
    "spotIamFleetRole": "arn:aws:iam::ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role"
  },
  "serviceRole": "arn:aws:iam::ACCOUNT_ID:role/AWSBatchServiceRole"
}
```

### Compute Environment: On-Demand

```json
{
  "computeEnvironmentName": "qar-ondemand-ce",
  "type": "MANAGED",
  "state": "ENABLED",
  "computeResources": {
    "type": "EC2",
    "minvCpus": 0,
    "maxvCpus": 64,
    "desiredvCpus": 0,
    "instanceTypes": ["m6i", "m5", "c6i", "c5"],
    "subnets": ["subnet-aaaa", "subnet-bbbb"],
    "securityGroupIds": ["sg-aaaa"],
    "instanceRole": "ecsInstanceRole"
  },
  "serviceRole": "arn:aws:iam::ACCOUNT_ID:role/AWSBatchServiceRole"
}
```

### Job Queue (Prefer Spot)

```json
{
  "jobQueueName": "qar-train-queue",
  "state": "ENABLED",
  "priority": 1,
  "computeEnvironmentOrder": [
    { "order": 1, "computeEnvironment": "qar-spot-ce" },
    { "order": 2, "computeEnvironment": "qar-ondemand-ce" }
  ]
}
```

Repeat the job queue for `qar-preprocess-queue` and `qar-savings-queue` or reuse a single shared queue if you prefer.

## Lambda Stubs

Minimal Lambda stubs are provided in `qar_pipeline/lambdas/`. They assume bucket names follow `qar-{airline}-data` and keys follow `incoming/{tail}/{batch_id}/file.csv`.

Files:
- `qar_pipeline/lambdas/parse_key.py`
- `qar_pipeline/lambdas/update_manifest.py`
- `qar_pipeline/lambdas/load_manifest.py`

## Manual Triggers

### Start Training

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT_ID:stateMachine:qar-train \
  --input '{"airline":"AA","batch_id":"2026-02-07"}'
```

### Start Savings

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT_ID:stateMachine:qar-savings \
  --input '{"airline":"AA","batch_id":"2026-02-07"}'
```

## Notes

- `manifest.json` is expected to list all tails and their input CSV keys for the airline batch.
- Set `MaxConcurrency` based on Batch compute capacity.
- Use object tags or metadata if you want stronger traceability and lineage.
- Replace `ACCOUNT_ID` with the numeric account ID for the fa-ei-cio account.
- For the EventBridge rule, either list all airline buckets explicitly or create one rule per airline bucket.
- For IAM least-privilege, replace `qar-*-data` with explicit bucket ARNs.
