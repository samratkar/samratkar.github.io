terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  s3_bucket_arn = "arn:aws:s3:::${var.airline_bucket_name}"
}

resource "aws_iam_role" "lambda_basic" {
  name                 = "${var.stack_name}-lambda-basic"
  permissions_boundary = var.permissions_boundary_arn == "" ? null : var.permissions_boundary_arn
  assume_role_policy   = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_exec" {
  role       = aws_iam_role.lambda_basic.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role" "eventbridge_invoke_sfn" {
  name                 = "${var.stack_name}-events-sfn"
  permissions_boundary = var.permissions_boundary_arn == "" ? null : var.permissions_boundary_arn
  assume_role_policy   = data.aws_iam_policy_document.events_assume.json
}

resource "aws_iam_role_policy" "eventbridge_invoke_sfn" {
  name = "${var.stack_name}-events-sfn"
  role = aws_iam_role.eventbridge_invoke_sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["states:StartExecution"]
        Resource = [aws_sfn_state_machine.preprocess.arn]
      }
    ]
  })
}

resource "aws_iam_role" "sfn" {
  name                 = "${var.stack_name}-sfn"
  permissions_boundary = var.permissions_boundary_arn == "" ? null : var.permissions_boundary_arn
  assume_role_policy   = data.aws_iam_policy_document.sfn_assume.json
}

resource "aws_iam_role_policy" "sfn_policy" {
  name = "${var.stack_name}-sfn"
  role = aws_iam_role.sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "batch:SubmitJob",
          "batch:DescribeJobs",
          "batch:TerminateJob"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["lambda:InvokeFunction"]
        Resource = [
          aws_lambda_function.parse_key.arn,
          aws_lambda_function.update_manifest.arn,
          aws_lambda_function.load_manifest.arn
        ]
      }
    ]
  })
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "events_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "${var.stack_name}-lambda-s3"
  role = aws_iam_role.lambda_basic.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = ["${local.s3_bucket_arn}/preprocess/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [local.s3_bucket_arn]
      }
    ]
  })
}

resource "aws_lambda_function" "parse_key" {
  function_name = "${var.stack_name}-parse-key"
  role          = aws_iam_role.lambda_basic.arn
  handler       = "parse_key.handler"
  runtime       = "python3.11"
  timeout       = 30
  memory_size   = 256
  filename      = "${path.module}/../lambdas/parse_key.zip"
}

resource "aws_lambda_function" "update_manifest" {
  function_name = "${var.stack_name}-update-manifest"
  role          = aws_iam_role.lambda_basic.arn
  handler       = "update_manifest.handler"
  runtime       = "python3.11"
  timeout       = 30
  memory_size   = 256
  filename      = "${path.module}/../lambdas/update_manifest.zip"
}

resource "aws_lambda_function" "load_manifest" {
  function_name = "${var.stack_name}-load-manifest"
  role          = aws_iam_role.lambda_basic.arn
  handler       = "load_manifest.handler"
  runtime       = "python3.11"
  timeout       = 30
  memory_size   = 256
  filename      = "${path.module}/../lambdas/load_manifest.zip"
}

resource "aws_sfn_state_machine" "preprocess" {
  name     = "${var.stack_name}-preprocess"
  role_arn = aws_iam_role.sfn.arn
  definition = templatefile("${path.module}/statemachines/preprocess.asl.json", {
    ParseKeyLambdaArn        = aws_lambda_function.parse_key.arn
    UpdateManifestLambdaArn  = aws_lambda_function.update_manifest.arn
    PreprocessJobQueue       = var.preprocess_job_queue
    PreprocessJobDefinition  = var.preprocess_job_definition
  })
  logging_configuration {
    level                  = "ALL"
    include_execution_data = true
    log_destination        = "${aws_cloudwatch_log_group.preprocess_sfn.arn}:*"
  }
}

resource "aws_sfn_state_machine" "training" {
  name     = "${var.stack_name}-training"
  role_arn = aws_iam_role.sfn.arn
  definition = templatefile("${path.module}/statemachines/training.asl.json", {
    LoadManifestLambdaArn = aws_lambda_function.load_manifest.arn
    TrainJobQueue         = var.train_job_queue
    TrainJobDefinition    = var.train_job_definition
  })
  logging_configuration {
    level                  = "ALL"
    include_execution_data = true
    log_destination        = "${aws_cloudwatch_log_group.training_sfn.arn}:*"
  }
}

resource "aws_sfn_state_machine" "savings" {
  name     = "${var.stack_name}-savings"
  role_arn = aws_iam_role.sfn.arn
  definition = templatefile("${path.module}/statemachines/savings.asl.json", {
    SavingsJobQueue      = var.savings_job_queue
    SavingsJobDefinition = var.savings_job_definition
  })
  logging_configuration {
    level                  = "ALL"
    include_execution_data = true
    log_destination        = "${aws_cloudwatch_log_group.savings_sfn.arn}:*"
  }
}

resource "aws_cloudwatch_log_group" "preprocess_sfn" {
  name              = "/aws/states/${var.stack_name}-preprocess"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "training_sfn" {
  name              = "/aws/states/${var.stack_name}-training"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "savings_sfn" {
  name              = "/aws/states/${var.stack_name}-savings"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "parse_key_lg" {
  name              = "/aws/lambda/${aws_lambda_function.parse_key.function_name}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "update_manifest_lg" {
  name              = "/aws/lambda/${aws_lambda_function.update_manifest.function_name}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "load_manifest_lg" {
  name              = "/aws/lambda/${aws_lambda_function.load_manifest.function_name}"
  retention_in_days = 30
}

resource "aws_cloudwatch_metric_alarm" "preprocess_failed" {
  alarm_name          = "${var.stack_name}-preprocess-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ExecutionsFailed"
  namespace           = "AWS/States"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.preprocess.arn
  }
}

resource "aws_cloudwatch_metric_alarm" "training_failed" {
  alarm_name          = "${var.stack_name}-training-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ExecutionsFailed"
  namespace           = "AWS/States"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.training.arn
  }
}

resource "aws_cloudwatch_metric_alarm" "savings_failed" {
  alarm_name          = "${var.stack_name}-savings-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ExecutionsFailed"
  namespace           = "AWS/States"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.savings.arn
  }
}

resource "aws_cloudwatch_metric_alarm" "parse_key_errors" {
  alarm_name          = "${var.stack_name}-parse-key-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  dimensions = {
    FunctionName = aws_lambda_function.parse_key.function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "update_manifest_errors" {
  alarm_name          = "${var.stack_name}-update-manifest-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  dimensions = {
    FunctionName = aws_lambda_function.update_manifest.function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "load_manifest_errors" {
  alarm_name          = "${var.stack_name}-load-manifest-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  dimensions = {
    FunctionName = aws_lambda_function.load_manifest.function_name
  }
}

resource "aws_cloudwatch_event_rule" "preprocess" {
  name        = "${var.stack_name}-preprocess"
  description = "Start preprocess workflow on new incoming CSVs"
  event_pattern = jsonencode({
    source      = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = {
        name = [var.airline_bucket_name]
      }
      object = {
        key = [
          {
            prefix = "incoming/"
          }
        ]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "preprocess" {
  rule      = aws_cloudwatch_event_rule.preprocess.name
  target_id = "PreprocessStateMachineTarget"
  arn       = aws_sfn_state_machine.preprocess.arn
  role_arn  = aws_iam_role.eventbridge_invoke_sfn.arn
}

output "preprocess_state_machine_arn" {
  value = aws_sfn_state_machine.preprocess.arn
}

output "training_state_machine_arn" {
  value = aws_sfn_state_machine.training.arn
}

output "savings_state_machine_arn" {
  value = aws_sfn_state_machine.savings.arn
}
