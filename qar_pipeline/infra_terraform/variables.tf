variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "stack_name" {
  type    = string
  default = "qar-pipeline"
}

variable "airline_bucket_name" {
  type = string
}

variable "permissions_boundary_arn" {
  type    = string
  default = ""
}

variable "preprocess_job_queue" {
  type    = string
  default = "qar-preprocess-queue"
}

variable "preprocess_job_definition" {
  type    = string
  default = "qar-preprocess-job:1"
}

variable "train_job_queue" {
  type    = string
  default = "qar-train-queue"
}

variable "train_job_definition" {
  type    = string
  default = "qar-train-job:1"
}

variable "savings_job_queue" {
  type    = string
  default = "qar-savings-queue"
}

variable "savings_job_definition" {
  type    = string
  default = "qar-savings-job:1"
}
