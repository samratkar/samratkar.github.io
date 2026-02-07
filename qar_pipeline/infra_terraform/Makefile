LAMBDA_DIR=../lambdas

.PHONY: zips init plan apply

zips:
	cd $(LAMBDA_DIR) && zip -q parse_key.zip parse_key.py
	cd $(LAMBDA_DIR) && zip -q update_manifest.zip update_manifest.py
	cd $(LAMBDA_DIR) && zip -q load_manifest.zip load_manifest.py

init:
	terraform init

plan:
	terraform plan

apply:
	terraform apply
