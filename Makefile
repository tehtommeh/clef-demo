# Thin wrappers over the commands you would otherwise retype constantly.
.PHONY: help download check verify up down logs test model-test shell stats clean

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

download:  ## Fetch model weights into ./models
	python3 scripts/download.py $$(grep '^MODEL_REPO=' .env | cut -d= -f2)

check:     ## Are there upstream model updates?
	python3 scripts/download.py --check

verify:    ## Do local weights still match the lock file?
	python3 scripts/download.py --verify

up:        ## Build and start the stack
	docker compose up -d --build

down:      ## Stop the stack
	docker compose down

logs:      ## Follow logs
	docker compose logs -f

test:      ## Run the endpoint smoke tests
	python3 scripts/smoke_test.py --wait 600

model-test: ## Answer-level correctness + latency checks
	python3 scripts/model_test.py

shell:     ## Shell into the API container
	docker compose exec api /bin/bash

stats:     ## GPU and container resource usage
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
	@docker stats --no-stream

clean:     ## Remove containers and images, keep the weights
	docker compose down --rmi local --volumes
