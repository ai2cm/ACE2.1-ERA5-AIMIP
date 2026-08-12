ENVIRONMENT_NAME = ace-aimip
POSTPROCESS_DIR = scripts/aimip_postprocessing
LOCAL_DIR = /tmp/aimip-ace/

create-env:
	conda env create -f environment.yml

train:
	bash scripts/run-ace-train.sh

evaluate:
	bash scripts/run-ace-evaluator-seed-selection.sh
	bash scripts/run-ace-evaluator-seed-selection-single.sh

fine-tune:
	bash scripts/run-ace-fine-tune-decoder-pressure-levels.sh

inference:
	bash scripts/run-ace-inference.sh

postprocess:
	cd $(POSTPROCESS_DIR) && conda run -n $(ENVIRONMENT_NAME) python postprocess.py $(ARGS)

ace2.2-build-plev-companion:
	bash scripts/run-ace2.2-build-plev-companion.sh

ace2.2-fine-tune:
	bash scripts/run-ace2.2-fine-tune-decoder-pressure-levels.sh

ace2.2-inference:
	bash scripts/run-ace2.2-inference.sh

test-postprocess:
	cd $(POSTPROCESS_DIR) && conda run -n $(ENVIRONMENT_NAME) python -m pytest test_postprocess.py -v --noconftest

clean:
	rm -rf $(LOCAL_DIR)

.PHONY: create-env train evaluate fine-tune inference postprocess test-postprocess clean ace2.2-build-plev-companion ace2.2-fine-tune ace2.2-inference
