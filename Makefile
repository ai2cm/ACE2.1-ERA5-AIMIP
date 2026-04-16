ENVIRONMENT_NAME = ace-aimip
POSTPROCESS_DIR = scripts/aimip_postprocessing
LOCAL_DIR = /tmp/aimip-ace/

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
	cd $(POSTPROCESS_DIR) && conda run -n $(ENVIRONMENT_NAME) python postprocess.py

test:
	cd $(POSTPROCESS_DIR) && conda run -n $(ENVIRONMENT_NAME) python -m pytest test_postprocess.py -v --noconftest

clean:
	rm -rf $(LOCAL_DIR)

.PHONY: train evaluate fine-tune inference postprocess test clean
