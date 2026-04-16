ENVIRONMENT_NAME = ace-aimip
POSTPROCESS_DIR = scripts/aimip_postprocessing
LOCAL_DIR = /tmp/aimip-ace/

postprocess:
	cd $(POSTPROCESS_DIR) && conda run -n $(ENVIRONMENT_NAME) python postprocess.py

test:
	cd $(POSTPROCESS_DIR) && conda run -n $(ENVIRONMENT_NAME) python -m pytest test_postprocess.py -v --noconftest

clean:
	rm -rf $(LOCAL_DIR)

.PHONY: postprocess test clean
