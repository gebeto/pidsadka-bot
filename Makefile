generate-config:
	poetry run python -m spacy init fill-config base_config.cfg config.cfg

generate-train:
	poetry run python trainer.py

train: generate-train
	poetry run python -m spacy train config.cfg --output ./output

test:
	poetry run python test.py
