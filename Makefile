# FIXME: does not work
start-venv:
	source ./venv/bin/activate

copy-data-sources:
	mkdir .data
	cp ~/Downloads/about_rodi.* ./.data

copy-env-files:
	cp ~/.env/.env .

start-server:
	python main.py

start-chainlit:
	chainlit run src/site_bot_opik/interface/chainlit.py

start-evaluate:
	python src/evaluations/evaluate.py


docker-compose-build:
	docker compose build --no-cache

docker-compose-up:
	docker compose up
	