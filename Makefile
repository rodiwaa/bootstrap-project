copy-data-sources:
	cp ~/Downloads/about_rodi.* ./.data

start-server:
	python main.py

start-chainlit:
	chainlit run src/site_bot_opik/interface/chainlit.py