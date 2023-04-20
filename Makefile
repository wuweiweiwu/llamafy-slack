env:
	. .venv/bin/activate

install:
	pip install -r requirements.txt

dump:
	pip freeze > requirements.txt

start:
	python3 app.py
