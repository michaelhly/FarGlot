notebook:
	poetry install
	poetry run ipython kernel install --user --name=farglot
	jupyter notebook --notebook-dir=notebooks