clean:
	rm -rf dist build __pycache__ *.egg-info

notebook:
	poetry install
	poetry run ipython kernel install --user --name=farglot
	jupyter notebook --notebook-dir=notebooks

publish:
	make clean
	poetry build
	poetry publish