PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests

# Compilation...

CYTHONSRC= $(wildcard bilearn/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.cpp)

inplace:
	$(PYTHON) setup.py build_ext -i

all: cython inplace

cython: $(CSRC)

clean:
	rm -f bilearn/*.c bilearn/*.cpp bilearn/*.html
	rm -f `find bilearn -name "*.pyc"`
	rm -f `find bilearn -name "*.so"`

%.cpp: %.pyx
	$(CYTHON) --cplus $<

# Tests...
#
test-code: inplace
	$(NOSETESTS) -s polylearn

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=bilearn bilearn

