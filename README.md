#bilearn: bilinear models for machine learning

[![Travis Status](https://travis-ci.org/vene/bilearn.svg?branch=master)](https://travis-ci.org/vene/bilearn)
[![Coveralls Status](https://coveralls.io/repos/vene/bilearn/badge.svg?branch=master&service=github)](https://coveralls.io/r/vene/bilearn)
[![CircleCI Status](https://circleci.com/gh/vene/bilearn.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/vene/bilearn/tree/master)

**bilearn** is a package implementing bilinear models for classification and regression.

## Important Links
HTML Documentation - http://github.org/vene/bilearn/

## Installation and Usage
The package by itself comes with a single module and an estimator. Before
installing the module you will need `numpy` and `scipy`.
To install the module execute:
```shell
$ python setup.py install
```
or 
```
pip install bilearn
```

If the installation is successful, and `scikit-learn` is correctly installed,
you should be able to execute the following in Python:

```python
>>> TODO
>>> from skltemplate import TemplateEstimator
>>> estimator = TemplateEstimator()
>>> estimator.fit(np.arange(10).reshape(10, 1), np.arange(10))
```
