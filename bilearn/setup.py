import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('bilearn', parent_package, top_path)

    config.add_extension('cd_fast', sources=['cd_fast.cpp'],
                         include_dirs=[numpy.get_include()])

    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
