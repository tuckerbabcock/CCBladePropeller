from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('ccblade_propeller/__init__.py').read(),
)[0]

setup(name='ccblade_propeller',
      version=__version__,
      author='Tucker Babcock and Daniel Ingraham',
      author_email='tuckerbabcock1@gmail.com',
      url='https://github.com/tuckerbabcock/InverterModel',
      license='MPL-2.0 License',
      description='OpenMDAO model for CCBlade.jl propeller',
      packages=[
          "ccblade_propeller"
      ],
      install_requires=[
          'juliapkg',
          'openmdao',
          'omjlcomps>=0.1.6'
      ],
      zip_safe=False
)
