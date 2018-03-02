from setuptools import setup

setup(name='hometrainer',
      version='0.1.1',
      description='Simple AlhpaZero implementation',
      url='https://github.com/FritzFlorian/hometrainer',
      author='Fritz Florian',
      license='MIT',
      packages=['hometrainer'],
      install_requires=[
          'tensorflow',
          'pyzmq',
          'numpy',
          'py-kim',
          'matplotlib'
      ],
      zip_safe=False)
