from setuptools import setup

requirements = [
    "pytorch",
    "scikit-image",
    "tqdm",
    "pillow",
    "torchvision",
    "matplotlib-base",
    "numpy"
]

setup(name='cornet_irl',
      version='0.1',
      description='CORnet-S but also trained on surface normals',
      url='http://github.com/bwest25/cornet_irl',
      author='Brody West',
      author_email='brodyw@mit.edu',
      install_requires=requirements,
      license='MIT',
      zip_safe=False)
