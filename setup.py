from setuptools import setup, find_packages

requirements = [
    "torch",
    "scikit-image",
    "tqdm",
    "pillow",
    "torchvision",
    "matplotlib",
    "numpy"
]

setup(name='cornet_irl',
      version='0.2',
      description='CORnet-S but also trained on surface normals',
      url='http://github.com/bwest25/cornet_irl',
      packages=find_packages(include=['cornet_irl']),
      author='Brody West',
      author_email='brodyw@mit.edu',
      install_requires=requirements,
      license='MIT',
      zip_safe=False)
