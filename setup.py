from setuptools import setup, find_packages

setup(
    name='atash',
    version='0.1.0',
    author='Bahman Amirsardary',
    author_email='bahman.amirsardary@mail.polimi.it',
    description='A library for satellite image processing and fire analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Bahman75/Geospatial-Processing',  
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'openeo',
        'matplotlib',
        'rasterio',
        'ipyleaflet',
        'ipywidgets',
        'scikit-learn',
        'scikit-image'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
