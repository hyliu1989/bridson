from setuptools import setup, find_packages


with open('README.rst') as f:
    description = f.read()


setup(
    name='bridson',
    url='http://github.com/emulbreh/bridson/',
    version='0.2.0',
    packages=find_packages(),
    license=u'MIT License',
    author=u'Johannes Dollinger, Hsiou-Yuan Liu',
    description=u'poisson disc sampling of N-dimensional sample domain (N<=3)',
    long_description=description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3'
    ]
)
