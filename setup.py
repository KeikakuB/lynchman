from setuptools import setup, find_packages
setup(
    name="lynchman",
    version="lynchman",
    py_modules=['lynchman'],
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[
        # Command Args
        'click',
        # SciPy
        'numpy',
        'scipy',
        'matplotlib',
        'ipython',
        'jupyter',
        'pandas',
        'sympy',
        'nose',
        # Audio File Metadata
        'mutagen',
        # Song File Analysis
        'librosa'
        ],
    entry_points='''
        [console_scripts]
        lynchman=lynchman:cli
    ''',
)
