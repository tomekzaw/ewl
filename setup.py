import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ewl',
    version='0.6.0',
    author='Piotr Kotara, Tomasz Zawadzki',
    author_email='piotrekkotara@gmail.com, tomekzawadzki98@gmail.com',
    description='A simple Python library to simulate and execute EWL quantum circuits on IBM Q.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tomekzaw/ewl',
    project_urls={
        'Bug Tracker': 'https://github.com/tomekzaw/ewl/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6',
)
