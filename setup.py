from setuptools import setup

setup(
    name='NLPdf',
    version='0.1.dev',
    packages=['pdf_processor'],
    long_description=open('README.md').read(),
    author='Katharina Wünsche',
    install_requires=[
    "PyPDF2",
    "nltk",
    "re",
    "spacy",
    "textract",
    "copy",
    "multiprocessing",
    "python-pdfbox",
    "statistics"
    ]

)
