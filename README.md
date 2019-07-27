# NLPdf
PDF Extractor using Natural Language Processing

## Quickstart
1. Download the repository
2. Install the requirements:
```console
pip3 install -r requirements.txt 
```
3. Load the language model for Spacy:
```console
python3 -m spacy download en
```
4. Copy the PDF files to be cleaned into the directory "PDFs"
5. Run the extraction tool:
```console
python3 run.py 
```
6. The output is written to the directory "output"

