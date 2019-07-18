import sys
import os
from PDF_Processor import PDF_Processor

#input directory containing the pdf files
inputdir = "PDFs/"

#output directory for the txt files
outputdir = "output/"


def getListOfFiles(dirName=os.getcwd()):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

filenames = getListOfFiles(dirName=inputdir)

pdfs = [filename for filename in filenames if filename.endswith(".pdf")]
pp = PDF_Processor()
pp.startThreads(pdfs, outputdir)
