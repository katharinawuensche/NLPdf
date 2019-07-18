from PyPDF2 import PdfFileReader
from nltk.corpus import words as nltkWords
import re
import spacy
import textract
import copy
import threading
import multiprocessing
from multiprocessing import Pool
import pdfbox
import statistics
import math

class PDF_Processor:
    #nlp = spacy.load("en")
    def __init__(self, filename=None):
        self.filename = filename
        nltkWords.ensure_loaded()
        self.correctWords = set(w.lower() for w in nltkWords.words())
        self.correctWords = list(self.correctWords)
        #self.nlp = spacy.load('en', disable=["parser", "textcat", "entity_ruler", "sentencizer", "merge_noun_chunks", "merge_subtokens")
        self.nlp = spacy.load("en")
        self.authors = None
        self.title = None
        for word in self.correctWords:
            if len(word) <= 1:
                self.correctWords.remove(word)
        for word in ["et", "al.", "acknowledgment", "acknowledgement", "3d", "pre", "post", "e.g.", "co2"]:
            self.correctWords.append(word)
        if self.filename:
            self.text = self.extract(self.filename)
            self.authors, self.title = self.getMetadata(self.filename)
            self.correctWords = self._addCorrectWords(self.text, self.correctWords)

    def _getCorrectWords(self):
        return self.correctWords

    def _addCorrectWords(self, text, correctWords):
        smallText = " ".join(set(re.split("\W", text))).lower()
        doc = list(self.nlp.pipe([smallText]))[0]
        incompleteWords = []
        for token in doc:
            if token.lemma_ in correctWords:
                correctWords.append(token.text.lower())
            else:
                incompleteWords.append(token.text.lower())
        # for token in doc.ents:
        #     if token.label_ in ["PERSON", "NORP", "FAC", "ORG"]:
        #         self.correctWords.append(token.text.lower())
        correctWords = list(set(correctWords))
        return correctWords#, incompleteWords

    def _groupParagraphs(self, doc):
        paragraphs = doc.split("\n\n")
        return paragraphs

    def _parenthetic_contents(self, string):
        """Generate parenthesized contents in string as pairs (level, contents).
         From: https://gist.github.com/constructor-igor/5f881c32403e3f313e6f"""
        stack = []
        for i, c in enumerate(string):
            if c == '(':
                stack.append(i)
            elif c == ')' and stack:
                start = stack.pop()
                yield [len(stack), string[start + 1: i]]

    def _removeTextInParentheses(self, text):
        if not text:
            return ""
        newText = text
        parentheses = list(self._parenthetic_contents(newText))
        outermost_par = [par for par in parentheses if par[0] == 0]
        for content in outermost_par:
            newText = newText.replace("(" + content[1] + ")", "")
            for entry in outermost_par:
                entry[1] = entry[1].replace("(" + content[1] + ")", "")
        return newText

    def _FigureTablePrep(self, doc):
        #newDoc = []
        newPage = ""
        for line in doc.splitlines():
            newLine = line
            if re.search(r"\. *(Figure|Fig.|Table) *[0-9]+(\.|\:)", line):
                re_result = re.search(r"(Figure|Fig.|Table) *[0-9]+(\.|\:)", line)
                newLine = line[:re_result.start()] + "\n" + line[re_result.start():]
            newPage += newLine + "\n"
        #newDoc.append(newPage)
        return newPage

    def _isWord(self, word, correctWords=None):
        initCorrWords = copy.copy(correctWords)
        if not correctWords:
            correctWords = self.correctWords
        return word.lower().lstrip().rstrip() in correctWords

    def _containsSpecialCharacter(self, word):
        greekpatt = r"[^a-zA-z0-9\:\-+!\" \.,;'\(\)]"
        return re.match(greekpatt, word)

    def _isNoise(self, line, correctWords=None):
        initCorrWords = copy.copy(correctWords)
        if not correctWords:
            correctWords = self.correctWords
        #doc = self.nlp(line)
        formulaWords = ["cid", "sin", "cos", "tan", "max", "min", "exp", "avg"]
        for word in re.split("\W", line):
            if word in formulaWords:
                continue
            if not self._containsSpecialCharacter(word) and self._isWord(word, correctWords=initCorrWords):
                #print("Lemma:", word.lemma_)
                return False
            #print("Kein Lemma:", word.lemma_)
        return True

    def _isWholeSentence(self, originalsentence, out=False):
        sentence = originalsentence.lstrip().rstrip()
        #sentence = self._removeInlineFormula(sentence)
        for char in "='`´’∗+":
            sentence = sentence.replace(char, "")
        for phrase in ["et al. ", "e.g. ", "Eq. ", "Fig. "]:
            sentence = sentence.replace(phrase, "")
        sentence = self._removeTextInParentheses(sentence)
        for match in re.findall("  +", sentence):
             sentence = sentence.replace(match, " ")
        sentence = sentence.lstrip().rstrip()
        doc = self.nlp(sentence)
        for sent in doc.sents:
            senttext = sent.text.lstrip().rstrip()
            if senttext.endswith("et al.") or senttext.endswith("e.g."):
                continue
            if not ((sent[0].text[0].isupper()) and (sent[-1].tag_ in ".:" or senttext[-1] in ".:?!") ):
                if out:
                    print("No sentence format:", sent, sent[-1].tag_, sent[0].text)
                return False, senttext
            containsSubject = len(list(filter(lambda token: token.dep_.count("subj") > 0 and token.pos_ != "VERB", sent))) > 0
            containsVerb = len(list(filter(lambda token: token.pos_ == "VERB", sent))) > 0
            #print(containsVerb, containsSubject)
            if not (containsSubject and containsVerb):
                if out:
                    print("No sentence word types:", sent)
                return False, senttext
        return True, ""

    def _isWholeParagraph(self, text, out=False):
        newText = self.removeLineBreaks(text)
        #check if every sentence is a whole sentence
        return self._isWholeSentence(newText, out)

    def _alluppercaseHeadline(self, line):
        wordsInLine = [word for word in re.split('[^(a-zA-z)]+', line) if len(word) > 0]
        if len(wordsInLine) == 0:
            return 0
        for char in line:
            if char.isalpha() and not char.isupper():
                return 0
        return 3
    def _alltitleHeadline(self, line):
        wordsInLine = [word for word in re.split('[^(a-zA-z)]+', line) if len(word) > 0]
        if len(wordsInLine) == 0:
            return 0
        for word in re.split('\W+', line):
            if word.isalpha() and not word.istitle():
                return 0
        return 2
    def _sometitleHeadline(self, line):
        wordsInLine = [word for word in re.split('[^(a-zA-z)]+', line) if len(word) > 0]
        if len(wordsInLine) == 0:
            return 0
        if wordsInLine[0].istitle(): #and line.split(" ")[-1].istitle():
            #nlpLine = self.nlp(line)
            nlpLine = list(self.nlp.pipe([line]))[0]
            for token in nlpLine:
                if token.pos_ in ["VERB", "NOUN", "ADJ"] and not token.text[0].istitle():
                    return 0
            return 1
        else:
            return 0

    def _mergeParagraphs(self, parA, parB):
        newParagraph = parA.lstrip().rstrip()
        if newParagraph.endswith("-"):
            newParagraph = newParagraph.rstrip("-")
        else:
            newParagraph += " "
        newParagraph += parB.lstrip().rstrip()
        return newParagraph

    def _cleanup(self, text):
        newText = ""
        for line in text.splitlines():
            #newline = line
            newline = line.rstrip().lstrip()
            replacement = " "
            if newline.endswith("-"):
                prefix = False
                for word in ["pre-", "post-", "over-", "under-"]:
                    if newline.endswith(word):
                        replacement = ""
                        prefix = True
                if not prefix:
                    replacement = ""
                    newline = newline.rstrip("-")
            newText += newline + replacement
        for match in re.findall("  +", newText):
            newText = newText.replace(match, " ")
        newText = newText.replace(" .", ".")
        newText = newText.replace(" ,", ",")
        return newText

    def _repairNumberedHeadlines(self, text):
        newText = text
        textlen = len(newText.splitlines())
        for idx in range(textlen):
            #print(idx, textlen)
            if idx >= textlen:
                break
            line = newText.splitlines()[idx]
            if len(line.rstrip().lstrip()) == 0:
                continue
            toreplace = line + "\n"
            if re.match("^[0-9]+\.$", line.rstrip().lstrip()):
                for jdx in range(idx+1, textlen):
                    nextLine = newText.splitlines()[jdx]
                    toreplace += nextLine + "\n"
                    idx = idx + 1
                    if len(nextLine.rstrip().lstrip()) == 0:
                        continue
                    if self._alluppercaseHeadline(nextLine) > 0 or self._alltitleHeadline(nextLine) > 0 or self._sometitleHeadline(nextLine) > 0:
                        newText = newText.replace(toreplace, line + " " + nextLine)
                        textlen = len(newText.splitlines())
                    break
        return newText

    def _replaceSpecialCharacters(self, text):
        newText = text.replace("ﬁ", "fi")
        newText = newText.replace("ďŹ", "fi")
        newText = newText.replace("ﬂ", "fl")
        newText = newText.replace("ďŹ", "fl")
        newText = newText.replace("ďŹ", "ff")
        newText = newText.replace("ﬀ", "ff")
        newText = newText.replace("ﬀe", "ffe")
        newText = newText.replace("ﬃ", "ffi")
        newText = newText.replace("ďŹ", "ffi")
        newText = newText.replace("â", "-")
        newText = newText.replace("â", "\'")
        newText = newText.replace("â", "\"")
        newText = newText.replace("¨ı", "ï")
        return newText

    def _clean(self):
        #self.findHeadlines(self.text)
        #print("-------------")
        self.symbols = []
        self.text = self._replaceSpecialCharacters(self.text)
        #print("Special Characters", len(self.text))
        self.text = self.findReferences(self.text)
        #print("Remove References", len(self.text))
        self.text = self._repairNumberedHeadlines(self.text)
        #print("Repair numbered Headlines", len(self.text))
        #self.text = self.removeDuplicateLines(self.text)

        self.text = self.findHeadersAndFooters(self.text)
        #print("Headers & Footers", len(self.text))
        self.text = self.findFigures(self.text)
        #print("Figures", len(self.text))
        self.text = self.findTables(self.text)
        #print("Tables", len(self.text))
        self.text = self.removeInlineReferences(self.text)
        #print("Inline References", len(self.text))
        self.text, self.symbols = self.findFormulas(self.text)
        #print("Formulas", len(self.text))
        self.text, self.inlineFormulas = self.findInlineFormula(self.text)
        #print(len(self.text))
        self.symbols += self.inlineFormulas
        #print(self.symbols)
        self.text = self.findNoise(self.text)
        self.nlp = spacy.load("en")
        self.text = self.removeAuthors(self.text)
        self.text = self.removeSymbols(self.text, self.symbols)

        self.text = self.removeDuplicateLines(self.text)
        #self.text = self.removeLineBreaks(self.text)
        #self.text = self.matchParagraphs(self.text, out=False)
        self.headlines = self.findHeadlines(self.text, out=False)
        self.chapters = self.groupChapters(self.text, self.headlines)
        self.text = self.joinChapters(self.chapters)
        self.text = self.strip(self.text, self.headlines, fromval=["Introduction", "Motivation"], toval=["References", "Acknowledgement", "Acknowledgment"])
        self.text = self.removeAdditionalInfo(self.text, ["Acknowledgment", "Acknowledgement", "Acknowledgments", "Acknowledgements", "Author Contributions", "Conflicts of Interest"])
        self.text = self.text.rstrip().lstrip()

    def extract(self, filename, method="pdfbox"):
        if method == "pdfbox":
            p = pdfbox.PDFBox()
            text = p.extract_text(filename)
            if len(text) == 0:
                method = "textract"

        if method == "textract":
            byte_text = textract.process(filename, encoding="utf-8", method="pdfminer")
            text = byte_text.decode("utf-8")

        return text

    def getMetadata(self, filename):
        with open(filename, 'rb') as file:
            pdf = PdfFileReader(file)
            authors = [name.lstrip().rstrip() for name in self._removeTextInParentheses(pdf.getDocumentInfo().author).split(", ") if len(name) > 0]
            title = pdf.getDocumentInfo().title
        return (authors, title)

    def strip(self, text, headlines, fromval=None, toval=None):
        startHL = None
        stopHL = None
        if fromval:
            startHLs = [hl for hl in headlines for fromh in fromval if fromh.lower() in hl[0].lower()]
            if len(startHLs) > 0:
                mindx = min([text.index(hl[0]) for hl in startHLs])
                startHL = [hl[0] for hl in startHLs if text.index(hl[0]) == mindx][0]

                #print("StartHLs:", startHLs)
                # print("maxIds:", maxIdx)
                # print("startHL:", startHL)
        else:
            if len(headlines) > 0:
                startHL = headlines[0][0]
        if toval:
            stopHLs = [hl for hl in headlines for toh in toval if toh.lower() in hl[0].lower()] #[hl for hl in headlines if toval.lower() in hl[0].lower()]
            if len(stopHLs) > 0:
                minIdx = min([text.index(hl[0]) for hl in stopHLs])
                stopHL = [hl[0] for hl in stopHLs if text.index(hl[0]) == minIdx][0]
        # else:
        #     if len(headlines) > 0:
        #         stopHL = headlines[-1][0]

        newText = text

        if startHL:
            #print("StartHL:", startHL)
            newText = newText[newText.index(startHL):]
        if stopHL:
            #print("StopHL", stopHL)
            newText = newText[:newText.rindex(stopHL)]

        return newText

    def findReferences(self, doc, remove=True):
        newDoc = doc
        if "References" in newDoc:
            newDoc = newDoc[:newDoc.rindex("References")]
        if "REFERENCES" in newDoc:
            newDoc = newDoc[:newDoc.rindex("REFERENCES")]
        return newDoc

    #Find headlines
    def findHeadlines(self, doc, remove=False, out=False):
        numberedHeadlinePattern = r"^([0-9\.\)]+|[IVX\.\)]+) +[A-Z]"
        potentialHeadlines = []
        ReferenceHeadlines = []
        realHeadlines = []
        lastHeadlines = []
        for rawline in doc.splitlines():
            line = rawline.lstrip().rstrip()
            if len(line) == 0 or not line[0].isalnum():
                continue
            headlineconfig = (max(self._alluppercaseHeadline(line), self._alltitleHeadline(line), self._sometitleHeadline(line)), not re.match(numberedHeadlinePattern, line) is None)
            if headlineconfig[0] > 0 or headlineconfig[1]:
                potentialHeadlines.append([line, headlineconfig])
                for word in ["Introduction", "Motivation", "Conclusio", "Discussion", "Summary", "Future Work"]:
                    if word.lower() in line.lower():
                        ReferenceHeadlines.append(headlineconfig)
                        #print("refLine: ", line)
                for word in ["References", "Acknowledgment", "Acknowledgement"]:
                    if word.lower() in line.lower():
                        lastHeadlines.append([line, headlineconfig])

        if out:
            print(potentialHeadlines)
            print(ReferenceHeadlines)

        for entry in potentialHeadlines:
            if entry[1] in ReferenceHeadlines:
                realHeadlines.append(entry)
            elif entry[1][1]: # and entry[1][0] in [ref[0] for ref in ReferenceHeadlines]
                match2 = re.match("^([0-9]+).", entry[0])
                if len(realHeadlines) == 0:
                    if int(match2.groups()[0]) == 1:
                        realHeadlines.append(entry)
                    continue

                match1 = re.match("^([0-9]+).", realHeadlines[-1][0])
                if match1 and match2:
                    if match1.groups()[0] == match2.groups()[0] or int(match1.groups()[0]) == int(match2.groups()[0]) - 1:
                        realHeadlines.append(entry)
                    else:
                        if out:
                            print(match1.groups(), match2.groups())
        for entry in lastHeadlines:
            realHeadlines.append(entry)
        if out:
            print(realHeadlines)
        return realHeadlines

    #Find figures
    def findFigures(self, doc, remove=True):
        doc = self._FigureTablePrep(doc)
        newPage = doc
        figurePattern = r"((\A|\n)\W*|\:|\.|>)(\W|[0-9])*(?=((Figure|Fig.) *[0-9]+(\.|\:)(.|\n){0,500}?\. *)(\n|\Z))"
        re_result = re.finditer(figurePattern, newPage)
        if re_result:
            for res in re_result:
                figureDescription = res.groups()[3]
                #print(figureDescription, "\n")
                if not remove:
                    newText = "<FIGURE: " + figureDescription + ">"
                else:
                    newText = ""
                newPage = newPage.replace(figureDescription, newText)
        return newPage

    #Find tables
    def findTables(self, doc, remove=True, correctWords=None):
        initCorrWords = copy.copy(correctWords)
        if not correctWords:
            correctWords = self.correctWords
        doc = self._FigureTablePrep(doc)
        newPage = doc
        tablePattern = r"((Table|Tbl.) *[0-9]+(\:)(.|\n){0,500}?\.)( *(\n|\Z)|<|\Z)"
        re_result = re.finditer(tablePattern, newPage)
        if re_result:
            for res in re_result:
                tableDescription = res.groups()[0]
                if res.groups()[3] == ".":
                    tableDescription += res.groups()[4]
                for line in newPage[newPage.index(tableDescription) + len(tableDescription):].splitlines():
                    if self._isNoise(line, correctWords=initCorrWords) or len(line) == 0:
                        tableDescription += line + "\n"
                        #print(line)
                    else:
                        break
                for line in reversed(newPage[:newPage.index(tableDescription)].splitlines()):
                    if self._isNoise(line, correctWords=initCorrWords) or len(line) == 0:
                        tableDescription = line + "\n" + tableDescription
                        #print(line)
                    else:
                        break
                #print("#", tableDescription)
                #print(res.groups())
                if not remove:
                    newText = "<TABLE: " + tableDescription + ">\n"
                else:
                    newText = ""
                newPage = newPage.replace(tableDescription, newText)
        #newDoc.append(newPage)
        return newPage

    #Find headers and footers
    def findHeadersAndFooters(self, doc, remove=True):
        newDoc = doc
        noDigitDoc = ''.join([c for c in doc if not c.isdigit()])
        for line in doc.splitlines():
            lineToFind = ''.join([c for c in line if not c.isdigit()])
            lineToReplace = "\n" + line + "\n"
            if(len(line) > 0 and noDigitDoc.count(lineToFind) > 2):
                #print(lineToFind)
                if remove:
                    newLine = "\n"
                else:
                    newLine = "\n<HEADER/FOOTER: " + line + ">\n"
                newDoc = newDoc.replace(lineToReplace, newLine)
        return newDoc

    #Remove duplicates with variations such as page numbers
    def findAdvDuplicates(self, doc, remove=True):
        footers = self.getFooterShape(doc)
        newDoc = []
        for page in doc:
            newPage = page
            for line in page.splitlines():
                nlpLine = self.nlp(line)
                footerFound = False
                for token in nlpLine:
                    if token.shape_ in footers:
                        footerFound = True
                if footerFound:
                    if not remove:
                        newPage = newPage.replace(line, "<FOOTER: " + line + ">")
                    else:
                        newPage = newPage.replace(line, "")
            newDoc.append(newPage)
        return newDoc
        #Helper function for finding footers
    def getFooterShape(self, doc):
        shapes = {}
        footer = []
        for page in doc:
            tempShapes = {}
            for line in page.splitlines():
                nlpLine = self.nlp(line)
                for token in nlpLine:
                    tempShapes[token.shape_] = tempShapes.get(token.shape_, 0) + 1
            for k, v in tempShapes.items():
                shapes[k] = shapes.get(k, 0) + 1
        for k, v in shapes.items():
            if v >= len(doc) - 1 and k.find("d") > -1: #Footer occurs on (almost) every page and contains page numbers
                footer += [k]
        return footer

    def findFormulas(self, text, remove=True, correctWords=None):
        initCorrWords = copy.copy(correctWords)
        if not correctWords:
            correctWords = self.correctWords
        newText = text
        greekpatt = r"([^a-zA-z\:\-+!\" \.,;'\(\)]|cid|max|min|avg|exp|sin|cos|\([0-9]+\))"
        lidx = 0
        textlen = len(newText.splitlines())
        symbols = []
        while lidx < textlen:
            if lidx < 0 or lidx >= len(newText.splitlines()):
                print("Invalid lidx:", lidx)
                break
            #print(lidx, textlen)
            oldline = newText.splitlines()[lidx]
            if "<Formula:" in oldline:
                lidx += 1
                continue
            if len(oldline) == 0 or not re.match(greekpatt, oldline) or not self._isNoise(oldline, correctWords=initCorrWords):
                lidx += 1
                continue
            line = oldline
            formula = line
            pointstr = ""
            for nextline in newText.splitlines()[lidx+1:]:
                lidx += 1
                if "<Formula:" in nextline or (len(nextline) > 0 and not self._isNoise(nextline, correctWords=initCorrWords)):
                    break
                formula += "\n" + nextline
                if "." in nextline:
                    pointstr = "."
            if "." in formula:
                pointstr = "."
            if remove:
                newLine = pointstr + "\n"
            else:
                newLine = "<Formula: " + formula.rstrip("\n") + ">" + pointstr
            #if not formula in newText:
                        #print(formula)
                        #print("----------------")
            newText = newText.replace(formula, newLine)
            for token in re.split(r'[ \n]', formula):
                if not (re.match(r'[0-9\.,]+', token)):
                    symbols.append(token)
            lidx -= len(formula.splitlines())
            lidx += len(newLine.splitlines()) + 1
            textlen = len(newText.splitlines())
        return (newText, symbols)

    def findNoise(self, doc, remove=True, correctWords=None):
        initCorrWords = copy.copy(correctWords)
        if not correctWords:
            correctWords = self.correctWords
        newDoc = doc
        for oldline in doc.splitlines():
            line = oldline
            lineToFind = "\n" + line + "\n"
            greekpatt = r"[^a-zA-z0-9\:\-+!\" .,;'\(\)]"
            for match in re.findall(greekpatt, line):
                 line = line.replace(match, "")
            if(len(line) > 0 and self._isNoise(line, correctWords=initCorrWords)):
                #print(lineToFind)
                pointstr = ""
                if line.rstrip().endswith("."):
                    pointstr = "."
                if remove:
                    newLine = pointstr + "\n"
                else:
                    newLine = "\n<NOISE: " + line + ">" + pointstr + "\n"
                newDoc = newDoc.replace(lineToFind, newLine)
        return newDoc

    def removeInlineReferences(self, doc):
        newDoc = doc
        figTableRefPattern = r"( \([ \n]*(see )?(e\.g\. )?(c\.f\. )?(cf )?(for example)?(Fig|Figure|fig|figure|Table|Tbl|Section|Sec)(\.)* *[0-9, ]+[ \n]*\))"
        for res in re.findall(figTableRefPattern, newDoc):
            #print(res)
            newDoc = newDoc.replace(res[0], "")
        #refPattern = r" *\[[0-9,\-]+?\]"
        refPattern = r" *\[.*?\]"
        countPattern1 = 0
        for res in re.findall(refPattern, newDoc):
            newDoc = newDoc.replace(res, "")
            countPattern1 += 1
        if countPattern1 > 0:
            return newDoc
        refPattern = r"([\.,:a-zA-Z]+)([∗\*0-9][0-9,]*)(.)"
        for res in re.finditer(refPattern, newDoc):
            #print(res.groups()[1])
            if not re.search(r"\.[0-9]+\.", "".join(res.groups())):
                #print("".join(res.groups()))
                newDoc = newDoc.replace(res.groups()[0]+res.groups()[1]+res.groups()[2], res.groups()[0]+res.groups()[2])

        remaining_bracket_pattern = r"( \([ \n]*(see)? *(e\.g\.)?(c\.f\.)?(cf)?(for example)?[ \,\n]*\))"
        for res in re.findall(remaining_bracket_pattern, newDoc):
            #print(res)
            newDoc = newDoc.replace(res[0], "")
        return newDoc

    def removeAuthors(self, doc, authors=None):
        if not authors:
            authors = self.authors
        if not authors:
            return doc
        newDoc = doc
        for line in newDoc.splitlines():
            if len(line) == 0:
                continue
            for author in authors:
                if author in line:
                    #print(line)
                    newDoc = newDoc.replace(line, "\n")
            #self.nlp = spacy.load('en')
            # lineDoc = self.nlp(line)
            # ents = " ".join([entity.label_ for entity in lineDoc.ents]) #if entity.label_ in ["NORP", "ORG", "PERSON", "NORP"]
            # #print([e.label_ for e in lineDoc.ents])
            # if not "PERSON" in ents and not "ORG" in ents and not "NORP" in ents:
            #     continue
            # ents = " ".join([entity.text for entity in lineDoc.ents])
            # #print(ents)
            # #print(line)
            # if len(ents.split(" ")) >= len(line.split(" ")) * 0.7:
            #     #print("Remove Author", line)
            #     newDoc = newDoc.replace(line, "\n")
            #print("--------------")
        return newDoc

    def removeLineBreaks(self, text):
        paragraphs = self._groupParagraphs(text)
        newText = ""
        #remove line breaks
        for paragraph in paragraphs:
            for idx, line in enumerate(paragraph.splitlines()):
                newLine = line.lstrip()
                if newLine.endswith("-") and idx < len(paragraph.splitlines()) -1 :
                    newLine = newLine.rstrip("-")
                else:
                    newLine = newLine.rstrip("\n")
                    newLine += " "
                #newLine = newLine.rstrip("\n")
                newText += newLine
            newText = newText.rstrip()
            newText += "\n\n"
        return newText

    def removeSymbols(self, text, symbols):
        newText = text
        for oldsymbol in set(symbols):
            symbol = oldsymbol.lstrip().rstrip()
            if symbol in "a()[]":
                continue
            if len(symbol) > 0 and not re.search(r"^\([0-9]+\)$", symbol) and not re.search(r"^[0-9]+\.[0-9]*$", symbol) and not symbol.lstrip().rstrip().isnumeric():
                if symbol.endswith("."):
                    pointstr = "."
                else:
                    pointstr = ""
                findstr = r"[\W\.,^]" + re.escape(symbol) + r"[\W\.,$]"
                for match in re.findall(findstr, newText):
                    newText = newText.replace(match, pointstr + " ")
        return newText

    def removeAdditionalInfo(self, text, sections):
        newText = text
        for section in sections:
            sectionToFind = section + ": "
            if sectionToFind in newText:
                print(sectionToFind, newText.rindex(sectionToFind))
                newText = newText[:newText.rindex(sectionToFind)]
        return newText

    def matchParagraphs(self, text, out=False):
        newText = text
        paragraphs = self._groupParagraphs(newText)
        maxLineNumber = len(paragraphs)
        idx = len(paragraphs) - 1
        #incompleteParagraphs = [paragraph for paragraph in  if not isWholeParagraph(paragraph)]
        while idx  >= 0:
            if idx >= len(paragraphs) or idx < 0:
                break
            paragraph = paragraphs[idx]
            if(self._isWholeParagraph(paragraph)[0]):
                if out:
                    print("Already whole paragraph:", idx)
                idx -= 1
                continue
            if(paragraph.lstrip()[0].islower()):
                idx -= 1
                continue
            if idx < len(paragraphs) - 1:
                nextidx = idx +1
                oldparagraph = paragraph
                newParagraph = paragraph
                oldWrongSent = ""
                while nextidx < len(paragraphs) and nextidx - idx <= maxLineNumber:
                    nextParagraph = paragraphs[nextidx]
                    endswithformula = False
                    if len(nextParagraph) > 0 and (nextParagraph[0].isalpha() and nextParagraph[0].isupper()):
                        idx -= 1
                        endswithformula = True
                    if out:
                        print(idx, nextidx, len(paragraphs), nextParagraph[:10])
                    if(self._isWholeParagraph(nextParagraph)[0] and not endswithformula):
                        if out:
                            print("---------- continue ----------")
                        break
                    if not endswithformula:
                        newParagraph = self._mergeParagraphs(newParagraph, nextParagraph)
                        oldparagraph = oldparagraph + "\n\n" + nextParagraph
                        isWhole, wrongSent = self._isWholeParagraph(newParagraph, out=out)
                    if endswithformula or isWhole:
                        #print(nextParagraph)
                        #print(newParagraph)
                        newText = newText.replace(oldparagraph, newParagraph)
                        paragraphs = self._groupParagraphs(newText)
                        idx += 1
                        if out:
                            print("OK:", newParagraph)
                        break

                    else:
                        if wrongSent == oldWrongSent:
                            break
                        else:
                            #print("NO: ", newParagraph)
                            nextidx += 1
                            oldWrongSent = wrongSent
                idx -= 1
        return newText

    def findInlineFormula(self, text, remove=True, correctWords=None):
        initCorrWords = copy.copy(correctWords)
        if not correctWords:
            correctWords = self.correctWords
        newText = text
        formulaPattern = r"((\S+ *)(\(cid\:[0-9]*\)|[\+\−\-\*\/\=\∗\≈\<\>\~\∼]+)( *\S+| *\(\S+\)))."
        inlineFormulas = []
        singleTerms = []
        for match in re.finditer(formulaPattern, newText):
            groups = match.groups()
            if self._isWord(groups[1], correctWords=initCorrWords) or self._isWord(groups[-1], correctWords=initCorrWords):
                continue
            if self._isWord(groups[0], correctWords=initCorrWords):
                continue
            if len([token for token in list(self.nlp.pipe([groups[0]]))[0].ents if token.label_ in ["PERSON", "ORG", "NORP", "FAC"]]) > 0:
                continue
            inlineFormulas.append(groups)
            if remove:
                toRemove = groups[0]
                if not toRemove.lstrip().rstrip() == "(":
                    newText = newText.replace("{}".format(toRemove), "")
                singleTerms.append(groups[1])
                singleTerms.append(groups[-1])
            #newText = newText.replace(match, "")
        #print(inlineFormulas)
        return (newText, singleTerms)

    def groupChapters(self, text, headlines):
        chapters = []
        #if len(headlines) == 0:
        #    return [text]
        for i in range(len(headlines)-1, -1, -1):
            fromLine = str(headlines[i][0])
            fromIdx = text.index(fromLine) + len(fromLine)
            if i < len(headlines)-1:
                toLine = headlines[i+1][0]
                toIdx = text.index(toLine)
                chapterText = self._cleanup(text[fromIdx:toIdx])
            else:
                chapterText = self._cleanup(text[fromIdx:])
            chapters = [fromLine + "\n" + chapterText.lstrip().rstrip()] + chapters
        return chapters

    def joinChapters(self, chapters):
        newText = ""
        for ch in chapters:
            if len(ch.splitlines()) > 1:
                newText += ch + "\n\n"
            else:
                newText += ch
        #"\n\n".join(self.chapters)
        return newText

    def removeDuplicateLines(self, text):
        newText = text
        found = 0
        duplicates = set([line for line in newText.splitlines() if newText.count(line) > 1 and len(line) > 0])
        for line in duplicates:
            # pattern = r"([\w]{5}[^\n])" + re.escape(line) + r"([^\n]\w{5})"
            # match = re.search(pattern, newText)
            # if match:
            #     #print(match.group())
            #     if newText.find(match.group()):
            #         found += 1
            #     newText = newText.replace(match.group(), "".join(match.groups()))
            ridx = newText.rfind(line.lstrip().rstrip())
            if ridx > -1:
                found += 1
                newText = newText[:ridx] + newText[ridx + len(line):]
        #print("Replaced {} duplicates".format(found))
        return newText

    def processWholePDF(self, inputfile, outputdir):
        ## TODO:
        single_filename = inputfile.split("/")[-1]
        print(single_filename)
        subdir = inputfile.split("/")[-2] + "/"
        path=inputfile
        tpp = Threaded_PDF_Processor(path, self)
        tpp._clean()
        if len(tpp.text) == 0:
            tpp = Threaded_PDF_Processor(path, self, method="textract")
            tpp._clean()
        if outputdir:
            newFile = open(outputdir + subdir + single_filename.rstrip(".pdf")+".final.txt", "w+")
            newFile.write(tpp.text)
        #else:
            #print(tpp.text)
        print("Finished file", single_filename)
        #print(tpp.text)

    def processFunc(self, filenames, outputdir=None):
        for filename in filenames:
            try:
                self.processWholePDF(filename, outputdir)
            except Exception as e:
                print("ERROR: ", e)

    def startThreads(self, filenames, outputdir=None, numproc=multiprocessing.cpu_count()):
        pool = Pool()
        numprocesses = numproc
        print("STARTING TO PROCESS {} FILES IN {} PROCESSES".format(len(filenames), numprocesses))
        for idx in range(numprocesses):
            subrange = [filenames[i] for i in range(idx, len(filenames), numprocesses)]
            print("PROCESS #{}, total {} files.".format(idx, len(subrange)))
            # fromidx = math.floor(idx * len(filenames)/numprocesses)
            # toidx = math.floor((idx + 1)* len(filenames)/numprocesses)
            pool.apply_async(self.processFunc, args=(subrange, outputdir))
        pool.close()
        pool.join()
                #t = threading.Thread(target=self.processWholePDF, args=(filename, outputdir))
                #t.start()
            #print("Error: unable to start thread")



class Threaded_PDF_Processor:
    def __init__(self, filename, pdfprocessor, method="pdfbox"):
        self.pdfprocessor = pdfprocessor
        self.correctWords = self.pdfprocessor._getCorrectWords()
        self.text = self.pdfprocessor.extract(filename, method=method)
        self._updateStatus()
        self.correctWords = self.pdfprocessor._addCorrectWords(self.text, self.correctWords)
        self.authors, self.title = self.pdfprocessor.getMetadata(filename)
        self._updateStatus()
    def _updateStatus(self):
        pass
    def _clean(self, out=True):
        self.symbols = []
        self.text = self.pdfprocessor._replaceSpecialCharacters(self.text)
        #self._updateStatus()
        self.text = self.pdfprocessor.findReferences(self.text)
        #self._updateStatus()
        self.text = self.pdfprocessor._repairNumberedHeadlines(self.text)
        #self._updateStatus()
        #self._updateStatus()
        self.text = self.pdfprocessor.findHeadersAndFooters(self.text)
        #self.text = self.removeDuplicateLines(self.text)
        self.text = self.pdfprocessor.findFigures(self.text)
        #self._updateStatus()
        self.text = self.pdfprocessor.findTables(self.text, correctWords=self.correctWords)

        #self._updateStatus()
        self.text = self.pdfprocessor.removeInlineReferences(self.text)
        #self._updateStatus()
        self.text, self.symbols = self.pdfprocessor.findFormulas(self.text, correctWords=self.correctWords)
        #self._updateStatus()
        self.text, self.inlineFormulas = self.pdfprocessor.findInlineFormula(self.text, correctWords=self.correctWords)
        self.symbols += self.inlineFormulas
        #self._updateStatus()
        #print(self.symbols)
        self.text = self.pdfprocessor.findNoise(self.text, correctWords=self.correctWords)
        #self._updateStatus()
        #self.nlp = spacy.load("en")
        self.text = self.pdfprocessor.removeAuthors(self.text, self.authors)
        #self._updateStatus()
        self.text = self.pdfprocessor.removeSymbols(self.text, self.symbols)
        #self._updateStatus()
        self.text = self.pdfprocessor.removeDuplicateLines(self.text)
        #self._updateStatus()
        #self.text = self.removeLineBreaks(self.text)
        #self.text = self.matchParagraphs(self.text, out=False)
        self.headlines = self.pdfprocessor.findHeadlines(self.text, out=False)
        #self._updateStatus()
        self.chapters = self.pdfprocessor.groupChapters(self.text, self.headlines)
        #self._updateStatus()
        self.text = self.pdfprocessor.joinChapters(self.chapters)
        #self._updateStatus()
        self.text = self.pdfprocessor.strip(self.text, self.headlines, fromval=["Introduction", "Motivation"], toval=["References", "Acknowledgement", "Acknowledgment"])
        #self._updateStatus()
        self.text = self.pdfprocessor.removeAdditionalInfo(self.text, ["Acknowledgment", "Acknowledgement", "Acknowledgments", "Acknowledgements", "Author Contributions", "Conflicts of Interest"])
        #self._updateStatus()
        self.text = self.text.rstrip().lstrip()

