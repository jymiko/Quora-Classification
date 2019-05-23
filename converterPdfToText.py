from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

def converterPdfToText(path):
    resourceManager = PDFResourceManager()
    resourceString = StringIO()
    codeType = 'utf-8'
    laParams = LAParams()
    converter = TextConverter(resourceManager, resourceString, codec = codeType, laparams = laParams)
    filePath = file(path, 'rb')
    interpreter = PDFPageInterpreter(resourceManager, converter)
    password = ""
    maxpages = 0
    caching = True
    pageNumber = set()
    for page in PDFPage.get_pages(filePath, maxpages = maxpages, password = password, caching = caching, check_extractable = True):
        interpreter.process_page(page)
    filePath.close()
    stringValue = resourceString.getvalue()
    resourceString.close()
    return stringValue
    
