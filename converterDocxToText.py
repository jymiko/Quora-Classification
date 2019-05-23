from docx import Document

def converterDocxToText(path):
    document = Document(path)
    return "\n".join([paragraph.text for paragraph in document.paragraphs ])