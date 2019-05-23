import nltk, os , subprocess, code, glob, re, traceback, sys, inspect
from time import clock, sleep
from pprint import pprint
import json
import zipfile
from converterPdfToText import converterPdfToText
from converterDocxToText import converterDocxToText

class exportToCSV:
    def _init_(self, fileName ='resultToCSV.txt', resetFile = False):
        headers = ['File Name',
                'Name',
                'Email Address',
                'Phone Number',
                'Institute','Years1',
                'Institute','Years2',
                'Institute','Years3',
                'Institute','Years4',
                'Experience',
                'Degrees',
                ]
        if not os.path.isfile(fileName) or resetFile:
            fileOut = open(fileName, 'w')
            fileOut.close()
        fileIn = open(fileName)
        inString = fileIn.read()
        fileIn.close()
