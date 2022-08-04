# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#
import csv
import pprint
import re

#most .csv files have the column labels in the first row
def extractlabels(rows, labelindex=0):
	return rows[labelindex]

#most .csv files have the data starting in the 2nd row
def extractdata(rows, start=1):
	return rows[start:]	

def cleanhtmltag(rows):
	clean = re.compile('<span style="color: red; ">|</span>')
	cleandData = re.sub(clean, '',rows)
	return cleandData
		
#build a dict to map a column label to a column index for efficiently accessing all the data
def columnmap(rows):
	column_labels = extractlabels(rows)
	mymap = {col:i for i, col in enumerate(column_labels)}
	return mymap

def loadCSVfromStream(csvstream, removespace=False, encoding="utf8", quoting='|'):
	rows = []
	csvreader = csv.reader(csvstream, 
		delimiter=',',				#most CSV using ',' to separate fields
		quotechar='|',				#special quote character used '|' 
		quoting=csv.QUOTE_MINIMAL
	)
	for r, row in enumerate(csvreader):
		if removespace:		#remove spaces from all fields (this is not what you always want)
			newrow = [field.replace(" ", "") for field in row]		#strip all space
		else:
			newrow = row
		rows.append(newrow)
	return rows

#load 'csvfilepath' as a .csv file and optionally remove white spaces
def loadCSV(csvfilepath, removespace=False, encoding="utf8", quoting='|'):
	rows = []
	with open(csvfilepath, 'r', newline='', encoding=encoding) as csvfile:
		rows = loadCSVfromStream(csvfile, removespace, encoding, quoting=quoting)
	return rows

def loadXML(filepath):
	with open(filepath, 'r', newline='', encoding="utf8") as xmlfile:
		content = xmlfile.read()
		clean = re.compile('<span style="color: red; ">|</span>')
		cleandXMLData = re.sub(clean, '', content)
	return cleandXMLData
#unit test
if __name__=='__main__':
	pp = pprint.PrettyPrinter(indent=1, width=120)

	inputdir = 'input/'
	csvdata = loadCSV(inputdir+'members.csv', removespace=True)
	pp.pprint(csvdata)

	labels = extractlabels(csvdata)
	data   = extractdata(csvdata)
