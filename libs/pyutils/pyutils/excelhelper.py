# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#

#!/usr/bin/python3
import argparse
import os, sys
#import numpy as np
from collections import defaultdict
import pprint
from datetime import datetime
import csv, codecs
#import difflib
#import re
#import openpyxl 		#this cannot read the old Excel format from DMS and have trouble with cell references
import xlrd
#import numpy as np

import pyutils.htmlhelper as html
import pyutils.dictutils as dictutils
import pyutils.dirutils as dirutils
import pyutils.loadcsv as loadcsv

kLogging=False
kOpenpyxl=False		#openpyxl|xlrd
kHTML=True
kShorterHTML=True

#
#https://stackoverflow.com/questions/16229610/xlrd-original-value-of-the-cell
#
def getmax(sheet):
	if kOpenpyxl:
		maxrow = sheet.max_row
		maxcol = sheet.max_column
	else:
		maxrow = sheet.nrows
		maxcol = sheet.ncols
	return maxrow, maxcol

def getsheet(wb, sheetname):
	sheet = None
	if kOpenpyxl:
		sheet = wb[sheetname]
	else:
		sheet = wb.sheet_by_name(sheetname)
	return sheet

def dumpcol(oursheet, column, maxrow):
	counter = 0
	for i in range(1, maxrow, 1):
		cell = oursheet.cell(row=i, column=column)
		print("cell[%s]: %s" % (cellid, cell.value))

def getcell(sheet, row, col):
	cell = sheet.cell_value(rowx=row, colx=col)
	return cell

def dump_column_labels(oursheet, maxcol, mycolumn='A', row=1, klogging=False):
	labels = []
	row = oursheet.row(row-1) 
	for idx, cell in enumerate(row):
		if klogging:
			print(cell.value)
		labels.append(cell.value)
	return labels

def extractrow(wb, oursheet, colindices, row, mapping=lambda wb, x: x):
	our_row = [None]*len(colindices)
	row = oursheet.row(row-1) 

	for i, column in enumerate(colindices):
		if column != None:
			cell = row[column]
			value = mapping(wb, cell)
		else:
			value = 'Unknown'
		our_row[i] = value
	return our_row

def extractcol(rows, col):
	column = []
	for row in rows[1:]:
		column.append(row[col])
	return column

def gettype(cell):
	cell_type_str = xlrd.sheet.ctype_text.get(cell.ctype, 'unknown type')
	return cell_type_str

def convertdate(wb, date_val):
	py_date = xlrd.xldate.xldate_as_datetime(date_val, wb.datemode)  #TODO: try xlrd.xldate_as_tuple
	return py_date

def validatephone(phone_tuple):
	valid = ((len(phone_tuple) >= 3) and 
			 (len(phone_tuple[0]) == 3) and 
			 (len(phone_tuple[1]) == 3) and 
			 (len(phone_tuple[2]) == 4))
	return valid

def convertphone(phone_val):
	#print(f"convertphone({phone_val})")
	fields = ''
	output = 'NA'

	if phone_val:
		if type(phone_val) is str:
			fields = phone_val.split('-')
		else:
			pstr = str(int(phone_val))
			fields = [pstr[:3], pstr[3:6], pstr[-4:]]

	if validatephone(fields):
		output = '-'.join(fields)

	return output

#map column lables to column indices
def map_collabels(label_lookup, columns):
	are_ints = all(map(lambda x: isinstance(x, int), columns))

	if are_ints:
		cols = columns
	else:
		cols = []
		for i, column in enumerate(columns):
			col = lookup_col(label_lookup, column)
		cols.append(col)
	return cols

def lookup_col(label_lookup, mycolumn, row=1):
	return label_lookup.get(mycolumn, None)

def isvalid(row):
	return any(x is not None for x in row)	

def extract(wb, sheet, columns, label_lookup, mapping=lambda wb, x: x):
	maxrow, maxcol = getmax(sheet)
	subset = []

	colindices = map_collabels(label_lookup, columns)

	for row in range(1, maxrow+1):
		ourrow = extractrow(wb, sheet, colindices, row, mapping=mapping)
		if isvalid(ourrow):
			subset.append(ourrow)

	return subset

def removecolumns(extracted, columns=[]):
	output = []
	for row in extracted:
		newrow = [data for col, data in enumerate(row) if col not in columns]
		output.append(newrow)
	return output

def remap_header(row, mapping):
	return row

def remap_row(row, mapping=lambda x: x):
	result = map(mapping, row)
	return result

def cell_mapper(wb, cell):
	typedispatch = {
		'xldate': convertdate,		#dates are encoded in strange way - convert to Python datetime
	}
	cell_type_str = gettype(cell)
	mapper = typedispatch.get(cell_type_str, lambda wb, x: x)
	cell = mapper(wb, cell.value)

	return cell

def extractnames(rows, col=0):
	names = extractcol(rows, col)
	return names

def extractemails(rows, col=1):
	emails = extractcol(rows, col)
	return emails

def extractphones(rows, col=2):
	phones = extractcol(rows, col)
	return phones

def extractcontacts(rows, cols):
	names  = extractnames(extracted, col=cols[0])
	emails = extractemails(extracted, col=cols[1])
	phones = extractphones(extracted, col=cols[2])
	return names, emails, phones

def writecontacts(csvfilepath, extracted):
	names, emails, phones = extractcontacts(extracted, (nameindex, emailindex, phoneindex))

	with open(csvfilepath, 'w', newline='', encoding="utf8") as csvfile:
		csvwriter = csv.writer(csvfile, 
			delimiter=',',				#most CSV using ',' to separate fields
            quotechar='|',				#special quote character used '|' 
            quoting=csv.QUOTE_MINIMAL
        )

		for r, name in enumerate(names):
			email = emails[r]
			phone = phones[r]
			csvwriter.writerow([name, email, phone])

def convert2Gmail(labels, labelmapper):
	labelmapper = {
		"Name": 			"Name",		#TODO: deal with Given Name, Family Name
		"Contact Email": 	"E-mail 1 - Value",
		"Phone #:":			"Phone 1 - Value",
	}
	workingdir = dirutils.workingdir(__file__)
	template = loadcsv.loadCSV(workingdir + 'input/gmail-template.csv')
	return template

def writecontacts4gmail(gmailfilepath, extracted):
	ourlabels = [
		"Name",		#TODO: deal with Given Name, Family Name
		"E-mail 1 - Value",
		"Phone 1 - Value",
	]
	template = convert2Gmail(extracted[0], lambda x: x)

	print(template[0])
	#print(template[1])
	#print(len(template[0]))
	#print(len(template[1]))

	nameindex  = template[0].index(ourlabels[0])
	emailindex = template[0].index(ourlabels[1])
	phoneindex = template[0].index(ourlabels[2])
	#print(nameindex, emailindex, phoneindex)

	with open(gmailfilepath, 'w', newline='', encoding="utf8") as csvfile:
		csvwriter = csv.writer(csvfile, 
			delimiter=',',				#most CSV using ',' to separate fields
            quotechar='|',				#special quote character used '|' 
            quoting=csv.QUOTE_MINIMAL
        )
		ncols = len(template[0])

		for r, row in enumerate(extracted):
			if r == 0:
				csvwriter.writerow(template[0])
			else:
				orow = template[1].copy()
				orow[nameindex] = row[0]
				orow[emailindex] = row[1]
				orow[phoneindex] = row[2]

				assert(len(orow) == ncols)
				csvwriter.writerow(orow)
		print(f"{len(extracted)} rows written")

	#for r, row in enumerate(extracted):
		#print(f"[{r}]={row[0]}")

def write1cell_html(cell, f):
	f.write("<td>")
	f.write(f"{cell}")
	f.write('</td>\n')

def write1row_html(row, f):
	#one row of images
	f.write('<tr align = \"center\">\n')

	for cell in row:
		write1cell_html(cell, f)

	f.write('</tr>\n')

def write_assignments(workingdir, cols):
	filename = 'mentors/applicants-assignments-sorted.csv'
	try:
		assigns = loadcsv.loadCSV(workingdir+filename)
	except:
		print(f"Cannot load '{filename}'")
	#pp.pprint(assigns)


if __name__=='__main__':
	pp = pprint.PrettyPrinter(indent=1, width=120)

	parser = argparse.ArgumentParser(description='procSearch.py')
	parser.add_argument('-excel', type=str, default='Excel/details_20190425_10.xlsx', help='.xlsx')
	parser.add_argument('-output', type=str, default='AutoLabel.xlsx', help='.xlsx')
	parser.add_argument('-sheetname', type=str, default='Form Responses 1', help='The active sheet')
	parser.add_argument('-columns', type=list, default=['Name', 'Contact Email', 'Phone #:'], nargs='+', help='The active sheet')
	args = parser.parse_args()

	workingdir = dirutils.workingdir(__file__)
	excel_file = args.excel
	excel_file = workingdir + 'input/Mentorship Application (Responses).xlsx'
	kEnableInfer = False

	output_dir = 'output/'

	dirutils.mkdirname(output_dir)
	#output_file = makeOutputName(excel_file)
	active_sheets = dict()

	#1: get the list of sheets & columns from <excel file>.meta
	our_sheets = [args.sheetname] 		#active sheet - where we can find our data
	our_columns = args.columns 			#list of columns we are interested in

	#2: load and open the Excel file
	wb = xlrd.open_workbook(excel_file)		#'xlrd' can load the older Excel format from DMS
	sheet_names = wb.sheet_names()
	sheet = wb.sheet_by_index(0)
	#print(sheet_names)

	#print(" D3 %s" % sheet.cell_value(2, 3))
	pydate = convertdate(wb, sheet.cell_value(1, 0))	#double check to see if date/time converted nicely
	print(pydate)
	print(" sheet_names: %s" % sheet_names)

	phone_val = convertphone(getcell(sheet, row=2, col=3))
	print(phone_val)
	print(f"row[1]={sheet.row_values(1)}")

	for name in sheet_names:
		sheet = getsheet(wb, name)
		if kLogging:
			print(" sheet title '%s'" % name)
	
		if name in set(our_sheets):
			active_sheets[name] = sheet

	if len(active_sheets) == 0:
		print(" no active sheets, quiting..")
		quit()

	#4: process all the sheets requested
	for name in active_sheets:
		sheet = getsheet(wb, name)
		maxrow, maxcol = getmax(sheet)

		#our_columns = [1, 2, 3]			#A, B, C for now
		our_columns = list(range(1, maxcol))		#extract all columns

		#4.1: extract column labels
		labels = dump_column_labels(sheet, maxcol, mycolumn='A', row=1, klogging=False)
		if kLogging:
			print("labels %s" % labels)
		label_lookup = {k: v for v, k in enumerate(labels)}

		#map column labels to column indices
		for col in our_columns:
			col_number = lookup_col(label_lookup,  col)
			if kLogging:
				print("'%s': column %d" %(col, col_number))

		#4.3: extract columns specified in the .meta file
		extracted = extract(wb, sheet, our_columns, label_lookup, mapping=cell_mapper)

		#4.4: sort by "Name"(column 0)
		extracted[1:] = sorted(extracted[1:], key=lambda row: row[0])

		if kLogging:
			pp.pprint(extracted)

		nameindex  = 0  #lookup_col(label_lookup, "Name") - 1
		emailindex = 1  #lookup_col(label_lookup, "Contact Email") - 1
		phoneindex = 2  #lookup_col(label_lookup, "Phone #:") - 1
		missing = []
		r = 0
		for row in extracted[1:]:
			#
			# do something here with 'row'
			#
			phone = row[phoneindex]
			cleaned = convertphone(phone)
			row[phoneindex] = cleaned
			#print(f"{row[nameindex]}, {cleaned}")
			#pp.pprint(row)
			r += 1
		print(f"{r} rows processed.")

		#2: extract the contact informations
		names, emails, phones = extractcontacts(extracted, (nameindex, emailindex, phoneindex))

		#print(names)
		#print(emails)
		#print(phones)

		#2.1: write contacts to .csv
		csvfilepath = 'applicants.csv'
		writecontacts(csvfilepath, extracted)

		gmailfilepath = 'applicants-gmail.csv'

		#2.1: write contacts to .csv in gmail format
		writecontacts4gmail(gmailfilepath, extracted)		#write 1 row for testing

		#3: find all the applicants without phone#s:
		missingphone = []
		for i, phone in enumerate(phones):
			if phone == 'NA':
				missingphone.append((names[i], emails[i]))
		print(f"missing phones: {missingphone}")		

		#4: reformat the whole dataset as HTML for browsing
		if kHTML:
			if kShorterHTML:
				extracted = removecolumns(extracted, columns=[emailindex, phoneindex])
				htmlfname = 'applicants-shorter.html'
			else:
				htmlfname = 'applicants.html'

			with open(htmlfname,'w', encoding="utf8") as f:
				html.writeHeader(f)
				html.writeTableHeader(f)

				for row in extracted[1:]:
					write1row_html(row, f)

				html.writeTableTrailer(f)
				html.writerTrailer(f)

		write_assignments(workingdir, [])
