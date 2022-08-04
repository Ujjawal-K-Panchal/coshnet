# Copyright (C) Manchor Ko - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Manchor Ko man961@yahoo.com, August 2019
#
import os, sys


def writeHeader(f):
	message  = "<!DOCTYPE html>\n"
	message += "<html>\n"
	message += "<body>\n"
	f.write(message)
	return

def writerTrailer(f):
	message  = "</body>\n"
	message += "</html>\n"
	f.write(message)
	return

def writeTableHeader(f):
	message = "<table style=\"width:100%\">\n";
	f.write(message)
	return

def writeTableTrailer(f):
	message = "</table>\n";
	f.write(message)
	return

def oneimg(filepath, width=200):
	f.write("<td><img src=")
	f.write('"')
	f.write(filepath)
	f.write('"')
	str = f' style="width:{width}px;\"'
	f.write('></td>\n')

def onefilename(f, filepath):
	f.write('<td>')
	f.write(filepath)
	f.write('</td>')

def perfile(
	filepath
):
	print(filepath)
	#one row of images
	f.write('<tr align = \"center\">\n')

	oneimg(filepath)

	prevbest = getMatchingRef(filepath, prevbestfolder)
	print('prevbest %s' % prevbest)
	oneimg(prevbest)

	reffile = getMatchingRef(filepath, reffolder)
	oneimg(reffile)

	shadingf = getShadingFile(filepath, bestfolder + '/shading')
	oneimg(shadingf)

	prevshadingf = getShadingFile(filepath, prevbestfolder + '/shading')
	oneimg(prevshadingf)

	refshading = getMatchingRef(shadingf, reffolder)
	oneimg(refshading)
	f.write('</tr>\n')

	# 2nd row for filename and distance value
	f.write('<tr align = \"center\">\n')
	onefilename(f, filepath)
	onefilename(f, prevbest)
	onefilename(f, reffile)

	onefilename(f, shadingf)
	onefilename(f, prevshadingf)
	onefilename(f, refshading)

	f.write('</tr>\n')

	return

# main()
if __name__ == '__main__':
	htmlfname = 'test.html'

	with open(htmlfname,'w') as f:
		writeHeader(f)
		writeTableHeader(f)

		#folder_iter(rootdir + bestfolder, perfile, deffolder_filter, pngfile_filter)

		writeTableTrailer(f)
		writerTrailer(f)
