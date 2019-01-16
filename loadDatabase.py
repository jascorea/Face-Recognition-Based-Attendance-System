#simple test program to create a data base of student name and their title
import pickle
import os
import numpy as np



## Automatically generated START
exists = os.path.isfile('/home/uncc_embed/Desktop/CV/ComputerVisionproject/myfile.pkl')
if exists:
	print("File exists")
	name=raw_input("Enter the name")
	mydict = {}	
	pkl_fileR = open('myfile.pkl', 'rb')
	mydict2 = pickle.load(pkl_fileR)
	pkl_fileR.close()
	if name in mydict2:
		print('duplicate')
	else:
		output = open('myfile.pkl', 'a')
		title=raw_input("Enter the title")
		mydict[name]=title
		pickle.dump(mydict, output)
		output.close()
		print('Student saved')
    
else:
	mydict={}
	name=raw_input("Enter the name")
	title=raw_input("Enter the title")	
	mydict[name]=title
	output = open('myfile.pkl', 'wb')
	pickle.dump(mydict, output)
	output.close()





#BDICT = {} ## Wiping the dictionary

## Usually in a loop
#firstrunDICT = True

#if firstrunDICT:
#    with open('DICT_ITEMS.txt', 'r') as dict_items_open:
#        BDICT = pickle.load(dict_items_open)
        #print BDICT

#check for duplicate keys
#if'Ad' in BDICT:
#	print ("Already exists")
#studentnames=[]
#for key, value in BDICT.iteritems() :
   #print key, value
#   studentnames.append(key)
#stdnpy=np.asarray(studentnames)
#print(stdnpy)