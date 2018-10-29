##############################################################################
# Import some libraries
##############################################################################
import glob
import os

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Ions and Photons\G4 L7 (Ion Trap Research)\G4-L7 LabVIEW"
	r"\Experimental Control VIs\branches")
files = glob.glob(p0 + '/**/*.vi', recursive=True)
p1 = r"D:\Subversion management"
p_listfile = p1 + r"\labVIEW file list.txt"
listfile = open(p_listfile, 'w', encoding='utf-8')
for i0, val0 in enumerate(files):
    h, t = os.path.split(val0)

    filename = t + '\n'
    filepath = val0 + '\n'
    listfile.write(filepath)
print(r"Number of .vi's =", len(files))
listfile.close

p_listfolder = p1 + r"\labVIEW folder list.txt"
listfolder = open(p_listfolder, 'w', encoding='utf-8')
for i0, val0 in enumerate(os.listdir(p0)):
    folderpath = val0 + '\n'
    listfolder.write(folderpath)
listfolder.close
