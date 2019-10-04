##############################################################################
# Import some libraries
##############################################################################
import glob
import os

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Ions and Photons\F3 L6 (Ion Trap Cleanroom)")
p1 = (r"D:\Ions and Photons\F4 L10 (Few Photon Metrology)")
p2 = (r"D:\Ions and Photons\F4 L8 (Photon Pair Source)")
p3 = (r"D:\Ions and Photons\F5 L10 (Quantum Photonics)")
p4 = (r"D:\Ions and Photons\F5 L9 (Ion Trap Development)")
p5 = (r"D:\Ions and Photons\G4 L7 (Ion Trap Research)")

vis0 = glob.glob(p0 + '/**/*.vi', recursive=True)
vis1 = glob.glob(p1 + '/**/*.vi', recursive=True)
vis2 = glob.glob(p2 + '/**/*.vi', recursive=True)
vis3 = glob.glob(p3 + '/**/*.vi', recursive=True)
vis4 = glob.glob(p4 + '/**/*.vi', recursive=True)
vis5 = glob.glob(p5 + '/**/*.vi', recursive=True)

files0 = glob.glob(p0 + '/**/*', recursive=True)
files1 = glob.glob(p1 + '/**/*', recursive=True)
files2 = glob.glob(p2 + '/**/*', recursive=True)
files3 = glob.glob(p3 + '/**/*', recursive=True)
files4 = glob.glob(p4 + '/**/*', recursive=True)
files5 = glob.glob(p5 + '/**/*', recursive=True)

p = r"D:\Subversion management"
p_listfile = p + r"\labVIEW file list.txt"
listfile = open(p_listfile, 'w', encoding='utf-8')
for i0, val0 in enumerate(files0):
    h, t = os.path.split(val0)

    filename = t + '\n'
    filepath = val0 + '\n'
    listfile.write(filepath)

listfile.close


print(r"Number of .vi's F3 L6 =", len(vis0))
print(r"Number of .vi's F4 L10 =", len(vis1))
print(r"Number of .vi's F4 L8 =", len(vis2))
print(r"Number of .vi's F5 L10 =", len(vis3))
print(r"Number of .vi's F5 L9 =", len(vis4))
print(r"Number of .vi's G4 L7 =", len(vis5))
print("total vi's =", len(vis0) +
      len(vis1) +
      len(vis2) +
      len(vis3) +
      len(vis4) +
      len(vis5))

print(r"Number of files F3 L6 =", len(files0))
print(r"Number of files F4 L10 =", len(files1))
print(r"Number of files F4 L8 =", len(files2))
print(r"Number of files F5 L10 =", len(files3))
print(r"Number of files F5 L9 =", len(files4))
print(r"Number of files G4 L7 =", len(files5))
print("total files =", len(files0) +
      len(files1) +
      len(files2) +
      len(files3) +
      len(files4) +
      len(files5))

p_listfolder = p + r"\labVIEW folder list.txt"
listfolder = open(p_listfolder, 'w', encoding='utf-8')
for i0, val0 in enumerate(os.listdir(p0)):
    folderpath = val0 + '\n'
    listfolder.write(folderpath)
listfolder.close
