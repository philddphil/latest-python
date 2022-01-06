import numpy as np

f0 = (r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717"
	r"\0\py data\arrival time files\ch0 slice 1.npy")

f1 = (r"C:\local files\Experimental Data\F5L10 SPADs Fastcom tech\20200717"
	r"\0\py data\arrival time files\ch0 slice 2.npy")

data0 = np.load(f0, allow_pickle=True)
data1 = np.load(f1, allow_pickle=True)

print(np.shape(data0))
print(np.shape(data1))

datas=[data0,data1]
datas.sort(key=len)

print(np.shape(datas[0]))
print(np.shape(datas[1]))
tot_lines = 0
for i0, v0 in enumerate(data1):
	tot_lines += len(v0)

print(tot_lines)