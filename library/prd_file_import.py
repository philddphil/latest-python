import glob
import numpy as np
import re
import os

from datetime import datetime


###############################################################################
# File loading defs
###############################################################################
# Load temperature log ########################################################
def load_T_log(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close
    t_sec = []
    t_date = []
    T = []
    for i0, val in enumerate(data[0:]):
        t_string = val.split("\t")[0]
        t_datetime = datetime.strptime(t_string, "%d/%m/%Y %H:%M:%S")
        t_sec = np.append(t_sec, t_datetime.timestamp())
        t_date = np.append(t_date, t_datetime)
        T = np.append(T, float(val.split("\t")[1]))
    return T, t_date


# Load a data set of Pressure values ##########################################
def load_Pressure(filepath):
    f = open(filepath, "r")
    lines = f.readlines()
    times = []
    for x in lines:
        times.append(x.split('\t')[1].rstrip())
    f.close()

    # read files for pressure values ##########################################
    data = np.genfromtxt(filepath)
    Ps = data[:, 0]

    # format time srings ######################################################
    ts = []
    for i0, val in enumerate(times[:-1]):
        t = str(times[i0])
        (h, m) = t.split('.')
        t = int(h) * 60 + int(m)
        ts.append(t)
    Δts = [0]
    for i0, val in enumerate(ts):
        Δt = ts[i0] - ts[0]
        Δts.append(Δt)
    return Δts, Ps


# Load data set from Psat measurement (note folder not file) ##################
def load_Psat(folderpath):
    P_file = folderpath + r'\P.txt'
    cps1_file = folderpath + r'\APD1.txt'
    cps2_file = folderpath + r'\APD2.txt'

    P_data = np.loadtxt(P_file)
    Ps = P_data.mean(axis=1)

    cps1_data = np.loadtxt(cps1_file)
    cps1 = cps1_data.mean(axis=1)
    cps2_data = np.loadtxt(cps2_file)
    cps2 = cps2_data.mean(axis=1)

    cps = cps1 + cps2
    return Ps, cps


# Load a data set of APD count rates ##########################################
def load_APD(filepath):
    data = np.loadtxt(filepath)
    t = data[:, 0]
    a = data[:, 1]
    b = data[:, 2]
    return t, a, b


# Load a Thorlabs PM100D logged power series ##################################
def load_PM100_log(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()
    t = []
    P = []
    for i0, val in enumerate(data[2:]):
        t_string = val.split("\t")[0]
        t_datetime = datetime.strptime(t_string, "%d/%m/%Y %H:%M:%S.%f   ")
        t = np.append(t, t_datetime.timestamp() * 1000)
        P = np.append(P, float(val.split("\t")[1]))
    return (t, P)


# Load spec file (.txt) #######################################################
def load_spec(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.readlines()
    data_line = 0
    skip_head = 0
    while skip_head == 0:
        try:
            float(data[data_line].split('\t')[0])
        except ValueError:
            data_line = data_line + 1
        else:
            skip_head = 1
    λ = []

    cts = np.zeros(shape=(len(data) - data_line,
                          len(data[data_line].split('\t')) - 1))
    for i0, val0 in enumerate(data[data_line:]):
        λ = np.append(λ, float(val0.split("\t")[0]))
        for i1, val1 in enumerate(val0.split("\t")[1:]):
            cts[i0][i1] = float(val1)

    return (λ, cts)


# Load all spec (.txt) in directory ###########################################
def load_spec_dir(dirpath):
    datafiles = glob.glob(dirpath + r'\*.txt')
    datafiles = natural_sort(datafiles)

    # Initialise lists of datasets
    λs = []
    ctss = []
    lbs = []
    count = 0

    # load each spec file, generate λ array cts array, label and plot name

    for i0, val0 in enumerate(datafiles[0:]):
        cts = []
        λ, cts = load_spec(val0)

        for i1, val1 in enumerate(cts[0, :]):
            lb = str(i1) + ' ' + os.path.basename(val0)
            # cts_crr = prd_data_proc.cos_ray_rem(cts[:, i1], 50)
            ctss.append(list(cts[:, i1]))
            λs.append(λ)
            lbs.append(lb)

            count += 1

    return λs, ctss, lbs


# Load SCM image #############################################################
def load_SCM_F5L10(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if 'X initial / V' in j0:
            x_init = float(data[i0].split("\t")[-1])
        if 'X final / V' in j0:
            x_fin = float(data[i0].split("\t")[-1])
        if 'X increment / V' in j0:
            x_res = float(data[i0].split("\t")[-1])
        if 'Y initial / V' in j0:
            y_init = float(data[i0].split("\t")[-1])
        if 'Y final / V' in j0:
            y_fin = float(data[i0].split("\t")[-1])
        if 'Y increment / V' in j0:
            y_res = float(data[i0].split("\t")[-1])
        if 'Y wait period / ms' in j0:
            data_start_line = i0 + 2

    x = np.linspace(x_init, x_fin, int(x_res))
    y = np.linspace(y_fin, y_init, int(y_res))
    img = np.loadtxt(data[data_start_line:])
    return (x, y, img)


# Load multiple .csvs ########################################################
def load_multicsv(directory):
    f1 = directory + r'\*.csv'
    files = glob.glob(f1)
    data_all = np.array([])
    for i1, val1 in enumerate(files[0:]):
        data = np.genfromtxt(val1, delimiter=',')
        data_all = np.append(data_all, data)

    return data_all


# Plot an image from a .csv  (saved by LabVIEW) ##############################
def img_csv(file, delim=',', sk_head=1):
    im = np.genfromtxt(file, delimiter=delim, skip_header=sk_head)
    im_size = np.shape(im)
    y = np.arange(im_size[0])
    x = np.arange(im_size[1])
    X, Y = np.meshgrid(x, y)
    coords = (X, Y)
    return (im, coords)


# Plot an image from a .txt (saved by labVIEW) ###############################
def img_labVIEW(file):
    im = np.loadtxt(file)
    im_size = np.shape(im)
    y = np.arange(im_size[0])
    x = np.arange(im_size[1])
    X, Y = np.meshgrid(x, y)
    coords = (X, Y)
    return (im, coords)


# Natural sort ###############################################################
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Load HH file (.txt) ########################################################
def load_HH(filepath):
    a = open(filepath, 'r', encoding='ascii')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if '#channels per curve' in j0:
            channels = float(data[i0 + 1])
        if '#ns/bin' in j0:
            τ_res0 = float(data[i0 + 1].split("\t")[0])
        if '#counts' in j0:
            data_start_line = i0 + 1

    τs = np.linspace(0, τ_res0 * (channels - 1), channels)
    all_cts = np.loadtxt(data[data_start_line:])
    cts = all_cts[:, 0]
    return (τs, cts)


# Load NEA 121 Profile data (.txt) ##########################################
def load_NEA_Prof(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if 'Period length' in j0:
            p_HMS = data[i0].split("\t")[1]
            p_HMS = p_HMS[1:-1]
            p_H = float(p_HMS.split(':')[0])
            p_M = float(p_HMS.split(':')[1])
            p_S = float(p_HMS.split(':')[2])
            p_s = 360 * p_H + 60 * p_M + p_S
        if 'Total number of periods' in j0:
            p_tot = float(data[i0].split("\t")[-1])
        if 'Trig time' in j0:
            t_init = data[i0].split("\t")[1]
        if 'Period:' in j0:
            data_start_line = i0 + 1
    ts = []
    dBs = []
    for i0, v0 in enumerate(data[data_start_line:]):
        ts.append(float(v0.split("\t")[0]))
        dB_temp = v0.split("\t")[3:]
        dBs.append([float(i0) for i0 in dB_temp])
    return ([p_s, p_tot, t_init], ts, dBs)


# Load NEA 121 Global data (.txt) ###########################################
def load_NEA_Glob(filepath):
    a = open(filepath, 'r', encoding='utf-8')
    data = a.readlines()
    a.close()
    for i0, j0 in enumerate(data):
        if 'Period length' in j0:
            p_HMS = data[i0].split("\t")[1]
            p_HMS = p_HMS[1:-1]
            p_H = float(p_HMS.split(':')[0])
            p_M = float(p_HMS.split(':')[1])
            p_S = float(p_HMS.split(':')[2])
            p_s = 360 * p_H + 60 * p_M + p_S
        if 'Total number of periods' in j0:
            p_tot = float(data[i0].split("\t")[-1])
        if 'Trig time' in j0:
            t_init = data[i0].split("\t")[1]
        if 'Period:' in j0:
            data_start_line = i0 + 1

    dBs_str = data[data_start_line].split("\t")[3:]
    fs_str = data[data_start_line - 1].split("\t")[3:]
    dBs = [float(i0) for i0 in dBs_str]
    fs = []
    for i0, v0 in enumerate(fs_str):
        if 'k' in v0.split()[1]:
            fs.append(1000 * float(v0.split()[0]))
        else:
            fs.append(float(v0.split()[0]))

    return [p_s, p_tot, t_init], fs, dBs
