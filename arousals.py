import edfdata

if __name__ == "__main__":
    print("Automatic detection of arousals")
    path = "/home/sasi/edf/SHHS"
#    edfFiles = loadAllEdf(path)
#    print("Loaded {} EDF files".format(len(edfFiles)))

#    signals, rates, Xs = edfdata.loadData(edfFiles)
    edfFiles = edfdata.loadPath(path)
    print("Loading data")
    window_seconds = 30
    signals, rates, Xs = edfdata.loadData(edfFiles, window_seconds)
    print("Loading labels")
    Y = edfdata.loadLabels(edfFiles, window_seconds)
