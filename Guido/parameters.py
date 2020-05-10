
instruments = ['GM']
classes = ['NOFX', 'TREM', 'DIST']

# features = ['ZCR', 'SpecRollOff', 'SpecCentr', 'SpecBandWidth', 'SpecContrast']
features = ['ZCR', 'SpecRollOff', 'SpecCentr', 'SpecBandWidth', 'SpecContrast', 'der']
n_features = 6

data_proportion = 0.1  # 1 = All
test_proportion = 0.2  # 20% test


win_length = 2000
hop_size = 2000
