import mirdata
import os

data_home = os.path.expanduser('~/mir_datasets/guitarset')

dataset = mirdata.initialize('guitarset', data_home=data_home)

dataset.download(partial_download=['annotations', 'audio_mic'])
dataset.validate()
