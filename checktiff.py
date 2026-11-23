import tifffile
import numpy as np

path = "C:\\Users\\nx\\Downloads\\app_samples\\sample1_24462488\\2020-09-28_VIIRS_Day.tif"

with tifffile.TiffFile(path) as tif:
    page = tif.pages[0]

    print("Shape:", page.shape)
    print("Dtype:", page.dtype)
    print("Photometric:", page.photometric)
    print("Description:", page.description)

data = np.load("C:\\Users\\nx\\Downloads\\app_samples\\sample1_24462488\\expected.npz")

for key in data.files:
    print(key, data[key].shape, data[key].dtype)