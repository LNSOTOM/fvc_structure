# import sample data
from skimage.data import cells3d

import napari

# create a `Viewer` and `Image` layer here
viewer, image_layer = napari.imshow(cells3d())

# print shape of image data
print(image_layer.data.shape)

# start the event loop and show the viewer
napari.run()