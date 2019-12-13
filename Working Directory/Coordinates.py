import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img =  mpimg.imread('../Map/main.png')

fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(img)

def onclick(event):
    print("{:.6f},{:.6f}".format(event.xdata / 44.0, event.ydata / 44.0))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()