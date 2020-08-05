import visdom
import numpy as np
vis = visdom.Visdom(port=7890)
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))