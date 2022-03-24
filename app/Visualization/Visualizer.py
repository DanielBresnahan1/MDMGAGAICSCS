from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from PIL import Image
import numpy as np
import os

xs = [1,1.5,2,2]
ys = [1,2,3,1]
zs = [0,1,2,0]

imgs = [Image.open("b.jpg"), Image.open("b.jpg"), Image.open("c.jpg"), Image.open("c.jpg")]
for img in imgs :
    img = np.asarray(img)

"""Takes a set of an array of threeples of the x y and z coordinates respectively in
the form [(x,y,z), (x,y,z), ect] and an array of numpy arrays of images for each point
and creates a 3D scatterplot with them with the images attatched to the points"""
class ImageAnnotations3D():
    """xyz is an array of threeples of the x y and z coordinates respectively in
    the form [(x,y,z), (x,y,z), ect], 
    imgs is an array of numpy arrays of images for each point"""
    def __init__(self, xyz, imgs):
        self.xyz = xyz
        self.imgs = imgs
        self.fig = plt.figure(figsize=(19.20, 10.80))
        self.c = ["b","r","g","gold"]
        self.ax3d = self.setAx3d()
        self.ax2d = self.setAx2d()
        self.annot = []
        for s,im in zip(self.xyz, self.imgs):
            x,y = self.proj(s)
            self.annot.append(self.image(im,[x,y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

        self.funcmap = {"button_press_event" : self.ax3d._button_press,
                        "motion_notify_event" : self.ax3d._on_move,
                        "button_release_event" : self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                        for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax3d, 
            calculate position in 2D in ax2d """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        imageZoom = self.zoomCalc(arr.size[0])
        im = offsetbox.OffsetImage(arr, zoom=imageZoom)
        im.image.axes = self.ax3d
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                            xycoords='data', boxcoords="offset points",
                            pad=0, arrowprops=dict(arrowstyle="->"))
        self.ax2d.add_artist(ab)
        return ab
    
    def zoomCalc(self, imageSize):
        """Scales images to a reasonable size based on their resolution"""
        imageZoom = 1
        while(imageSize * imageZoom > 50):
            imageZoom = imageZoom/2
        return imageZoom

    def update(self,event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                        np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s,ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)
                
    def setAx3d(self):
        x =[]
        y =[]
        z =[]
        counter = 0
        for counter in range(len(self.xyz)):
            x.append(self.xyz[counter][0])
            y.append(self.xyz[counter][1])
            z.append(self.xyz[counter][2])
        ax3 = self.fig.add_subplot(111, projection=Axes3D.name)
        ax3.scatter(x, y, z, marker="o")
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        return ax3
        
    def setAx2d(self):
        # Create a dummy axes to place annotations to
        ax2 = self.fig.add_subplot(111,frame_on=False) 
        ax2.axis("off")
        ax2.axis([0,1,0,1])
        return ax2

ia = ImageAnnotations3D(np.c_[xs,ys,zs],imgs)


plt.show()