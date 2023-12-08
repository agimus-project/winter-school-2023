import matplotlib.pyplot as plt
import numpy as np

def plotBicopter(x,f=None,pltAx=None):
    '''
    Plot a bicopter of a 2D matplotlib, with arrows showing the thrust.
    '''
    if pltAx is None:
        pltAx=plt.gca()

    g=10
    span = .2
    mass = 2.
    force_scale=.5/mass/g

    f = [mass*g/2,mass*g/2] if f is None or len(f)==0 else f

    a, b, th = x[:3]
    c, s = np.cos(th), np.sin(th)
    fr,fl = f[0]*force_scale, f[1]*force_scale
    refs = [
        pltAx.arrow(a + span * c, b + span * s,
                    -s * fr, c * fr, head_width=.05),
        pltAx.arrow(a - span * c, b - span * s,
                    -s * fl, c * fl, head_width=.05,color='r')
    ]
    return refs


def plotBicopterSolution(xs, pltAx=None,show=False):
    '''
    Plot a sequence of bicopters by calling iterativelly plotBicopter.
    If need be, create the figure window.
    '''
    if show=='interactive':
        plt.ion()
    if pltAx is None:
        f,pltAx = plt.subplots(1,1, figsize=(6.4, 6.4))
    for x in xs:
        plotBicopter(x,[],pltAx)
    pltAx.axis([-2, 2., -2., 2.])
    if show==True:
        plt.show()

        
class ViewerBicopter:
    '''
    Wrapper on meshcat to display a 3d object representing a (quad) bicopter.
    Call display(x) to display one frame, and displayTrajectories for a sequence.
    '''
    def __init__(self):
        '''
        Init using example-robot-data to get the mesh of the quadcopter Hector,
        and meshcat to render it.
        '''
        import example_robot_data as robex
        from utils.meshcat_viewer_wrapper import MeshcatVisualizer
        import pinocchio as pin

        hector=robex.load('hector')
        self.viz=MeshcatVisualizer(hector)
        self.gname = self.viz.getViewerNodeName(self.viz.visual_model.geometryObjects[0],
                                                pin.VISUAL)

    def display(self,x):
        '''
        Display one pose of the quadcopter in meshcat.
        '''
        import pinocchio as pin
        M = pin.SE3(pin.utils.rotate('x',-x[2]),
                    np.array([0,x[0],x[1]]))
        self.viz.applyConfiguration(self.gname,M)


    def displayTrajectory(self,xs,timeStep):
        '''Display an animation showing a trajectory of a bicopter,
        xs is a list-type object containing bicopter states [x,z,th]
        and timeStep is used to control the lag of the animation. 
        '''
        import time
        for x in xs:
            self.display(x)
            time.sleep(timeStep)
