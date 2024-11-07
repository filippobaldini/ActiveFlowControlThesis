from dolfin import *
import numpy as np


class JetBCValue(UserExpression):
    '''
    Implements Boundary condition function for velocity
    onto our jet, taking as input frequency of the impulse and amplitude (Q) that are
    provided from the aget at each actuation

    Length_before_control, step height and control_width are the physical parameters
    necessary to evaluate the function at the physical point x = (x[0], x[1]) of the jet BC
    '''
    def __init__(self, time, length_before_control, step_height, control_width, a1, a2, b, frequency, Q, **kwargs):
        
        # frequancy of the sinusoidal profile
        self.freq = frequency
        
        self.time = time
        
        # lensgth of the segment that lies between the inlet and the jet (9.8 in our case)
        self.length_before_contr = length_before_control
        
        self.step_height = step_height
        
        self.control_width = control_width

        # ampltude of the jet control (can be also negative for suction)
        self.Q = Q

        #tuning parameters for individuating the best min_max interval for our jet and our frequency
        self.a1 = a1
        self.a2 = a2
        self.b = b
        
        super().__init__(self, **kwargs) 

    def eval(self, values, x):

        x_min = self.length_before_contr
        
        y = self.step_height   
        
        x_max = x_min + self.control_width

    # evaluation of the profile at point x                                           
        values[0] = self.a1*self.Q*(x[0]-x_min)*(x_max-x[0])/self.control_width**2 * abs(sin((self.a2*self.freq+self.a2/2.0)*2*pi*self.time))
        values[1] = self.a1*self.Q*(x[0]-x_min)*(x_max-x[0])/self.control_width**2 * abs(sin((self.a2*self.freq+self.a2/2.0)*2*pi*self.time))

    def value_shape(self):
        return (2, )