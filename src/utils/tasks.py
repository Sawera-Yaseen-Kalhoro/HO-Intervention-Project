import numpy as np
from utils.functions import wrap_angle,kinematics,jacobian

class Task:
    '''
        Base class representing an abstract Task.
    '''
    def __init__(self, name, desired):
        '''
            Constructor.
        
            Arguments:
            name (string): title of the task
            desired (Numpy array): desired sigma (goal)
        '''
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.feedforward_velocity = np.zeros((6,1))
        self.K = np.eye(self.sigma_d.shape[0])
        self.active = True
        self.status = False # indicate whether the task is enabled; not to be confused with activation function

        
    def update(self, q, base_pos, base_ori):
        '''
            Method updating the task variables (abstract).

            Arguments:
            q (Numpy array): manipulator joint positions
            base_pos (Numpy array): position (x and y) of the mobile base
            base_ori (float): orientation (yaw) of the mobile base
        '''
        pass

    def setDesired(self, value):
        ''' 
            Method setting the desired sigma.

            Arguments:
            value (Numpy array): value of the desired sigma (goal)
        '''
        self.sigma_d = value

    def getDesired(self):
        '''
            Method returning the desired sigma.
        '''
        return self.sigma_d

    def getJacobian(self):
        '''
            Method returning the task Jacobian.
        '''
        return self.J
  
    def getError(self):
        '''
            Method returning the task error (tilde sigma).
        '''  
        return self.err
    
    def setFeedforwardVelocity(self, velocity):
        '''
            Method to set the feedforward velocity vector.

            Arguments:
            velocity (Numpy array): value of the feedforward velocity
        '''
        self.feedforward_velocity = velocity

    def getFeedforwardVelocity(self):
        '''
            Method to get the feedforward velocity vector.
        '''
        return self.feedforward_velocity

    def setGainMatrix(self, K):
        '''
            Method to set gain matrix K.

            Arguments:
            K (Numpy array): value of the gain matrix K
        '''
        self.K = K

    def getGainMatrix(self):
        '''
            Method to get gain matrix K.
        '''
        return self.K
    
    def isActive(self, q):
        '''
            Method to check if the task is active.

            Arguments:
            q (Numpy array): joint positions
        '''
        return self.active
    
    def setStatus(self, status):
        '''
            Method to enable or disable the task.

            Arguments:
            status (bool): status of the task (True means enabled)
        '''
        self.status = status

    def getStatus(self):
        '''
            Method to get the status (enabled/disabled) of the task.
        '''
        return self.status

  
#### NEW TASKS FOR PROJECT #####################

class EEPosition(Task):
    '''
        Subclass of Task, representing end-effector position task.
    '''
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((3, 6))
        self.feedforward_velocity = np.zeros((3, 1))  
        self.K = np.eye(3)

    def update(self, q, base_pos, base_ori):
        ee_pos, _ = kinematics(q.flatten(),base_pos.flatten(),base_ori)
        # Update task Jacobian
        self.J = jacobian(q.flatten(),base_pos.flatten(),base_ori)[:3] # first 3 rows of the EE Jacobian

        # Update task error
        self.err = self.getDesired() - ee_pos

class EEOrientation(Task):
    '''
        Subclass of Task, representing end-effector orientation task.
    '''
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((1, 6))  
        self.feedforward_velocity = np.zeros((1, 1)) 
        self.K = np.eye(1)  

    def update(self, q, base_pos, base_ori):
        _, ee_ori = kinematics(q.flatten(),base_pos.flatten(),base_ori)

        # Update task Jacobian
        self.J = jacobian(q.flatten(),base_pos.flatten(),base_ori)[-1] # last row of the EE Jacobian, corresponding to yaw

        # Update task error
        self.err = self.getDesired() - ee_ori
        self.err[0,0] = wrap_angle(self.err[0,0])

class EEConfiguration(Task):
    '''
        Subclass of Task, representing end-effector configuration task.
    '''
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((4,6)) 
        self.feedforward_velocity = np.zeros((4, 1))  
        self.K = np.eye(4)

    def update(self, q, base_pos, base_ori):
        ee_pos, ee_ori = kinematics(q.flatten(),base_pos.flatten(),base_ori)

        # Update task Jacobian
        # position
        J_pos = jacobian(q.flatten(),base_pos.flatten(),base_ori)[:3] # first 3 rows of the EE Jacobian
        J_ori = jacobian(q.flatten(),base_pos.flatten(),base_ori)[-1] # last row of the EE Jacobian, corresponding to yaw

        self.J = np.vstack((J_pos, J_ori))

        # Update task error
        ee_pos_d = self.getDesired()[:3]
        ee_ori_d = self.getDesired()[-1]

        # position
        err_pos =  ee_pos_d - ee_pos

        # orientation
        err_ori = ee_ori_d - ee_ori
        # err_ori[0,0] = wrap_angle(err_ori[0,0])

        # combined
        self.err = np.vstack((err_pos,err_ori))

class BaseOrientation(Task):
    '''
        Subclass of Task, representing base orientation.
    '''
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((1, 6))  
        self.feedforward_velocity = np.zeros((1, 1)) 
        self.K = np.eye(1)  

    def update(self, q, base_pos, base_ori):
        # Update task Jacobian
        self.J = np.zeros((1,6))
        self.J[0,0] = 1.0

        # Update task error
        self.err = self.getDesired() - np.array([[base_ori]])
        self.err[0,0] = wrap_angle(self.err[0,0])

class BasePosition(Task):
    '''
        Subclass of Task, representing base position.
    '''
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((2, 6))  
        self.feedforward_velocity = np.zeros((2, 1)) 
        self.K = np.eye(2)  

    def update(self, q, base_pos, base_ori):
        # Update task Jacobian
        self.J = np.zeros((2,6))
        self.J[:,:2] = jacobian(q.flatten(),base_pos.flatten(),base_ori)[:2,:2]

        # Update task error
        self.err = self.getDesired() - base_pos

class JointPosition(Task):
    ''' 
    Subclass of Task, representing the manipulator joint position task.
    '''
    def __init__(self, name, desired, joint):
        '''
            Constructor.
        
            Arguments:
            name (string): title of the task
            desired (Numpy array): desired sigma (goal)
            joint (int): manipulator joint of interest (1 until 4)
        '''
        super().__init__(name, desired)
        self.J = np.zeros((1,6)) # Initialize with proper dimensions
        self.err = np.zeros((1,1)) # Initialize with proper dimensions
        self.feedforward_velocity = np.zeros_like(desired)
        self.K = np.eye(len(desired))
        self.joint = joint-1 # converted to index
        
    def update(self, q, base_pos, base_ori):
        # Update task Jacobian
        self.J = np.zeros((1,6))
        self.J[0,self.joint+2] = 1

        # Update task error
        self.err = self.getDesired() - q[self.joint]
        self.err[0,0] = wrap_angle(self.err[0,0])

class JointLimits(Task):
    '''
        Subclass of Task, representing joint limits task.
    '''
    def __init__(self, name, limits, joint):
        '''
            Constructor.
        
            Arguments:
            name (string): title of the task
            limits (Numpy array): lower and upper limit of the joint position
            joint (int): manipulator joint of interest (1 until 4)
        '''
        super().__init__(name, np.zeros((0,1))) # no sigma_d
        self.J = np.zeros((1,6))
        self.err = 1 # xdot
        self.qmin = limits[0] # lower limit
        self.qmax = limits[1] # upper limit
        self.ra = 0.02 # activation threshold
        self.rd = 0.04 # deactivation threshold

        self.feedforward_velocity = 0 # not really relevant but added to maintain generality
        self.K = np.eye(1)

        self.active = 0 # initially set to False (inactive)

        self.joint = joint-1 # converted to index

        self.status = True # always enable this task

    def update(self, q, base_pos, base_ori):
        # Update task Jacobian
        self.J = np.zeros((1,6))
        self.J[0,self.joint+2] = 1

        # Update task error
        self.err = np.array([[self.active]])

    def isActive(self, q): #override the base class
        qi = q[self.joint,0]
        if (self.active == 0):
            if (qi >= self.qmax-self.ra):
                self.active = -1
            elif (qi <= self.qmin+self.ra):
                self.active = 1
        elif (self.active == -1) and (qi <= self.qmax-self.rd):
            self.active = 0
        elif (self.active == 1) and (qi >= self.qmin+self.rd):
            self.active = 0
            
        return self.active
    