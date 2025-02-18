import numpy as np
from math import cos, sin

class Pose(np.ndarray):
    """
    Definition of a robot pose interface from where all the particular poses of different DOF inherit.
    This class defines a robot pose AxB as the pose of the B-Frame expressed in the A-Frame coordinates.
    """

    def oplus(AxB, BxC):
        """
        Given a Pose object *AxB* (the self object) and a Pose object *BxC*, it returns the compounded Pose object *AxC*.

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} = \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C}
            :label: eq-oplus

        **This is a pure virtual method that must be implemented by a child class.**

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """
        pass

    def J_1oplus(AxB, BxC):
        """
        Jacobian of the pose compounding operation (eq. :eq:`eq-oplus`) with respect to the first pose:

        .. math::
            J_{1\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Ax_B}
            :label: eq-J1oplus

        The method returns a numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object) and the :math:`2^{nd}` pose *BxC*.

        **This is a pure virtual method that must be implemented by a child class.**

        :param BxC: 2nd pose
        :returns: Evaluation of the :math:`J_{1\\oplus}` Jacobian of the pose compounding operation with respect to the first pose (eq. :eq:`eq-J1oplus`)
        """
        pass

    def J_2oplus(AxB):
        """
        Jacobian of the pose compounding operation (:eq:`eq-oplus`) with respect to the second pose:

        .. math::
            J_{2\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Bx_C}
            :label: eq-J2oplus

        The method returns a numerical matrix containing the evaluation of the Jacobian for the :math:`1^{st}` pose *AxB* (the self object).

        **This is a pure virtual method that must be implemented by a child class.**

        :returns: Evaluation of the :math:`J_{2\\oplus}` Jacobian of the pose compounding operation with respect to the second pose (eq. :eq:`eq-J2oplus`)
        """
        pass

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self object):

        .. math::
            ^Bx_A = \\ominus ^Ax_B
            :label: eq-ominus

        **This is a pure virtual method that must be implemented by a child class.**

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus`)
        """
        pass

    def J_ominus(AxB):
        """
        Jacobian of the inverse pose compounding operation (:eq:`eq-oplus`) with respect the pose *AxB* (the self object):

        .. math::
            J_{\\ominus}=\\frac{\\partial  \\ominus ^Ax_B}{\\partial ^Ax_B}
            :label: eq-Jominus

        Returns the numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object).
        **This is a pure virtual method that must be implemented by a child class.**

        :returns: Evaluation of the :math:`J_{\\ominus}` Jacobian of the inverse pose compounding operation with respect to the pose (eq. :eq:`eq-Jominus3dof`)
        """

        pass


class Pose3D(Pose):
    """
    Definition of a robot pose in 3 DOF (x, y, yaw). The class inherits from a ndarray.
    This class extends the ndarray with the :math:`oplus` and :math:`ominus` operators and the corresponding Jacobians.
    """
    # def __new__(cls, input_array):
    #     """
    #     Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarray numpy class.
    #
    #     :param input_array: array used to initialize the class
    #     :returns: the instance of a Pose3D class object
    #     """
    #     assert input_array.shape == (3, 1), "mean must be a 3x1 vector"
    #
    #     # Input array is an already formed ndarray instance
    #     # We first cast to be our class type
    #     obj = np.asarray(input_array).view(cls)
    #     # Finally, we must return the newly created object:
    #     return obj

    def __new__(cls, input_array=np.array([[0.0, 0.0, 0.0]]).T):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a Pose4D class object
        """

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj
    def __init__(self, input_array=np.array([[0.0, 0.0, 0.0]]).T):

        assert input_array.shape == (3, 1), "mean must be a 3x1 vector"

    def oplus(AxB, BxC):
        """
        Given a Pose3D object *AxB* (the self object) and a Pose3D object *BxC*, it returns the Pose3D object *AxC*.

        .. math::
            \\mathbf{{^A}x_B} &= \\begin{bmatrix} ^Ax_B & ^Ay_B & ^A\\psi_B \\end{bmatrix}^T \\\\
            \\mathbf{{^B}x_C} &= \\begin{bmatrix} ^Bx_C & ^By_C & & ^B\\psi_C \\end{bmatrix}^T \\\\

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} &= \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C} =
            \\begin{bmatrix}
                ^Ax_B + ^Bx_C  \\cos(^A\\psi_B) - ^By_C  \\sin(^A\\psi_B) \\\\
                ^Ay_B + ^Bx_C  \\sin(^A\\psi_B) + ^By_C  \\cos(^A\\psi_B) \\\\
                ^A\\psi_B + ^B\\psi_C
            \\end{bmatrix}
            :label: eq-oplus3dof

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """
        
        # define AxC
        AxC = np.empty((3,1))
        
        # compute the direct compounding
        AxC[0] = AxB[0] + BxC[0]*cos(AxB[2]) - BxC[1]*sin(AxB[2])
        AxC[1] = AxB[1] + BxC[0]*sin(AxB[2]) + BxC[1]*cos(AxB[2])
        AxC[2] = AxB[2] + BxC[2]

        return AxC

    def J_1oplus(AxB, BxC):
        """
        Jacobian of the pose compounding operation (eq. :eq:`eq-oplus3dof`) with respect to the first pose:

        .. math::
            J_{1\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Ax_B} =
            \\begin{bmatrix}
                1 & 0 &  -^Bx_C \\sin(^A\\psi_B) - ^By_C \\cos(^A\\psi_B) \\\\
                0 & 1 &  ^Bx_C \\cos(^A\\psi_B) - ^By_C \\sin(^A\\psi_B) \\\\
                0 & 0 & 1
            \\end{bmatrix}
            :label: eq-J1oplus3dof

        The method returns a numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object) and the :math:`2^{nd}` posepose *BxC*.

        :param BxC: 2nd pose
        :returns: Evaluation of the :math:`J_{1\\oplus}` Jacobian of the pose compounding operation with respect to the first pose (eq. :eq:`eq-J1oplus3dof`)
        """

        J1 = np.array([[1,0,-BxC[0,0]*sin(AxB[2,0])-BxC[1,0]*cos(AxB[2,0])],
                       [0,1,BxC[0,0]*cos(AxB[2,0])-BxC[1,0]*sin(AxB[2,0])],
                       [0,0,1]])

        return J1

    def J_2oplus(AxB):
        """
        Jacobian of the pose compounding operation (:eq:`eq-oplus3dof`) with respect to the second pose:

        .. math::
            J_{2\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Bx_C} =
            \\begin{bmatrix}
                \\cos(^A\\psi_B) & -\\sin(^A\\psi_B) & 0  \\\\
                \\sin(^A\\psi_B) & \\cos(^A\\psi_B) & 0  \\\\
                0 & 0 & 1
            \\end{bmatrix}
            :label: eq-J2oplus3dof

        The method returns a numerical matrix containing the evaluation of the Jacobian for the :math:`1^{st} posepose *AxB* (the self object).

        :returns: Evaluation of the :math:`J_{2\\oplus}` Jacobian of the pose compounding operation with respect to the second pose (eq. :eq:`eq-J2oplus3dof`)
        """

        J2 = np.array([[cos(AxB[2,0]), -sin(AxB[2,0]), 0],
                       [sin(AxB[2,0]), cos(AxB[2,0]), 0],
                       [0,0,1]])

        return J2

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self objetc):

        .. math::
            ^Bx_A = \\ominus ^Ax_B =
            \\begin{bmatrix}
                -^Ax_B \\cos(^A\\psi_B) - ^Ay_B \\sin(^A\\psi_B) \\\\
                ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                -^A\\psi_B
            \\end{bmatrix}
            :label: eq-ominus3dof

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus3dof`)
        """

        # define BxA
        BxA = np.empty((3,1))

        # compute inverse compounding
        BxA[0] = -AxB[0]*cos(AxB[2]) - AxB[1]*sin(AxB[2])
        BxA[1] = AxB[0]*sin(AxB[2]) - AxB[1]*cos(AxB[2])
        BxA[2] = -AxB[2]

        return Pose3D(BxA)

    def J_ominus(AxB):
        """
        Jacobian of the inverse pose compounding operation (:eq:`eq-oplus3dof`) with respect the pose *AxB* (the self object):

        .. math::
            J_{\\ominus}=\\frac{\\partial  \\ominus ^Ax_B}{\\partial ^Ax_B} =
            \\begin{bmatrix}
                -\\cos(^A\\psi_B) & -\\sin(^A\\psi_B) &  ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                \\sin(^A\\psi_B) & -\\cos(^A\\psi_B) &  ^Ax_B \\cos(^A\\psi_B) + ^Ay_B \\sin(^A\\psi_B) \\\\
                0 & 0 & -1
            \\end{bmatrix}
            :label: eq-Jominus3dof

        Returns the numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object).

        :returns: Evaluation of the :math:`J_{\\ominus}` Jacobian of the inverse pose compounding operation with respect to the pose (eq. :eq:`eq-Jominus3dof`)
        """


        J = np.array([[-cos(AxB[2,0]), -sin(AxB[2,0]), AxB[0,0]*sin(AxB[2,0])-AxB[1,0]*cos(AxB[2,0])],
                      [sin(AxB[2,0]), -cos(AxB[2,0]), AxB[0,0]*cos(AxB[2,0])+AxB[1,0]*sin(AxB[2,0])],
                      [0,0,-1]])

        return J