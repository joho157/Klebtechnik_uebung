import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root, fsolve
import time

class SandwichSolution:
    """
    Solution Class for Sandwich Structure

    Stores the computed solutions during the solving process of a Sandwich system. 
    Provides methods for plotting iteration progress, stresses, energy release rates, 
    and edge stress analysis.

    Attributes
    ----------
    ERR1 : ndarray
        Energy release rate for the first crack mode
    ERR2 : ndarray
        Energy release rate for the second crack mode
    aArray : ndarray
        Array of crack lengths
    sigzz : ndarray
        Array of normal stresses [MPa]
    tauxz : ndarray
        Array of shear stresses [MPa]
    dx : ndarray
        Array of x-coordinates of the crack tip [mm]
    FailureLoad_Iterated : ndarray
        Array of failure loads for each iteration [N]
    Crack_Iterated : ndarray
        Array of crack lengths for each iteration [mm]
    n_Iteration : int
        Number of iterations
    Title: string 
        The Title of the generated plot
    dx2 : ndarray
        Array of x-coordinates for the second data set [mm]
    tauxz2 : ndarray
        Array of shear stresses for the second data set [MPa]
    sigzz2 : ndarray
        Array of normal stresses for the second data set [MPa]
    sigzz_edge : ndarray, optional
        Array of normal stresses along the edge of the sandwich structure [MPa]
    tauxz_edge : ndarray, optional
        Array of shear stresses along the edge of the sandwich structure [MPa]
    L_linspace : ndarray, optional
        Array of different overlapping lengths for the sandwich structure [mm].

    Methods
    -------
    __init__(self, ERR1=None, ERR2=None, aArray=None, sigzz=None, tauxz=None, dx=None, 
             FailureLoad_Iterated=None, Crack_Iterated=None, n_Iteration=None, sigzz_edge=None, 
             tauxz_edge=None, L_linspace=None):
        Initializes the SandwichSolution class with optional attributes for energy release rates,
        stresses, crack lengths, failure loads, iteration count, and edge stresses.

    plot_iter(self):
        Plots the iteration progress of failure load and crack size over the iterations.

    plot_stress(self):
        Plots the normal and shear stresses in the overlapping region of the sandwich joint.

    plot_multi_stress(self, dx2, tauxz2, sigzz2):
        Plots the normal and shear stresses for two different data sets over the x-coordinate of the overlap.

    plot_ERR(self):
        Plots the energy release rates for both crack modes over the crack length.

    plot_stress_edge(self, stress_component="both"):
        Plots the edge stresses of the sandwich structure for diffrent overlapping lenghts.
    """

    def __init__(self, ERR1=None, ERR2=None, aArray=None, sigzz=None, tauxz=None, dx=None, 
                 FailureLoad_Iterated=None, Crack_Iterated=None, n_Iteration=None, sigzz_edge=None,tauxz_edge=None, L_linspace=None):
        """
        Initializes the SandwichSolution class with the following properties: ERR1, ERR2, aArray, 
        sigzz, tauxz, dx, FailureLoad_Iterated, Crack_Iterated, n_Iteration.
        
        These properties can be defined during the calculation process, and are not required for initialization.

        Parameters
        ----------
        ERR1 : ndarray, optional
            Energy release rate for the first crack mode
        ERR2 : ndarray, optional
            Energy release rate for the second crack mode
        aArray : ndarray, optional
            Array of crack lengths
        sigzz : ndarray, optional
            Array of normal stresses [MPa]
        tauxz : ndarray, optional
            Array of shear stresses [MPa]
        dx : ndarray, optional
            Array of x-coordinates of the crack tip [mm]
        FailureLoad_Iterated : ndarray, optional
            Array of failure loads for each iteration [N]
        Crack_Iterated : ndarray, optional
            Array of crack lengths for each iteration [mm]
        n_Iteration : int, optional
            Number of iterations
        
        """
        self.ERR1 = ERR1
        self.ERR2 = ERR2
        self.aArray = aArray
        self.sigzz = sigzz
        self.tauxz = tauxz
        self.dx = dx
        self.FailureLoad_Iterated = FailureLoad_Iterated
        self.Crack_Iterated = Crack_Iterated
        self.n_Iteration = n_Iteration
        self.sigzz_edge=sigzz_edge
        self.tauxz_edge=tauxz_edge
        self.L_linspace=L_linspace

    def plot_iter(self):
        """
        Plots the iteration progress, showing how the failure load and crack size evolve over each iteration.

        This method is called after the iteration process has been completed and the results stored in the
        `SandwichSolution` class. It generates a plot with failure load on the primary y-axis and crack size 
        on the secondary y-axis.

        The plot includes scatter plots for failure load and crack size versus iteration index.

        Raises
        ------
        ValueError
            If no iteration data is available in `FailureLoad_Iterated` or `Crack_Iterated`.
        """
        n_space = np.arange(self.n_Iteration + 1)
        
        plt.figure()
        fig, host = plt.subplots(layout='constrained') 
        ax2 = host.twinx()

        host.set_ylim(min(np.concat([self.FailureLoad_Iterated * 0.9, self.FailureLoad_Iterated * 1.1])), 
                      max(np.concat([self.FailureLoad_Iterated * 0.9, self.FailureLoad_Iterated * 1.1])))

        ax2.set_ylim(min(np.concat([self.Crack_Iterated * 0.9, self.Crack_Iterated * 1.1])), 
                      max(np.concat([self.Crack_Iterated * 0.9, self.Crack_Iterated * 1.1])))

        host.set_xlabel("Iteration")
        host.set_ylabel("Failure Load [N]")
        ax2.set_ylabel("Crack Size [mm]")

        host.scatter(n_space, self.FailureLoad_Iterated, label="Failure Load [N]", color='blue')
        ax2.scatter(n_space, self.Crack_Iterated, label="Crack Size [mm]", color='red')

        plt.grid(True) 
        fig.legend()       
        plt.show()

    def plot_stress(self, Title = None):
        """
        Plots the normal stress (sigma_zz) and shear stress (tau_xz) over the x-coordinate of the overlap.

        This method is called when stresses are available in the `SandwichSolution` class. It generates a plot
        with the stresses on the y-axis and the x-coordinates of the overlap on the x-axis.

        Attributes
        ----------
        Title: string, optional
            The Title of the generated plot

        Raises
        ------
        ValueError
            If no stress data is available in `sigzz` or `tauxz`, or if no x-coordinates are available in `dx`.
        """
        plt.figure()
        plt.plot(self.dx, self.tauxz, label='shear stress')
        plt.plot(self.dx, self.sigzz, label='normal stress')
        plt.legend()
        plt.xlabel('x-coordinate of overlap [mm]')
        plt.ylabel('Stresses [MPa]')
        plt.grid(True)
        plt.axis([self.dx[0], self.dx[-1], np.min([self.tauxz, self.sigzz]), np.max([self.tauxz, self.sigzz])])
        plt.xlabel('x-coordinate of overlap [mm]')
        plt.ylabel('Stresses [MPa]')
        if Title != None:
            plt.title(Title)
        plt.grid(True)
        plt.show()

    def plot_multi_stress(self, dx2, tauxz2, sigzz2):
        """
        Plots the normal and shear stresses for two different data sets over the x-coordinate of the overlap.

        This method allows comparison of two stress profiles (normal and shear stresses) on the same plot.
        It requires the additional x-coordinates, normal stresses, and shear stresses of the second data set.

        Parameters
        ----------
        dx2 : ndarray
            Array of x-coordinates for the second data set [mm].
        tauxz2 : ndarray
            Array of shear stresses for the second data set [MPa].
        sigzz2 : ndarray
            Array of normal stresses for the second data set [MPa].

        Raises
        ------
        ValueError
            If stress or x-coordinate data is not available for either the first or second data set.

        Notes
        -----
        The plot includes:
        - Shear stress and normal stress for the first data set.
        - Shear stress and normal stress for the second data set.
        - A common x-axis for comparison of stresses across the two data sets.
        """
        if self.dx is None or self.tauxz is None or self.sigzz is None:
            raise ValueError("First data set (dx, tauxz, sigzz) is incomplete.")
        if dx2 is None or tauxz2 is None or sigzz2 is None:
            raise ValueError("Second data set (dx2, tauxz2, sigzz2) is incomplete.")

        plt.figure()
        plt.plot(self.dx, self.tauxz, label='shear stress: 1')
        plt.plot(self.dx, self.sigzz, label='normal stress: 1')
        plt.plot(dx2, tauxz2, label='shear stress: 2')
        plt.plot(dx2, sigzz2, label='normal stress: 2')
        plt.legend()
        plt.xlabel('x-coordinate of overlap [mm]')
        plt.ylabel('Stresses [MPa]')
        plt.grid(True)
        plt.axis([self.dx[0], self.dx[-1], 
                  min(np.min(self.tauxz), np.min(self.sigzz), np.min(tauxz2), np.min(sigzz2)), 
                  max(np.max(self.tauxz), np.max(self.sigzz), np.max(tauxz2), np.max(sigzz2))])
        plt.show()
        
    def plot_ERR(self):
        """
        Plots the energy release rates (ERR1 and ERR2) for both crack modes over the crack length.

        This method is called when energy release rates are available in the `SandwichSolution` class. 
        It generates a scatter plot with crack length on the x-axis and energy release rate on the y-axis.

        Raises
        ------
        ValueError
            If no energy release rate data is available in `ERR1` or `ERR2`, or if no crack lengths are available in `aArray`.
        """
        plt.figure()
        plt.scatter(self.aArray, self.ERR1, label="ERR 1")
        plt.scatter(self.aArray, self.ERR2, label="ERR 2")
        plt.legend()
        plt.xlabel('x-coordinate of Crack [mm]')
        plt.ylabel('Energy Release Rate [MPa]')
        plt.grid(True)
        plt.show()

    def plot_stress_edge(self,stress_component="both"):
        """
        Plots the edge stresses of the sandwich structure.

        This method visualizes the computed stresses along the edge of the structure for the normal stress (`sigzz_edge`),
        shear stress (`tauxz_edge`), or both, based on the specified `stress_component`.

        Parameters
        ----------
        stress_component : str, optional
            Specifies which stress component to plot. Options are:
            - `"both"`: Plots both `sigzz_edge` and `tauxz_edge` (default).
            - `"tauxz"`: Plots only the shear stress (`tauxz_edge`).
            - `"sigzz"`: Plots only the normal stress (`sigzz_edge`).
        """
        plt.figure()
        if stress_component=="both":            
            plt.plot(self.L_linspace, self.sigzz_edge, label="sigzz edge")
            plt.plot(self.L_linspace, self.tauxz_edge, label="tauxz edge")
        if stress_component=="tauxz":
            plt.plot(self.L_linspace, self.tauxz_edge, label="tauxz edge")
        if stress_component=="sigzz":
            plt.plot(self.L_linspace, self.sigzz_edge, label="sigzz edge")
        plt.legend()
        plt.xlabel(' Overlap length [mm]')
        plt.ylabel('Stress [MPa]')
        plt.grid(True)
        plt.show()



# Define Sandwich Class with the following properties: Joint-Konfiguration, 2 Adherend Materials, 1 Adhesive Material, Stress Criteria, and Energy Criteria
# Define Sandwich Class with the following properties: Joint-Konfiguration, 2 Adherend Materials, 1 Adhesive Material, Stress Criteria, and Energy Criteria
class Sandwich:
    """
    Base Class for Sandwich-Layers:
    Provides geometry, material attributes, and methods
    for Crack Initialization of Sandwich-Type Structures 

    Attributes
    ----------
    name: str
        Name of the Joint-Type
    h1: float
        Height of 1st Adherend
    h2: float
        Height of 2nd Adherend
    t: float
        Height of the Adhesive
    L0: float
        Overlapping Length of the two Adherends
    b: float
        Width of the Configuration     
    StressCrit: str
        Criteria for Stress (optional)
    EnergyCrit: str
        Criteria for Energy (optional)
    SF: ndarray
        Section forces
    L: float
        Overlapping length (with Crack)
    x0: float
        Point of evaluation
    BL: bool
        Side of the Sandwich Layer (True: Left, False: Right) for stresses
    F: float
        Force on the Joint
    delta_a: float
        Finite Crack-Length
    a: float
        Crack Length
    maxIter: int
        Maximal Iterations
    Finit: float
        Start point for the force to begin calculation
    tol: float
        Tolerance for error: If the change of the Force from one to another Iteration is smaller than Tolerance, calculation is finished
    L_linspace: ndarray
        List of different overlapping lengths

    Methods
    -------
    __init__(self, name, h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit):
        Initializes a Sandwich instance with specified geometry, material attributes, and criteria for stress and energy.
    
    linearelasticsandwich_EVP(self):
        Solves Eigenvalue-Problem for Sandwich-Type Models with the given Configuration.

    linearelasticsandwich_stress(self, SF, L, x0, BL, write_Data=False):
        Calculates the stresses in a Sandwich joint based on section forces, geometry, and material properties.
    
    calc_Krenk(self, F, delta_a, BL, write_Data=False):
        Calculates energy criteria for a Sandwich joint based on force and crack length.
    
    energyCrit(self, ERRbar1, ERRbar2, EnergyCrit):
        Computes the energy criterion (e.g., Griffith, Linear Interaction, Gpsi) based on the provided error values.
    
    calc_stressCrit(self, F, a, BL):
        Calculates stress criterion based on user-defined criteria, crack length, and side of the sandwich layer.
    
    optimizeffm(self, BL, maxIter, Finit, tol, write_Data=False):
        Optimizes the Failure Mode of a Sandwich Structure using an iterative process.

    brittlenessnumber(self, F, BL):        
        Calculates the Brittle Number to assess the appropriateness of the calculation.

    edge_stress(self, L_linspace, F=0, BL=True):
        Computes the edge stresses of a sandwich structure for differnet overlapping lenghts.
    """

    def __init__(self, name, h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit = None, EnergyCrit = None):
        """
        Initialize Sandwich-System by User-Input of h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit.

        Parameters
        ----------
        name : str
            Name of the Joint-Type
        h1 : float
            Height of 1st Adherend
        h2 : float
            Height of 2nd Adherend
        t : float
            Height of the Adhesive
        L0 : float
            Overlapping Length of the two Adherends
        b : float
            Width of the Configuration
        Adherend1 : Material
            Material properties of the 1st Adherend
        Adherend2 : Material
            Material properties of the 2nd Adherend
        Adhesive : Material
            Material properties of the Adhesive
        StressCrit : str
            Criteria for Stress (e.g., "Max Principal", "Linear Interaction", etc.) (optional)
        EnergyCrit : str
            Criteria for Energy (e.g., "Griffith", "Linear Interaction", etc.) (optional)
        """

        self.name = name        
        self.h1 = h1
        self.h2 = h2
        self.t = t
        self.b = b
        self.L0 = L0
        self.Adherend1 = Adherend1
        self.Adherend2 = Adherend2
        self.Adhesive = Adhesive 
        self.EnergyCrit = EnergyCrit
        self.StressCrit = StressCrit 
        self.linearelasticsandwich_EVP()
        self.Sol = SandwichSolution()

    def linearelasticsandwich_EVP(self):
        """
        Solves the Eigenvalue Problem for Sandwich-Type Models with the given configuration of adherends and adhesive.
        
        This method computes the eigenvalues and eigenvectors for the system, which are later used for stress and energy calculations.
        """

        Ea = self.Adhesive.E
        nua = self.Adhesive.nu
        h1 = self.h1
        h2 = self.h2
        t = self.t
        Ga = self.Adhesive.G

        # ABD-Matrices of the adherends material
        M1 = self.Adherend1.M
        M2 = self.Adherend2.M

        # Constants
        D1 = M1[0] * M1[2] - M1[1]**2
        D2 = M2[0] * M2[2] - M2[1]**2        
        
        Eas = Ea / (1 - nua**2)
        
        a1 = Ga / t * (1 / D1 * (M1[2] - (h1 + t) * M1[1] + (0.5 * (h1 + t))**2 * M1[0]) + 1 / D2 * (M2[2] + (h2 + t) * M2[1] + (0.5 * (h2 + t))**2 * M2[0]))
        a2 = Ga / t * (1 / D1 * (M1[1] - (0.5 * (h1 + t)) * M1[0]) + 1 / D2 * (M2[1] + (0.5 * (h2 + t)) * M2[0]))
        a3 = Ga / 2 * (1 / (M2[3] * M2[4]) - 1 / (M1[3] * M1[4]))
        b1 = Eas / t * (1 / (M1[3] * M1[4]) + 1 / (M2[3] * M2[4]))
        b2 = Eas / t * (M1[0] / D1 + M2[0] / D2)
        b3 = Eas / t * (1 / D1 * (M1[1] - (0.5 * (h1 + t)) * M1[0]) + 1 / D2 * (M2[1] + (0.5 * (h2 + t)) * M2[0]))

        A = np.array([[0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, a1, 0, -a2, 0, a3, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, b3, 0, -b2, 0, b1, 0]])

        # Eigenvectors and Eigenvalues
        Eval, Evect = la.eig(A)
        self.Eval = Eval
        self.Evect = Evect

    def linearelasticsandwich_stress(self, SF, L, x0, BL, write_Data=False):
        """
            Calculates the stresses in a Sandwich joint based on section forces, geometry, and material properties.

            Parameters
            ----------
            SF : ndarray
                Section forces at the joint
            L : float
                Length of the Sandwich layer
            x0 : float
                Evaluation point for stresses
            BL : bool
                Side of the Sandwich Layer (True: Left, False: Right)
            write_Data : bool, optional
                Whether to store the results in the SandwichSolution object (default is False)

            Returns
            -------
            sigzz : ndarray
                Normal stress at the evaluation point
            tauxz : ndarray
                Shear stress at the evaluation point
        """
        n = 100  # number of evaluation points        
            
        T11 = SF[0]
        V11 = SF[1]
        M11 = SF[2]
        T12 = SF[3]
        V12 = SF[4]
        M12 = SF[5]   
        T21 = SF[6]
        V21 = SF[7]
        M21 = SF[8]
        T22 = SF[9]
        V22 = SF[10]
        M22 = SF[11]                      

        M1 = self.Adherend1.M
        M2 = self.Adherend2.M
        Ea = self.Adhesive.E
        nua = self.Adhesive.nu
        h1 = self.h1
        h2 = self.h2
        t = self.t
        Eval = self.Eval
        Evect = self.Evect
        D1 = M1[0] * M1[2] - M1[1]**2
        D2 = M2[0] * M2[2] - M2[1]**2
        Ga = self.Adhesive.G
        Eas = Ea / (1 - nua**2)
        
        evZ = Evect[:, Eval == 0].real
        ewR = Eval[(np.imag(Eval) == 0) & (np.real(Eval) != 0)].real
        evR = Evect[:, (np.imag(Eval) == 0) & (np.real(Eval) != 0)].real
        ewC = Eval[np.imag(Eval) > 0]
        evC = Evect[:, np.imag(Eval) > 0]

        # edge conditions
        sR = np.zeros_like(ewR)
        sC = np.zeros_like(ewC)
        sR[ewR > 0] = -L / 2
        sR[ewR < 0] = L / 2
        sC[np.real(ewC) > 0] = -L / 2
        sC[np.real(ewC) < 0] = L / 2 

        tau0_int = np.concatenate(([evZ[0, 0] * L],
                                    evR[0, :] / np.transpose(ewR) * (np.exp(ewR * L / 2) - np.exp(ewR * (-L / 2))) * np.exp(np.transpose(ewR) * np.transpose(sR)),
                                    np.exp(np.transpose(np.real(ewC)) * np.transpose(sC)) * 1 / (np.real(np.transpose(ewC))**2 + np.imag(np.transpose(ewC))**2) * np.exp(-np.real(np.transpose(ewC)) * L / 2) * ((np.exp(np.real(np.transpose(ewC)) * L) - 1) * np.cos(np.imag(np.transpose(ewC)) * L / 2) * (np.real(np.transpose(ewC)) * np.real(evC[0, :]) + np.imag(np.transpose(ewC)) * np.imag(evC[0, :])) + (1 + np.exp(np.real(np.transpose(ewC)) * L)) * np.sin(np.imag(np.transpose(ewC)) * L / 2) * (np.imag(np.transpose(ewC)) * np.real(evC[0, :]) - np.real(np.transpose(ewC)) * np.imag(evC[0, :]))),
                                    np.exp(np.transpose(np.real(ewC)) * np.transpose(sC)) * 1 / (np.real(np.transpose(ewC))**2 + np.imag(np.transpose(ewC))**2) * np.exp(-np.real(np.transpose(ewC)) * L / 2) * ((np.exp(np.real(np.transpose(ewC)) * L) - 1) * np.cos(np.imag(np.transpose(ewC)) * L / 2) * (np.real(np.transpose(ewC)) * np.imag(evC[0, :]) - np.imag(np.transpose(ewC)) * np.real(evC[0, :])) + (1 + np.exp(np.real(np.transpose(ewC)) * L)) * np.sin(np.imag(np.transpose(ewC)) * L / 2) * (np.imag(np.transpose(ewC)) * np.imag(evC[0, :]) + np.real(np.transpose(ewC)) * np.real(evC[0, :])))))

        # Discretization of x-coordinate
        
        dt = np.linspace(-L / 2, L / 2, n)
        dt = dt[:, np.newaxis]

        Dres = np.zeros((7, 7 * len(dt)))

        for k in range(len(dt)):

            tmp = np.zeros((7, 7))
            sR = np.zeros_like(ewR)
            sC = np.zeros_like(ewC.imag)

            sR[ewR > 0] = dt[k] - L / 2
            sR[ewR < 0] = dt[k] + L / 2

            sC[np.real(ewC) > 0] = dt[k] - L / 2
            sC[np.real(ewC) < 0] = dt[k] + L / 2

            tmp[:, 0] = evZ[:, 0]

            for j in range(len(ewR)):
                tmp[:, j + 1] = evR[:, j] * np.exp(ewR[j] * sR[j])

            for j in range(len(ewC)):
                tmp[:, 1 + len(ewR) + j] = np.exp(np.real(ewC[j]) * sC[j]) * (
                    np.real(evC[:, j]) * np.cos(np.imag(ewC[j]) * dt[k]) - np.imag(evC[:, j]) * np.sin(np.imag(ewC[j]) * dt[k])
                )
                tmp[:, 1 + len(ewR) + len(ewC) + j] = np.exp(np.real(ewC[j]) * sC[j]) * (
                    np.imag(evC[:, j]) * np.cos(np.imag(ewC[j]) * dt[k]) + np.real(evC[:, j]) * np.sin(np.imag(ewC[j]) * dt[k])
                )

            Dres[:, k * 7:(k + 1) * 7] = tmp

        r_r = Dres[:, -7:]
        r_l = Dres[:, 0:7]

        LHS = np.vstack([
            tau0_int,
            r_r[1, :] - Ga / 2 * (1 / (M2[3] * M2[4]) - 1 / (M1[3] * M1[4])) * r_r[3, :],
            r_l[1, :] - Ga / 2 * (1 / (M2[3] * M2[4]) - 1 / (M1[3] * M1[4])) * r_l[3, :],
            r_r[5, :] - Eas / t * (1 / (M2[3] * M2[4]) + 1 / (M1[3] * M1[4])) * r_r[3, :],
            r_l[5, :] - Eas / t * (1 / (M2[3] * M2[4]) + 1 / (M1[3] * M1[4])) * r_l[3, :],
            r_r[6, :] - Eas / t * (1 / (M2[3] * M2[4]) + 1 / (M1[3] * M1[4])) * r_r[4, :] -
            Eas / t * (1 / D1 * (M1[1] - (0.5 * (h1 + t)) * M1[0]) + 1 / D2 * (M2[1] + (0.5 * (h2 + t)) * M2[0])) * r_r[0, :],
            r_l[6, :] - Eas / t * (1 / (M2[3] * M2[4]) + 1 / (M1[3] * M1[4])) * r_l[4, :] -
            Eas / t * (1 / D1 * (M1[1] - (0.5 * (h1 + t)) * M1[0]) + 1 / D2 * (M2[1] + (0.5 * (h2 + t)) * M2[0])) * r_l[0, :]
        ])

        RHS = np.vstack([
            [T11 - T12],
            [Ga / t * (1 / D2 * ((M2[2] + (0.5 * (h2 + t)) * M2[1]) * T22 - (M2[1] + (0.5 * (h2 + t)) * M2[0]) * M22) +
                    1 / D1 * ((-M1[2] + (0.5 * (h1 + t)) * M1[1]) * T12 - (-M1[1] + (0.5 * (h1 + t)) * M1[0]) * M12))],
            [Ga / t * (1 / D2 * ((M2[2] + (0.5 * (h2 + t)) * M2[1]) * T21 - (M2[1] + (0.5 * (h2 + t)) * M2[0]) * M21) +
                    1 / D1 * ((-M1[2] + (0.5 * (h1 + t)) * M1[1]) * T11 - (-M1[1] + (0.5 * (h1 + t)) * M1[0]) * M11))],
            [Eas / t * (1 / D1 * (-M1[1] * T12 + M1[0] * M12) - 1 / D2 * (-M2[1] * T22 + M2[0] * M22))],
            [Eas / t * (1 / D1 * (-M1[1] * T11 + M1[0] * M11) - 1 / D2 * (-M2[1] * T21 + M2[0] * M21))],
            [Eas / t * (M1[0] / D1 * V12 - M2[0] / D2 * V22)],
            [Eas / t * (M1[0] / D1 * V11 - M2[0] / D2 * V21)]
        ])

        # Solving for constants 
        C = la.solve(LHS, RHS)
        
        tau0 = np.zeros((len(dt), 1))
        sigma0 = np.zeros((len(dt), 1))

        for k in range(len(dt)):
            stress = np.matmul(Dres[:, k * 7:(k + 1) * 7], C)

            if np.isreal(stress[0]) & np.isreal(stress[3]):
                tau0[k] = stress[0].real   
                sigma0[k] = stress[3].real
            else:
                print("Warning complex values")

        # Stress at the point of crack onset
        dt1 = np.concatenate(dt)

        if write_Data:
            self.Sol.sigzz = sigma0
            self.Sol.tauxz = tau0
            self.Sol.dx = dt1

        if BL:       
            sigzz = interp1d(dt1, sigma0, kind='linear', axis=0)(-L / 2 + x0)
            tauxz = interp1d(dt1, tau0, kind='linear', axis=0)(-L / 2 + x0)
        else:
            sigzz = interp1d(dt1, sigma0, kind='linear', axis=0)(L / 2 - x0)
            tauxz = interp1d(dt1, tau0, kind='linear', axis=0)(L / 2 - x0)  
        sigzz=np.squeeze(sigzz)
        tauxz=np.squeeze(tauxz)

        return sigzz, tauxz

    def calc_Krenk(self, F, delta_a, BL, write_Data=False,nInc=5):  
        """
        Calculates energy criteria for a Sandwich joint based on force and crack length.

        Parameters
        ----------
        F : float
            Force applied to the joint
        delta_a : float
            Crack length
        BL : bool
            Side of the Sandwich Layer (True: Left, False: Right)
        write_Data : bool, optional
            Whether to store the results in the SandwichSolution object (default is False)
        nInc : integer
            Number of Increments for integration

        Returns
        -------
        ERRbar1 : float
            Average energy from normal stress
        ERRbar2 : float
            Average energy from shear stress
        """      
        Adhesive = self.Adhesive        

        #define Functions for GaußQuadratur
        def calc_ERR1(inc_a):
            SF = self.load(F, inc_a)            
            sigzz, tauxz = self.linearelasticsandwich_stress(SF, self.L0 - inc_a, 0, BL)
            ERR1 = 0.5 * self.t / Adhesive.E * sigzz ** 2
            return ERR1
        
        def calc_ERR2(inc_a):
            SF = self.load(F, inc_a)            
            sigzz, tauxz = self.linearelasticsandwich_stress(SF, self.L0 - inc_a, 0, BL)
            ERR2 = 0.5 * self.t / Adhesive.G * tauxz ** 2
            return ERR2
        #Define Gaußpoints and wheightfactors        
        x , w = np.polynomial.legendre.leggauss(nInc)

        summe1 = 0
        summe2 = 0

        # integrate with transformation of the Integrationrange from [0,delta_a] to [-1,+1] 
        for i in range(nInc):
            summe1 = summe1 + delta_a / 2 * calc_ERR1(delta_a/2 * x[i] + delta_a / 2)* w[i]
            summe2 = summe2 + delta_a / 2 * calc_ERR2(delta_a/2 * x[i] + delta_a / 2) * w[i]

        ERRbar1=summe1 / delta_a
        ERRbar2=summe2 / delta_a
        
        return ERRbar1, ERRbar2

    def energyCrit(self, ERRbar1, ERRbar2, EnergyCrit):
        """
        Computes the energy criterion (e.g., Griffith, Linear Interaction, Gpsi) based on the provided error values.

        Parameters
        ----------
        ERRbar1 : float
            Average energy from normal stress
        ERRbar2 : float
            Average energy from shear stress
        EnergyCrit : str
            Criteria for Energy (e.g., "Griffith", "Linear Interaction", etc.)

        Returns
        -------
        criteria : float
            Energy criterion result based on the selected method
        """
        G_Ic = self.Adhesive.G_Ic

        match EnergyCrit:
            case "Griffith":
                gsig = (ERRbar1 + ERRbar2) / G_Ic

            case "Linear Interaction": 
                G_IIc = self.Adhesive.G_IIc
                gsig = ERRbar1 / G_Ic + ERRbar2 / G_IIc

            case "Gpsi":
                psi = np.atan(np.sqrt(ERRbar1 / ERRbar2))
                Lambda = 1 - 2 / np.pi * np.arctan(np.sqrt(G_IIc / G_Ic - 1))
                Gbarc = G_Ic * (1 + np.tan((1 - Lambda) * psi) ** 2)
                gsig = (ERRbar1 + ERRbar2) / Gbarc

            case _:
                print("Error: Chosen Energy Criteria is not available")

        return gsig

    def calc_stressCrit(self, F, a, BL):
        """
        Calculates stress criterion based on user-defined criteria, crack length, and side of the sandwich layer.

        Parameters
        ----------
        F : float
            Force applied to the joint
        a : float
            Crack length
        BL : bool
            Side of the Sandwich Layer (True: Left, False: Right)

        Returns
        -------
        stress : float
            Stress criterion result based on the selected method
        """       

        if a < 0 or a > self.L0:  # allow only positive crack lengths
            fsig = 1
        else:         
            SF = self.load(F, 0)    
            sigzz, tauxz = self.linearelasticsandwich_stress(SF, self.L0, a, BL)    

            match self.StressCrit:
                case "Linear Interaction":
                    fsig = tauxz / self.Adhesive.tau_c + (sigzz / self.Adhesive.sig_c) - 1

                case "Max Principal":
                    fsig = (sigzz / 2 + np.sqrt((sigzz / 2) ** 2 + (tauxz) ** 2)) / self.Adhesive.sig_c - 1

                case "Quadratic Interaction":
                    fsig = (tauxz / self.Adhesive.tau_c) ** 2 + (sigzz / self.Adhesive.sig_c) ** 2 - 1

                case "Maximum Peel":
                    fsig = sigzz / self.Adhesive.sig_c - 1

                case "Maximum Shear":
                    fsig = tauxz / self.Adhesive.tau_c - 1

                case _:
                    print("Error: Chosen Stress Criteria is not available")
                    fsig = 1

        return fsig
       
    def optimizeffm(self, BL, maxIter, Finit, tol, write_Data=False):
        """
        Optimizes the Failure Mode of a Sandwich Structure using an iterative process.

        This method computes the failure load and crack length by solving the stress and energy criteria
        through an iterative procedure. It starts with an initial guess for the failure load and crack size
        and iteratively updates them until the convergence criteria are met.

        Parameters
        ----------
        BL : object
            The Sandwich Beam or structure object that contains information about the geometry and material properties.
        maxIter : int
            The maximum number of iterations for the optimization procedure.
        Finit : float
            The initial guess for the applied load in Newtons.
        tol : float
            The tolerance for convergence, defined as the relative change in failure load between iterations.
        write_Data : bool, optional
            If True, stores the results of each iteration in the `FailureLoad_Iterated` and `Crack_Iterated` attributes of the solution class. Default is False.

        Returns
        -------
        Ff : float
            The optimized failure load (N) at the point where the energy criterion is satisfied.
        af : float
            The optimized crack length (mm) at the point where the energy criterion is satisfied.
        """
        timer_start = time.time()

        F = np.zeros((maxIter, 1))
        delta_a = np.zeros((maxIter, 1))
        zeit = np.zeros(maxIter)

        # Helper function for the upcoming line
        def Ffun(F):        
            fsig = self.calc_stressCrit(F, 0, BL)            
            return fsig    

        F[0] = fsolve(Ffun, Finit)
        delta_a[0][0] = 1e-04
        zeit[0] = time.time() - timer_start  

        print("Iteration:", 0, "F=", F[0][0], "N, a=", delta_a[0][0], "mm, took", zeit[0], "s \n")

        ERRbar1, ERRbar2 = self.calc_Krenk(F[0][0], delta_a[0][0], BL)
        gsig = self.energyCrit(ERRbar1, ERRbar2, self.EnergyCrit)

        if gsig >= 1:
            # If energy criterion fulfilled, failure load is found
            Ff = F[0]
            af = delta_a[0]
            n = 0
        else:
            # Iterative procedure
            for n in range(1, maxIter):
                timer_start = time.time()            

                # Update applied load                
                ERRbar1, ERRbar2 = self.calc_Krenk(F[n-1][0], delta_a[n-1][0], BL)
                gsig = self.energyCrit(ERRbar1, ERRbar2, self.EnergyCrit)
                F[n][0] = np.sqrt(1 / gsig) * F[n-1][0]            

                SF = self.load(F[n][0], 0)

                # Evaluate stress criterion for new load
                def delta_aFun(a):                
                    fsig = self.calc_stressCrit(F[n][0], a, BL)                    
                    return fsig            

                delta_a[n] = fsolve(delta_aFun, delta_a[n-1][0])
                zeit[n] = (time.time() - timer_start)

                print("Iteration:", n, "F=", F[n][0], "N, a=", delta_a[n][0], "mm, took", zeit[n], "s \n")

                # Convergence reached?
                if abs(F[n] - F[n-1]) / F[n] < tol:
                    Ff = F[n]
                    af = delta_a[n]
                    maxI = n
                    break            

                # Maximum number of iterations reached
                if n == maxIter - 1:
                    Ff = 1e10
                    af = 1e10
                    print("optimize_ffm: Maximum number of iterations reached")

        if write_Data:
            self.calc_Krenk(F[n, 0], delta_a[n, 0], BL, True) 
            SF = self.load(F[n, 0], 0)
            self.linearelasticsandwich_stress(SF, self.L0, 0, BL, True)           
            self.Sol.FailureLoad_Iterated = F[0:n+1, 0]
            self.Sol.Crack_Iterated = delta_a[0:n+1, 0]
            self.Sol.n_Iteration = n

        return Ff, af
    
    def brittlenessnumber(self, F, BL):
        """
        Calculates the Brittle Number to assess the appropriateness of the calculation.

        The Brittle Number is a dimensionless quantity used to estimate the likelihood of brittle failure
        based on the applied load and crack length. It incorporates the energy release rate and stress criteria.

        Parameters
        ----------
        F : float
            The applied load (N) at which the brittleness number is calculated.
        BL : object
            The Sandwich Beam or structure object, used to provide the necessary properties for calculation.

        Returns
        -------
        mu : float
            The calculated Brittle Number, which indicates the potential for brittle failure.
        """
        ERRbar1, ERRbar2 = self.calc_Krenk(F, 1e-5, BL)
        gsig = self.energyCrit(ERRbar1, ERRbar2, self.EnergyCrit)
        mu = (self.calc_stressCrit(F, 0, BL) + 1) ** 2 / gsig
        return mu
    
    def edge_stress(self, L_linspace, F=0, BL=True):

        """
        Computes the edge stresses of a sandwich structure for differnet overlapping lenghts.

        This method calculates the normal stress (`sigzz_edge`) and shear stress (`tauxz_edge`) at the edge of a sandwich structure on the edge for different overlapping lenghts defined by `L_linspace`. The calculations are based on linear elasticity theory.

        Parameters
        ----------
        L_linspace : array-like
            Array of overlapping lengths.
        F : float, optional
            Applied force (N). Default is 0.
        BL : bool, optional
            Flag indicating the side of the sandwich layer (True: Left, False: Right). Default is True.

        Returns
        -------
        tuple
            - `sigzz_edge` (array-like): Normal stresses along the edge.
            - `tauxz_edge` (array-like): Shear stresses along the edge.
        """

        sigzz_edge = np.zeros_like(L_linspace)
        tauxz_edge = np.zeros_like(L_linspace)

        SF = self.load(F, 0)
        

        i = 0
        
        for L in L_linspace:            
            sigzz_edge[i], tauxz_edge[i] = self.linearelasticsandwich_stress(SF, L, 0, BL, False)
            i = i+1

        self.Sol.tauxz_edge=tauxz_edge
        self.Sol.sigzz_edge=sigzz_edge
        self.Sol.L_linspace=L_linspace
        
        return sigzz_edge, tauxz_edge            

    

## Joint-Types

class Sandwich_Layer(Sandwich):
    """
    Subclass of Sandwich:
    Class for basic Sandwich Layer Properties and with given Section Forces.

    

    Inherits from the Sandwich class and uses specialized material properties and formulas for loading.

    T-Joint is defined by:
    ----------
    h1 : float
        Height of the first adherend layer (mm).
    h2 : float
        Height of the second adherend layer (mm).
    t : float
        Thickness of the adhesive layer (mm).
    L0 : float
        Overlapping length (mm).
    b : float
        Overlap width (mm).
    SF : liste
        List of Section Forces [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22] (N).
    F : float
        Force on the joint (N).
    a : float
        Crack length (mm).

    Methods
    -------
    _init__(self, h1, h2, t, L0, b, SF, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit):
        Initialize Sandwich-Layer with user input for material properties, Section Forces and geometric parameters.
    
    load(self, F, a):
        Dummie Function to be consistent.    
    """
    def __init__(self, h1, h2, t, L0, b, SF, Adherend1, Adherend2, Adhesive, StressCrit = None, EnergyCrit = None):
        """
        Initialize T_Joint with user input for material properties and geometric parameters.
        
        Parameters
        ----------
        h1 : float
            Height of the first adherend layer (mm).
        h2 : float
            Height of the second adherend layer (mm).
        t : float
            Thickness of the adhesive layer (mm).
        L0 : float
            Overlapping length (mm).
        b : float
            Overlap width (mm).
        SF : liste
            List of Section Forces [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22] (N).
        Adherend1 : Material_Adherend
            Material properties for the first adherend.
        Adherend2 : Material_Adherend
            Material properties for the second adherend.
        Adhesive : Material_Adhesive
            Material properties for the adhesive.
        StressCrit : function (optional)
            Stress criterion function to check the failure condition.
        EnergyCrit : function (optional)
            Energy criterion function to check the failure condition.
        """
        super().__init__("Sandwich-Layer", h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit)  
        self.SF = SF

    def load(self, F, a):
            """
            Return by user given Section Forces. Needed to be consistent

            Parameters (not used could)
            ----------
            F : float
                The applied force (N) on the joint.
            a : float
                The crack length (mm).

            Returns
            -------
            list
                A list containing the section forces: [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22].
                These represent the normal forces, shear forces, and bending moments at various sections.
            """            
            return self.SF


class T_Joint(Sandwich):
    """
    Subclass of Sandwich:
    Class for T-Joint Properties and Calculation of Section Forces.

    A T-Joint is a joint configuration in which two adherend layers are bonded with an adhesive layer in 
    the shape of the letter 'T'. This class computes the section forces (e.g., bending moments, shear forces) 
    based on the applied load and crack length in the joint. It assumes that the T-joint undergoes deformation 
    under loading and calculates the forces at different sections.

    Inherits from the Sandwich class and uses specialized material properties and formulas for loading.

    T-Joint is defined by:
    ----------
    h1 : float
        Height of the first adherend layer (mm).
    h2 : float
        Height of the second adherend layer (mm).
    t : float
        Thickness of the adhesive layer (mm).
    L0 : float
        Overlapping length (mm).
    a0 : float
        External length (mm).
    b : float
        Overlap width (mm).
    F : float
        Force on the joint (N).
    a : float
        Crack length (mm).

    Methods
    -------
    _init__(self, h1, h2, t, L0, a0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit):
        Initialize T_Joint with user input for material properties and geometric parameters.
    
    load(self, F, a):
        Calculates section forces for the T-Joint under applied load and crack length.    
    """


    def __init__(self, h1, h2, t, L0, a0, b, Adherend1, Adherend2, Adhesive, StressCrit = None, EnergyCrit = None):
        """
        Initialize T_Joint with user input for material properties and geometric parameters.
        
        Parameters
        ----------
        h1 : float
            Height of the first adherend layer (mm).
        h2 : float
            Height of the second adherend layer (mm).
        t : float
            Thickness of the adhesive layer (mm).
        L0 : float
            Overlapping length (mm).
        a0 : float
            External length (mm).
        b : float
            Overlap width (mm).
        Adherend1 : Material_Adherend
            Material properties for the first adherend.
        Adherend2 : Material_Adherend
            Material properties for the second adherend.
        Adhesive : Material_Adhesive
            Material properties for the adhesive.
        StressCrit : function (optional)
            Stress criterion function to check the failure condition.
        EnergyCrit : function (optional)
            Energy criterion function to check the failure condition.
        """
        super().__init__("T-Joint", h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit)  
        self.a0 = a0

    
    def load(self, F, a):
        """
        Calculates section forces for the T-Joint under applied load and crack length.

        This method computes the bending moment and shear force at various sections of the T-Joint 
        based on the applied force and crack length. It assumes a simple model with applied forces and 
        moments at specific locations along the joint.

        Parameters
        ----------
        F : float
            The applied force (N) on the joint.
        a : float
            The crack length (mm).

        Returns
        -------
        list
            A list containing the section forces: [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22].
            These represent the normal forces, shear forces, and bending moments at various sections.
        """
        L0 = self.L0
        b = self.b
        V11 = F / b
        M11 = 5 * V11
        V22 = F / b
        M22 = V22 * (5 + L0 - a)
        T11 = M12 = T21 = V21 = M21 = T22 = T12 = V12 = 0 
        
        return [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22]


class SLJ_GR(Sandwich):
    """
    Subclass of Sandwich:
    Class for Single Lap Joint (SLJ) with Goland-Reissner (GR) theory and Calculation of Section Forces.
    
    The Goland-Reissner (GR) theory is applied to model the behavior of Single Lap Joints (SLJ) with 
    adhesive layers, considering the deformation and stress distributions across the adhesive and adherends. 
    This class calculates the section forces (e.g., bending moments, shear forces) based on the applied 
    load and crack length in the joint.

    Inherits from the Sandwich class and uses specialized material properties and formulas for loading.

    SLJ_GR is a special type of Joint defined by:
    ----------
    h1 : float
        Height of the first adherend layer (mm).
    h2 : float
        Height of the second adherend layer (mm).
    t : float
        Thickness of the adhesive layer (mm).
    L0 : float
        Overlapping length (mm).
    b : float
        Overlap width (mm).
    F : float
        Force on the joint (N).
    a : float
        Crack length (mm).
    Methods
    -------
    _init__(self, h1, h2, t, L0, a0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit):
        Initialize SLJ_GR with user input for material properties and geometric parameters.
    
    load(self, F, a):
        Calculates section forces for the SLJ_GR under applied load and crack length.    
    """
    def __init__(self, h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit = None, EnergyCrit = None):
        """
        Initialize SLJ_GR with user input for material properties and geometric parameters.
        
        Parameters
        ----------
        h1 : float
            Height of the first adherend layer (mm).
        h2 : float
            Height of the second adherend layer (mm).
        t : float
            Thickness of the adhesive layer (mm).
        L0 : float
            Overlapping length (mm).
        b : float
            Overlap width (mm).
        Adherend1 : Material_Adherend
            Material properties for the first adherend.
        Adherend2 : Material_Adherend
            Material properties for the second adherend.
        Adhesive : Material_Adhesive
            Material properties for the adhesive.
        StressCrit : function (optional)
            Stress criterion function to check the failure condition.
        EnergyCrit : function (optional)
            Energy criterion function to check the failure condition.
        """
        super().__init__("SLJ_GR", h1, h2, t, L0, b, Adherend1, Adherend2, Adhesive, StressCrit, EnergyCrit)

    def load(self, F, a):
        """
        Calculates section forces for the SLJ_GR under applied load and crack length.

        This method computes the bending moment and lateral force at various sections of the SLJ_GR joint 
        based on the applied force and crack length. It also computes the generalized resistance factors 
        using the Goland-Reissner (GR) theory.

        Parameters
        ----------
        F : float
            The applied force (N) on the joint.
        a : float
            The crack length (mm).

        Returns
        -------
        list
            A list containing the section forces: [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22].
            These represent the normal forces, shear forces, and bending moments at various sections.
        """
        M1 = self.Adherend1.M
        h1 = self.h1
        L0 = self.L0
        b = self.b
        t = self.t

        nu = 1 - 2 * M1[3] / M1[0]
        E = M1[0] / h1 * (1 - nu**2)

        # Factors for bending moment and lateral force
        kM = 1 / (1 + 2 * np.sqrt(2) * np.tanh(np.sqrt((3 * (1 - nu**2)) / 2) * (L0 - a) / (2 * h1) * np.sqrt(F / (E * b * h1))))
        kV = ((1 - kM) * h1 + t) / (L0 - a)

        # Bending moment and lateral force
        M0 = kM * h1 / 2 * F / b
        V0 = kV * F / b

        # General loading
        T11 = F / b
        V11 = V0
        M11 = M0
        T22 = F / b
        V22 = V0
        M22 = -M0

        # All other loads zero
        T12 = V12 = M12 = T21 = V21 = M21 = 0
        
        return [T11, V11, M11, T12, V12, M12, T21, V21, M21, T22, V22, M22]

