class Material_Adherend:
    """
    Class for Material Properties of Adherends.

    An adherend material is characterized by the following attributes:
    - name: Name of the material.
    - E: Young's Modulus.
    - nu: Poisson's ratio.
    - M: ABD-Matrix (optional, can be calculated for isotropic materials).
    
    Methods
    -------
    __init__(self, name, E, nu, M=None):
        Initializes the material properties for the adherend.
    
    computeABD_isotrop(self, h):
        Computes the ABD-Matrix for isotropic materials, given the height (h).
    """
    
    def __init__(self, name, E, nu, M=None):
        """
        Initializes the material properties for the adherend.

        Parameters
        ----------
        name : str
            The name of the material.
        E : float
            Young's Modulus of the material.
        nu : float
            Poisson's ratio of the material.
        M : list, optional
            The ABD matrix for the material (default is None). If provided, it will be used directly.
        """
        self.name = name
        self.E = E
        self.nu = nu
        self.M = M

    def computeABD_isotrop(self, h):
        """
        Computes the ABD matrix for isotropic materials.

        The ABD matrix is important for analyzing the sandwich structure behavior.
        This function computes the A, B, D, A55, and k values for an isotropic material.

        Parameters
        ----------
        h : float
            The height (thickness) of the adherend material.

        Returns
        -------
        list
            The computed ABD matrix as a list [A11, B11, D11, A55, k].
        
        Notes
        -----
        The ABD matrix is used in sandwich beam analysis, but it is not strictly a material property.
        Consider moving this functionality to the `Sandwich` class or using a helper function.
        """
        A11 = self.E * h / (1 - self.nu**2)
        B11 = 0.
        D11 = self.E * h**3 / (12 * (1 - self.nu**2))
        A55 = self.E * h / (2 * (1 + self.nu))
        k = 5 / 6
        self.M = [A11, B11, D11, A55, k]
        return self.M



class Material_Adhesive:
    """
    Class for Material Properties of Adhesive.

    An adhesive material is characterized by the following attributes:
    - name: Name of the material.
    - E: Young's Modulus.
    - nu: Poisson's ratio.
    - G_Ic: Mode I critical energy release rate.
    - G_IIc: Mode II critical energy release rate (optional).
    - sig_c: Critical normal stress (optional).
    - tau_c: Critical shear stress (optional).
    
    Methods
    -------
    __init__(self, name, E, nu, G_Ic, G_IIc=None, sig_c=None, tau_c=None, G=None):
        Initializes the material properties for the adhesive.
    """

    def __init__(self, name, E, nu, G_Ic=None, G_IIc=None, sig_c=None, tau_c=None, G=None):
        """
        Initializes the material properties for the adhesive.

        Parameters
        ----------
        name : str
            The name of the material.
        E : float
            Young's Modulus of the material.
        nu : float
            Poisson's ratio of the material.
        G_Ic : float, optional
            Mode I critical energy release rate of the adhesive.
        G_IIc : float, optional
            Mode II critical energy release rate of the adhesive (default is None).
        sig_c : float, optional
            Critical normal stress of the adhesive (default is None).
        tau_c : float, optional
            Critical shear stress of the adhesive (default is None).
        G : float, optional
            Shear modulus of the adhesive (if not provided, it is calculated as E / (2 * (1 + nu))).
        """
        self.name = name
        self.E = E
        self.nu = nu
        self.G_Ic = G_Ic
        self.G_IIc = G_IIc
        self.sig_c = sig_c
        self.tau_c = tau_c
        if G is None:
            self.G = E / (2 * (1 + nu))
        else:
            self.G = G
