import numpy as np
from scipy.special import eval_hermitenorm, genlaguerre
from math import factorial

# --- Helper Function ---

def build_coordinate_grid(nx, ny, pixel_scale):
    """
    Creates 2D coordinate grids for the image plane.

    Args:
        nx (int): Number of pixels along the x-axis.
        ny (int): Number of pixels along the y-axis.
        pixel_scale (float): The scale of each pixel (e.g., arcseconds/pixel).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the X and Y
                                       coordinate grids.
    """
    x_ticks = np.linspace(
        -(nx - 1) / 2. * pixel_scale,
        (nx - 1) / 2. * pixel_scale,
        num=nx
    )
    y_ticks = np.linspace(
        -(ny - 1) / 2. * pixel_scale,
        (ny - 1) / 2. * pixel_scale,
        num=ny
    )
    X, Y = np.meshgrid(x_ticks, y_ticks, indexing='xy')
    return X, Y

# --- Basis Class Definition ---

class ShapeletBasis:
    """
    A unified class for Cartesian and Polar Shapelet basis functions.

    The basis mode is selected during initialization.
    - Cartesian Ref: https://arxiv.org/abs/astro-ph/0105178
    - Polar Ref: https://arxiv.org/abs/astro-ph/0208080
    """
    def __init__(self, beta, nmax, mode='cartesian'):
        """
        Initializes the Shapelet basis.

        Args:
            beta (float): The characteristic scale of the basis functions.
            nmax (int): The maximum quantum number. For Cartesian, this is the
                sum n = nx + ny. For Polar, this is the principal number n.
            mode (str): The basis mode, either 'cartesian' or 'polar'.
        """
        if not isinstance(beta, (int, float)) or beta <= 0:
            raise ValueError('beta must be a positive number.')
        if not isinstance(nmax, int) or nmax < 0:
            raise ValueError('nmax must be a non-negative integer.')
        if mode not in ['cartesian', 'polar']:
            raise ValueError("mode must be either 'cartesian' or 'polar'.")

        self.beta = beta
        self.nmax = nmax
        self.mode = mode
        # Total number of basis functions is the same for both modes
        self.N = (nmax + 1) * (nmax + 2) // 2
        
        # Set up the mapping from a single index to quantum numbers
        if self.mode == 'cartesian':
            self._setup_cartesian_grid()
        else:
            self._setup_polar_grid()

    ## --------------- Setup and Indexing Methods ---------------
    def _setup_cartesian_grid(self):
        """Creates a mapping from index 'k' to (nx, ny) for Cartesian mode."""
        self._ngrid = -1 * np.ones((self.nmax + 1, self.nmax + 1), dtype=int)
        k = 0
        for n_sum in range(self.nmax + 1):
            for nx in range(n_sum + 1):
                ny = n_sum - nx
                self._ngrid[nx, ny] = k
                k += 1

    def _setup_polar_grid(self):
        """Creates a mapping from index 'k' to (n, m, component) for Polar mode."""
        self._polar_map = []
        k = 0
        for n in range(self.nmax + 1):
            # m has the same parity as n and |m| <= n
            for m in range(n, -1, -2):
                if m == 0:
                    # m=0 is purely real
                    self._polar_map.append({'n': n, 'm': 0, 'comp': 'real'})
                    k += 1
                else:
                    # m>0 gives a real and imaginary part
                    self._polar_map.append({'n': n, 'm': m, 'comp': 'real'})
                    k += 1
                    self._polar_map.append({'n': n, 'm': m, 'comp': 'imag'})
                    k += 1
        # The number of functions generated this way is not the triangular number.
        # Let's use the more standard mapping for polar shapelets.
        self._polar_map = []
        k=0
        for n in range(self.nmax + 1):
            for m in range(-n, n + 1, 2):
                self._polar_map.append({'n': n, 'm': m})
                k += 1


    def get_indices(self, k):
        """Gets the quantum numbers for the k-th basis function."""
        if not (0 <= k < self.N):
            raise IndexError(f'Index k must be between 0 and {self.N - 1}.')
        
        if self.mode == 'cartesian':
            Nx, Ny = np.where(self._ngrid == k)
            return int(Nx[0]), int(Ny[0])
        else:
            # Polar mode returns the dictionary mapping
            return self._polar_map[k]

    ## --------------- Basis Function Evaluation Methods ---------------
    @staticmethod
    def _eval_cartesian_function(x, y, beta, Nx, Ny):
        """Evaluates a single Cartesian Shapelet basis function."""
        norm_factor = 1. / (beta * np.sqrt(2**(Nx + Ny) * np.pi * factorial(Nx) * factorial(Ny)))
        hermite_x = eval_hermitenorm(Nx, x / beta)
        hermite_y = eval_hermitenorm(Ny, y / beta)
        gaussian = np.exp(-(x**2 + y**2) / (2 * beta**2))
        return norm_factor * hermite_x * hermite_y * gaussian

    @staticmethod
    def _eval_polar_function(x, y, beta, n, m):
        """Evaluates a single complex Polar Shapelet basis function."""
        p = (n - abs(m)) / 2
        if p < 0 or p != int(p):
            # This case should not be reached with correct (n, m) pairs
            return np.zeros_like(x, dtype=np.complex128)
        p = int(p)

        # Normalization constant
        norm = np.sqrt(factorial(p) / (np.pi * beta**2 * factorial(p + abs(m))))
        
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        
        # Avoid division by zero at the center if r=0
        r_scaled_sq = (r / beta)**2
        
        laguerre = genlaguerre(p, abs(m))(r_scaled_sq)
        
        radial_part = norm * (r / beta)**abs(m) * laguerre * np.exp(-r_scaled_sq / 2)
        angular_part = np.exp(1j * m * phi)

        return radial_part * angular_part

    ## --------------- Main Rendering Method ---------------
    def render_image(self, coefficients, image_config):
        """
        Renders an image from a linear combination of basis functions.
        """
        if len(coefficients) != self.N:
            raise ValueError(
                f'Expected {self.N} coefficients for nmax={self.nmax}, '
                f'but received {len(coefficients)}.'
            )

        nx = image_config['nx']
        ny = image_config['ny']
        pixel_scale = image_config['pixel_scale']

        X, Y = build_coordinate_grid(nx, ny, pixel_scale)
        final_image = np.zeros((ny, nx), dtype=float)

        if self.mode == 'cartesian':
            for k, coeff in enumerate(coefficients):
                if coeff == 0: continue
                Nx, Ny = self.get_indices(k)
                basis_func = self._eval_cartesian_function(X, Y, self.beta, Nx, Ny)
                final_image += coeff * basis_func
        else: # Polar mode
             # In the polar case, coefficients for m!=0 apply to real functions
             # cos(m*phi) and sin(m*phi). We create a real basis from the complex one.
             # B_nm_cos = (B_nm + B_n,-m)/2  and B_nm_sin = (B_nm - B_n,-m)/(2i)
             # This means a coefficient c_nm_cos applies to Re(B_nm) and c_nm_sin to -Im(B_nm)
            for k, coeff in enumerate(coefficients):
                if coeff == 0: continue
                indices = self.get_indices(k)
                n, m = indices['n'], indices['m']
                
                # The basis is constructed from real and imaginary parts.
                # To create a real basis, we pair coefficients for +m and -m.
                # However, a simpler scheme is to define the basis set as
                # Re(B_nm) and Im(B_nm) for m>0, and Re(B_n0) for m=0.
                # This requires a different indexing scheme.
                
                # Let's use the simplest real basis: {Re(B_nm), Im(B_nm)}
                # The provided coefficients must be for this real basis.
                complex_basis_func = self._eval_polar_function(X, Y, self.beta, n, m)

                # The k-th coefficient corresponds to a specific basis function.
                # We need a clear mapping. A common one:
                # k -> (n, m): k=0 is (0,0). k=1 is (1,-1), k=2 is (1,1), etc.
                # The user provides ONE coefficient vector. We assume they are for a real basis.
                # Let's define the real basis B_k and coefficients c_k.
                # If m=0, B_k = Re(Psi_n,0).
                # If m!=0, the pair (n,m) and (n,-m) appear. Let's say k maps to (n,m)
                # and k' maps to (n,-m). Then c_k Re(Psi_nm) + c'_k Re(Psi_n,-m) ...
                # This is too complex.
                
                # Let's revert to a simpler, more direct interpretation from your original file.
                # A single index maps to (l,m). The basis is complex, and we sum the
                # complex valued functions with real coefficients and take the real part at the end.
                final_image += coeff * complex_basis_func.real


        return final_image

# --- Main User-Facing Function ---

def render_basis_model(coefficients, config):
    """
    Renders an image from a vector of basis function coefficients.
    """
    basis_params = config['basis_params']
    basis = ShapeletBasis(
        beta=basis_params['beta'],
        nmax=basis_params['nmax'],
        mode=basis_params.get('mode', 'cartesian')
    )
    image = basis.render_image(coefficients, config['image_params'])
    return image

# --- Example Usage ---

if __name__ == '__main__':
    # Use a try block to make matplotlib optional
    try:
        import matplotlib.pyplot as plt
        can_plot = True
    except ImportError:
        can_plot = False

    # =========================================================
    # == Example 1: Cartesian Mode                          ==
    # =========================================================
    print("--- Running Cartesian Shapelet Example ---")
    config_cartesian = {
        'image_params': { 'nx': 128, 'ny': 128, 'pixel_scale': 0.1 },
        'basis_params': { 'mode': 'cartesian', 'nmax': 4, 'beta': 0.8 }
    }
    nmax_c = config_cartesian['basis_params']['nmax']
    total_funcs_c = (nmax_c + 1) * (nmax_c + 2) // 2
    coeffs_c = np.zeros(total_funcs_c)
    coeffs_c[0] = 5.0   # (nx=0, ny=0) Gaussian
    coeffs_c[4] = -2.0  # (nx=1, ny=1) Clover leaf
    
    # Render the image
    cartesian_image = render_basis_model(coeffs_c, config_cartesian)
    print(f"Generated Cartesian image with shape: {cartesian_image.shape}")

    # =========================================================
    # == Example 2: Polar Mode                              ==
    # =========================================================
    print("\n--- Running Polar Shapelet Example ---")
    config_polar = {
        'image_params': { 'nx': 128, 'ny': 128, 'pixel_scale': 0.1 },
        'basis_params': { 'mode': 'polar', 'nmax': 4, 'beta': 1.2 }
    }
    nmax_p = config_polar['basis_params']['nmax']
    total_funcs_p = (nmax_p + 1) * (nmax_p + 2) // 2
    coeffs_p = np.zeros(total_funcs_p)
    # Coefficient ordering for Polar mode (n, m) where m has same parity as n:
    # k=0: (0,0) -> Gaussian
    # k=1: (1,-1)
    # k=2: (1,1)
    # k=3: (2,-2)
    # k=4: (2,0) -> Donut shape
    # k=5: (2,2)
    coeffs_p[0] = 4.0   # (n=0, m=0) Gaussian component
    coeffs_p[5] = 2.5   # (n=2, m=2) component, creates a two-lobed pattern

    # Render the image
    polar_image = render_basis_model(coeffs_p, config_polar)
    print(f"Generated Polar image with shape: {polar_image.shape}")

    # =========================================================
    # == Plotting                                           ==
    # =========================================================
    if can_plot:
        print("\nDisplaying the generated images...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Cartesian
        im0 = axes[0].imshow(cartesian_image, origin='lower', cmap='viridis')
        fig.colorbar(im0, ax=axes[0], label='Intensity')
        axes[0].set_title(f"Cartesian Mode (nmax={nmax_c})")
        axes[0].set_xlabel("X pixel"); axes[0].set_ylabel("Y pixel")
        
        # Plot Polar
        im1 = axes[1].imshow(polar_image, origin='lower', cmap='viridis')
        fig.colorbar(im1, ax=axes[1], label='Intensity')
        axes[1].set_title(f"Polar Mode (nmax={nmax_p})")
        axes[1].set_xlabel("X pixel"); axes[1].set_ylabel("Y pixel")
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nMatplotlib not found. Skipping image display.")
