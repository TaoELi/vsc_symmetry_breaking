# This class calculates the polariton spectrum for a given set of parameters

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import matplotlib
import columnplots as clp
import sys

cupy_installed = False
try:
    import cupy as cp
    cupy_installed = True
except:
    print("cupy is not installed, using numpy instead")
    cp = np

hartree_to_cminv = 219474.63


# predefined parameter set for different cavity setups in a dictionary
cavity_1d_sin = {"ngridx_1d": 1080,
                 "ngridy_1d": 1,
                 "nmodex_1d": 200,
                 "nmodey_1d": 1,
                 "omega_perp": 2320.0,
                 "domegax_1d": 10.0,
                 "domegay_1d": 10.0,
                 "g0": 2.0,
                 "omega_m": 2320.0,
                 "pattern": "sin1d_250_0.05",
                 "enlarge": False}

cavity_1d_gaussian = {"ngridx_1d": 1080,
                      "ngridy_1d": 1,
                      "nmodex_1d": 200,
                      "nmodey_1d": 1,
                      "omega_perp": 2320.0,
                      "domegax_1d": 10.0,
                      "domegay_1d": 10.0,
                      "g0": 2.0,
                      "omega_m": 2320.0,
                      "pattern": "gaussian1d_0.3",
                      "enlarge": False}

cavity_1d_gaussian_perturb = {"ngridx_1d": 1080,
                      "ngridy_1d": 1,
                      "nmodex_1d": 200,
                      "nmodey_1d": 1,
                      "omega_perp": 2320.0,
                      "domegax_1d": 10.0,
                      "domegay_1d": 10.0,
                      "g0": 2.0,
                      "omega_m": 2320.0,
                      "pattern": "gp1d_0.03_0.1",
                      "enlarge": False}

cavity_1d_tilting = {"ngridx_1d": 1080,
                      "ngridy_1d": 1,
                      "nmodex_1d": 200,
                      "nmodey_1d": 1,
                      "omega_perp": 2320.0,
                      "domegax_1d": 10.0,
                      "domegay_1d": 10.0,
                      "g0": 2.0,
                      "omega_m": 2320.0,
                      "pattern": "tilting1d_600_0.8",
                      "enlarge": False}

cavity_2d_uniform = {"ngridx_1d": 120,
                     "ngridy_1d": 120,
                     "nmodex_1d": 60,
                     "nmodey_1d": 60,
                     "omega_perp": 2320.0,
                     "domegax_1d": 30.0,
                     "domegay_1d": 30.0,
                     "g0": 0.5,
                     "omega_m": 2320.0,
                     "pattern": "uniform",
                     "enlarge": False}

cavity_2d_sin = {"ngridx_1d": 120,
                     "ngridy_1d": 120,
                     "nmodex_1d": 60,
                     "nmodey_1d": 60,
                     "omega_perp": 2320.0,
                     "domegax_1d": 30.0,
                     "domegay_1d": 30.0,
                     "g0": 0.5,
                     "omega_m": 2320.0,
                     "pattern": "sin2d_250_1.0",
                     "enlarge": False}

cavity_2d_gaussian = {"ngridx_1d": 120,
                     "ngridy_1d": 120,
                     "nmodex_1d": 60,
                     "nmodey_1d": 60,
                     "omega_perp": 2320.0,
                     "domegax_1d": 30.0,
                     "domegay_1d": 30.0,
                     "g0": 0.5,
                     "omega_m": 2320.0,
                     "pattern": "gaussian2d_0.25",
                     "enlarge": False}

cavity_2d_tilting = {"ngridx_1d": 120,
                     "ngridy_1d": 120,
                     "nmodex_1d": 60,
                     "nmodey_1d": 60,
                     "omega_perp": 2320.0,
                     "domegax_1d": 30.0,
                     "domegay_1d": 30.0,
                     "g0": 0.5,
                     "omega_m": 2320.0,
                     "pattern": "tilting2d_250",
                     "enlarge": False}

cavity_2d_usr = {"ngridx_1d": 120,
                     "ngridy_1d": 120,
                     "nmodex_1d": 60,
                     "nmodey_1d": 60,
                     "omega_perp": 2320.0,
                     "domegax_1d": 30.0,
                     "domegay_1d": 30.0,
                     "g0": 0.5,
                     "omega_m": 2320.0,
                     "pattern": "spongebob.png",
                     "enlarge": False}


def renormalize_spectrum(spectrum, nx):
    for i in range(nx):
        sum_value = np.sum(spectrum[i, :])
        if np.abs(sum_value) > 1e-10:
            spectrum[i, :] /= sum_value

class PolaritonSpectrumCalc:
    def __init__(self, cavity_params):

        pattern = cavity_params["pattern"]
        enlarge = cavity_params["enlarge"]

        # definition of the photon modes
        self.omega_perp = cavity_params["omega_perp"]  # cm^-1
        self.domegax_1d = cavity_params["domegax_1d"]  # cm^-1
        self.domegay_1d = cavity_params["domegay_1d"]  # cm^-1
        self.nmodex_1d = cavity_params["nmodex_1d"]
        self.nmodey_1d = cavity_params["nmodey_1d"]
        if enlarge:
            self.nmodex_1d *= 2
            self.nmodey_1d *= 2
            self.domegax_1d /= 2.0
            self.domegay_1d /= 2.0

        # definition of the molecular exciton
        self.omega_m = cavity_params["omega_m"]  # cm^-1

        # definition of the molecular grid points in units of Lx, Ly
        self.ngridx_1d = cavity_params["ngridx_1d"]
        self.ngridy_1d = cavity_params["ngridy_1d"]

        # obtain the molecular distribution for the original cell
        g0_distri = self.set_molecular_distribution(pattern=pattern)
        print("molecular distribution = \n", g0_distri)

        # enlarge to check if the spectrum is sensitive to the grid size
        if enlarge:
            g0_distri_enlarged = np.zeros((self.ngridx_1d * 2, self.ngridy_1d * 2))
            g0_distri_enlarged[0:self.ngridx_1d, 0:self.ngridy_1d] = g0_distri
            g0_distri_enlarged[0:self.ngridx_1d, self.ngridy_1d:] = g0_distri
            g0_distri_enlarged[self.ngridx_1d:, 0:self.ngridy_1d] = g0_distri
            g0_distri_enlarged[self.ngridx_1d:, self.ngridy_1d:] = g0_distri
            g0_distri = g0_distri_enlarged
            self.ngridx_1d *= 2
            self.ngridy_1d *= 2

        self.xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d  # in units of Lx
        self.ygrid_1d = np.linspace(1.0, self.ngridy_1d - 1.0, self.ngridy_1d) / self.ngridy_1d  # in units of Ly
        if self.ngridy_1d == 1:
            self.ygrid_1d = np.array([0.0])
        self.xgrid_2d, self.ygrid_2d = np.meshgrid(self.xgrid_1d, self.ygrid_1d)
        self.xgrid_2d = np.reshape(self.xgrid_2d, -1)
        self.ygrid_2d = np.reshape(self.ygrid_2d, -1)
        self.ngrid = np.size(self.xgrid_2d)
        print("x grid point array in 1D = ", self.xgrid_1d)
        print("y grid point array in 1D = ", self.ygrid_1d)
        # print("x grid point array in 2D = \n", self.xgrid_2d)
        # print("y grid point array in 2D = \n", self.ygrid_2d)
        print("number of grid points = ", self.ngrid)

        # kx and ky in units of 1/Lx or 1/Ly
        #self.kx_grid_1d = 2.0 * np.pi * np.array([i + 1.0 for i in range(self.nmodex_1d)])
        #self.ky_grid_1d = 2.0 * np.pi * np.array([i + 1.0 for i in range(self.nmodey_1d)])
        #self.kx_grid_1d = 2.0 * np.pi * np.array([i + 1.0 for i in range(self.nmodex_1d)] + [-i for i in range(self.nmodex_1d)])
        self.kx_grid_1d = 2.0 * np.pi * np.array([-self.nmodex_1d//2+i for i in range(self.nmodex_1d//2)] + [i+1 for i in range(self.nmodex_1d//2)])
        if self.nmodey_1d == 1:
            self.ky_grid_1d = 2.0 * np.pi * np.array([1.0])
        else:
            self.ky_grid_1d = 2.0 * np.pi * np.array([-self.nmodey_1d//2+i for i in range(self.nmodey_1d//2)] + [i+1 for i in range(self.nmodey_1d//2)])
        self.kx_grid_2d, self.ky_grid_2d = np.meshgrid(self.kx_grid_1d, self.ky_grid_1d)
        self.kx_grid_2d = np.reshape(self.kx_grid_2d, -1)
        self.ky_grid_2d = np.reshape(self.ky_grid_2d, -1)
        self.nmode = np.size(self.kx_grid_2d)
        print("kx grid point array in 1D = ", self.kx_grid_1d)
        print("ky grid point array in 1D = ", self.ky_grid_1d)
        # print("kx grid point array in 2D = \n", self.kx_grid_2d)
        # print("ky grid point array in 2D = \n", self.ky_grid_2d)
        print("number of photon modes = ", self.nmode)

        # definition of the cavity mode frequency array
        self.omega_parallel = ((self.kx_grid_2d / 2.0 / np.pi * self.domegax_1d)**2 + (self.ky_grid_2d / 2.0 / np.pi * self.domegay_1d)**2)**0.5
        self.omega_parallel = np.reshape(self.omega_parallel, -1)
        # print("omega_parallel = ", self.omega_parallel)
        self.omega_k = (self.omega_perp**2 + self.omega_parallel**2)**0.5
        # print("omega_k = ", self.omega_k)

        # definition of the cavity mode function: g0 * np.exp(i * k_parallel * r_parallel)
        self.g0 = cavity_params["g0"]  # in units of cm-1
        if enlarge:
            # enlarge both x and y directions, so the system is four-times larger
            self.g0 /= 2.0
        # now, if there are some grid points with no molecule, we need to set the corresponding g to zero
        self.g0_distri = np.reshape(g0_distri, -1)
        self.g = np.zeros((self.nmode, self.ngrid), dtype=np.complex128)
        for i in range(self.ngrid):
            x, y = self.xgrid_2d[i], self.ygrid_2d[i]
            self.g[:, i] = self.g0 * np.exp(1j * (self.kx_grid_2d * x + self.ky_grid_2d * y)) * self.g0_distri[i]
        # print("mode function = \n", self.g)

        self.check_inplane_translation_symmetry()

        print("constructing the Hamiltonian...")
        self.H_QED = np.zeros((self.ngrid + self.nmode, self.ngrid + self.nmode), dtype=np.complex128)
        self.construct_hamiltonian(self.H_QED)

        print("begin diagonalization")
        # diagonalize the Hamiltonian to get the eigenvalues and eigenvectors -> polariton spectrum
        self.eigenvalues, self.eigenvectors = self.solve_eigenvalue(self.H_QED)

        # get the photonic weight for each eigenstate
        self.ph_weight_summed = self.get_photon_weight_summed()

    def check_inplane_translation_symmetry(self):
        # check the in-plane translational symmetry breaking along two dimensions
        print("analyzing the in-plane translational symmetry breaking along the x dimension...")
        sum_array = np.zeros((self.nmodex_1d, self.nmodex_1d), dtype=np.complex128)
        for j in range(self.nmodex_1d):
            kx = self.kx_grid_1d[j]
            for k in range(self.nmodex_1d):
                kxp = self.kx_grid_1d[k]
                sum_value = np.exp(1j * (kx * self.xgrid_1d)) * np.exp(-1j * (kxp * self.xgrid_1d))
                sum_array[j, k] = np.sum(sum_value)
        sum_array = np.abs(sum_array / self.ngridx_1d - np.eye(self.nmodex_1d))
        print("mean error along x is", np.mean(sum_array))

        print("analyzing the in-plane translational symmetry breaking along the y dimension...")
        sum_array = np.zeros((self.nmodey_1d, self.nmodey_1d), dtype=np.complex128)
        for j in range(self.nmodey_1d):
            ky = self.ky_grid_1d[j]
            for k in range(self.nmodey_1d):
                kyp = self.ky_grid_1d[k]
                sum_value = np.exp(1j * (ky * self.ygrid_1d)) * np.exp(-1j * (kyp * self.ygrid_1d))
                sum_array[j, k] = np.sum(sum_value)
        sum_array = np.abs(sum_array / self.ngridy_1d - np.eye(self.nmodey_1d))
        print("mean error along y is", np.mean(sum_array))

    def construct_hamiltonian(self, H_QED):
        current_time = time.perf_counter()
        # definition of the Hamiltonian
        # the first block is the Hamiltonian of the molecular exciton
        for i in range(self.ngrid):
            H_QED[i, i] = self.omega_m
        # the second block is the Hamiltonian of the cavity modes
        for i in range(self.nmode):
            H_QED[self.ngrid + i, self.ngrid + i] = self.omega_k[i]
        # the off-diagonal block is the light-matter coupling term of the Hamiltonian
        for i in range(self.ngrid):
            for j in range(self.nmode):
                H_QED[i, self.ngrid + j] = self.g[j, i]
                H_QED[self.ngrid + j, i] = np.conjugate(self.g[j, i])
        # print("Hamiltonian = \n", np.real(self.H_QED))
        print("time cost for constructing the Hamiltonian = ", time.perf_counter() - current_time)
        # print("shape of Hamiltonian = ", self.H_QED.shape)

    def solve_eigenvalue(self, H_QED):
        start_time = time.perf_counter()
        H_QED = cp.asarray(H_QED, dtype=cp.complex64)
        # diagonalize the Hamiltonian to get the eigenvalues and eigenvectors -> polariton spectrum
        eigenvalues, eigenvectors = cp.linalg.eigh(H_QED)
        if cupy_installed:
            # convert the eigenvalues and eigenvectors back to numpy arrays
            eigenvalues = cp.asnumpy(eigenvalues)
            eigenvectors = cp.asnumpy(eigenvectors)
        end_time = time.perf_counter()
        print("eigenvalues from numpy", eigenvalues)
        print("\n time cost = ", end_time - start_time)
        return eigenvalues, eigenvectors

    def set_molecular_distribution(self, pattern=None):
        distri = np.ones((self.ngridx_1d, self.ngridy_1d))
        if pattern == "uniform":
            pass
        elif ".png" in pattern or ".jpg" in pattern:
            img = Image.open(pattern)
            print("the original image is")
            # plt.imshow(img)
            # plt.show()
            print("resized image is")
            img = img.resize((self.ngridx_1d, self.ngridy_1d))
            # plt.imshow(img)
            # plt.show()
            bw_img = img.convert("L")
            bw_img = np.array(bw_img)
            bw_img_1d = np.reshape(bw_img, -1)
            avg_color = np.median(bw_img_1d)
            print("median color is ", avg_color)
            idx_sm = bw_img < avg_color
            idx_bg = bw_img >= avg_color
            bw_img[idx_sm] = 255
            bw_img[idx_bg] = 0.0
            print("black and white image is...")
            # plt.imshow(bw_img, cmap=cm.gray)
            # plt.show()
            distri = bw_img / 255.0 * 2.0

        elif pattern == "circles":
            # set circle areas to be zero
            for i in range(self.ngridx_1d):
                for j in range(self.ngridy_1d):
                    if (i - self.ngridx_1d / 2.0)**2 + (j - self.ngridy_1d / 2.0)**2 < (self.ngridy_1d*0.3)**2:
                        distri[i, j] = 0.0
        elif pattern == "random":
            distri = np.random.rand(self.ngridx_1d, self.ngridy_1d)
            distri[distri < 0.5] = 0.0
            distri[distri >= 0.5] = 1.0
        elif pattern == "kspace" and self.ngridy_1d == 1:
            # provide a given k-space distribution along x, obtain the corresponding real-space distribution
            omega_parallel = 250.0  # cm^-1
            n = int(omega_parallel / self.domegax_1d)  # ratio indicates the order of the normal mode
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            print("xgrid_1d shape = ", xgrid_1d.shape)
            print("distri shape = ", distri.shape)
            distri[:, 0] += 0.1 * np.sin(2.0 * np.pi * xgrid_1d * n)
            # sigma = 0.3
            # additional_distri = np.exp(-0.5 * (xgrid_1d - 0.5)**2 / sigma**2) / sigma / (2.0 * np.pi)**0.5
            # distri[:, 0] += additional_distri
            # distri /= np.mean(distri)
        elif "sin1d" in pattern and self.ngridy_1d == 1:
            omega_parallel = float(pattern.split("_")[1])
            amplitude = float(pattern.split("_")[2])
            print("## will apply a sin function with freq = ", omega_parallel, "cm^-1 and relative amplitude ", amplitude)
            n = int(omega_parallel / self.domegax_1d)  # ratio indicates the order of the normal mode
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            print("xgrid_1d shape = ", xgrid_1d.shape)
            print("distri shape = ", distri.shape)
            distri[:, 0] += amplitude * np.sin(2.0 * np.pi * xgrid_1d * n)
            print("### For this sin1d distribution, mean value is", np.mean(distri))
            distri /= np.mean(distri)

        elif "tilting1d" in pattern and self.ngridy_1d == 1:
            omega_parallel = float(pattern.split("_")[1])
            amplitude = float(pattern.split("_")[2])
            print("## will apply a tilting with freq = ", omega_parallel, "cm^-1 and relative amplitude ", amplitude)

            n = int(omega_parallel / self.domegax_1d)  # ratio indicates the order of the normal mode
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d

            distri[:, 0] += amplitude * np.sin(2.0 * np.pi * xgrid_1d * n)

            distri /= np.mean(distri)

        elif "gaussian1d" in pattern and self.ngridy_1d == 1:
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            sigma = float(pattern.split("_")[1])
            gaussian_distri = np.exp(-0.5 * (xgrid_1d - 0.5)**2 / sigma**2) / sigma / (2.0 * np.pi)**0.5
            print("gaussian distri sum is", np.sum(gaussian_distri))
            print("original uniform distri sum is", np.sum(distri))
            distri[:, 0] = gaussian_distri
            print("### For this Gaussian1D distribution, mean value is", np.mean(distri))
            distri /= np.mean(distri)

        elif "gp1d" in pattern and self.ngridy_1d == 1:
            amplitude = float(pattern.split("_")[2])
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            #sigma = float(pattern.split("_")[1])
            #gaussian_distri = 1.0 + amplitude * np.exp(-0.5 * (xgrid_1d - 0.5)**2 / sigma**2) / sigma / (2.0 * np.pi)**0.5
            #print("gaussian distri sum is", np.sum(gaussian_distri))
            #print("original uniform distri sum is", np.sum(distri))
            n_lst = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43]
            for n in n_lst:
                distri[:, 0] += amplitude * np.sin(2.0 * np.pi * xgrid_1d * n) / len(n_lst)
            print("### For this Gaussian1D distribution, mean value is", np.mean(distri))
            distri /= np.mean(distri)

        elif "sin2dx" in pattern and self.ngridy_1d > 1:
            omega_parallel = float(pattern.split("_")[1])
            amplitude = float(pattern.split("_")[2])
            print("## will apply a sin function with freq = ", omega_parallel, "cm^-1 and relative amplitude ", amplitude, " along the x axis")
            n = int(omega_parallel / self.domegax_1d)  # ratio indicates the order of the normal mode
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            for i in range(self.ngridy_1d):
                distri[:, i] += amplitude * np.sin(2.0 * np.pi * xgrid_1d * n)

        elif "sin2d" in pattern and self.ngridy_1d > 1:
            omega_parallel = float(pattern.split("_")[1])
            amplitude = float(pattern.split("_")[2])
            print("## will apply a sin function with freq = ", omega_parallel, "cm^-1 and relative amplitude ", amplitude, " along the x & y axes")
            n = int(omega_parallel / self.domegax_1d)  # ratio indicates the order of the normal mode
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            ygrid_1d = np.linspace(1.0, self.ngridy_1d - 1.0, self.ngridy_1d) / self.ngridy_1d
            xgrid_2d, ygrid_2d = np.meshgrid(xgrid_1d, ygrid_1d)
            distri += amplitude * np.sin(2.0 * np.pi * xgrid_2d * n) * np.sin(2.0 * np.pi * ygrid_2d * n)
            print("### For this sin2d distribution, mean value is", np.mean(np.mean(distri)))
            distri /= np.mean(np.mean(distri))

        elif "tilting2d" in pattern and self.ngridy_1d > 1:
            omega_parallel = float(pattern.split("_")[1])
            amplitude = 0.1
            print("## will apply a sin function with freq = ", omega_parallel, "cm^-1 and relative amplitude ", amplitude, " along the x & y axes")
            n = int(omega_parallel / self.domegax_1d)  # ratio indicates the order of the normal mode
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            ygrid_1d = np.linspace(1.0, self.ngridy_1d - 1.0, self.ngridy_1d) / self.ngridy_1d
            xgrid_2d, ygrid_2d = np.meshgrid(xgrid_1d, ygrid_1d)
            distri = amplitude * np.sin(2.0 * np.pi * xgrid_2d * n) * np.sin(2.0 * np.pi * ygrid_2d * n)
            distri[distri > 0.0] = 1.0
            distri[distri <= 0.0] = 0.0
            distri /= np.mean(np.mean(distri))

        elif "gaussian2dx" in pattern and self.ngridy_1d > 1:
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            sigma = float(pattern.split("_")[1])
            gaussian_distri = np.exp(-0.5 * (xgrid_1d - 0.5) ** 2 / sigma ** 2) / sigma / (2.0 * np.pi) ** 0.5
            for i in range(self.ngridy_1d):
                distri[:, i] += gaussian_distri
            distri /= np.mean(distri)

        elif "gaussian2d" in pattern and self.ngridy_1d > 1:
            xgrid_1d = np.linspace(1.0, self.ngridx_1d - 1.0, self.ngridx_1d) / self.ngridx_1d
            ygrid_1d = np.linspace(1.0, self.ngridy_1d - 1.0, self.ngridy_1d) / self.ngridy_1d
            xgrid_2d, ygrid_2d = np.meshgrid(xgrid_1d, ygrid_1d)
            sigma = float(pattern.split("_")[1])
            gaussian_distri = np.exp(-0.5 * ((xgrid_2d - 0.5) ** 2 + (ygrid_2d - 0.5) ** 2) / sigma ** 2) / sigma / (2.0 * np.pi) ** 0.5
            distri = gaussian_distri
            print("### For this gaussian2d distribution, mean value is", np.mean(np.mean(distri)))
            distri /= np.mean(np.mean(distri))

        return distri**0.5

    def get_photon_weight_summed(self):
        # here we analyze the eigenstates and check the photonic component of each state
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        ph_weight_summed = np.sum(ph_weight, axis=0)
        return ph_weight_summed

    def calc_2d_spectrum_compare_analytic_sin(self, resolution_cminv=1.0, rabi=131.0,
                                          analytic_coeff=1e5, savefile="spectrum.pdf"):
        # x-axis should be the in-plane photon frequency
        omega_parallel_max = np.max(self.omega_parallel) + 30.0
        omega_parallel_min = 0.0
        # y-axis should be the polariton frequency
        omega_polariton_max = np.max(self.eigenvalues) + 30.0
        omega_polariton_min = np.min(self.eigenvalues) - 60.0
        # create a spectrum with a certain resolution
        resolution_cminv_x = self.domegax_1d
        nx = int((omega_parallel_max - omega_parallel_min) / resolution_cminv_x)
        ny = int((omega_polariton_max - omega_polariton_min) / resolution_cminv)
        spectrum = np.zeros((nx, ny))
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        current_time = time.perf_counter()
        '''
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            x = int((self.omega_parallel[i] - omega_parallel_min) / resolution_cminv)
            for j in range(self.eigenvalues.size):
                y = int((self.eigenvalues[j] - omega_polariton_min) / resolution_cminv)
                spectrum[x, y] += ph_weight_i[j]
        '''
        x_lst = (self.omega_parallel - omega_parallel_min) / resolution_cminv_x
        x_lst = x_lst.astype(int)
        y_lst = (self.eigenvalues - omega_polariton_min) / resolution_cminv
        y_lst = y_lst.astype(int)
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            for j in range(self.eigenvalues.size):
                spectrum[x_lst[i], y_lst[j]] += ph_weight_i[j]
        print("time cost for making spectrum = ", time.perf_counter() - current_time)

        # identify the Rabi splitting at resonance conditions
        idx_max = np.argmax(spectrum[1, :])
        spectrum2 = np.array(spectrum)
        spectrum2[1, idx_max] = 0.0
        idx_max2 = np.argmax(spectrum2[1, :])
        # rabi = np.abs(idx_max - idx_max2) * resolution_cminv
        # print("Rabi splitting = ", rabi, "cm^-1")
        #spectrum /= np.max(np.max(spectrum))
        # spectrum /= np.reshape(np.sum(spectrum, axis=1), (-1, 1))
        # renormalize_spectrum(spectrum, nx)
        spectrum += 1e-8  # to avoid the issue to show 0 in log scale
        # now we plot the spectrum :)
        # for imshow, rotate the spectrum for 90 degrees
        spectrum = np.rot90(spectrum)
        extent = [omega_parallel_min, omega_parallel_max, omega_polariton_min, omega_polariton_max]

        # make sure the imshow to be interpolated and in log scale
        fig, axes = clp.initialize(2, 1, width=4.3, height=4.3*0.618*1.2, fontname="Arial",
                                   fontsize=12, return_fig_args=True, sharey=True)

        ax = axes[1]

        ax.imshow(spectrum, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(1e-4, 1.),
                  extent=extent)

        # add lines for the molecular excitations and cavity photon frequencies
        omega_p = np.linspace(omega_parallel_min, omega_parallel_max, nx)
        size = np.size(omega_p)
        xm = np.linspace(omega_parallel_min, omega_parallel_max, size)
        ym = np.ones(size) * self.omega_m
        yc = (omega_p ** 2 + self.omega_perp ** 2) ** 0.5
        clp.plotone([xm, xm], [ym, yc], ax, colors=["w--", "m--"], showlegend=False, lw=1.0)
        '''
        # add lines for the analytic solution of the polariton branches
        omega_up = (ym + yc) / 2.0 + (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        omega_lp = (ym + yc) / 2.0 - (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        clp.plotone([xm, xm], [omega_up, omega_lp], ax, colors=["k--", "k--"], showlegend=False, lw=0.5, xlim=[xm[0], xm[-1]])
        # add lines for the analytic solution of the side polariton branches
        omega_side_up = (ym + yc) / 2.0 + (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        omega_side_lp = (ym + yc) / 2.0 - (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        clp.plotone([xm - 250.0, xm - 250.0], [omega_side_up, omega_side_lp], ax, colors=["w--", "w--"],
                    xlim = [xm[0], xm[-1]],
                    showlegend=False, lw=0.5)
        clp.plotone([xm + 250.0, xm + 250.0], [omega_side_up, omega_side_lp], ax, colors=["w--", "w--"],
                    xlim = [xm[0], xm[-1]],
                    showlegend=False, lw=0.5)
        '''

        ax = axes[0]

        # we now provide an analytic solution of the polariton 2D spectrum
        spectrum_analytic = np.zeros((nx, ny))
        omega_dis = 250.0  # cm^-1
        idx_omega_dis = int(omega_dis / resolution_cminv_x)

        # first, we plot the main polariton branches
        omega_up = (ym + yc) / 2.0 + (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        omega_lp = (ym + yc) / 2.0 - (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        ph_weight_up = (ym - omega_up)**2 / (rabi**2 / 4.0 + (ym - omega_up)**2 / 4.0)
        ph_weight_lp = (ym - omega_lp)**2 / (rabi**2 / 4.0 + (ym - omega_lp)**2 / 4.0)
        idx_omega_up = np.array((omega_up - omega_polariton_min) / resolution_cminv, dtype=int)
        idx_omega_lp = np.array((omega_lp - omega_polariton_min) / resolution_cminv, dtype=int)
        for i in range(nx):
            spectrum_analytic[i, idx_omega_up[i]] = ph_weight_up[i]
            spectrum_analytic[i, idx_omega_lp[i]] = ph_weight_lp[i]
            # recalculate the coeff according to the analytical equation
            coeff = 0.05**2 / 4.0 * (rabi / 2.0) ** 4.0
            #print("the analytical pre coeff is", coeff)
            # also plot the side polariton branches
            '''
            if i + idx_omega_dis < nx:
                # the right upper branch
                spectrum_analytic[i + idx_omega_dis, idx_omega_up[i]] = ph_weight_up[i] * coeff / ((omega_up[i + idx_omega_dis] - omega_up[i]) * (omega_up[i + idx_omega_dis] - omega_lp[i]))**2.0
                # the right lower branch
                spectrum_analytic[i + idx_omega_dis, idx_omega_lp[i]] = ph_weight_lp[i] * coeff / ((omega_lp[i + idx_omega_dis] - omega_up[i]) * (omega_lp[i + idx_omega_dis] - omega_lp[i]))**2.0
            if i - idx_omega_dis >= 0:
                # the left upper branch
                spectrum_analytic[i - idx_omega_dis, idx_omega_up[i]] = ph_weight_up[i] * coeff / ((omega_up[i - idx_omega_dis] - omega_up[i]) * (omega_up[i - idx_omega_dis] - omega_lp[i]))**2.0
                # the left lower branch
                spectrum_analytic[i - idx_omega_dis, idx_omega_lp[i]] = ph_weight_lp[i] * coeff / ((omega_lp[i - idx_omega_dis] - omega_up[i]) * (omega_lp[i - idx_omega_dis] - omega_lp[i]))**2.0
            '''
            if i + idx_omega_dis < nx:
                # the right upper branch
                spectrum_analytic[i + idx_omega_dis, idx_omega_up[i]] = ph_weight_up[i + idx_omega_dis] * coeff / ((omega_up[i + idx_omega_dis] - omega_up[i]) * (omega_up[i + idx_omega_dis] - omega_lp[i]))**2.0
                # the right lower branch
                spectrum_analytic[i + idx_omega_dis, idx_omega_lp[i]] = ph_weight_lp[i + idx_omega_dis] * coeff / ((omega_lp[i + idx_omega_dis] - omega_up[i]) * (omega_lp[i + idx_omega_dis] - omega_lp[i]))**2.0
            if i - idx_omega_dis >= 0:
                # the left positive frequency upper branch
                spectrum_analytic[i - idx_omega_dis, idx_omega_up[i]] = ph_weight_up[i - idx_omega_dis] * coeff / ((omega_up[i - idx_omega_dis] - omega_up[i]) * (omega_up[i - idx_omega_dis] - omega_lp[i]))**2.0
                # the left positive frequency lower branch
                spectrum_analytic[i - idx_omega_dis, idx_omega_lp[i]] = ph_weight_lp[i - idx_omega_dis] * coeff / ((omega_lp[i - idx_omega_dis] - omega_up[i]) * (omega_lp[i - idx_omega_dis] - omega_lp[i]))**2.0
            if i < idx_omega_dis:
                spectrum_analytic[idx_omega_dis - i, idx_omega_up[i]] += ph_weight_up[idx_omega_dis - i] * coeff / ((omega_up[idx_omega_dis - i] - omega_up[i]) * (omega_up[idx_omega_dis - i] - omega_lp[i])) ** 2.0
                spectrum_analytic[idx_omega_dis - i, idx_omega_lp[i]] += ph_weight_lp[idx_omega_dis - i] * coeff / ((omega_lp[idx_omega_dis - i] - omega_up[i]) * (omega_lp[idx_omega_dis - i] - omega_lp[i])) ** 2.0

        #spectrum_analytic /= np.max(np.max(spectrum_analytic))
        #spectrum_analytic /= np.reshape(np.sum(spectrum_analytic, axis=1), (-1, 1))
        #renormalize_spectrum(spectrum_analytic, nx)
        spectrum_analytic += 1e-8  # to avoid the issue to show 0 in log scale

        # for imshow, rotate the spectrum for 90 degrees
        spectrum_analytic = np.rot90(spectrum_analytic)

        im1 = ax.imshow(spectrum_analytic, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(1e-4, 1.),
                  extent=extent)

        clp.plotone([xm, xm], [ym, yc], ax, colors=["w--", "m--"], showlegend=False, lw=1.0)

        # add text on top of the two figures
        axes[0].text(0.02, 0.86, "(a) analytical", transform=axes[0].transAxes, color="w", fontsize=12)
        axes[1].text(0.02, 0.86, "(b) numerical", transform=axes[1].transAxes, color="w", fontsize=12)


        # add colorbar
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.19, 0.02, 0.75])
        fig.colorbar(im1, cax=cbar_ax)
        # show ticks explicitly
        for i in range(2):
            axes[i].tick_params(color='c', labelsize='medium', width=1)
        # add labels
        axes[1].set_xlabel(r"$\omega_{\parallel}$ (cm$^{-1}$)", fontsize=12)
        axes[0].set_ylabel(r"$\omega_{\rm pol}$ (cm$^{-1}$)", fontsize=12)
        axes[1].set_ylabel(r"$\omega_{\rm pol}$ (cm$^{-1}$)", fontsize=12)

        clp.adjust(tight_layout=True, savefile=savefile)

    def calc_2d_spectrum_compare_analytic_general(self, resolution_cminv=1.0, rabi=131.0,
                                                   analytic_coeff=1e5, savefile="spectrum_general.pdf"):
        # x-axis should be the in-plane photon frequency
        omega_parallel_max = np.max(self.omega_parallel) + 30.0
        omega_parallel_min = 0.0
        # y-axis should be the polariton frequency
        omega_polariton_max = np.max(self.eigenvalues) + 30.0
        omega_polariton_min = np.min(self.eigenvalues) - 60.0
        # create a spectrum with a certain resolution
        resolution_cminv_x = self.domegax_1d
        nx = int((omega_parallel_max - omega_parallel_min) / resolution_cminv_x)
        ny = int((omega_polariton_max - omega_polariton_min) / resolution_cminv)
        spectrum = np.zeros((nx, ny))
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        current_time = time.perf_counter()
        '''
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            x = int((self.omega_parallel[i] - omega_parallel_min) / resolution_cminv)
            for j in range(self.eigenvalues.size):
                y = int((self.eigenvalues[j] - omega_polariton_min) / resolution_cminv)
                spectrum[x, y] += ph_weight_i[j]
        '''
        x_lst = (self.omega_parallel - omega_parallel_min) / resolution_cminv_x
        x_lst = x_lst.astype(int)
        y_lst = (self.eigenvalues - omega_polariton_min) / resolution_cminv
        y_lst = y_lst.astype(int)
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            for j in range(self.eigenvalues.size):
                spectrum[x_lst[i], y_lst[j]] += ph_weight_i[j]
        print("time cost for making spectrum = ", time.perf_counter() - current_time)

        # identify the Rabi splitting at resonance conditions
        idx_max = np.argmax(spectrum[1, :])
        spectrum2 = np.array(spectrum)
        spectrum2[1, idx_max] = 0.0
        idx_max2 = np.argmax(spectrum2[1, :])
        # rabi = np.abs(idx_max - idx_max2) * resolution_cminv
        # print("Rabi splitting = ", rabi, "cm^-1")
        #spectrum /= np.reshape(np.sum(spectrum, axis=1), (-1,1))
        #spectrum /= np.max(np.max(spectrum))
        # renormalize_spectrum(spectrum, nx)
        spectrum += 1e-8  # to avoid the issue to show 0 in log scale
        # now we plot the spectrum :)
        # for imshow, rotate the spectrum for 90 degrees
        spectrum = np.rot90(spectrum)
        extent = [omega_parallel_min, omega_parallel_max, omega_polariton_min, omega_polariton_max]

        # plot the spatial density distribution and fit it with multiple sin and cos functions
        fig, axes = clp.initialize(3, 1, width=4.3, height=4.3, fontname="Arial",
                                     fontsize=12, return_fig_args=True)

        xs, ys = [], []
        # first, we plot the coupling constant distribution in the 1D plane
        pattern = np.reshape(self.g0_distri, (self.ngridx_1d,self. ngridy_1d))
        x, y = self.xgrid_1d, pattern[:, 0]**2  # g0 = sqrt(rho(x))
        xs.append(x)
        ys.append(y - 1.0)

        clp.plotone(xs, ys, axes[0], showlegend=False)

        # calculate its fourier transform
        sp = np.fft.fft(y-1.0)
        freq = np.fft.fftfreq(np.size(x), x[1]-x[0])
        clp.plotone([freq], [np.abs(sp)], axes[1], showlegend=False)

        # recover the original lineshape from the spectrum
        # y_recover = np.fft.ifft(sp, len(sp))
        y_recover = np.zeros(len(sp), dtype=np.complex128)
        for idx, freq_i in enumerate(freq[:int(len(freq)//2)]):
            delta = np.abs(sp[idx] / (len(sp))) * 2.0
            print("The associated delta for freq = ", freq_i, " is ", delta)
            y_recover += delta * np.sin(2.0 * np.pi * freq_i * x)
        clp.plotone([x], [y_recover], axes[2], showlegend=True)

        clp.adjust(tight_layout=True)

        # make sure the imshow to be interpolated and in log scale
        fig, axes = clp.initialize(2, 1, width=4.3, height=4.3*0.618*1.2, fontname="Arial",
                                   fontsize=12, return_fig_args=True, sharey=True)

        ax = axes[1]

        vmin, vmax = 1e-6, 1.0

        ax.imshow(spectrum, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(vmin, vmax),
                  extent=extent)

        # add lines for the molecular excitations and cavity photon frequencies
        omega_p = np.linspace(omega_parallel_min, omega_parallel_max, nx)
        size = np.size(omega_p)
        xm = np.linspace(omega_parallel_min, omega_parallel_max, size)
        ym = np.ones(size) * self.omega_m
        yc = (omega_p ** 2 + self.omega_perp ** 2) ** 0.5
        clp.plotone([xm, xm], [ym, yc], ax, colors=["w--", "m--"], showlegend=False, lw=1.0)

        ax = axes[0]

        # we now provide an analytic solution of the polariton 2D spectrum
        spectrum_analytic = np.zeros((nx, ny))

        # first, we plot the main polariton branches
        omega_up = (ym + yc) / 2.0 + (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        omega_lp = (ym + yc) / 2.0 - (rabi ** 2 / 4.0 + (yc - ym) ** 2 / 4.0) ** 0.5
        ph_weight_up = (ym - omega_up)**2 / (rabi**2 / 4.0 + (ym - omega_up)**2 / 4.0)
        ph_weight_lp = (ym - omega_lp)**2 / (rabi**2 / 4.0 + (ym - omega_lp)**2 / 4.0)
        idx_omega_up = np.array((omega_up - omega_polariton_min) / resolution_cminv, dtype=int)
        idx_omega_lp = np.array((omega_lp - omega_polariton_min) / resolution_cminv, dtype=int)

        freq = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43]

        for i in range(nx):
            for idx, freq_i in enumerate(freq):
            #for idx, freq_i in enumerate(freq[1:int(len(freq) // 2)]):
                if True:
                    #delta = np.abs(sp[idx] / (len(sp))) * 2.0
                    delta = 0.1 / 13.0
                    omega_dis = freq_i * resolution_cminv_x
                    idx_omega_dis = int(omega_dis / resolution_cminv_x)
                    coeff = delta**2 / 4.0 * (rabi / 2.0) ** 4.0
                    # also plot the side polariton branches according the Fourier power spectrum of the real-space density distribution
                    if i + idx_omega_dis < nx:
                        # the right upper branch
                        spectrum_analytic[i + idx_omega_dis, idx_omega_up[i]] += ph_weight_up[i + idx_omega_dis] * coeff / ((omega_up[i + idx_omega_dis] - omega_up[i]) * (omega_up[i + idx_omega_dis] - omega_lp[i]))**2.0
                        # the right lower branch
                        spectrum_analytic[i + idx_omega_dis, idx_omega_lp[i]] += ph_weight_lp[i + idx_omega_dis] * coeff / ((omega_lp[i + idx_omega_dis] - omega_up[i]) * (omega_lp[i + idx_omega_dis] - omega_lp[i]))**2.0
                    if i - idx_omega_dis >= 0:
                        # the left upper branch
                        spectrum_analytic[i - idx_omega_dis, idx_omega_up[i]] += ph_weight_up[i - idx_omega_dis] * coeff / ((omega_up[i - idx_omega_dis] - omega_up[i]) * (omega_up[i - idx_omega_dis] - omega_lp[i]))**2.0
                        # the left lower branch
                        spectrum_analytic[i - idx_omega_dis, idx_omega_lp[i]] += ph_weight_lp[i - idx_omega_dis] * coeff / ((omega_lp[i - idx_omega_dis] - omega_up[i]) * (omega_lp[i - idx_omega_dis] - omega_lp[i]))**2.0
                    if idx_omega_dis - i > 0 and idx_omega_dis - i < nx:
                        print("debugging the general part", idx_omega_dis, i) if i == 0 else None
                        spectrum_analytic[idx_omega_dis - i, idx_omega_up[i]] += ph_weight_up[idx_omega_dis - i] * coeff / ((omega_up[idx_omega_dis - i] - omega_up[i]) * (omega_up[idx_omega_dis - i] - omega_lp[i])) ** 2.0
                        spectrum_analytic[idx_omega_dis - i, idx_omega_lp[i]] += ph_weight_lp[idx_omega_dis - i] * coeff / ((omega_lp[idx_omega_dis - i] - omega_up[i]) * (omega_lp[idx_omega_dis - i] - omega_lp[i])) ** 2.0
            # add the diagonal terms: the unperturbed polariton branches
            spectrum_analytic[i, idx_omega_up[i]] = ph_weight_up[i]
            spectrum_analytic[i, idx_omega_lp[i]] = ph_weight_lp[i]
            # calculate the coeff according to the analytical equation

        #renormalize_spectrum(spectrum_analytic, nx)
        spectrum_analytic += 1e-8  # to avoid the issue to show 0 in log scale
        #spectrum_analytic /= np.max(np.max(spectrum_analytic))
        #spectrum_analytic /= np.reshape(np.sum(spectrum_analytic, axis=1), (-1,1))

        # for imshow, rotate the spectrum for 90 degrees
        spectrum_analytic = np.rot90(spectrum_analytic)

        im1 = ax.imshow(spectrum_analytic, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(vmin, vmax),
                  extent=extent)

        clp.plotone([xm, xm], [ym, yc], ax, colors=["w--", "m--"], showlegend=False, lw=1.0)

        # add text on top of the two figures
        axes[0].text(0.02, 0.86, "(a) analytical", transform=axes[0].transAxes, color="w", fontsize=12)
        axes[1].text(0.02, 0.86, "(b) numerical", transform=axes[1].transAxes, color="w", fontsize=12)


        # add colorbar
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.19, 0.02, 0.75])
        fig.colorbar(im1, cax=cbar_ax)
        # show ticks explicitly
        for i in range(2):
            axes[i].tick_params(color='c', labelsize='medium', width=1)
        # add labels
        axes[1].set_xlabel(r"$\omega_{\parallel}$ (cm$^{-1}$)", fontsize=12)
        axes[0].set_ylabel(r"$\omega_{\rm pol}$ (cm$^{-1}$)", fontsize=12)
        axes[1].set_ylabel(r"$\omega_{\rm pol}$ (cm$^{-1}$)", fontsize=12)

        clp.adjust(tight_layout=True, savefile=savefile)

    def calc_2d_spectrum(self, resolution_cminv=1.0, omega_polariton_min=None, omega_polariton_max=None):
        # x-axis should be the in-plane photon frequency
        omega_parallel_max = np.max(self.omega_parallel) + 30.0
        omega_parallel_min = 0.0
        # y-axis should be the polariton frequency
        if omega_polariton_max is None:
            omega_polariton_max = np.max(self.eigenvalues) + 30.0
        if omega_polariton_min is None:
            omega_polariton_min = np.min(self.eigenvalues) - 60.0
        # create a spectrum with a certain resolution
        resolution_cminv_x = self.domegax_1d
        nx = int((omega_parallel_max - omega_parallel_min) / resolution_cminv_x)
        ny = int((omega_polariton_max - omega_polariton_min) / resolution_cminv)
        spectrum = np.zeros((nx, ny))
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        current_time = time.perf_counter()
        '''
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            x = int((self.omega_parallel[i] - omega_parallel_min) / resolution_cminv)
            for j in range(self.eigenvalues.size):
                y = int((self.eigenvalues[j] - omega_polariton_min) / resolution_cminv)
                spectrum[x, y] += ph_weight_i[j]
        '''
        x_lst = (self.omega_parallel - omega_parallel_min) / resolution_cminv_x
        x_lst = x_lst.astype(int)
        y_lst = (self.eigenvalues - omega_polariton_min) / resolution_cminv
        y_lst = y_lst.astype(int)
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            for j in range(self.eigenvalues.size):
                spectrum[x_lst[i], y_lst[j]] += ph_weight_i[j]
        print("time cost for making spectrum = ", time.perf_counter() - current_time)

        # identify the Rabi splitting at resonance conditions
        idx_max = np.argmax(spectrum[1, :])
        spectrum2 = np.array(spectrum)
        spectrum2[1, idx_max] = 0.0
        idx_max2 = np.argmax(spectrum2[1, :])
        rabi = np.abs(idx_max - idx_max2) * resolution_cminv
        print("Rabi splitting = ", rabi, "cm^-1")
        # spectrum /= np.max(np.max(spectrum))
        # renormalize_spectrum(spectrum, nx)
        spectrum += 1e-8  # to avoid the issue to show 0 in log scale
        # now we plot the spectrum :)
        # for imshow, rotate the spectrum for 90 degrees
        spectrum = np.rot90(spectrum)
        extent = [omega_parallel_min, omega_parallel_max, omega_polariton_min, omega_polariton_max]
        return spectrum, extent, rabi

    def calc_2d_spectrum_compare_others(self, resolution_cminv=1.0,
                                        analytic_coeff=1e5, savefile="spectrum.pdf",
                                        polariton_lst=[], labels=[]):
        # x-axis should be the in-plane photon frequency
        omega_parallel_max = np.max(self.omega_parallel) + 30.0
        omega_parallel_min = 0.0
        # y-axis should be the polariton frequency
        omega_polariton_max = np.max(self.eigenvalues) + 30.0
        omega_polariton_min = np.min(self.eigenvalues) - 60.0
        # create a spectrum with a certain resolution
        resolution_cminv_x = self.domegax_1d
        nx = int((omega_parallel_max - omega_parallel_min) / resolution_cminv_x)
        ny = int((omega_polariton_max - omega_polariton_min) / resolution_cminv)
        spectrum = np.zeros((nx, ny))
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        current_time = time.perf_counter()
        '''
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            x = int((self.omega_parallel[i] - omega_parallel_min) / resolution_cminv)
            for j in range(self.eigenvalues.size):
                y = int((self.eigenvalues[j] - omega_polariton_min) / resolution_cminv)
                spectrum[x, y] += ph_weight_i[j]
        '''
        x_lst = (self.omega_parallel - omega_parallel_min) / resolution_cminv_x
        x_lst = x_lst.astype(int)
        y_lst = (self.eigenvalues - omega_polariton_min) / resolution_cminv
        y_lst = y_lst.astype(int)
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            for j in range(self.eigenvalues.size):
                spectrum[x_lst[i], y_lst[j]] += ph_weight_i[j]
        print("time cost for making spectrum = ", time.perf_counter() - current_time)

        # identify the Rabi splitting at resonance conditions
        idx_max = np.argmax(spectrum[1, :])
        spectrum2 = np.array(spectrum)
        spectrum2[1, idx_max] = 0.0
        idx_max2 = np.argmax(spectrum2[1, :])
        rabi = np.abs(idx_max - idx_max2) * resolution_cminv
        print("Rabi splitting = ", rabi, "cm^-1")
        #spectrum /= np.max(np.max(spectrum))

        # renormalize_spectrum(spectrum, nx)
        spectrum += 1e-8  # to avoid the issue to show 0 in log scale
        # now we plot the spectrum :)
        # for imshow, rotate the spectrum for 90 degrees
        spectrum = np.rot90(spectrum)
        extent = [omega_parallel_min, omega_parallel_max, omega_polariton_min, omega_polariton_max]

        spectrum /= np.max(np.max(spectrum))
        # make sure the imshow to be interpolated and in log scale
        n_subfig = 1 + len(polariton_lst)
        fig, axes = clp.initialize(n_subfig, 1, width=4.3, height=4.3*0.618*1.2*n_subfig/2.0, fontname="Arial",
                                   fontsize=12, return_fig_args=True, sharey=True)

        ax = axes[0]

        im1 = ax.imshow(spectrum, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(1e-3, 1.),
                  extent=extent)

        for idx, polariton in enumerate(polariton_lst):
            ax = axes[idx + 1]
            spectrum_local, extent_local, rabi_local = polariton.calc_2d_spectrum(resolution_cminv=resolution_cminv,
                                                                                  omega_polariton_min=omega_polariton_min,
                                                                                  omega_polariton_max=omega_polariton_max)
            spectrum_local /= np.max(np.max(spectrum_local))

            ax.imshow(spectrum_local, interpolation="quadric",
                      cmap=cm.hot,
                      norm=matplotlib.colors.LogNorm(1e-3, 1.),
                      extent=extent_local)

        # add text on top of the figures
        labels_pre = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        for i, label in enumerate(labels):
            label_tot = labels_pre[i] + " " + label
            axes[i].text(0.02, 0.86, label_tot, transform=axes[i].transAxes, color="w", fontsize=12)

            # add lines for the molecular excitations and cavity photon frequencies
            omega_p = np.linspace(omega_parallel_min, omega_parallel_max, nx)
            size = np.size(omega_p)
            xm = np.linspace(omega_parallel_min, omega_parallel_max, size)
            ym = np.ones(size) * self.omega_m
            yc = (omega_p ** 2 + self.omega_perp ** 2) ** 0.5
            for ax in axes:
                clp.plotone([xm, xm], [ym, yc], ax, colors=["w--", "m--"], showlegend=False, lw=1.0)

        # add colorbar
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.16, 0.02, 0.75])
        fig.colorbar(im1, cax=cbar_ax)
        # show ticks explicitly
        for i in range(n_subfig):
            axes[i].tick_params(color='c', labelsize='medium', width=1,
                                direction="in", bottom=True, top=True, left=True, right=True)
            axes[i].set_ylabel(r"$\omega_{\rm pol}$ (cm$^{-1}$)", fontsize=12)
        # add labels
        axes[-1].set_xlabel(r"$\omega_{\parallel}$ (cm$^{-1}$)", fontsize=12)

        clp.adjust(tight_layout=True, savefile=savefile)

    def plot_eigenstates(self):
        plt.subplot(121)
        for idx, omega in enumerate(self.eigenvalues):
            w = self.ph_weight_summed[idx]
            x1 = np.linspace(0, w, 100)
            x2 = np.linspace(w, 1, 100)
            y1 = np.ones(100) * omega
            y2 = np.ones(100) * omega
            plt.plot(x1, y1, "r-")
            plt.plot(x2, y2, "k-")
        # also plot the density distribution of all the eigenstates
        plt.subplot(122)
        hist, bins = np.histogram(self.eigenvalues, bins=200, density=True)
        widths = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2
        # Create horizontal bar plot
        plt.barh(center, hist, height=widths)
        plt.show()

    def get_ph_weights_at_in_plane_angle(self, n=0):
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        weight_distr = []
        for idx, omega in enumerate(self.eigenvalues):
            # get the photonic weight at k_parallel for each eigenstate
            w = ph_weight[n, idx]
            weight_distr.append(w)
        weight_distr = np.array(weight_distr)
        weight_distr = np.sort(weight_distr)
        return weight_distr

    def get_spectrum(self, resolution_cminv=5.0):

        # first, we try to plot the coupling constant distribution in the 2D plane
        pattern = np.reshape(self.g0_distri, (self.ngridx_1d, self.ngridy_1d))
        # pattern = np.rot90(pattern)
        plt.imshow(pattern, cmap=cm.gray, extent=[0, 1, 0, 1])
        plt.show()

        # second, we plot the polariton spectrum as a function of the in-plane photon frequency
        ax = clp.initialize(1, 1, width=4.3, height=4.3, fontname="Arial")

        # x-axis should be the in-plane photon frequency
        omega_parallel_max = np.max(self.omega_parallel) + 30.0
        omega_parallel_min = 0.0

        # y-axis should be the polariton frequency
        omega_polariton_max = np.max(self.eigenvalues) + 30.0
        omega_polariton_min = np.min(self.eigenvalues) - 30.0

        # create a spectrum with a certain resolution
        resolution_cminv_x = self.domegax_1d
        nx = int((omega_parallel_max - omega_parallel_min) / resolution_cminv_x)
        ny = int((omega_polariton_max - omega_polariton_min) / resolution_cminv)
        spectrum = np.zeros((nx, ny))
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        current_time = time.perf_counter()
        '''
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            x = int((self.omega_parallel[i] - omega_parallel_min) / resolution_cminv)
            for j in range(self.eigenvalues.size):
                y = int((self.eigenvalues[j] - omega_polariton_min) / resolution_cminv)
                spectrum[x, y] += ph_weight_i[j]
        '''
        x_lst = (self.omega_parallel - omega_parallel_min) / resolution_cminv_x
        x_lst = x_lst.astype(int)
        y_lst = (self.eigenvalues - omega_polariton_min) / resolution_cminv
        y_lst = y_lst.astype(int)
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            for j in range(self.eigenvalues.size):
                spectrum[x_lst[i], y_lst[j]] += ph_weight_i[j]

        print("time cost for making spectrum = ", time.perf_counter() - current_time)
        # renormalize the spectrum with the maximal value to unity
        # spectrum /= np.max(spectrum)
        # renormalize_spectrum(spectrum, nx)
        spectrum += 1e-8
        # now we plot the spectrum :)
        # for imshow, rotate the spectrum for 90 degrees
        spectrum = np.rot90(spectrum)
        # make sure the imshow to be interpolated and in log scale
        ax.imshow(spectrum, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(1e-3, 1.),
                  extent=[omega_parallel_min, omega_parallel_max, omega_polariton_min, omega_polariton_max])

        # add lines for the molecular excitations and cavity photon frequencies
        omega_p = np.linspace(omega_parallel_min, omega_parallel_max, 100)
        size = np.size(omega_p)
        xm = np.linspace(omega_parallel_min, omega_parallel_max, size)
        ym = np.ones(size) * self.omega_m
        yc = (omega_p ** 2 + self.omega_perp ** 2) ** 0.5
        clp.plotone([xm, xm], [ym, yc], ax, colors=["w--", "m--"], showlegend=False)

        clp.adjust()

    def calc_3d_spectrum(self, resolution_cminv=5.0, savefile="3d_spectrum.pdf"):
        '''
        This function is used to calculate the 3D spectrum of the polariton system
        z-axis is the photonic weight of each eigenstate
        x-axis is the in-plane photon frequency along the x direction of the cavity mirror
        y-axis is the in-plane photon frequency along the y direction of the cavity mirror
        '''

        # x-axis or y-axis should be the in-plane photon frequency
        omega_parallel_x = np.reshape(self.kx_grid_2d / 2.0 / np.pi * self.domegax_1d, -1)
        omega_parallel_y = np.reshape(self.ky_grid_2d / 2.0 / np.pi * self.domegay_1d, -1)
        omega_parallel_x_max = np.max(omega_parallel_x) + 30.0
        omega_parallel_x_min = -omega_parallel_x_max
        omega_parallel_y_max = np.max(omega_parallel_y) + 30.0
        omega_parallel_y_min = -omega_parallel_y_max

        # z-axis should be the polariton frequency
        omega_polariton_max = np.max(self.eigenvalues) + 30.0
        omega_polariton_min = np.min(self.eigenvalues) - 30.0

        # create a spectrum with a certain resolution
        resolution_cminv_x = self.domegax_1d
        nx = int((omega_parallel_x_max - omega_parallel_x_min) / resolution_cminv_x) 
        ny = int((omega_parallel_y_max - omega_parallel_y_min) / resolution_cminv_x) 
        nz = int((omega_polariton_max - omega_polariton_min) / resolution_cminv)
        spectrum = np.zeros((nx, ny, nz))
        weight = np.abs(self.eigenvectors) ** 2
        ph_weight = weight[self.ngrid:, :]
        current_time = time.perf_counter()

        x_lst = (omega_parallel_x - omega_parallel_x_min) / resolution_cminv_x
        x_lst = x_lst.astype(int)
        y_lst = (omega_parallel_y - omega_parallel_y_min) / resolution_cminv_x
        y_lst = y_lst.astype(int)
        z_lst = (self.eigenvalues - omega_polariton_min) / resolution_cminv
        z_lst = z_lst.astype(int)
        print("shape of xlist ylist zlist", np.shape(x_lst), np.shape(y_lst), np.shape(z_lst))
        print("nx, ny, nz = ", nx, ny, nz)
        for i in range(self.nmode):
            # get the photon weights of different eigenstates for each photon mode
            ph_weight_i = ph_weight[i, :]
            for j in range(self.eigenvalues.size):
                # get the x and y axis index from the i-th photon mode
                spectrum[x_lst[i], y_lst[i], z_lst[j]] += ph_weight_i[j]

        # renormalize_spectrum(spectrum, nx)
        spectrum += 1e-8  # to avoid the issue to show 0 in log scale
        # spectrum /= np.max(np.max(np.max(spectrum)))

        print("time cost for making spectrum = ", time.perf_counter() - current_time)

        extent = [omega_parallel_x_min, omega_parallel_x_max, omega_polariton_min, omega_polariton_max]

        # make sure the imshow to be interpolated and in log scale
        fig, axes = clp.initialize(2, 1, width=4.3, height=4.3 * 0.618*1.2, fontname="Arial",
                                   fontsize=12, return_fig_args=True, sharey=True)

        ax = axes[0]

        spectrum_slice_x = spectrum[:, int(nx//2)-1, :]
        spectrum_slice_y = spectrum[:, int(nx//4), :]
        spectrum_slice_x = np.rot90(spectrum_slice_x)
        spectrum_slice_y = np.rot90(spectrum_slice_y)
        spectrum_slice_x /= np.max(np.max(spectrum_slice_x))
        spectrum_slice_y /= np.max(np.max(spectrum_slice_y))

        im1 = ax.imshow(spectrum_slice_x, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(1e-3, 1.),
                  extent=extent)

        ax = axes[1]

        ax.imshow(spectrum_slice_y, interpolation="quadric",
                  cmap=cm.hot,
                  norm=matplotlib.colors.LogNorm(1e-3, 1.),
                  extent=extent)

        # add text on top of the figures
        labels = ["(a)", "(b)"]
        for i, label in enumerate(labels):
            axes[i].text(0.02, 0.8, label, transform=axes[i].transAxes, color="w", fontsize=12)

            # add lines for the molecular excitations and cavity photon frequencies
            omega_p = np.linspace(omega_parallel_x_min, omega_parallel_x_max, nx)
            size = np.size(omega_p)
            xm = np.linspace(omega_parallel_x_min, omega_parallel_x_max, size)
            ym = np.ones(size) * self.omega_m
            yc = (omega_p ** 2 + self.omega_perp ** 2) ** 0.5
            if i == 2:
                yc = (omega_p ** 2 * 2 + self.omega_perp ** 2) ** 0.5
            clp.plotone([xm, xm], [ym, yc], axes[i], colors=["w--", "m--"], showlegend=False, lw=1.0)

        # add colorbar
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.98, 0.25, 0.02, 0.65])
        fig.colorbar(im1, cax=cbar_ax)
        # show ticks explicitly
        for i in range(2):
            axes[i].tick_params(color='c', labelsize='medium', width=1,
                                direction="in", bottom=True, top=True, left=True, right=True)
            axes[i].set_ylabel(r"$\omega_{\rm pol}$ (cm$^{-1}$)", fontsize=12)
        # add labels
        axes[1].set_xlabel(r"$\omega_{x}$ (cm$^{-1}$)", fontsize=12)
        axes[0].text(0.14, 0.8, r"$\omega_{y}$ = 30 cm$^{-1}$", fontsize=12, transform=axes[0].transAxes, color="w")
        axes[1].text(0.14, 0.8, r"$\omega_{y}$ = 450 cm$^{-1}$", fontsize=12, transform=axes[1].transAxes, color="w")

        clp.adjust(tight_layout=True, savefile=savefile)

    def plot_molecular_distribution_from_polaritons(self, polaritons=[], savefile="molecular_distribution_2d.pdf"):
        n_figure = len(polaritons)

        if self.ngridy_1d > 1:
            # This is a 2D plot
            fig, axes = clp.initialize(1, n_figure, width=2. * n_figure, height=2., fontname="Arial",
                                       fontsize=12, return_fig_args=True)
            for idx, polariton in enumerate(polaritons):
                # first, we plot the coupling constant distribution in the 2D plane
                pattern = np.reshape(polariton.g0_distri, (polariton.ngridx_1d, polariton.ngridy_1d))**2
                print("maximal is", np.max(pattern))
                clp.plotone([], [], axes[idx], showlegend=False)
                # pattern = np.rot90(pattern)
                im1 = axes[idx].imshow(pattern, cmap=cm.binary, extent=[0, 1, 0, 1],
                                       norm=matplotlib.colors.Normalize(0.0, 2.))
                axes[idx].set_xlabel(r"$x$ [$L_x = 0.333$ mm]", fontsize=12)
            axes[0].set_ylabel(r"$y$ [$L_y = 0.333$ mm]", fontsize=12)
            # add a colorbar
            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes([0.99, 0.19, 0.02, 0.75])
            fig.colorbar(im1, cax=cbar_ax)

            clp.adjust(tight_layout=True, savefile=savefile)

        elif self.ngridy_1d == 1:
            # This is a 1D plot
            fig, ax = clp.initialize(1, 1, width=4.3, height=4.3, fontname="Arial",
                                    fontsize=12, return_fig_args=True)

            xs, ys = [], []
            for idx, polariton in enumerate(polaritons):
                # first, we plot the coupling constant distribution in the 1D plane
                pattern = np.reshape(polariton.g0_distri, (polariton.ngridx_1d, polariton.ngridy_1d))**2
                print("maximal is", np.max(pattern))
                x, y = polariton.xgrid_1d, pattern[:, 0]
                xs.append(x)
                ys.append(y)
            clp.plotone(xs, ys, ax, showlegend=False)

            clp.adjust(tight_layout=True, savefile=savefile)


def plot_1d_eigenstate_distr(ngrid_1d=1080, nmode_1d=201):
    # reset the default number of grid points per dimension
    cavity_1d_sin["ngridx_1d"] = ngrid_1d
    # also reset g0 so that the Rabi splitting is the same for all the cases
    cavity_1d_sin["g0"] = 2.0 * (1080 / ngrid_1d) ** 0.5
    # reset the default number of photon modes per dimension
    cavity_1d_sin["nmodex_1d"] = nmode_1d
    # also reset the spacing between each photon mode
    cavity_1d_sin["domegax_1d"] = 2000.0 / nmode_1d

    # reset the default number of grid points per dimension
    cavity_1d_gaussian["ngridx_1d"] = ngrid_1d
    # also reset g0 so that the Rabi splitting is the same for all the cases
    cavity_1d_gaussian["g0"] = 2.0 * (1080 / ngrid_1d) ** 0.5
    # reset the default number of photon modes per dimension
    cavity_1d_gaussian["nmodex_1d"] = nmode_1d
    # also reset the spacing between each photon mode
    cavity_1d_gaussian["domegax_1d"] = 2000.0 / nmode_1d

    # Task debug: plot the photon weight distribution among eigenstates for 1D systems corresponding to kp = 0 cavity mode for a few different systems
    axes = clp.initialize(1, 1, width=4.3, height=4.3 * 0.618, fontsize=12,
                          labelsize=14,)
                          #labelthem=True, labelthemPosition=[-0.05, 1.05])
    # plot the first part
    ax = axes#[0]

    patterns = ["sin1d_250_0.0", "gaussian1d_0.45", "gaussian1d_0.35", "gaussian1d_0.25"]
    labels_gaussian = [r"homogeneous", r"Gaussian $\sigma$ = 0.45 mm", r"Gaussian $\sigma$ = 0.35 mm",
                       r"Gaussian $\sigma$ = 0.25 mm"]
    polaritons_gaussian = []
    ws = []
    for idx, pattern in enumerate(patterns):
        cavity_1d_gaussian["pattern"] = pattern
        polariton = PolaritonSpectrumCalc(cavity_params=cavity_1d_gaussian)
        polaritons_gaussian.append(polariton)
        w = polariton.get_ph_weights_at_in_plane_angle(n=nmode_1d//2) + polariton.get_ph_weights_at_in_plane_angle(n=nmode_1d//2+1)
        ws.append(w)
    x = np.arange(np.size(w))
    xs = [x / (np.max(x) + 1) * 100.0] * len(ws)
    clp.plotone(xs, ws, ax, colors=["k-o"] * len(xs),
                labels=labels_gaussian, xlim=[88, 100], ylim=[-0.004, 0.08],
                xlabel="No. eigenstate [%]", ylabel="weight of mode at $k_{\parallel}=0$")
    # reset color to obey a colormap
    colormap = plt.cm.hot
    colors = [colormap(i) for i in np.linspace(0, 0.6, len(xs))]
    for i, j in enumerate(ax.lines):
        j.set_color(colors[i])
    ax.legend(prop={'size': 10}, edgecolor="k")

    clp.adjust(tight_layout=True, savefile="photon_weight_distr.pdf")
    return 0

def plot_1d_modulation(ngrid_1d=1080, nmode_1d=100):
    '''
    This function is used to plot the polariton spectrum for 1D waveguide + sin modulation
    '''
    # reset the default number of grid points per dimension
    cavity_1d_sin["ngridx_1d"] = ngrid_1d
    # also reset g0 so that the Rabi splitting is the same for all the cases
    cavity_1d_sin["g0"] = 2.0 * (1080 / ngrid_1d)**0.5
    # reset the default number of photon modes per dimension
    cavity_1d_sin["nmodex_1d"] = nmode_1d
    # also reset the spacing between each photon mode
    cavity_1d_sin["domegax_1d"] = 2000.0 / nmode_1d

    # reset the default number of grid points per dimension
    cavity_1d_gaussian["ngridx_1d"] = ngrid_1d
    # also reset g0 so that the Rabi splitting is the same for all the cases
    cavity_1d_gaussian["g0"] = 2.0 * (1080 / ngrid_1d) ** 0.5
    # reset the default number of photon modes per dimension
    cavity_1d_gaussian["nmodex_1d"] = nmode_1d
    # also reset the spacing between each photon mode
    cavity_1d_gaussian["domegax_1d"] = 2000.0 / nmode_1d


    # Task zero: plot the polariton spectrum for 1D waveguide + sin modulation and compare it with the analytic solution
    polariton = PolaritonSpectrumCalc(cavity_params=cavity_1d_sin)
    polariton.calc_2d_spectrum_compare_analytic_sin(resolution_cminv=1.0, analytic_coeff=2.89e5,
                                                savefile="1d_analytic.pdf", rabi=131.0)


    # Task one: plot the polariton spectrum for 1D waveguide + general modulation and compare it with the analytic solution
    polariton = PolaritonSpectrumCalc(cavity_params=cavity_1d_gaussian_perturb)
    polariton.calc_2d_spectrum_compare_analytic_general(resolution_cminv=1.0, analytic_coeff=2.89e5,
                                                         savefile="1d_analytic_general.pdf", rabi=131.0)

    # Task two: plot the polariton spectrum for 1D waveguide + sin modulation at different modulation amplitudes
    patterns = ["sin1d_250_0.0", "sin1d_250_0.2", "sin1d_250_0.4", "sin1d_250_0.6"]
    labels_amp = [r"$\delta$ = 0.0", r"$\delta$ = 0.2", r"$\delta$ = 0.4", r"$\delta$ = 0.6"]
    polaritons_amp = []
    for pattern in patterns:
        cavity_1d_sin["pattern"] = pattern
        polariton = PolaritonSpectrumCalc(cavity_params=cavity_1d_sin)
        polaritons_amp.append(polariton)
    polaritons_amp[0].calc_2d_spectrum_compare_others(resolution_cminv=1.0,
                                                  savefile="1d_compare_amp.pdf",
                                                  polariton_lst=polaritons_amp[1:],
                                                  labels=labels_amp)


    # Task three: plot the polariton spectrum for 1D waveguide + sin modulation at different modulation frequencies
    patterns = ["sin1d_100_0.3", "sin1d_200_0.3", "sin1d_300_0.3"]
    labels_freq = [r"$k_x$ = 100 cm$^{-1}$", r"$k_x$ = 200 cm$^{-1}$", r"$k_x$ = 300 cm$^{-1}$"]
    polaritons_freq = []
    for pattern in patterns:
        cavity_1d_sin["pattern"] = pattern
        polariton = PolaritonSpectrumCalc(cavity_params=cavity_1d_sin)
        polaritons_freq.append(polariton)
    polaritons_freq[0].calc_2d_spectrum_compare_others(resolution_cminv=1.0,
                                                  savefile="1d_compare_freq.pdf",
                                                  polariton_lst=polaritons_freq[1:],
                                                  labels=labels_freq)

    # Task four: plot the polariton spectrum for 1D waveguide + gaussian modulation at different modulation phases
    patterns = ["gaussian1d_0.45", "gaussian1d_0.35", "gaussian1d_0.25"]
    labels_gaussian = [r"$\sigma$ = 0.45 mm", r"$\sigma$ = 0.35 mm", r"$\sigma$ = 0.25 mm"]
    polaritons_gaussian = []
    for pattern in patterns:
        cavity_1d_gaussian["pattern"] = pattern
        polariton = PolaritonSpectrumCalc(cavity_params=cavity_1d_gaussian)
        polaritons_gaussian.append(polariton)
    polaritons_gaussian[0].calc_2d_spectrum_compare_others(resolution_cminv=1.0,
                                                  savefile="1d_compare_gaussian.pdf",
                                                  polariton_lst=polaritons_gaussian[1:],
                                                  labels=labels_gaussian)

    # Finally, plot the molecular distribution for these different cases
    fig, axes = clp.initialize(3, 1, width=3.2, height=3.2*0.618*3, fontname="Arial",
                               fontsize=12, return_fig_args=True,
                               sharey=True, sharex=True, labelsize=14,
                               labelthem=True, labelthemPosition=[-0.02, 1.1], LaTeX=True)

    colors = ["0.5", clp.red, "k--", clp.yellow]
    xs, ys = [], []
    for idx, polariton in enumerate(polaritons_amp):
        # first, we plot the coupling constant distribution in the 1D plane
        pattern = np.reshape(polariton.g0_distri**2, (polariton.ngridx_1d, polariton.ngridy_1d))
        print("maximal is", np.max(pattern))
        x, y = polariton.xgrid_1d, pattern[:, 0]
        xs.append(x)
        ys.append(y)
    clp.plotone(xs, ys, axes[0], colors=colors, labels=labels_amp, lw=1)

    xs, ys = [], []
    for idx, polariton in enumerate(polaritons_freq):
        # first, we plot the coupling constant distribution in the 1D plane
        pattern = np.reshape(polariton.g0_distri**2, (polariton.ngridx_1d, polariton.ngridy_1d))
        print("maximal is", np.max(pattern))
        x, y = polariton.xgrid_1d, pattern[:, 0]
        xs.append(x)
        ys.append(y)
    clp.plotone(xs, ys, axes[1], colors=colors, labels=labels_freq, lw=1, ylabel=r"density distribution $\rho(x)$ [$N/L_x$]")

    xs, ys = [], []
    for idx, polariton in enumerate(polaritons_gaussian):
        # first, we plot the coupling constant distribution in the 1D plane
        pattern = np.reshape(polariton.g0_distri**2, (polariton.ngridx_1d, polariton.ngridy_1d))
        print("maximal is", np.max(pattern))
        x, y = polariton.xgrid_1d, pattern[:, 0]
        xs.append(x)
        ys.append(y)
    clp.plotone(xs, ys, axes[2], colors=colors, labels=labels_gaussian, lw=1,
                xlabel="$x$ [$L_x = 1$ mm]", xlim=[0, 1], ylim=[0, 1.9])

    clp.adjust(tight_layout=True, savefile="molecular_distribution_1d.pdf")


def plot_2d_modulation(resolution_cminv=5.0, ngrid_1d=120, nmode_1d=30):
    # reset the default number of grid points per dimension
    cavity_2d_uniform["ngridx_1d"] = ngrid_1d
    cavity_2d_uniform["ngridy_1d"] = ngrid_1d
    cavity_2d_sin["ngridx_1d"] = ngrid_1d
    cavity_2d_sin["ngridy_1d"] = ngrid_1d
    cavity_2d_gaussian["ngridx_1d"] = ngrid_1d
    cavity_2d_gaussian["ngridy_1d"] = ngrid_1d
    cavity_2d_tilting["ngridx_1d"] = ngrid_1d
    cavity_2d_tilting["ngridy_1d"] = ngrid_1d
    cavity_2d_usr["ngridx_1d"] = ngrid_1d
    cavity_2d_usr["ngridy_1d"] = ngrid_1d
    # also reset g0 so that the Rabi splitting is the same for all the cases
    cavity_2d_uniform["g0"] = 0.5 * 120 / ngrid_1d
    cavity_2d_sin["g0"] = 0.5 * 120 / ngrid_1d
    cavity_2d_gaussian["g0"] = 0.5 * 120 / ngrid_1d
    cavity_2d_tilting["g0"] = 0.5 * 120 / ngrid_1d
    cavity_2d_usr["g0"] = 0.5 * 120 / ngrid_1d
    # reset the default number of photon modes per dimension
    cavity_2d_uniform["nmodex_1d"] = nmode_1d
    cavity_2d_uniform["nmodey_1d"] = nmode_1d
    cavity_2d_sin["nmodex_1d"] = nmode_1d
    cavity_2d_sin["nmodey_1d"] = nmode_1d
    cavity_2d_gaussian["nmodex_1d"] = nmode_1d
    cavity_2d_gaussian["nmodey_1d"] = nmode_1d
    cavity_2d_tilting["nmodex_1d"] = nmode_1d
    cavity_2d_tilting["nmodey_1d"] = nmode_1d
    cavity_2d_usr["nmodex_1d"] = nmode_1d
    cavity_2d_usr["nmodey_1d"] = nmode_1d
    # also reset the spacing between each photon mode
    cavity_2d_uniform["domegax_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_uniform["domegay_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_sin["domegax_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_sin["domegay_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_gaussian["domegax_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_gaussian["domegay_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_tilting["domegax_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_tilting["domegay_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_usr["domegax_1d"] = 30.0 * 60.0 / nmode_1d
    cavity_2d_usr["domegay_1d"] = 30.0 * 60.0 / nmode_1d
    # get the polariton spectrum for 2D cavity + sin modulation, gaussian modulation, or a random pattern!
    polariton1 = PolaritonSpectrumCalc(cavity_params=cavity_2d_uniform)
    #polariton1.calc_3d_spectrum(resolution_cminv=resolution_cminv, savefile="3d_spectrum_uniform.pdf")
    polariton2 = PolaritonSpectrumCalc(cavity_params=cavity_2d_sin)
    #polariton2.calc_3d_spectrum(resolution_cminv=resolution_cminv, savefile="3d_spectrum_sin.pdf")
    polariton3 = PolaritonSpectrumCalc(cavity_params=cavity_2d_gaussian)
    #polariton3.calc_3d_spectrum(resolution_cminv=resolution_cminv, savefile="3d_spectrum_gaussian.pdf")
    polariton4 = PolaritonSpectrumCalc(cavity_params=cavity_2d_usr)
    #polariton4.calc_3d_spectrum(resolution_cminv=5.0, savefile="3d_spectrum_usr.pdf")

    polaritons = [polariton1, polariton2, polariton3, polariton4]
    polariton1.plot_molecular_distribution_from_polaritons(polaritons=polaritons, savefile="molecular_distribution_2d.pdf")

    labels = ["homogeneous", "sin modulation", "Gaussian distribution", "cartoon pattern"]
    polaritons[0].calc_2d_spectrum_compare_others(resolution_cminv=5.0,
                                                  savefile="2d_compare.pdf",
                                                  polariton_lst=polaritons[1:],
                                                  labels=labels)

if __name__ == "__main__":
    # Simple 1D calculation with affordable computational cost
    plot_1d_modulation(ngrid_1d=1080, nmode_1d=200)
    plot_1d_eigenstate_distr(ngrid_1d=1080, nmode_1d=200)
    # This is calculated in my personal computer: i7-14700K + 32GB RAM + RTX 3090
    plot_2d_modulation(resolution_cminv=5.0, ngrid_1d=120, nmode_1d=60)
