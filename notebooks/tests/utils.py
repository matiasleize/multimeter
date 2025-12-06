import numpy as np
from scipy.special import erfcinv
from scipy.stats import chi2, multivariate_normal, gamma
from scipy.special import erfcinv, erfinv, erf, erfc
from scipy.linalg import sqrtm, inv, norm
from scipy.interpolate import interp1d


from math import sqrt

def n_eff_to_L_iso(n: float, d: int = 2, sigma2: float = 0.5, gauss_scale: str = '1sigma') -> float:
    """
    Distancia euclídea L entre las medias de dos Gaussianas D-dimensionales
    idénticas e isotrópicas (Sigma = sigma2 * I), tal que sus contornos 
    que en 1D corresponden a ±nσ se 'toquen'.

    Parámetros
    ----------
    d : int
        Dimensión (D).
    n : float
        'Número de sigmas' 1D (es decir, p = erf(n / sqrt(2))).
    sigma2 : float, opcional
        Varianza isotrópica por eje (default 0.5).

    Devuelve
    --------
    L : float
        Distancia euclídea entre medias.
    """


    if gauss_scale == '1sigma':
        # Probabilidad 1D asociada a ±nσ
        p = erf(n / sqrt(2.0)) #is the same, less general
        #p = chi2.cdf(n**2, df=1)  # Cumulative distribution function for chi2Q
    elif gauss_scale == '2sigma':
        p = chi2.cdf(n**2, df=2)  # Cumulative distribution function for chi2

    # Cuantil chi-cuadrado con d grados de libertad
    m2 = chi2.ppf(p, df=d)              # m^2 = χ²_{d,p}
    m  = sqrt(m2)
    # En isotrópico: L = 2 * σ * m, con σ = sqrt(sigma2)
    return 2.0 * sqrt(sigma2) * m




def L_iso_to_n_eff(L: float, d: int = 2, sigma2: float = 0.5, gauss_scale: str = '1sigma') -> float:
    """
    Distancia euclídea L entre las medias de dos Gaussianas D-dimensionales
    idénticas e isotrópicas (Sigma = sigma2 * I), tal que sus contornos 
    que en 1D corresponden a ±nσ se 'toquen'.

    Parámetros
    ----------
    d : int
        Dimensión (D).
    n : float
        'Número de sigmas' 1D (es decir, p = erf(n / sqrt(2))).
    sigma2 : float, opcional
        Varianza isotrópica por eje (default 0.5).

    Devuelve
    --------
    L : float
        Distancia euclídea entre medias.
    """
    factor = np.linspace(0, 8, 100)
    n_eff_to_L_iso_values = np.array([n_eff_to_L_iso(d=d, n=f, sigma2=sigma2, gauss_scale=gauss_scale) for f in factor])
    aux = interp1d(n_eff_to_L_iso_values, factor, bounds_error=False, fill_value="extrapolate")
    return aux(L)


def PTE_to_n_sigma(PTE: float) -> float:
    return np.sqrt(2)*erfcinv(PTE)

def n_sigma_to_PTE(n_sigma: float) -> float:
    return erfc(n_sigma/np.sqrt(2))



def PTE_to_L(PTE: float, Nd: int, d: int = 2, sigma2: float = 0.5) -> float:
    factor = np.linspace(0, 8, 100)
    PTE_arr = np.zeros(len(factor))

    if Nd == 2:
        for i in range(len(factor)):
            mean_A = factor[i] * np.array([-1/2., 0.])
            mean_B = factor[i] * np.array([1/2, 0.])

            cov_A = sigma2 * np.eye(d)
            cov_B = cov_A.copy()
            cov_C = cov_A.copy()

            r_1 = np.dot(sqrtm(np.linalg.inv(cov_A + cov_B)), (mean_A - mean_B))

            # Example usage:
            beta_L = np.linalg.norm(r_1)**2
            PTE_arr[i] = chi2(d).sf(beta_L)
        return interp1d(PTE_arr, factor, bounds_error=False, fill_value="extrapolate")(PTE)


    elif Nd == 3:
        for i in range(len(factor)):
            mean_A = factor[i] * np.array([-1/2, 0.])
            mean_B = factor[i] * np.array([1/2, 0.])
            mean_C = factor[i] * np.array([0., sqrt(3)/2])

            cov_A = np.eye(d) / 2
            cov_B = cov_A.copy()
            cov_C = cov_A.copy()


            r_1 = np.dot(sqrtm(np.linalg.inv(cov_A + cov_B)), (mean_A - mean_B))
            r_2 = np.dot(sqrtm(np.linalg.inv(cov_A + cov_C)), (mean_A - mean_C))
            r_3 = np.dot(sqrtm(np.linalg.inv(cov_B + cov_C)), (mean_B - mean_C))
            
            beta_L = (np.linalg.norm(r_1)**2 + np.linalg.norm(r_2)**2 + np.linalg.norm(r_3)**2)/3
            PTE_arr[i] = gamma(a=d,scale=1).sf(beta_L)
        return interp1d(PTE_arr, factor, bounds_error=False, fill_value="extrapolate")(PTE)
    
    else:
        print('Invalid number of distributions!')
        pass


def L_to_PTE(L: float, Nd: int, d: int = 2, sigma2: float = 0.5) -> float:
    PTE_arr = np.logspace(-25, 0, 100)
    L_arr = np.zeros(len(PTE_arr))

    for i, PTE in enumerate(PTE_arr):
        L_arr[i] = PTE_to_L(PTE, Nd=Nd, d=d, sigma2=sigma2)
    
    return interp1d(L_arr, PTE_arr, bounds_error=False, fill_value="extrapolate")(L)




if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n_sigma = np.linspace(0, 5, 500)
    N_eff_2 = np.zeros(len(n_sigma))
    N_eff_3 = np.zeros(len(n_sigma))
    for i, n in enumerate(n_sigma):
        PTE = n_sigma_to_PTE(n_sigma=n)
        L_2 = PTE_to_L(PTE=PTE, Nd=2)
        N_eff_2[i] = L_iso_to_n_eff(d=2, L=L_2)

        L_3 = PTE_to_L(PTE=PTE, Nd=3)
        N_eff_3[i] = L_iso_to_n_eff(d=3, L=L_3)

    plt.figure()
    plt.plot(n_sigma, N_eff_2, label='Nd = 2')
    plt.plot(n_sigma, N_eff_3, label='Nd = 3')
    plt.plot(n_sigma, n_sigma, 'k--', label='1:1 line')
    plt.grid()
    plt.xlabel(r'$N_\sigma$')
    plt.ylabel(r'$N_\text{eff}$')
    plt.legend()
    plt.savefig('n_sigma_to_n_eff.png', dpi=200)
    plt.show()


else:
    print('--- Using 1sigma scale ---')
    for n in [1, 2, 3, 4, 5]:
        print(n, n_eff_to_L_iso(d=2, n=n))   # d=2, sigma^2=0.5

    print('--- Using 2sigma scale ---')
    for n in [1, 2, 3, 4, 5]:
        print(n, n_eff_to_L_iso(d=2, n=n, gauss_scale='2sigma'))   # d=2, sigma^2=0.5

    print('--- L to N_eff ---')
    for L in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(L, L_iso_to_n_eff(d=2, L=L))   # d=2, sigma^2=0.5

    print('--- PTE to N_sigma ---')
    for PTE in [0.32, 0.05, 0.003, 0.00006]:
        print(PTE, PTE_to_n_sigma(PTE=PTE))   # d=2
    
    print('--- N_sigma to PTE ---')
    for n_sigma in [1, 2, 3, 4]:
        print(n_sigma, n_sigma_to_PTE(n_sigma=n_sigma))   # d=2

    print('--- PTE to L (Nd=2) ---')
    for PTE in [0.32, 0.05, 0.003, 0.00006]:
        print(PTE, PTE_to_L(PTE=PTE, Nd=2))   # d=2, Nd=2

    print('--- L to PTE (Nd=2) ---')
    for L in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(L, L_to_PTE(L=L, Nd=2))   # d=2, Nd=2

    print('--- PTE to L (Nd=3) ---')
    for PTE in [0.32, 0.05, 0.003, 0.00006]:
        print(PTE, PTE_to_L(PTE=PTE, Nd=3))   # d=2, Nd=3

    print('--- L to PTE (Nd=3) ---')
    for L in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(L, L_to_PTE(L=L, Nd=3))   # d=2, Nd=3
