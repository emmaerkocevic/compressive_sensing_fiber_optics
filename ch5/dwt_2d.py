import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.optimization.sparsity import spgl1


### sample image: 16-by-16 pixels checkerboard ###
block_size = 4
rows, cols = 16, 16
checkerboard = np.zeros((rows, cols))

for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):
        if (i // block_size + j // block_size) % 2 == 0:
            checkerboard[i:i+block_size, j:j+block_size] = 1

plt.figure()
plt.imshow(checkerboard, cmap='gray')
plt.title('ground truth')


def cs_dwt_reconstruction(m, n, f_2D, correlation, L, wavelet, level, sigma):
    np.random.seed(0)
    Phi = np.random.randn(m, n)
    Phiop = pylops.MatrixMult(Phi)

    if correlation == True:
        Smop = pylops.Smoothing1D(nsmooth=L, dims=n)
        Phiop = Phiop * Smop

    plt.figure()
    plt.imshow(Phiop * np.eye(n))
    plt.title('Phi')

    nx, ny = f_2D.shape
    Psiop_t = pylops.signalprocessing.DWT2D((nx, ny), wavelet=wavelet, level=level)
    x_2D = Psiop_t * f_2D

    Psiop = Psiop_t.T

    plt.figure()
    plt.imshow(Psiop * np.eye(n))
    plt.title('Psi')

    plt.figure()
    plt.imshow(x_2D, cmap='gray')
    plt.title('sparse representation')

    x = x_2D.flatten()

    Aop = Phiop * Psiop

    plt.figure()
    plt.imshow(Aop * np.eye(n))
    plt.title('A = Phi Psi')

    y = Aop * x

    xinv = spgl1(Aop, y, sigma=sigma)[0]
    info = spgl1(Aop, y, sigma=sigma)[2]
    # finv, xinv, info = spgl1(A=Phiop, b=y, SOp=Psiop_t, sigma=sigma)

    xinv_2D = xinv.reshape(16, 16)
    finv_2D = Psiop * xinv_2D

    plt.figure()
    plt.plot(info["xnorm1"], label="L1-norm")
    plt.plot(info["rnorm2"], label="L2-norm")
    plt.plot(info["lambdaa"], label="epsilon")
    plt.plot(info["rnorm2"]+info["lambdaa"]+info["xnorm1"], label="cost")
    plt.legend()

    plt.figure()
    plt.imshow(finv_2D, cmap='gray')
    plt.title('reconstruction')

    plt.show()


cs_dwt_reconstruction(m=100, n=256, f_2D=checkerboard, correlation=True, L=8, wavelet='db1', level=2, sigma=0.1)
