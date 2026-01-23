import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from scipy.integrate import odeint

class FlagSolver:
    def __init__(self, N=64, L=1.0, T=10.0, dt=0.01, C=0.1, delta=1e-4):
        self.N = N
        self.L = L
        self.T = T
        self.dt = dt
        self.C = C
        self.delta = delta  # Parámetro de regularización Krasny
        
        # Nodos de Chebyshev
        self.s = np.cos(np.pi * np.arange(N) / (N - 1)) * L
        # Matriz de diferenciación
        self.D = self.cheb_diff_matrix(N, L)
        
    def cheb_diff_matrix(self, N, L):
        """Matriz de diferenciación para nodos de Chebyshev"""
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        D = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i,j] = ((-1)**(i+j)) / (x[i] - x[j])
        
        for i in range(N):
            D[i,i] = -np.sum(D[i,:])
        
        return D * (2/L)
    
    def krasny_regularization(self, u):
        """Regularización tipo Krasny para evitar singularidades"""
        return u / (1 + (u/self.delta)**2)
    
    def flag_rhs(self, y, t):
        """Right-hand side de las ecuaciones de la bandera"""
        theta = y[:self.N]
        theta_t = y[self.N:]
        
        # Tensión (condición de inextensibilidad)
        theta_s = self.D @ theta
        sigma = -0.5 * theta_s**2
        
        # Términos de flexión y tensión
        theta_ss = self.D @ theta_s
        theta_ssss = self.D @ self.D @ theta_ss
        tension_term = self.D @ (sigma * theta_s)
        
        # Acoplamiento fluido (simplificado)
        x_t = np.sin(theta)  # Componentes velocidad
        y_t = np.cos(theta)
        fluid_term = -self.C * (x_t * np.sin(theta) - y_t * np.cos(theta))
        
        # Ecuación principal regularizada
        theta_tt = -theta_ssss + tension_term + fluid_term
        theta_tt = self.krasny_regularization(theta_tt)
        
        return np.concatenate([theta_t, theta_tt])
    
    def solve(self, theta0=None):
        """Resuelve la evolución temporal"""
        if theta0 is None:
            theta0 = 0.1 * np.sin(np.pi * self.s / self.L)
        
        y0 = np.concatenate([theta0, np.zeros(self.N)])
        t = np.arange(0, self.T, self.dt)
        
        sol = odeint(self.flag_rhs, y0, t)
        return sol, t

# Ejemplo de uso
if __name__ == "__main__":
    solver = FlagSolver(N=64, T=5.0, C=0.5)
    sol, t = solver.solve()
    
    # Visualización
    plt.figure(figsize=(12, 4))
    plt.plot(solver.s, sol[-1, :solver.N])
    plt.title('Configuración final de la bandera')
    plt.xlabel('s')
    plt.ylabel('θ(s)')
    plt.grid(True)
    plt.show()


    fig,ax = plt.subplots()
    for i, solvi in enumerate(sol[::10,:solver.N]):
        ax.plot(solver.s,solvi , label=f't={t[i*50]:.2f}s')
    ax.set_xlabel('s')
    ax.set_ylabel('θ(s,t)')