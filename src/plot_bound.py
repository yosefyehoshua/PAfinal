import numpy as np
import matplotlib.pyplot as plt

def generate_problem_data(m, n):
    A = np.random.randn(m, n)
    b = np.random.rand(m) + np.linalg.norm(A, axis=1) + 1  # Ensure b_i > ||a_i||
    return A, b

def f(A, b, x):
    return -np.sum(np.log(b - np.dot(A, x)))

def grad_f(A, b, x):
    return np.dot(A.T, 1 / (b - np.dot(A, x)))

def hess_f(A, b, x):
    D = np.diag(1 / (b - np.dot(A, x))**2)
    return np.dot(A.T, np.dot(D, A))

def newtons_method(A, b, x0, tol=1e-6, max_iter=100, alpha = 1e-4, beta = 0.5):
    x = x0


    for i in range(max_iter):
        gradient = grad_f(A, b, x)
        hessian = hess_f(A, b, x)
        delta_x = np.linalg.solve(hessian, -gradient)
        
        # Line search with Armijo condition to ensure feasibility and sufficient decrease
        t = 1.0
        while np.any(b - np.dot(A, x + t * delta_x) <= 0) or f(A, b, x + t * delta_x) > f(A, b, x) + alpha * t * np.dot(gradient, delta_x):
            t *= beta
        
        x = x + t * delta_x

        if np.linalg.norm(gradient) < tol:
            break

    return x, i + 1

# Problem sizes
problems = [
    (100, 50),    # m = 100, n = 50
    (1000, 5),  # m = 1000, n = 5
    (10, 500)    # m = 10, n = 500
]

num_instances = 50

results = []
alpha = 1e-4  # Armijo condition parameter
beta = 0.5    # Step size reduction factor
eta = (1-2*alpha)/4
gamma = (alpha*beta*(eta**2))/(1+eta)
epsilon = 1e-6

for m, n in problems:
    for _ in range(num_instances):
        A, b = generate_problem_data(m, n)
        x0 = np.zeros(n)
        x_star, _ = newtons_method(A, b, x0, tol=epsilon, alpha=alpha, beta=beta)
        p_star = f(A, b, x_star)

        # Generate a feasible initial point
        x0 = np.random.randn(n)
        while np.any(b - np.dot(A, x0) <= 0):
            x0 = np.random.randn(n)
        
        f0 = f(A, b, x0)
        _, iterations = newtons_method(A, b, x0, tol=epsilon, alpha=alpha, beta=beta)
        
        results.append((f0 - p_star, iterations, m, n))

# Convert results to numpy array for easier indexing
results = np.array(results)

# Plot the results
plt.figure(figsize=(10, 6))

markers = {(100, 50): 'o', (1000, 5): 's', (10, 500): 'd'}
labels = {(100, 50): 'm = 100, n = 50', (1000, 5): 'm = 1000, n = 500', (10, 500): 'm = 10, n = 500'}

for (m, n), marker in markers.items():
    mask = (results[:, 2] == m) & (results[:, 3] == n)
    plt.scatter(results[mask, 0], results[mask, 1], label=labels[(m, n)], marker=marker)

# Plot the self-concordant bound
f0_vals = np.linspace(min(results[:, 0]), max(results[:, 0]), 100)

bounds = f0_vals/gamma + np.log(np.log(1/epsilon))
plt.plot(f0_vals, bounds, label='Self-Concordant Bound', color='r', linestyle='--')

plt.xlabel(r'$f(x^{(0)}) - p^*$')
plt.ylabel('Iterations')
plt.title('Number of Newton Iterations for Self-Concordant Functions')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()