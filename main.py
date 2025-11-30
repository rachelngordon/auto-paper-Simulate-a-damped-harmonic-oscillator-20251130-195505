# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0  # mass (kg)
k = 1.0  # spring constant (N/m)
omega_n = np.sqrt(k / m)  # natural frequency
c_crit = 2 * np.sqrt(k * m)  # critical damping coefficient

# Damping cases
cases = {
    'underdamped': 0.5 * c_crit,   # ζ = 0.5 < 1
    'critical': c_crit,            # ζ = 1
    'overdamped': 2.0 * c_crit     # ζ = 2 > 1
}

# Initial conditions
x0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity

# Time discretization
t_max = 20.0
num_steps = 2000
t = np.linspace(0, t_max, num_steps)
dt = t[1] - t[0]

# RK4 integrator for the system: y = [x, v]

def rk4_step(y, dt, c):
    def dydt(y):
        x, v = y
        dxdt = v
        dvdt = -(c / m) * v - (k / m) * x
        return np.array([dxdt, dvdt])
    k1 = dydt(y)
    k2 = dydt(y + 0.5 * dt * k1)
    k3 = dydt(y + 0.5 * dt * k2)
    k4 = dydt(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Containers for results
results = {}
for name, c in cases.items():
    y = np.array([x0, v0])
    xs = []
    vs = []
    for _ in t:
        xs.append(y[0])
        vs.append(y[1])
        y = rk4_step(y, dt, c)
    results[name] = {
        'x': np.array(xs),
        'v': np.array(vs)
    }

# Plot displacement vs time
plt.figure(figsize=(8, 5))
for name, data in results.items():
    plt.plot(t, data['x'], label=name)
plt.title('Displacement vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('displacement_vs_time.png')
plt.close()

# Plot phase portrait (velocity vs displacement)
plt.figure(figsize=(8, 5))
for name, data in results.items():
    plt.plot(data['x'], data['v'], label=name)
plt.title('Phase Portrait')
plt.xlabel('Displacement (m)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('phase_portrait.png')
plt.close()

# Primary numeric answer: critical damping coefficient
print('Answer:', c_crit)

