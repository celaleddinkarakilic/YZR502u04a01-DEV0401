import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Robot Parametreleri
L1, L2 = 0.10, 0.05
m1, m2 = 0.050, 0.025
I1, I2 = 0.0000416, 0.0000104
g = 9.81

#Kontrolcü Katsayıları
Kp = np.diag([250, 250])
Ki = np.diag([40, 40])
Kd = np.diag([35, 35])


def inverse_kinematics(x, y):
    d2 = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    d2 = np.clip(d2, -1, 1)
    q2 = np.arccos(d2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def get_matrices_user(q, dq):
    q1, q2 = q
    dq1, dq2 = dq

    # M(Q)
    M11 = m1 * (L1 / 2) ** 2 + I1 + m2 * (L1 ** 2 + (L2 / 2) ** 2 + 2 * L1 * (L2 / 2) * np.cos(q2)) + I2
    M12 = m2 * ((L2 / 2) ** 2 + 0.5 * L1 * L2 * np.cos(q2)) + I2
    M21 = M12
    M22 = m2 * (L2 / 2) ** 2 + I2
    M = np.array([[M11, M12], [M21, M22]])

    # C(Q, Q_dot)
    h = 0.5 * m2 * L1 * L2 * np.sin(q2)
    C = np.array([[-h * dq2, -h * (dq1 + dq2)],
                  [h * dq1, 0]])

    # G(Q)
    G1 = (0.5 * m1 * L1 + m2 * L1) * g * np.cos(q1) + 0.5 * m2 * L2 * g * np.cos(q1 + q2)
    G2 = 0.5 * m2 * L2 * g * np.cos(q1 + q2)
    G = np.array([G1, G2])

    return M, C, G


def robot_ode(state, t):
    q, dq, e_int = state[0:2], state[2:4], state[4:6]

    #İstenilen çembersel yörünge
    xc, yc, R, freq = 0.07, 0.07, 0.03, 2.5
    xd, yd = xc + R * np.cos(freq * t), yc + R * np.sin(freq * t)
    qd = inverse_kinematics(xd, yd)

    #İstenilen Hız
    dt = 0.001
    qd_next = inverse_kinematics(xc + R * np.cos(freq * (t + dt)), yc + R * np.sin(freq * (t + dt)))
    dqd = (qd_next - qd) / dt

    # --- Inverse Dynamics + PID Control Law ---
    M, C, G = get_matrices_user(q, dq)
    error = qd - q
    derror = dqd - dq

    #PID
    u = 0 + Kp @ error + Ki @ e_int + Kd @ derror
    tau = M @ u + C @ dq + G

    #İleri Dinamik
    ddq = np.linalg.solve(M, tau - C @ dq - G)
    return np.concatenate([dq, ddq, error])


t = np.linspace(0, 10, 1000)
initial_q = inverse_kinematics(0.10, 0.07)  # Starting point
initial_state = np.concatenate([initial_q, [0, 0, 0, 0]])
sol = odeint(robot_ode, initial_state, t)

accel = np.array([robot_ode(s, ti)[2:4] for s, ti in zip(sol, t)])
x_act = L1 * np.cos(sol[:, 0]) + L2 * np.cos(sol[:, 0] + sol[:, 1])
y_act = L1 * np.sin(sol[:, 0]) + L2 * np.sin(sol[:, 0] + sol[:, 1])

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Cartesian Circle
axs[0, 0].plot(x_act, y_act, 'b', label='Gerçek')
axs[0, 0].plot(0.07 + 0.03 * np.cos(2.5 * t), 0.07 + 0.03 * np.sin(2.5 * t), 'r--', label='Referans')
axs[0, 0].set_title('Dairesel Yörünge');
axs[0, 0].axis('equal');
axs[0, 0].legend()

axs[0, 1].plot(t, sol[:, 0], label='q1');
axs[0, 1].plot(t, sol[:, 1], label='q2')
axs[0, 1].set_title('Mafsal Açıları (rad)');
axs[0, 1].legend()

axs[1, 0].plot(t, sol[:, 2], label='dq1');
axs[1, 0].plot(t, sol[:, 3], label='dq2')
axs[1, 0].set_title('Mafsal Hızları (rad/s)');
axs[1, 0].legend()

axs[1, 1].plot(t, accel[:, 0], label='ddq1');
axs[1, 1].plot(t, accel[:, 1], label='ddq2')
axs[1, 1].set_title('Mafsal İvmeleri (rad/s²)');
axs[1, 1].legend()

plt.tight_layout()
plt.show()