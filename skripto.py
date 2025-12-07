import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

df = pd.read_csv('output/output - mars - smallbig.txt', header=None, names=['jday', 'elong', 'velo'])
# df = df.head(5000)
# df = df.iloc[::5]

# ========================================
# EKSTREMUMOJ
# ========================================

sign = np.diff(np.sign(df['velo'])) != 0
sign = np.concatenate(([False], sign))
points_sign = df[sign]
df_max = points_sign[df['velo'].shift(1)[sign] > 0].copy()
df_min = points_sign[df['velo'].shift(1)[sign] < 0].copy()

# ========================================
# omega_t
# ========================================

omega_t = np.mean(np.diff(df_max['elong']) / np.diff(df_max['jday']))
print(f"omega_t = {omega_t}")

# ========================================
# omega_a
# ========================================

omega_a = 2 * 360 / np.mean(np.diff(df_max['jday']))
print(f"omega_a = {omega_a}")

# ========================================
# alpha
# ========================================

df_analysis = pd.DataFrame({
    'jday_start': df_min["jday"].iloc[:-1].values,
    'elong_start': df_min["elong"].iloc[:-1].values % 360,
    'elong_end': df_min["elong"].iloc[1:].values % 360,
})

df_analysis['elong_diff'] = (df_analysis['elong_end'] - df_analysis['elong_start']) % 360
df_analysis['elong_diff'] = df_analysis['elong_diff'].apply(lambda x: x if x >= 0 else x + 360)

sign_elong = np.diff(np.sign(np.diff(df_analysis['elong_diff']))) > 0
sign_elong = np.concatenate(([False], sign_elong, [False]))
points_sign_elong = df_analysis[sign_elong]

final_min = points_sign_elong['elong_end'].min()
final_max = points_sign_elong['elong_start'].max()

print(f"alpha in [{final_min}°; {final_max}°]")

# ========================================
# e
# ========================================

R = 1.0

mask_A = (df["elong"] % 360 >= final_min) & (df["elong"] % 360 <= final_max)
df_A = df[mask_A][["jday", "elong"]]

mean_e = 0
count = 0

for i, A in df_A.iterrows():
    diff = (df["elong"] - A["elong"] + 180) % 360 - 180
    mask = (abs(diff) > 90) & (abs(diff) < 180)
    df_E = df[mask][["jday", "elong"]]

    if df_E.empty:
        continue

    E = df_E.iloc[(df_E["jday"] - A["jday"]).abs().argmin()]

    delta_lambda = (E["elong"] - A["elong"] + 180) % 360 - 180
    delta = omega_t * (E["jday"] - A["jday"])
    epsilon = 180 - (delta + delta_lambda)

    if abs(epsilon) < 1e-3 or abs(delta_lambda) < 1e-3:
        continue

    e = R * sin(radians(epsilon)) / sin(radians(delta_lambda))
    
    mean_e += abs(e)
    count += 1

mean_e /= count

print(f"e = {mean_e}")

# ========================================
# r
# ========================================

df_S1 = df_max["elong"].diff() % 360
df_S2 = df_min["elong"].diff() % 360

delta_lambda = (df_S1.mean() + df_S2.mean()) / 2

r = sin(radians(delta_lambda / 2))

print(f"r = {r}")

# ========================================
# ANTAUXDIROJ
# ========================================

R = 1
alpha = (final_min + final_max) / 2
theta_t = -93
theta_a = -114

jday = df["jday"]
df_real = df["elong"]

x = R * np.cos(np.deg2rad(omega_t * jday + theta_t)) + r * np.cos(np.deg2rad(omega_a * jday + theta_a)) + e * np.cos(np.deg2rad(alpha))
y = R * np.sin(np.deg2rad(omega_t * jday + theta_t)) + r * np.sin(np.deg2rad(omega_a * jday + theta_a)) + e * np.sin(np.deg2rad(alpha))
df_pred = np.rad2deg(np.unwrap(np.arctan2(y, x)))

# ========================================
# ERAROJ
# ========================================

angular_gap = (df_pred - df_real)
mean_deviation = np.mean(np.abs(angular_gap))
print(f"Averagxa angula eraro: {mean_deviation}°")

# ========================================
# MONTRADO
# ========================================

plt.figure(figsize=(10, 7))

plt.plot(jday, df_pred)
plt.plot(jday, df_real)

plt.show()

# ========================================
# OPTIMIZADO DE PARAMETROJ
# ========================================

def model(params, jday, R):
    omega_t, omega_a, alpha, e, r, theta_t, theta_a = params

    alpha_rad   = np.deg2rad(alpha)
    theta_t_rad = np.deg2rad(theta_t)
    theta_a_rad = np.deg2rad(theta_a)

    arg_t = np.deg2rad(omega_t * jday) + theta_t_rad
    arg_a = np.deg2rad(omega_a * jday) + theta_a_rad

    x = R * np.cos(arg_t) + r * np.cos(arg_a) + e * np.cos(alpha_rad)
    y = R * np.sin(arg_t) + r * np.sin(arg_a) + e * np.sin(alpha_rad)

    elonge = np.rad2deg(np.unwrap(np.arctan2(y, x)))
    return elonge

def cost(params, jday, df_real, R):
    elonge_pred = model(params, jday, R)
    residuals = elonge_pred - df_real
    return np.sum(residuals**2)

res = minimize(
    fun=cost,
    x0=np.array([omega_t, omega_a, alpha, mean_e, r, theta_t, theta_a]),
    args=(jday, df_real, R),
    method="BFGS",
    options={"maxiter": 10000, "disp": True}
)

print("Optimumigitaj parametroj :", list(res.x))
