# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 12:11:04 2025

@author: anapa
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

# Modelos matemáticos
def linear_model(x, a, b):
    return a * x + b

def exponential_model(x, a, b):
    return a * np.exp(b * x)

def logarithmic_model(x, a, b):
    return a * np.log(x + 1) + b

def polynomial_model(x, a, b, c):
    return a * x**2 + b * x + c

# R²
def calculate_r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1 - ss_res / ss_tot

# Carregar dados
dados_chuva = pd.read_csv(r"C:\Users\anapa\OneDrive - UNIVALI\Doutorado\AnaliseChuva\Dados_Chuva.csv", sep=';')
dados_chuva['datetime'] = pd.to_datetime(dados_chuva[['Year', 'Month', 'Day', 'Hour']])
dados_chuva.set_index('datetime', inplace=True)

# Limpar e agrupar
dados_chuva = dados_chuva[['Time_Step', 'Cd']].dropna()
dados_chuva['Grouped_Time_Step'] = (dados_chuva['Time_Step'] // 1) * 1
agrupado = dados_chuva.groupby('Grouped_Time_Step')['Cd'].mean().reset_index()

x_data = agrupado['Grouped_Time_Step']
y_data = agrupado['Cd']

# Modelos candidatos
modelos = {
    'Linear': (linear_model, [1, 1]),
    'Exponential': (exponential_model, [1, 0.1]),
    'Logarithmic': (logarithmic_model, [1, 1]),
    'Polynomial': (polynomial_model, [1, 1, 1]),
}

melhor_r2 = -np.inf
melhor_modelo = None
melhor_nome = ""
melhor_params = None
melhor_cov = None

# Ajustar modelos
for nome, (modelo, p0) in modelos.items():
    try:
        params, cov = curve_fit(modelo, x_data, y_data, p0=p0, maxfev=10000)
        y_pred = modelo(x_data, *params)
        r2 = calculate_r2(y_data, y_pred)

        if r2 > melhor_r2:
            melhor_r2 = r2
            melhor_modelo = modelo
            melhor_params = params
            melhor_cov = cov
            melhor_nome = nome
    except Exception as e:
        print(f"Erro ajustando modelo {nome}: {e}")

# Equação do modelo
if melhor_nome == "Linear":
    eq = f"y = {melhor_params[0]:.3f}x + {melhor_params[1]:.3f}"
elif melhor_nome == "Exponential":
    eq = f"y = {melhor_params[0]:.3f}e^({melhor_params[1]:.3f}x)"
elif melhor_nome == "Logarithmic":
    eq = f"y = {melhor_params[0]:.3f}ln(x + 1) + {melhor_params[1]:.3f}"
elif melhor_nome == "Polynomial":
    eq = f"y = {melhor_params[0]:.3f}x² + {melhor_params[1]:.3f}x + {melhor_params[2]:.3f}"
else:
    eq = "Unknown model"

# Preparar curva de previsão
x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit = melhor_modelo(x_fit, *melhor_params)

# Calcular intervalo de confiança (95%)
alpha = 0.05
n = len(x_data)
p = len(melhor_params)
dof = max(0, n - p)  # graus de liberdade
t_val = t.ppf(1 - alpha/2, dof)

# Derivar erro padrão da previsão
J = np.zeros((len(x_fit), p))
delta = 1e-8  # pequeno incremento para derivada numérica

for i in range(p):
    p1 = melhor_params.copy()
    p2 = melhor_params.copy()
    p1[i] -= delta
    p2[i] += delta
    y1 = melhor_modelo(x_fit, *p1)
    y2 = melhor_modelo(x_fit, *p2)
    J[:, i] = (y2 - y1) / (2 * delta)

# Variância da predição
pred_var = np.sum(J @ melhor_cov * J, axis=1)
ci = t_val * np.sqrt(pred_var)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x_data, y_data, label="Mean Cd (5h interval)", color="cadetblue", alpha=0.7)
plt.plot(x_fit, y_fit, label="Fitting Curve", color="red", linestyle="--")
plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='silver', alpha=0.2, label="95% Confidence Interval")

# Rótulos do eixo x como intervalos
xticks = np.arange(0, 48, 3)
plt.xticks(ticks=xticks, fontsize=12)

plt.xlim(0, 48)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

# Informações no gráfico
plt.text(
    0.05, 0.95,
    f"Model: {melhor_nome}\n{eq}\nR² = {melhor_r2:.4f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
)

# Estética final
plt.xlabel("Time Step (grouped, hours)", fontsize=14)
plt.ylabel("Mean Cd concentration (mg/L)", fontsize=14)
plt.legend()
plt.legend(fontsize=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Salvar o gráfico antes de exibir
output_path = r"C:\Users\anapa\OneDrive - UNIVALI\Doutorado\AnaliseChuva\Fitting Curve\Agrupado\artigo"
plt.savefig(os.path.join(output_path, "Cu_grouped_5H.png"), dpi=300)
plt.show()
