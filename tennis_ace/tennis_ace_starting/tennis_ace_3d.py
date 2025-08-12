import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())

# Features e target
features = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
outcome = df[['Winnings']]

# Split
x_train, x_test, y_train, y_test = train_test_split(features, outcome, train_size=0.8, random_state=6)

# Modelo
ols = LinearRegression()
ols.fit(x_train, y_train)
print("R²:", ols.score(x_test, y_test))

# Scatter 3D dos dados reais
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test['BreakPointsOpportunities'], 
           x_test['FirstServeReturnPointsWon'], 
           y_test, alpha=0.5, label="Dados Reais")

# Criar malha para o plano
x_surf, y_surf = np.meshgrid(
    np.linspace(x_test['BreakPointsOpportunities'].min(), x_test['BreakPointsOpportunities'].max(), 50),
    np.linspace(x_test['FirstServeReturnPointsWon'].min(), x_test['FirstServeReturnPointsWon'].max(), 50)
)

# Prever em cima da malha
mesh_features = np.c_[x_surf.ravel(), y_surf.ravel()]
z_surf = ols.predict(mesh_features).reshape(x_surf.shape)

# Plano da regressão
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3)

# Labels
ax.set_xlabel('BreakPointsOpportunities')
ax.set_ylabel('FirstServeReturnPointsWon')
ax.set_zlabel('Winnings')
plt.legend()
plt.show()