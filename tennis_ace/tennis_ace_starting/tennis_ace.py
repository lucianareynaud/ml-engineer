import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

df = pd.read_csv('tennis_stats.csv')
print(df.head())

x = df[['BreakPointsOpportunities']]
y = df[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

ols = LinearRegression()
ols.fit(x_train, y_train)
y_predict = ols.predict(x_test)
ols.score(x_test, y_test)
print(ols.score(x_test, y_test))

plt.scatter(x_test, y_test, alpha=0.5, label="Dados Reais")
plt.plot(x_test, y_predict, color='red', linewidth=2, label='Regressão Linear' )
plt.xlabel('BreakingPointsOpportunities')
plt.ylabel('Winnings')
plt.legend()
plt.show()





