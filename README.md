# train_titanic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
data = pd.read_csv('train_titanic.csv')
data.head()
data.shape
print("number of rows:",data.shape[0])
print("number of rows:",data.shape[1])
data.info()
stats = data.describe()
print(stats)
first_fares = data['Fare'][data['Pclass']==1]
first_fares
first_means = round(np.mean(first_fares),2)
first_means
first_median = round(np.median(first_fares),2)
first_median
first_conf = np.round(np.percentile(first_fares, [2.5, 97.5]), 2)
first_conf
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(first_fares)
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Distribution of the fare of the tickets in the first class")
plt.show()
third_fares = data['Fare'][data['Pclass']==3]
third_fares
third_mean = round(np.mean(third_fares), 2)
third_median = round(np.median(third_fares), 2)
third_conf = np.round(np.percentile(third_fares, [2.5, 97.5]), 2)
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(third_fares)
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Distribution of the fare of the tickets in the third class")
plt.show()
x = ['First Class', 'Second Class']
y = [np.mean(data["Survived"][data["Pclass"]==1]),
np.mean(data["Survived"][data["Pclass"]==3])]
plt.bar(x, y)
plt.ylabel("Survival Rate")
plt.title("Survival Rate for people in the first and third classes")
plt.show()
First_Class_Sample = np.array([np.mean(data[data["Pclass"]==1].sample(20)["Survived"].v
third_Class_Sample = np.array([np.mean(data[data["Pclass"]==3].sample(20)["Survived"].v
plt.subplots(1, 2, figsize = (10, 5))
plt.subplot(1,2, 1)
sn.distplot(First_Class_Sample)
plt.title("First-Class Sample Distribution")
plt.xlabel("Survival Rate")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
sn.distplot(third_Class_Sample)
plt.title("Third-Class Sample Distribution")
plt.xlabel("Survival Rate")
plt.ylabel("Frequency")
plt.show()
effect = np.mean(First_Class_Sample) - np.mean(third_Class_Sample)
sigma_first = np.std(First_Class_Sample)
sigma_third = np.std(third_Class_Sample)
sigma_difference = np.sqrt((sigma_first**2)/len(First_Class_Sample) + (sigma_third**2)
/len(third_Class_Sample))
z_score = effect / sigma_difference
z_score
import scipy.stats as st
st.norm.sf(abs(z_score))*2
