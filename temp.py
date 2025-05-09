import seaborn as sns 
import matplotlib.pyplot as plt

tdf = sns.load_dataset('titanic')
print(tdf[['age']].head())
print(tdf.shape)
print(tdf.isnull().sum())
print(tdf.info())

sns.countplot(x='survived', data=tdf)
plt.show()

