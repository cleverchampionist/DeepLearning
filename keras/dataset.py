import seaborn as sns 
iris = sns.load_dataset('iris')

import seaborn as sns;
sns.set()
sns.pairplot(iris, hue='species', height=3)