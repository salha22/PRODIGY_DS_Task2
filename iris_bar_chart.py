import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the species column
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Create a bar chart for the species distribution
sns.countplot(x='species', data=data)
plt.title('Distribution of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()



