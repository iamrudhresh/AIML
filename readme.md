1. Write a python program that uses the decision tree algorithm from the sklearn library to classify the iris dataset. [Note: load the iris data set from sklearn]

```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

2. Write a python program that uses the random forest algorithm from the sklearn library to classify the iris dataset. [Note: load the iris data set from sklearn]
```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```
3. Build simple Neural Network model, by creating the own dataset using numpy library and evaluate the performance of the model on new test data
```py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(0)
X = np.random.randn(1000, 2)  # 1000 samples, 2 features
y = np.random.randint(0, 2, size=1000)  # 2 classes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
4. Write a python program to build the Linear Regression model. Calculate MSE and R-Squared value and Plot the actual and predicted values. 
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate R-Squared value
r_squared = r2_score(y_test, y_pred)
print("R-Squared value:", r_squared)

# Plot actual vs predicted values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Actual vs Predicted values')
plt.legend()
plt.show()
```
5. Implement k-Nearest Neighbor algorithm to classify the Iris Dataset.
```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNeighborsClassifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
6. Write a python program to build the SVM classifier model to classify whether a fruit is orange or mango. 
```py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample data: weight in grams, color (1 for orange, 0 for mango)
X = np.array([[100, 1], [150, 1], [200, 1], [120, 0], [170, 0], [220, 0]])
y = np.array([1, 1, 1, 0, 0, 0])  # 1 for orange, 0 for mango

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X, y)

# Predictions for new data points
new_fruits = np.array([[130, 1], [180, 0], [210, 1]])
predicted_labels = svm_classifier.predict(new_fruits)

# Map predicted labels to fruit names
fruit_names = {1: 'Orange', 0: 'Mango'}
predicted_fruits = [fruit_names[label] for label in predicted_labels]

# Print the predicted fruits
for i, fruit in enumerate(predicted_fruits):
    print(f"New fruit {i+1} is predicted as: {fruit}")

# Accuracy on training data
train_accuracy = accuracy_score(y, svm_classifier.predict(X))
print("Accuracy on training data:", train_accuracy)
```

7. Build deep learning Neural Network model by creating two hidden layers and check the accuracy of prediction.
```py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(0)
X = np.random.randn(1000, 2)  # 1000 samples, 2 features
y = np.random.randint(0, 2, size=1000)  # 2 classes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build deep learning neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
8. Write a python program to use naive Bayes model to predict whether a person will buy_computer or not. Also display the accuracy of prediction. 
```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Age': [25, 35, 45, 20, 30, 40, 50, 60],
    'Income': ['Low', 'Low', 'Medium', 'Low', 'Medium', 'High', 'Medium', 'High'],
    'Student': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes'],
    'Credit_Rating': ['Fair', 'Good', 'Fair', 'Good', 'Fair', 'Good', 'Good', 'Fair'],
    'Buy_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df[['Income', 'Student', 'Credit_Rating']])

# Combine numerical and encoded categorical variables
X = pd.concat([df[['Age']], df_encoded], axis=1)
y = df['Buy_Computer']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

9. Implement BFS and DFS
```py
from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node, neighbor):
        if node not in self.graph:
            self.graph[node] = []
        self.graph[node].append(neighbor)

    def bfs(self, start_node):
        visited = set()
        queue = deque([start_node])
        traversal = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                traversal.append(node)
                visited.add(node)
                neighbors = self.graph.get(node, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return traversal

    def dfs(self, start_node):
        visited = set()
        traversal = []

        def dfs_helper(node):
            traversal.append(node)
            visited.add(node)
            neighbors = self.graph.get(node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs_helper(neighbor)

        dfs_helper(start_node)
        return traversal

# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(2, 6)

print("BFS traversal:", g.bfs(0))
print("DFS traversal:", g.dfs(0))
```
10.i) Create a pandas data frame for the string “He is a good person” and display the dataframe.
ii) Create a pandas data frame for car_data which displays the list of car names and their corresponding prices. Display list of car names and display the details of second car using index.
iii) Append the new car_data to the existing data frame.
iv) Update any of the data in data frame.
v) Store the data frame to the csv file.
vi) Reading a csv file and storing in a another dataframe and display it.
```py
import pandas as pd

# i) Create a pandas data frame for the string “He is a good person” and display the dataframe.
string_data = "He is a good person"
df_string = pd.DataFrame({"Text": [string_data]})
print("DataFrame for the string:")
print(df_string)
print()

# ii) Create a pandas data frame for car_data which displays the list of car names and their corresponding prices. 
#     Display list of car names and display the details of second car using index.
car_data = {
    "Car_Name": ["Toyota", "Honda", "Ford"],
    "Price": [25000, 30000, 27000]
}
df_cars = pd.DataFrame(car_data)
print("DataFrame for car_data:")
print(df_cars)
print()

# Display list of car names
print("List of car names:")
print(df_cars['Car_Name'].tolist())
print()

# Display details of second car using index
print("Details of the second car:")
print(df_cars.iloc[1])
print()

# iii) Append the new car_data to the existing data frame.
new_car_data = {
    "Car_Name": ["BMW", "Mercedes"],
    "Price": [45000, 55000]
}
df_new_cars = pd.DataFrame(new_car_data)
df_cars = df_cars.append(df_new_cars, ignore_index=True)
print("DataFrame after appending new car data:")
print(df_cars)
print()

# iv) Update any of the data in data frame.
df_cars.at[2, 'Price'] = 28000  # Updating the price of Ford
print("DataFrame after updating data:")
print(df_cars)
print()

# v) Store the data frame to the csv file.
df_cars.to_csv("car_data.csv", index=False)
print("DataFrame stored to 'car_data.csv'")

# vi) Reading a csv file and storing in another dataframe and display it.
df_read = pd.read_csv("car_data.csv")
print("\nDataFrame read from 'car_data.csv':")
print(df_read)
```
11.i) Create a 2-D array with 3 rows and 4 colums using numpy and display its shape, itemsize and datatype and reshape the array as 4 rows and 3 columns.
ii) Create a 1-D array named profit with set of values. Similarly Create another 1-D array named sales. Calculate Profit Margin Ratio. [Formula: profit/sales]
iii) Use matplotlib library to plot a graph by taking any random set of values for x & y.
iv) Reading any csv file and storing in a dataframe
v) Use matplotlib library to plot a scatter plot with two different classes specifying different color for classes.
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# i) Create a 2-D array with 3 rows and 4 columns using numpy
array_2d = np.arange(12).reshape(3, 4)
print("Original array:")
print(array_2d)
print("Shape:", array_2d.shape)
print("Itemsize:", array_2d.itemsize)
print("Datatype:", array_2d.dtype)

# Reshape the array as 4 rows and 3 columns
array_2d_reshaped = array_2d.reshape(4, 3)
print("\nArray after reshaping:")
print(array_2d_reshaped)

# ii) Create a 1-D array named profit and sales, and calculate Profit Margin Ratio
profit = np.array([100, 150, 200, 250])
sales = np.array([1000, 2000, 3000, 4000])
profit_margin_ratio = profit / sales
print("\nProfit Margin Ratio:", profit_margin_ratio)

# iii) Use matplotlib library to plot a graph by taking any random set of values for x & y
x = np.random.randint(0, 100, size=10)
y = np.random.randint(0, 100, size=10)
plt.scatter(x, y)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot of Random Values')
plt.show()

# iv) Reading any csv file and storing in a dataframe
df = pd.read_csv("example.csv")  # Replace "example.csv" with the filename
print("\nDataFrame read from CSV:")
print(df)

# v) Use matplotlib library to plot a scatter plot with two different classes specifying different color for classes
# Example scatter plot with two classes "class1" and "class2"
class1_x = np.random.normal(0, 1, 50)
class1_y = np.random.normal(0, 1, 50)
class2_x = np.random.normal(3, 1, 50)
class2_y = np.random.normal(3, 1, 50)

plt.scatter(class1_x, class1_y, color='blue', label='Class 1')
plt.scatter(class2_x, class2_y, color='red', label='Class 2')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Two Classes')
plt.legend()
plt.show()
```
12.i)Create a pandas data frame for calories_data which displays the list of calories_consumed and  calories_burnt daywise.[Output contain 4 columns index value, day, calories_consumed and  calories_burnt] 
ii) Add an additional column calories_remining and calculate it and display[Note: calories remining = calories consumed - calories burnt] 
iii) Display calories_consumed and  calories_burnt daywise. Here, The days should not be a separate column. Days should act as a index to the dataframe.   
iv) Store the data frame to the csv file.
v)  Display the pandas version.
```py
import pandas as pd

# i) Create a pandas data frame for calories_data which displays the list of calories_consumed and  calories_burnt daywise.
calories_data = {
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "calories_consumed": [2000, 2200, 2500, 2300, 2400],
    "calories_burnt": [500, 600, 550, 700, 650]
}
df = pd.DataFrame(calories_data)
print("DataFrame with calories consumed and burnt daywise:")
print(df)
print()

# ii) Add an additional column calories_remaining and calculate it and display
df['calories_remaining'] = df['calories_consumed'] - df['calories_burnt']
print("DataFrame with additional column 'calories_remaining':")
print(df)
print()

# iii) Display calories_consumed and  calories_burnt daywise. Here, The days should not be a separate column. Days should act as an index to the dataframe.
df.set_index('day', inplace=True)
print("DataFrame with days as index:")
print(df[['calories_consumed', 'calories_burnt']])
print()

# iv) Store the data frame to the csv file.
df.to_csv("calories_data.csv")
print("DataFrame stored to 'calories_data.csv'")
print()

# v) Display the pandas version.
print("Pandas version:", pd.__version__)
```
