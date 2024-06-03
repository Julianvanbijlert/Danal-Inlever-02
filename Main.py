# Assignment 2
import pandas as pd

# Specify the path to your CSV file
csv_path = "query_product.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, encoding="latin1")

df
df['relevance'].describe()
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named data_frame
# and you want to visualize a column named 'column_name'

# Create a histogram of the column's values
plt.hist(df['relevance'], bins=5)

# Add labels and title
plt.xlabel('Relevance')
plt.ylabel('Frequency')
plt.title('Histogram of Relevance Score')

# Display the plot
plt.show()
from sklearn.model_selection import train_test_split


# Set the training size
training_size = 50000

# Split the DataFrame into training and test sets
train_data, test_data = train_test_split(df, test_size=(len(df) - training_size), random_state=42)

# Display the training data
train_data

# Display the test data
test_data
def check_words(row):
    words_column1 = set(row['product_title'].lower().split())
    words_column2 = set(row['search_term'].lower().split())
    return int(words_column2.issubset(words_column1))

train_data['all_words_in_title'] = train_data.apply(check_words, axis=1)
test_data['all_words_in_title'] = test_data.apply(check_words, axis=1)

train_data
train_data[train_data['all_words_in_title'] == True]
train_data['all_words_in_title'].describe()
import statsmodels.api as sm

# Define the feature and target variables
X = train_data['all_words_in_title']
y = train_data['relevance']

# Add a constant term to the feature variable
X = sm.add_constant(X)

# Create a linear regression model
model = sm.OLS(y, X)

# Fit the model to the data
results = model.fit()

# Print the model summary
print(results.summary())
from sklearn.metrics import r2_score

X_test = test_data['all_words_in_title']
y_test = test_data['relevance']

X_test = sm.add_constant(X_test)

y_pred = results.predict(X_test)

r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)
import matplotlib.pyplot as plt
from collections import Counter

weight_counter = Counter(y_test)
weights = [weight_counter[i]/10 for i in y_test]

plt.scatter(test_data['all_words_in_title'], y_test, label='Actual', s=weights)
plt.plot(test_data['all_words_in_title'], y_pred, color='red', label='Fitted Line')
plt.xlabel('Do all query terms occur in the product title?')
plt.ylabel('Relevance Score')
plt.title('Linear Regression: Fitted Line')
plt.legend()
plt.show()
