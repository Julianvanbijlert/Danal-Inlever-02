# Assignment 2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# Specify the path to your CSV file
csv_path = "query_product.csv"
pd_path = 'product_descriptions.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, encoding="latin1")
product_description_text = pd.read_csv(pd_path, encoding='latin1')

# Merge the two datasets on product_uid
merged_data = pd.merge(df, product_description_text, on='product_uid')

# For simplicity, let's assume we are using 'relevance' as the target for regression
# and creating a binary target for classification
X = merged_data[['product_uid']].values  # Using product_uid as feature for simplicity
y_reg = merged_data['relevance'].values
y_log = (merged_data['relevance'] > 2.5).astype(int)  # Binary target based on relevance threshold

# Split the data into training and testing sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=50000, random_state=42)
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=50000, random_state=42)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_reg_train, y_reg_train)
y_reg_pred = lin_reg.predict(X_reg_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_log_train, y_log_train)
y_log_pred = log_reg.predict(X_log_test)

# Evaluate Linear Regression
mse = mean_squared_error(y_reg_test, y_reg_pred)
print("Linear Regression Mean Squared Error:", mse)

# Evaluate Logistic Regression
accuracy = accuracy_score(y_log_test, y_log_pred)
print("Logistic Regression Accuracy:", accuracy)
print(classification_report(y_log_test, y_log_pred))

# Plot results for Linear Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_reg_test, y_reg_test, color='blue', label='Actual')
plt.plot(X_reg_test, y_reg_pred, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Product UID')
plt.ylabel('Relevance')
plt.legend()

# Plot results for Logistic Regression
plt.subplot(1, 2, 2)
plt.scatter(X_log_test, y_log_test, color='blue', label='Actual')
plt.scatter(X_log_test, y_log_pred, color='red', marker='x', label='Predicted')
plt.title('Logistic Regression')
plt.xlabel('Product UID')
plt.ylabel('Is Expensive')
plt.legend()

plt.show()

df
df['relevance'].describe()

df.head()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def filter_stopwords_from_csv(input_file, output_file, text_column):
    # Lees de CSV in een DataFrame
    df = pd.read_csv(input_file, encoding='latin1')

    # Haal de Engelse stopwoorden op
    stop_words = set(stopwords.words('english'))

    def filter_stopwords(text):
        # Tokenize de tekst in woorden
        words = word_tokenize(text)
        # Filter de stopwoorden uit de lijst met woorden
        filtered_words = [word for word in words if word.lower() not in stop_words]
        # Voeg de gefilterde woorden samen tot een string
        return ' '.join(filtered_words)

    # Pas de filterfunctie toe op de opgegeven tekstkolom
    df[text_column] = df[text_column].astype(str).apply(filter_stopwords)

    # Schrijf de gewijzigde DataFrame naar een nieuw CSV-bestand
    df.to_csv(output_file, index=False)



# Voorbeeldgebruik
input_file = 'product_descriptions.csv'  # CSV-bestand met beschrijvingen
output_file = 'Updated_product_description.csv'  # CSV-bestand waar de gefilterde beschrijvingen worden opgeslagen
text_column = "product_description"  # Naam van de kolom die de beschrijvingen bevat
filter_stopwords_from_csv(input_file, output_file, text_column)

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
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

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



print('begin')
df = pd.read_csv('query_product.csv', encoding='latin1')
print('klaar met 1')
id_list = df['id'].tolist()
print('klaar met 2')
product_uid_list = df['product_uid'].tolist()
print('klaar met 3')
product_title_list = df['product_title'].tolist()
print('klaar met 4')
search_term_list = df['search_term'].tolist()
print('klaar met 5')
relevance_list = df['relevance'].tolist()
print('klaar met lezen')


def count_frequencies(column):   
    lijst = dict()               
    for key in column:          
        #print('item' + str(n) + 'van de' + str(len(column)))
        if key not in lijst:    
            lijst[key] = 1      
        else:                    
            lijst[key] += 1     
    return lijst                



result1 = count_frequencies(product_uid_list)
result2 = count_frequencies(search_term_list)

print("term frequency from productid:", result1)
print("term frequency from searchterms:", result2)






plt.scatter(test_data['all_words_in_title'], y_test, label='Actual', s=weights)
plt.plot(test_data['all_words_in_title'], y_pred, color='red', label='Fitted Line')
plt.xlabel('Do all query terms occur in the product title?')
plt.ylabel('Relevance Score')
plt.title('Linear Regression: Fitted Line')
plt.legend()
plt.show()

