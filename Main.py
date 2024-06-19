import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm

# Specify the paths to your CSV files
csv_path = "query_product.csv"
pd_path = 'Updated_product_description.csv'
training_size = 50000

# Read the CSV files into pandas DataFrames
df = pd.read_csv(csv_path, encoding="latin1")
product_description_text = pd.read_csv(pd_path, encoding='latin1')

# Merge the two datasets on product_uid
merged_data = pd.merge(df, product_description_text, on='product_uid')

# Function to compute if all search terms are in the product name
def all_terms_in_name(row):
    search_terms = set(row['search_term'].lower().split())
    product_name = set(row['product_title'].lower().split())
    return int(search_terms.issubset(product_name))

# Function to compute the percentage of search terms in the product name
def percent_terms_in_name(row):
    search_terms = set(row['search_term'].lower().split())
    product_name = set(row['product_title'].lower().split())
    return len(search_terms.intersection(product_name)) / len(search_terms)

# Function to compute the term frequency of the product
def term_frequency(column):
    freq = column.value_counts().to_dict()
    return column.map(freq)

# Function to compute the percentage of similar words in text
def percent_similar_words(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    return csim[0, 1]

# Function to compute if all search terms are in the product description
def all_terms_in_description(row):
    search_terms = set(row['search_term'].lower().split())
    description = set(row['product_description'].lower().split())
    return int(search_terms.issubset(description))

# Function to compute the percentage of search terms in the description
def percent_terms_in_description(row):
    search_terms = set(row['search_term'].lower().split())
    description = set(row['product_description'].lower().split())
    return len(search_terms.intersection(description)) / len(search_terms)
# Apply functions to create new features
merged_data['all_terms_in_name'] = merged_data.apply(all_terms_in_name, axis=1)
merged_data['percent_terms_in_name'] = merged_data.apply(percent_terms_in_name, axis=1)
merged_data['term_frequency_product'] = term_frequency(merged_data['product_uid'])
merged_data['percent_similar_words_name'] = merged_data.apply(lambda row: percent_similar_words(row['search_term'], row['product_title']), axis=1)
merged_data['all_terms_in_description'] = merged_data.apply(all_terms_in_description, axis=1)
merged_data['percent_terms_in_description'] = merged_data.apply(percent_terms_in_description, axis=1)
merged_data['description_length'] = merged_data['product_description'].apply(lambda x: len(x.split()))

# Default relevance score for missing values
merged_data['relevance_score'] = merged_data['relevance'].fillna(2)

# Define binary target for logistic regression
merged_data['relevance_binary'] = (merged_data['relevance'] > 2.5).astype(int)

# Split the data for logistic regression
X_log = merged_data[['product_uid']].values  # Using product_uid as a placeholder feature
y_log = merged_data['relevance_binary'].values
X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_log, y_log, test_size=training_size, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_log_train, y_log_train)
y_log_pred = log_reg.predict(X_log_test)

# Compute logistic regression metrics
log_accuracy = accuracy_score(y_log_test, y_log_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_log_test, y_log_pred, average='binary')

# Add logistic regression metrics to the dataset
merged_data['log_accuracy'] = log_accuracy
merged_data['log_precision'] = precision
merged_data['log_recall'] = recall
merged_data['log_f1'] = f1

# Print a summary of the dataset
print("Summary of the dataset with new features:")
print(merged_data.describe())

# Define feature matrix X and target vector y for linear regression
features = [
    'all_terms_in_name',
    'percent_terms_in_name',
    'relevance_score',
    'term_frequency_product',
    'percent_similar_words_name',
    'all_terms_in_description',
    'percent_terms_in_description',
    'description_length',
    'log_accuracy',
    'log_precision',
    'log_recall',
    'log_f1'
]
X = merged_data[features]
y = merged_data['relevance']

# Print the first few rows of the feature matrix
print("\nFirst few rows of the feature matrix (X):")
print(X.head())

# Add a constant term to the feature matrix for statsmodels
X = sm.add_constant(X)

# Create a linear regression model using statsmodels
model = sm.OLS(y, X)

# Fit the model to the data
results = model.fit()

# Print the model summary
print("\nLinear Regression Model Summary:")
print(results.summary())

# Plot results for Linear Regression
plt.figure(figsize=(12, 5))
plt.scatter(range(len(y_log_test)), y_log_test, color='blue', label='Actual')
plt.scatter(range(len(y_log_pred)), y_log_pred, color='red', marker='x', label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Sample index')
plt.ylabel('Relevance')
plt.legend()
plt.show()



# different code --------------------------------
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
#filter_stopwords_from_csv(input_file, output_file, text_column)

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
train_data, test_data = train_test_split(df, test_size=training_size, random_state=42)




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

