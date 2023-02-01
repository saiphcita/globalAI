import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load the CSV file into a pandas dataframe
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1dFdlvgmyXfN3SriVn5Byv_BNtyroICxdgrQKBzuMA1U/export?format=csv&id=1dFdlvgmyXfN3SriVn5Byv_BNtyroICxdgrQKBzuMA1U&gid=0")


# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['message'], df['sentiment'], test_size=0.2)

# Preprocess the labels by encoding them as integer values
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
test_labels = encoder.transform(test_labels)

# Tokenize the messages using the TensorFlow Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
new_train_data=[]
#clean data
for d in train_data:
  d=str(d)
  new_train_data.append(d)

new_test_data=[]
for d in test_data:
  d=str(d)
  new_test_data.append(d)


tokenizer.fit_on_texts(new_train_data)
train_data = tokenizer.texts_to_matrix(new_train_data)
test_data = tokenizer.texts_to_matrix(new_test_data)
