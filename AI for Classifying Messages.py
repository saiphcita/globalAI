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

#train model
tokenizer.fit_on_texts(new_train_data)
train_data = tokenizer.texts_to_matrix(new_train_data)
test_data = tokenizer.texts_to_matrix(new_train_data)
