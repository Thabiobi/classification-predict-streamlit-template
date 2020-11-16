"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

# for plots
import plotly.graph_objects as go
from PIL import Image 

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# cleaning text fucntion
def clean_tweets(message):
	""" This function removes punctions and url's from the message"""
    
	#change all words into lower case
	message = message.lower()
    
	#replace website links
	url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	web = 'url-web'
	message = re.sub(url, web, message)
    
	#removing puntuation and digits
	message  = "".join([char for char in message if char not in string.punctuation])
	message = re.sub('[0-9]+', '', message)
    
	#removing stopwords
	nltk_stopword = nltk.corpus.stopwords.words('english')
	message = ' '.join([item for item in message.split() if item not in nltk_stopword])
    
	return message


# more text cleaning
def cleaning (text):
	"""this function lemmatizes the message"""
    
	text = re.sub(r'[^\w\s]','',text, re.UNICODE)
	text = text.lower()

	lemmatizer = WordNetLemmatizer()
	text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
	text = [lemmatizer.lemmatize(token, "v") for token in text]

	text = " ".join(text)
	text = re.sub('ãââ', '', text)
    
	return text


# sentiment prediction
def statement(sentiment):
	"""" """
	# 'anti' text statement
	if sentiment == -1:
		st.success("tweet does not believe in man-made climate change")
	# 'neutral' text statement
	if sentiment == 0:
		st.success("tweet neither supports nor refutes the belief of man-made climate change")
	# 'pro' text sentiment
	if sentiment == 1:
		st.success("tweet upports the belief of man-made climate change")
	# 'news' text statement
	if sentiment == 2:
		st.success("tweet links to factual news about climate change")
	return 	


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","Information", "EDA", "Prediction", "About us"]
	st.sidebar.subheader("Navigation")
	selection = st.sidebar.selectbox("select a page", options)

	# Building out the "Home" page
	if selection == "Home":
		st.title("Lets Do Some Classification")
		st.markdown("we fight climate change one tweet at a time")
		header_image = Image.open('resources/6-climatechange.jpg')
		st.image(header_image, width=500)
		st.subheader("Fighting Climate Change with AI")

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "EDA" page
	if selection == "EDA":
		st.title("Exploratory Data Analysis")
		if st.checkbox('Sentiment Hashtags'):
			st.markdown("popular Hashtags")
			#sentiment choice
			sentimentchoice = st.radio("Choose and option", ("don't believe in man-made climate change", "neither supports nor refutes the belief of man-made", "do believe in man-made climate change", "are news related to climate change"))
			
			if sentimentchoice == "don't believe in man-made climate change":
				st.image(Image.open('resources/imgs/hashtags on the anti-sentiment.png'), caption='Popular hashtags for negative tweet sentiments', use_column_width=True)
				
			if sentimentchoice == "neither supports nor refutes the belief of man-made":
				st.image(Image.open('resources/imgs/hashtags on the neutral sentiment.png'), caption='Popular hashtags with neutral tweet sentiments')
			
			if sentimentchoice == "do believe in man-made climate change":
				st.image(Image.open('resources/imgs/hashtags on the pro sentiment.png'), caption="Popular hashtags for positive tweet sentiments", use_column_width=True)

			if sentimentchoice == "are news related to climate change": 
				st.image(Image.open('resources/imgs/hashtags on the news setiment.png'), caption="Popular hashtags for factual/news tweet sentiments", use_column_width=True)




	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	
	
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
