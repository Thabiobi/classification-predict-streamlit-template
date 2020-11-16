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
from PIL import Image 

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC




# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
	
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	st.title("Tweet Classification")
	st.subheader("Climate change tweet classification")

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
		if st.checkbox("Tweets per Sentiment"):
			st.markdown("Class Distribution")
			st.markdown("-1: **Negative**(anti-climate change tweet)")
			st.markdown("0: **Neutral**")
			st.markdown("1: **Positive**(pro climate change tweet)")
			st.markdown("2: **Factual/News** related tweet")
			st.image(Image.open('resources/imgs/class distribution.jpeg'), caption='Class Distribution', use_column_width=True)
			st.markdown("Looking at the distribution we are able to see that the data is imbalanced, most tweets are skewed to the Pro sentiment category supporting the belief of man-made climate change.")

		if st.checkbox('Wordcloud Analysis'):
			Wordcloud = st.radio("choose an option", ("Negative popular words", "Neutral popular words", "Positive popular words", "Factual/News popular words"))
			if Wordcloud == "Negative popular words":
				st.image(Image.open('resources/imgs/anti climate change wordcloud.jpeg'), caption=None, use_column_width=True)
			if Wordcloud == "Neutral popular words":
				st.image(Image.open('resources/imgs/neutral wordcloud.jpeg'), caption=None, use_column_width=True)
			if Wordcloud == "Positive popular words":
				st.image(Image.open('resources/imgs/pro wordcloud.jpeg'), caption=None, use_column_width=True)
			if Wordcloud == "Factual/News popular words":
				st.image(Image.open('resources/imgs/news wordcloud.jpeg'), caption=None, use_column_width=True)
			st.markdown("The size of the word indicates the relevance in the tweet.")
			st.markdown("The most popular words in all four classes are climate change, global warming and belief.")
			st.markdown("The pro and anti groups include a number of words that might be expected in each group.")
			st.markdown("In the word clouds there is evidence of noisy text which include words such as https, webside, co and RT. These do not assist us in our classification, rather they add noise, we will have another look at it when the noise have been removed.")
		
		if st.checkbox('Retweets'):
			st.markdown("Twitter allows a user to retweet, or RT another users tweets. Retweeting is a great way for creating trends.")
			st.image(Image.open('resources/imgs/retweets per sentiment class.jpeg'), caption=None, use_column_width=True)
			st.markdown("The Pro sentiment class seems to have more tweets retweeted with over 5000 retweets. while other sentiment classes have less than 2000 retweets. looks like evryone is retweeting positive climate change tweets more than others.")
			
		if st.checkbox('Sentiment Hashtags'):
			st.markdown("popular Hashtags")
			#sentiment choice
			sentimentchoice = st.radio("Choose an option", ("don't believe in man-made climate change", "neither supports nor refutes the belief of man-made", "do believe in man-made climate change", "are news related to climate change"))
			
			if sentimentchoice == "don't believe in man-made climate change":
				st.image(Image.open('resources/imgs/hashtags on the anti-sentiment.png'), caption='Popular hashtags for negative tweet sentiments', use_column_width=True)
				
			if sentimentchoice == "neither supports nor refutes the belief of man-made":
				st.image(Image.open('resources/imgs/hashtags on the neutral sentiment.png'), caption='Popular hashtags with neutral tweet sentiments')
			
			if sentimentchoice == "do believe in man-made climate change":
				st.image(Image.open('resources/imgs/hashtags on the pro sentiment.png'), caption="Popular hashtags for positive tweet sentiments", use_column_width=True)

			if sentimentchoice == "are news related to climate change": 
				st.image(Image.open('resources/imgs/hashtags on the news setiment.png'), caption="Popular hashtags for factual/news tweet sentiments", use_column_width=True)
			st.markdown("We can see that the top 5 hashtags have similar words like Climate, climate change, Trump and Before the flood, although there seem to be words here that are irrelevant, eg. single words like 'tcot' and single letters like 'A', 'p2',")


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Understanding sentiment predictions
		st.image(Image.open('resources/imgs/understanding predictions table.png'), caption=None, use_column_width=True)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		
		modelChoice = st.radio("Choose a model", ("Logistic Regression","Linear SVC"))   
		#if st.button("Classify"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			#prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))
	
		if modelChoice == 'Logistic Regression':
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			predictor = joblib.load(open(os.path.join("resources/logreg_model.pickle"),"rb"))
			prediction = predictor.predict(vect_text)
			#when model has ran succefully, it will print out predictions
			if prediction[0] == -1:
				st.success('tweet has been classified to show non believe in man made climate change')
			elif prediction[0] == 0:
				st.success('tweet has been classified to being belief nor non belief in man made climate change')
			elif prediction[0] == 1:
				st.success('tweet has been classified to show belief in man made climate change')
			else:
				st.success('tweet has ben classified as factual/news about climate change')
			st.success("Text Categorized as: {}".format(prediction))

		if modelChoice == 'Linear SVC':
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			#load pkl file with model and make predictions
			predictor = joblib.load(open(os.path.join("resources/lin_svc_model.pickle"),"rb"))
			prediction = predictor.predict(vect_text)
			#when model has ran succefully, it will print out predictions
			if prediction[0] == -1:
				st.success('tweet has been classified to show non believe in man made climate change')
			elif prediction[0] == 0:
				st.success('tweet has been classified to being belief nor non belief in man made climate change')
			elif prediction[0] == 1:
				st.success('tweet has been classified to show belief in man made climate change')
			else:
				st.success('tweet has ben classified as factual/news about climate change')
			st.success("Tweet Classified as:{}".format(prediction))


	# Building out the "About Us" page
	if selection == "About Us":
		st.image(Image.open('resources/imgs/EDSA_logo.png'),caption=None, use_column_width=True)
		st.subheader("we are Explore Data Science Academy students. we happen to be the only all ladies group from the classifcation sprint:grin:")

		



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
