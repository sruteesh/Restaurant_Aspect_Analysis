# Restaurant_Analysis

Needs python3.5

# Install the requirements
sudo pip3.5 install -r requirements.txt

# Download NLTK data
Open a Python console and do the following:
```
>>> import nltk
>>> nltk.download()
showing info http://nltk.github.com/nltk_data/
```
In the GUI window that opens simply press the 'Download' button to download all corpora or go to the 'Corpora' tab and only download the ones you need/want.

# Requires Sentiment Models to work
Get the models from sentiment neuron repo and copy them to the model folder inside sentiment_neuron
```
git clone https://github.com/openai/generating-reviews-discovering-sentiment.git
sudo cp -r generating-reviews-discovering-sentiment/model Restaurant_Aspect_Analysis/sentiment_neuron/model
```
# Hosting the API
Change the IP in the file api_v2.py to the IP (<hosted_ip>) you want to host and then do
```
python3.5 api_v2.py
```
the API should be running at your <hosted_ip>

# Example Request
```
curl -X POST "http://<hosted_ip>:8085/aspects/" -H "Content-Type:application/json" -d '{"text":["Beautiful views of Munnar","Nice location","Perfect place!"],"domain":"hotels"}'
```
