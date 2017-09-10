# Restaurant_Analysis

Needs python3.5

# Install the requirements
sudo pip3.5 install -r requirements

# Requires Sentiment Models to work
Get the models from sentiment neuron repo and copy them to the model folder inside sentiment_neuron
```
git clone https://github.com/openai/generating-reviews-discovering-sentiment.git
sudo cp -r generating-reviews-discovering-sentiment/model Restaurant_Analysis/sentiment_neuron/model
```

# Example Request
```
curl -X POST "http://<hosted_ip>:8085/aspects/" -H "Content-Type:application/json" -d '{"text":["Beautiful views of Munnar","Nice location","Perfect place!"],"domain":"hotels"}'
```
