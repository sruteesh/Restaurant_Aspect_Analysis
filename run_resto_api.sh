# Do git lfs
if [ ! -d "Restaurant_Aspect_Analysis" ]; then
	git lfs clone https://github.com/sruteesh/Restaurant_Aspect_Analysis.git
	cd Restaurant_Aspect_Analysis
else
	git lfs pull
fi;


# Install the requirements
pip3.5 install -r requirements.txt

# Works for tensorflow 1.1.0, use

pip3.5 intstall -U https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl for python3.5

# Download NLTK data using the following script
python3.5 download_nltk_data.py

# Requires Sentiment Models to work
#Get the models from sentiment neuron repo and copy them to the model folder inside sentiment_neuron

cd sentiment_neuron
if [ ! -d "model" ]; then
	tar -xf model.tar.gz
	cd ..
fi;

# Hosting the API
# Change the IP in the file api_v2.py to the IP (<hosted_ip>) you want to host and then do
python3.5 api_v2.py