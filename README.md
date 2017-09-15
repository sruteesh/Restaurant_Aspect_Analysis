# Resturant Aspect Analysis
Get aspects and sentiment of the reviews

# Clone the Repo using git lfs

# Install git lfs if not already installed
```
wget https://github.com/git-lfs/git-lfs/releases/download/v2.3.0/git-lfs-linux-amd64-2.3.0.tar.gz
tar -xf git-lfs*
cd git-lfs*
./install.sh 
```
If you intend to use a pre-trained model you should clone the repository with the lfs plugin

```
git lfs clone https://github.com/sruteesh/Restaurant_Aspect_Analysis.git
cd Restaurant_Aspect_Analysis
```  
If you have already cloned the repository without lfs, you can download the missing files with :

`git lfs pull`

# Install the requirements
`pip3.5 install -r requirements.txt`

# Works for tensorflow 1.1.0, (installs CPU version)

`pip3.5 install -U https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl` for python3.5

# GPU version
`#pip3.5 install - U https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl`

# Download NLTK data using the following script
`python3.5 download_nltk_data.py`

# Requires Sentiment Models to work
# Uncompress the sentiment models 
```
cd sentiment_neuron
if [ ! -d "model" ]; then
	tar -xf model.tar.gz
	cd ..
fi;
```

# Hosting the API
# Change the IP in the file api_v2.py to the IP (<hosted_ip>) you want to host and then do
`python3.5 api_v2.py`

# Run the following script to skip the above steps
`./run_resto_api.sh`

# Example Request to the API
```
curl -X POST "http://<hosted_ip>:8085/aspects/" -H "Content-Type:application/json" -d '{"text":["Beautiful views of Munnar","Nice location","Perfect place!"],"domain":"hotels"}'
```

