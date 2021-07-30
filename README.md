# PatentStuff
## Checkpoints
This contains the checkpoint files that are used to save our model in tensorflow.
## Class Descriptions
This contains the dictionaries that contain scraped descriptions of each patent classification, or are randomly sampled patents from each patent classification.
## Demo

The first step is to clone this github repo:
```git clone git@github.com:juliusgeo/PatentStuff.git```
Then you need to make sure that you have the following installed (versions included are the only platform that this demo has been tested on):
- flask (1.1.2)
- flask_cors (3.0.10)
- numpy (1.18.5)
- gensim (3.8.3)
- nltk (3.5)
- pandas (1.1.2)
- tensorflow (2.3.0)


Then simply navigate your terminal to ```demo/server``` and then ```flask run``` and it should start the demo server listening on port 5000 (if it does not you need to change line 2069 in ```demo/client/index.html```to reflect a modified port value). 
Once the demo server is started, simply open the file ```demo/client/index.html``` in your web browser and enjoy the demo!
## Legacy
This contains all of the legacy code used for previous iterations of this project.
## Notebooks
This contains all of our notebooks which you can view on github. 
