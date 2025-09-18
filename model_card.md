# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
### Who made it?
I, Nolan Mellott, created and chose the model for this project.
### What Type of Model is it?
I decided to go with a Random Forest Classifier model for this project as it is a nice overall performer model that
does not require intricate tuning to prove effective and I have the most experience with this model.
### Training Details
For random forest classifiers tuning n_estimators and max_depth are the easiest parameters to tune by hand. I simply
did a "binary search" to find an optimal set of values.
## Intended Use
This model should be used to infer one's salary based off a handful of demographical attributes. The model can be used
when an organization wants to understand what influences salary for potential (and current) customers.
## Data
### Dataset Origin
The data was obtained from
### Training/Evaluation Data
I split the data with a 75-25 split to break it into training and test datasets. I used OneHotEncoding for categorical 
features and label binarizering on the labels. I did not stratify the data features.

## Metrics
The metrics I used were precision, recall, and F1 score. My model scored around 0.74, 0.62, and 0.67 in each metric 
respectfully.

## Ethical Considerations
One thing to note with this dataset is that it has a large bias towards lower salaries due to the bias in economics in
the US.

## Caveats and Recommendations
One large caveat to this model is that I did not find the best performing model nor did I do any real hyperparameter 
tuning so the model is not as optimized as it could be.