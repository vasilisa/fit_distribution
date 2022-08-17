# Code for custom code recipe fitdistribution (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import *
from dataiku import pandasutils as pdu
import pandas as pd
import statsmodels.api as sm
import numpy as np


# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_dataset_name = get_input_names_for_role('main')[0]
# The dataset objects themselves can then be created like this:
input_dataset = dataiku.Dataset(input_dataset_name)

# For outputs, the process is the same:
output_dataset_name    = get_output_names_for_role('main_output')[0]
output_dataset         = dataiku.Dataset(output_dataset_name)


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
dist_name = get_recipe_config().get('distribution')

### ERROR CHECKING OF USER INPUTS ###

# Check that x, y and z axis correspond to column names
names = ["NegativeBinomial","Poisson","Tweedie"]
if (dist_name not in names):
    raise KeyError("Unknown distribution")
    
    
# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the data
# dataset = dataiku.Dataset("loss_data_prepared_prepared")
df   = input_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the project variable with the information about the distribution 
# dist_name = (dataiku.get_custom_variables()["distribution"])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
endog  = df['PurePremium'].values
exog   = df.drop('PurePremium', axis=1).values

# add a constnat
exog = sm.add_constant(exog)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# specify a model
# set up the model based on the project variable 

if(dist_name=="NegativeBinomial"):
    family = sm.families.NegativeBinomial()
elif(dist_name=="Poisson"):
    family = sm.families.Poisson()
elif(dist_name=="Tweedie"):
    family = sm.families.Tweedie()
else:
    raise NameError('Unknown Distribution name')
    
    
model   = sm.GLM(endog, exog, family=family)
results = model.fit()
print(results.summary())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y_hat = model.predict(results.params)
df["loss_predicted"] = y_hat

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
output_dataset.write_with_schema(df)