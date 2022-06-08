
![1](https://user-images.githubusercontent.com/102172479/172487815-8c9955eb-f3da-44c0-b29a-5401d9424e2a.jpeg)

# **Predicting Home Value using Zillow data**



**PROJECT DESCRIPTION**

- The Zillow Data Science team was asked to predict the values of single unit properties. By analyze property attributes in relation to their 2017 assessed tax value, I have develop a model for predicting that value based on those attributes, and leave with recommendations for how to improve future predictions. Since Zillow's estimate of home value is one of the primary drivers of website traffic, having a reliable estimate is paramount. Any improvement I can make on the previous model will help us better position ourselfs in the real estate technology marketplace, keeping us as the number one name in real estate technology.

**GOALS**

- My goal is to create a reliable model for predicting property values. Using in depth data analysis of Zillows property data from 2017. I will use exploratory analysis techniques to identify the key drivers of the assessed tax value for those properties, then use machine learning algorithms to create a model capable of predicting tax values based on features of the property.

**Questions to think about:**

- Is there a significant correlation between the number of bathrooms in a home and it's value?

- Is there a significant correlation between the number of bedrooms in a home and it's value?

- Is there a significant correlation between the square footage of a home and it's value?

- Which has a greater effect on home values, number of bedrooms, number of bathrooms, or square footage?

- A three bedrooms and two bathrooms home is often considered ideal for many buyers. Could this be a useful categorical feature?

**DATA DICTIONARY**

| Feature    | Definition                                          | Data\_Type |
| ---------- | --------------------------------------------------- | ---------- |
| tax\_value | Unique parcel identifier                            | float64    |
| bedrooms   | Number of bedrooms                                  | float64    |
| bathrooms  | Number of bathrooms (includes half baths)           | float64    |
| sqft       | Property structure square footage                   | float64    |
| age        | Age of the structure (from yearbuilt todate)        | float64    |
| fips       | County associated with property                     | float64    |


| Target               | Definition            | Data Type |
| -------------------- | --------------------- | --------- |
| Assessed\_Value\_usd | Value of the property | float64   |


| FIPS codes | Description        |
| ---------- | ------------------ |
| 6037       | Los Angeles County |
| 6059       | Orange County      |
| 6111       | Ventura County     |

- **What are FIPS codes?**

Answer:
Federal Information Processing Standards (FIPS), now known as Federal Information Processing Series, are numeric codes assigned by the National Institute of Standards and Technology (NIST). Typically, FIPS codes deal with US states and counties. US states are identified by a 2-digit number, while US counties are identified by a 3-digit number. For example, a FIPS code of 06071, represents California -06 and San Bernardino County -071.


# PROJECT PLANNIG

I used Zillow.csv data

## Acquire

- Acquire data from the Codeup Database using my own function to automate this process. This function is saved in acquire.py file.

## Prepare

- Clean and prepare data for preparation step. Split dataset into train, validate, test. Separate target from features and scale the selected features. Create a function to automate the process. The function is saved in a prepare.py module.

## Exacute66

This fuctions is a combination of my explore, evaluate and model

- #### Explore

    - Visualize all combinations of variables.Define two hypotheses, set an alpha, run the statistical tests needed, document findings and takeaways.

- #### Evaluate

    - This function takes in a set of independent variable values, the corresponding set of dependent variable values, and a set of predictions for the dependent variable. it then displays a plot of residuals for the given values. 

- #### Model

    - Extablishing and evaluating a baseline model.
    - Document various algorithms and/or hyperparameters you tried along with the evaluation code and results.
    - Evaluate the models using the standard techniques: computing the evaluation metrics (SSE, RMSE, and/or MSE)
    - Choose the model that performs the best.
    - Evaluate the best model (only one) on the test dataset.

# AUDIENCE

- Managers and Excutive of Zillow.

# INITIAL IDEAS/ HYPOTHESES STATED

- 𝐻𝑜 : There is no difference in the average of assessed_value_usd for the properties with 3 bedrooms vs 2 bedrooms
- 𝐻𝑎 : There is significant difference in the average of assessed_value_usd for the properties with 3 bedrooms vs 2 bedrooms

# INSTRUCTIONS FOR RECREATING PROJECT

 - Read this README.md
 - Create a env.py file that has (user, host, password) in order to get the database
 - Download the aquire.py, prepare.py, execute66.py and Final_Report.ipynb into your working directory
 - Run the Final_Report.ipynb notebook

# DELIVER:

- A report in the form of a jupter notebook presentation. (The presentation summarizes my findings about the drivers of the single unit property values.)

- A github repository containing my work.

- README file contains project description and goals, data dictionary, project planning, initial ideas/hypotheses, instructions to recreate project.

- Individual modules, .py files, that hold your functions to acquire, prepare and execute66(explore,execute,model) your data.
