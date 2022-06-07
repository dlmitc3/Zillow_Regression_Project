import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt

 #=========================EXPLORE==================================#


def correlation_test(data_for_category_1, data_for_category_2, alpha=.05):
    '''
    This function takes in data for two variables and performs a pearsons r statistitical test for correlation. 
    It outputs to the console values for r and p and compares them to a given alpha value, then outputs to the 
    console whether or not to reject the null hypothesis based on that comparison. 
    '''
    # display hypotheses
    print(f'H0: There is no linear correlation between {data_for_category_1.name} and {data_for_category_2.name}.')
    print(f'H1: There is a linear correlation between {data_for_category_1.name} and {data_for_category_2.name}.')
    # conduct the stats test and store values for p and r
    r, p = stats.pearsonr(data_for_category_1, data_for_category_2)
    # display the p and r values
    print('\nr = ', round(r, 2))
    print('p = ', round(p, 3))
    # compare p to alpha, display whether to reject the null hypothesis
    if p < alpha:
        print('\nReject H0')
    else:
        print('\nFail to Reject H0')

def value_by_bathrooms(train):
    '''
    This function takes in the zillow train sample and uses seaborn to create box plots of 
    tax_value for each number of bathrooms that exists in the sample. 
    '''
    # establish figure size
    plt.figure(figsize=(10,8))
    # create the plot
    sns.boxplot(data=train,
                  x='bathrooms',
                  y='tax_value')
    # establish title
    plt.title('Value by Number of Bathrooms')
    # display the plot
    plt.show()

def sqft_vs_value(train):
    '''
    This function takes in the zillow train sample and uses seaborn to create a scatter plot
    of tax_value vs square feet, with a best-fit regression line. 
    '''
    # create the plot
    sns.lmplot(x='sqft', 
               y='tax_value', 
               data=train.sample(1000, random_state=42), 
               line_kws={'color': 'red'})
    # establish the title
    plt.title('Value by Square Footage')
    # display the plot
    plt.show()

def value_by_bedrooms(train):
    '''
    This function takes in the zillow train sample and uses seaborn to create box plots of the 
    distribution of tax_value for each number of bedrooms that exists in the sample. 
    '''
    # establish the figure size
    plt.figure(figsize=(10,8))
    # create the plot
    sns.boxplot(data=train,
                  x='bedrooms',
                  y='tax_value')
    # establish plot title
    plt.title('Value by Number of Bedrooms')
    # display the plot
    plt.show() 

def value_correlations(train):
    '''
    This functino takes in the zillow train sample and uses pandas and seaborn to create a
    ordered list and heatmap of the correlations between the various quantitative feeatures and the target. 
    '''
    # create a dataframe of correlation values, sorted in descending order
    corr = pd.DataFrame(train.corr().abs().tax_value).sort_values(by='tax_value', ascending=False)
    # rename the correlation column
    corr.columns = ['correlation (abs)']
    # establish figure size
    plt.figure(figsize=(10,8))
    # creat the heatmap using the correlation dataframe created above
    sns.heatmap(corr, annot=True)
    # establish a plot title
    plt.title('Features\' Correlation with Value')
    # display the plot
    plt.show()


#============================EVALUATE==============================#


def plot_residuals(x, y, y_hat):
    '''
    this function takes in a set of independent variable values, the corresponding set of 
    dependent variable values, and a set of predictions for the dependent variable. it then displays a plot
    of residuals for the given values. 
    '''
    plt.scatter(x, y - y_hat)
    plt.axhline(y = 0, ls = ':')
    plt.show()

def regression_errors(y, y_hat):
     
    SSE = ((y - y_hat) ** 2).sum()
    TSS = SSE_baseline = ((y.mean() - y_hat) ** 2).sum()
    ESS = TSS - SSE
    MSE = mean_squared_error(y, y_hat)
    RMSE = sqrt(MSE)
    
    print(f'SSE: {SSE}')
    print(f'ESS: {ESS}')
    print(f'TSS: {TSS}')
    print(f'MSE: {MSE}')
    print(f'RMSE: {RMSE}')
     
        
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    
    SSE_baseline = ((y - y.mean()) ** 2).sum()
    MSE_baseline = SSE_baseline / len(y)
    RMSE_baseline = sqrt(MSE_baseline)
        
    print(f'Baseline SSE: {SSE_baseline}')
    print(f'Baseline MSE: {MSE_baseline}')
    print(f'Baseline RMSE: {RMSE_baseline}')

    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, y_hat):
    
    SSE = ((y - y_hat) ** 2).sum()
    SSE_baseline = ((y - y.mean()) ** 2).sum()

    if SSE < SSE_baseline:
        return True
    else:
        return False



#===========================MODEL==================================#

import pandas as pd
import sklearn as sk
from math import sqrt
from sklearn.linear_model import LinearRegression, LassoLars 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def determine_regression_baseline(train, target):
    '''
    This function takes in a train sample and a continuous target variable label and 
    determines whether the mean or median performs better as a baseline prediction. 
    '''
    # create empty dataframe for storing prediction results
    results = pd.DataFrame(index=train.index)
    # assign actual values for the target variable
    results['actual'] = train[target]
    # assign a baseline using mean
    results['baseline_mean'] = train[target].mean()
    # assign a baseline using median
    results['baseline_median']= train[target].median()
    
    # get RMSE values for each potential baseline
    RMSE_baseline_mean = sqrt(sk.metrics.mean_squared_error(results.actual, results.baseline_mean))
    RMSE_baseline_median = sqrt(sk.metrics.mean_squared_error(results.actual, results.baseline_median))
    
    # compare the two RMSE values; drop the lowest performer and assign the highest performer to baseline variable
    if RMSE_baseline_median < RMSE_baseline_mean:
        results = results.drop(columns='baseline_mean')
        results['RMSE_baseline'] = RMSE_baseline_median
        baseline = 'median'
    else:
        results = results.drop(columns='baseline_median')
        results['RMSE_baseline'] = RMSE_baseline_mean
        baseline = 'mean'
    # print the results
    print(f'The highest performing baseline is the {baseline} target value.')

def run_baseline(train,
                 validate,
                 target,
                 model_number,
                 model_info,
                 model_results):
    '''
    This function performs the operations required for storing information about baseline performance for
    a regression model.
    '''

    y_train = train[target]
    y_validate = validate[target]

    # identify model number
    model_number = 'baseline'
    #identify model type
    model_type = 'baseline'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)


    # establish baseline predictions for train sample
    y_pred = baseline_pred = pd.Series(train[target].mean()).repeat(len(train))

    # create a dictionary containing information about the baseline's performance on train
    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
    # append that dictionary to the model_results dataframe
    model_results = model_results.append(dct, ignore_index=True)


    # establish baseline predictions for validate sample
    y_pred = baseline_pred = pd.Series(validate[target].mean()).repeat(len(validate))

    # create a dictionary containing information about the baseline's performance on validate
    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'RMSE',
           'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
    # append that dictionary to the model results dataframe
    model_results = model_results.append(dct, ignore_index=True)
    
    # reset the model_number to 0 to be changed in each subsequent modeling iteration
    model_number = 0
    
    return model_number, model_info, model_results

def run_OLS(train, validate, target, model_number, model_info, model_results):
    '''
    This function creates various OLS regression models and stores infomation about their performance
    for later evaluation.
    '''

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age']
    features3 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111']
    features4 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft']
    features5 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools']
    features6 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools', 'scaled_lot_sqft']
    feature_combos = [features1, features2, features3, features4, features5, features6]

    for features in feature_combos:

        # establish model number
        model_number += 1

        #establsh model type
        model_type = 'OLS linear regression'

        # store info about the model

        # create a dictionary containing the features and hyperparamters used in this model instance
        dct = {'model_number': model_number,
               'model_type': model_type,
               'features': features}
        # append that dictionary to the model_info dataframe
        model_info = model_info.append(dct, ignore_index=True)

        #split the samples into x and y
        x_train = train[features]
        y_train = train[target]

        x_validate = validate[features]
        y_validate = validate[target]

        # create the model object and fit to the training sample
        linreg = LinearRegression(normalize=True).fit(x_train, y_train)

        # make predictions for the training sample
        y_pred = linreg.predict(x_train)
        sample_type = 'train'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)

        # make predictions for the validate sample
        y_pred = linreg.predict(x_validate)
        sample_type = 'validate'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results

def run_LassoLars(train, validate, target, model_number, model_info, model_results):
    '''
    This function creates various LASSO + LARS regression models and stores infomation about their performance
    for later evaluation.
    '''

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age']
    features3 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111']
    features4 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'garage_sqft']
    features5 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools']
    features6 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111',
                 'scaled_garage_sqft', 'scaled_pools', 'scaled_lot_sqft']
    feature_combos = [features1, features2, features3, features4, features5, features6]

    # set alpha hyperparameter
    alpha = 1

    for features in feature_combos:


        # establish model number
        model_number += 1

        #establsh model type
        model_type = 'LASSO + LARS'

        # store info about the model

        # create a dictionary containing the features and hyperparameters used in this model instance
        dct = {'model_number': model_number,
               'model_type': model_type,
               'features': features,
               'alpha': alpha}
        # append that dictionary to the model_info dataframe
        model_info = model_info.append(dct, ignore_index=True)

        #split the samples into x and y
        x_train = train[features]
        y_train = train[target]

        x_validate = validate[features]
        y_validate = validate[target]

        # create the model object and fit to the training sample
        linreg = LassoLars(alpha=alpha).fit(x_train, y_train)

        # make predictions for the training sample
        y_pred = linreg.predict(x_train)
        sample_type = 'train'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)

        # make predictions for the validate sample
        y_pred = linreg.predict(x_validate)
        sample_type = 'validate'

        # store information about model performance
        # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
        dct = {'model_number': model_number, 
               'sample_type': sample_type, 
               'metric_type': 'RMSE',
               'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
        model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results

def run_PolyReg(train, validate, target, model_number, model_info, model_results):
    '''
    This function creates various Polynomial Regression models and stores infomation about their performance
    for later evaluation.
    '''

    features1 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft']
    features2 = ['scaled_bedrooms', 'scaled_bathrooms', 'scaled_sqft', 'scaled_age', 
                 'enc_fips_06059', 'enc_fips_06111', 'scaled_garage_sqft', 'scaled_pools', 'scaled_lot_sqft']
    feature_combos = [features1, features2]

    for features in feature_combos:
        for degree in range(2,6):

            # establish model number
            model_number += 1

            #establsh model type
            model_type = 'Polynomial Regression'

            # store info about the model

            # create a dictionary containing the features and hyperparameters used in this model instance
            dct = {'model_number': model_number,
                   'model_type': model_type,
                   'features': features,
                   'degree': degree}
            # append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            #split the samples into x and y
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]

            # create a polynomial features object
            pf = PolynomialFeatures(degree=degree)

            # fit and transform the data
            x_train_poly = pf.fit_transform(x_train)
            x_validate_poly = pf.fit_transform(x_validate)

            # create the model object and fit to the training sample
            linreg = LinearRegression().fit(x_train_poly, y_train)

            # make predictions for the training sample
            y_pred = linreg.predict(x_train_poly)
            sample_type = 'train'

            # store information about model performance
            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': sample_type, 
                'metric_type': 'RMSE',
                'score': sqrt(sk.metrics.mean_squared_error(y_train, y_pred))}
            model_results = model_results.append(dct, ignore_index=True)

            # make predictions for the validate sample
            y_pred = linreg.predict(x_validate_poly)
            sample_type = 'validate'

            # store information about model performance
            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': sample_type, 
                'metric_type': 'RMSE',
                'score': sqrt(sk.metrics.mean_squared_error(y_validate, y_pred))}
            model_results = model_results.append(dct, ignore_index=True)
        
    return model_number, model_info, model_results


def final_test_model18(train, test, target):
    '''
    This function recreates the regression model previously found to perform with the smallest error, then 
    evaluates that model on the test sample and prints the resulting RMSE.
    '''
    # establish x-train with the appropriate set of features
    x_train = train[['scaled_bedrooms',
                    'scaled_bathrooms',
                    'scaled_sqft',
                    'scaled_age',
                    'enc_fips_06059',
                    'enc_fips_06111',
                    'scaled_garage_sqft',
                    'scaled_pools',
                    'scaled_lot_sqft']]
    # establish y train as the target values
    y_train = train[target]
    
    # establish x-test with the appropriate set of features
    x_test = test[['scaled_bedrooms',
                    'scaled_bathrooms',
                    'scaled_sqft',
                    'scaled_age',
                    'enc_fips_06059',
                    'enc_fips_06111',
                    'scaled_garage_sqft',
                    'scaled_pools',
                    'scaled_lot_sqft']]
    # establish y_test as the target values
    y_test = test[target]

    # create a polynomial features object
    pf = PolynomialFeatures(degree=4)

    # fit and transform x_train and x_test
    x_train_poly = pf.fit_transform(x_train)
    x_test_poly = pf.fit_transform(x_test)
    
    # create and fit the model on the training data
    linreg = LinearRegression(normalize=True).fit(x_train_poly, y_train)
    # create predictions on the test sample
    y_pred = linreg.predict(x_test_poly)
    # compute the rmse performance metric
    RMSE = sqrt(mean_squared_error(y_test, y_pred))
    #display the results
    print('Model 3 RMSE: ', '${:,.2f}'.format(RMSE))

def display_model_results(model_results):
    '''
    This function takes in the model_results dataframe created in the Model stage of the 
    Zillow Regression analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index grouped by metric_type then sample_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)

def get_best_model_results(model_results, n_models=3):
    '''
    This function takes in the model_results dataframe created in the Modeling stage of the 
    TelCo Churn analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function identifies the {n_models} models with the highest scores for the given metric
    type, as measured on the validate sample.
    It returns a dataframe of information about those models' performance in the tidy data format
    (as described above). 
    The resulting dataframe can be fed into the display_model_results function for convenient display formatting.
    '''
    # create an array of model numbers for the best performing models
    # by filtering the model_results dataframe for only validate scores
    best_models = (model_results[(model_results.sample_type == 'validate')]
                                                 # sort by score value in ascending order
                                                 .sort_values(by='score', 
                                                              ascending=True)
                                                 # take only the model number for the top n_models
                                                 .head(n_models).model_number
                                                 # and take only the values from the resulting dataframe as an array
                                                 .values)
    # create a dataframe of model_results for the models identified above
    # by filtering the model_results dataframe for only the model_numbers in the best_models array
    # TODO: make this so that it will return n_models, rather than only 3 models
    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results