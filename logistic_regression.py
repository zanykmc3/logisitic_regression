import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from sklearn import linear_model
style.use('fivethirtyeight')


def normalize_input(features):
    norm_feat = np.empty_like(features,dtype=float)
    for i in range(0,features.shape[1]):

        total = features[:,i].sum()
        max = np.amax(features[:,i])
        min = np.amin(features[:,i])
        avg = total / len(features[:,i])
        range_col = max - min

        for k in range(0,len(features[:,i])):
            norm_feat[k,i] = (features[k,i] - avg)/range_col

    return norm_feat




def gradient_descent(coefficients, learning_rate, target, features, lambda_const):
    # derivative
    predictions = predict(features, coefficients)
    updates = coefficients

    # update coefficient values
    for i in range(len(coefficients)):
        if(i==0):
            cost_error = (target-predictions)
            feat = np.reshape(features[:, i], (-1, 1))

            cost_deriv_one = cost_error
            cost_deriv_one = cost_deriv_one * feat
            cost_deriv_one = cost_deriv_one.sum()
            change = (learning_rate * (-1/len(target))*cost_deriv_one)

        else:
            cost_error = (target - predictions)
            feat = np.reshape(features[:, i], (-1, 1))

            cost_deriv_one = cost_error
            cost_deriv_one = cost_deriv_one * feat
            cost_deriv_one = cost_deriv_one.sum()
            change = (learning_rate * (-1 / len(target)) * cost_deriv_one) - ((learning_rate*(lambda_const/len(target)))*coefficients[i])


        temp_other = coefficients[i] - change
        updates[i] = temp_other

    coefficients = updates

    return coefficients


def predict(features, coefficients):
    # theta_zero is y-intercept, theta_one is slope
    predictions = np.zeros((features.shape[0],1))
    trans = np.transpose(coefficients)

    for x in range(0,features.shape[0]):
        var = 0
        if(features.shape[1]>1):
            x_val = np.transpose(features[x,:])
            exponent = np.matmul(trans,x_val)
            var = np.exp(-exponent)
            val = 1 / (1+var)
            predictions[x] = val

    return predictions

def cost_function(predictions, target, lambda_const, features, coefficients):
    # cost function to determine performance of theta_zero and theta_one
    cost = 0
    sum_val = 0
    reg_val = 0

    for i in range(0,len(predictions)):
        sum_val = sum_val + target[i]*np.log(predictions[i])+(1-target[i])*np.log(1-predictions[i])
    for j in range(0,features.shape[1]):
        reg_val = reg_val + coefficients[j]**2

    cost = sum_val / (-len(target)) + reg_val * (lambda_const / (-2*len(target)))

    return cost


def main():
    # linear regression equation (univariate)
    learning_rate = 0.0001
    lambda_const = 2
    i=0

    # fake dataset (binary classification)
    dataset = np.array(([1,0],[2,0],[4,1],[8,1]))

    # split data into inputs and outputs (column 0 is inputs, column 1 is outputs)
    features = dataset[:, :dataset.shape[1]-1]
    target = dataset[:, dataset.shape[1]-1:]
    target = np.reshape(target, (-1, 1))

    # create starting coeff array
    num_coeff = features.shape[1]+1
    coefficients = np.zeros((num_coeff,1))
    norm_features = normalize_input(features)
    intercept_col = np.ones((features.shape[0],1))
    n_features = np.column_stack([intercept_col,norm_features])

    cost = 1.0
    while cost > .01:
       coefficients = gradient_descent(coefficients, learning_rate, target, n_features, lambda_const)
       predictions = predict(n_features, coefficients)
       cost = cost_function(predictions, target, lambda_const, features, coefficients)
       i=i+1


    print(predictions)
    print("GD Coefficients:")
    print(coefficients)
    print('\n')



    # sklearn coeff for reference
    regr = linear_model.LogisticRegression()
    regr.fit(features, target)
    pred = regr.predict(features)

    print("Sklearn Coefficients:")
    print(regr.coef_)
    print(regr.intercept_)
    print('\n')


main()