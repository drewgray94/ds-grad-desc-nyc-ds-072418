
# Gradient of Matrices

Recall that we previously fit a regression model y=mx+b to predict domestic gross sales from a movies budget.  
Expanding this example, we can imagine being in charge of managing a large production and we might want to figure out what other factors such as the actors, director, genre, running length or other features were most predictive of the movie's success in the box office.

Here gradient descent is slightly more complicated as we're dealing with the multivariate case. As a result, we can take the derivative (or gradient) with respect to different variables. The underlying intuition is that we want to move in the direction of the steepest descent in hopes that we can find the global minimum. We then do this through a series of successive steepest steps forward until we are satisfied with the result.


### 1. Import the data. It's stored in a file called 'movie_data_detailed_with_ols.xlsx'.


```python
import pandas as pd
%matplotlib inline
```


```python
df = None #Your code here
```

### 2. Scatter Plot <a id="scatter"></a>  
Create a Scatter Plot of the budget and  Domestic Gross (domgross) along with the model column's predictions.


```python
#Your Code here
```

As you can see, we get far better results then a simple straight line! Let's start to further investigate how this happens under the hood.

Gradient Descent works by finding the steepest slope downhill and taking a step in that direction.  
From there, you then iterate and take another step in the steepest direction from that point.  
This continues on until you converge on a minimum solution.

To write our gradient descent algorithm, we'll need a few things:
- An array of [coefficient] weights for the polynomial model. 
- An error function to evaluate the current iteration's model.
- A specified step size coefficient
- A precision parameter

## 3. Define the problem.
Create an X (multidimensional) and Y variable.


```python
X = None #Your code goes here
y = None #Your code goes here
```

### 4.  Predicting!  
**Create a function predict(X,w) that takes in a matrix of data and a vector of coefficient weights and returns a vector of predicted values.**

$x_1\bullet w_1 + x_2\bullet w_2 + x_3\bullet w_3 + ... = y_1+y_2+y_3+...$


```python
def predict(X, w, b=False):
    """Takes in a matrix X of features, and w,
    a vector of coefficients for how much each feature is weighed in the current model.
    w[-1] is the constant additive value by default. To remove this feature specify b=False."""
    return None
```

### 5. Write an error function to calculate the residual sum of squares for a given model.  
Your function should take in 3 inputs:
 * a list of x values
 * a list of y values (corresponding to the x values passed)
 * a list of $\hat{y}$ values produced by the model (corresponding to the x values passed)


```python
def rss(y, y_hat):
    #Your code goes here
    return None

#rss(df['domgross'], df['Model'])
```

### 6. Gradient Descent, take 2!

Here's a pseudocode outline:  

* Initialize a vector of weights for your model.
* Then calculate the error for this model using the RSS.
* From there, calculate the gradient at this point. 
* Use the gradient along with your step size coefficient to update your vector of weights.
* Iterate! Continue this process until the model converges to an arbitrary precison.  

Okay then, let's have at it!  
**Define a function grad_desc() that takes in 7 parameters (shown below) and returns a vector of optimized coefficients for the model.**
* X #Matrix of Data
* Y #Variable interested in modelling
* Precision
* Max Iterations
* Initialization Vector (Initial weights to start the algorithm at)
* An error/loss function (Ultimately this is how we measure the effectiveness of our optimization)
* A prediction function (This function should take your function weights along with the original data and return predictions; see the previous question)

**Have your function print updated weights every so often so you can preview what's going on under the hood.**


```python
import numpy as np
```


```python
def grad_desc(x, y, precision, max_iters, w, rss, predict):
    previous_step_size = 1 #Arbitrary
    iteration = 0 #iteration counter
#    while :
   
        #Calculate Nearby Points
       
        #Calculate the Gradient
        #Look at weights surrounding our current position.
        
        #Calculate the RSS error for this surrounding weights-space.
        
        
        #Move opposite the gradient by some step size
        
    print("Gradient descent converged. Local minimum identified at:")
    print('Iteration {} \nCurrent weights:\n{} \nRSS Produced: {}'.format(iteration, w, rss(y, predict(X, w))))
    return None
```

### 7. Try Running Your New Function
Use your new function to train a regression model.


```python
#Run your function!
```

## 8. Use the coefficient weights from your model and create a new column 'Predictions' using your predict function.


```python
#Your code goes here
```

### 9. Scatter Plot <a id="scatter"></a>  
Create a Scatter Plot of the budget and  Domestic Gross (domgross) along with the model column's predictions and those of your new model.


```python
#Your code goes here
```
