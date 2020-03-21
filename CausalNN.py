import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class DNN_reg(BaseEstimator,RegressorMixin):
    def __init__(self,number_of_neurons,shape):
        self.__number_of_neurons = number_of_neurons
        self.__feature_cols = [tf.feature_column.numeric_column("X", shape=shape)]
        self.__dn_reg = tf.estimator.DNNRegressor(number_of_neurons,feature_columns=self.__feature_cols)

    def fit(self,X,y):
        self.__dn_reg = tf.estimator.DNNRegressor(self.__number_of_neurons,feature_columns=self.__feature_cols)
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"X": X}, y=y, num_epochs=30, batch_size=50, shuffle=True)
        self.__dn_reg.train(input_fn=input_fn)
    def predict(self,X):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"X": X}, y=None, shuffle=False)
        answers = []
        for i in self.__dn_reg.predict(input_fn=test_input_fn):
          answers.append(i['predictions'][0])
        return np.array(answers)



if __name__ == '__main__':
    dnn = DNN_reg([100,20],12)
    print("Ok")