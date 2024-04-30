#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


# In[2]:


# Helper functions

def avg_col(col):
    return sum(col)/len(col)

def std_col(col):
    mean = avg_col(col)
    variance = sum([((x - mean) ** 2) for x in col]) / len(col)
    std = variance ** 0.5
    return std


# In[3]:


# Functions for normalization and scaling

def shift_scale(series):
    return (series - min(series))/(max(series))

def zero_mean_unit_variance(series):
    return (series - avg_col(series)) / (std_col(series))


# In[4]:


def vis_distribution_housing(dataset):
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k in dataset.keys():
        sns.boxplot(y=k, data=dataset, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[ ]:


# Functions for machine learning tasks


# In[5]:


def train_test_split(data, test_size=0.3):
    """
    My implementation of train_test_split without using sk-learn
    default test size = 30%
    """

    # Shuffle the data
    data = data.sample(frac = 1)
    
    # Find splitting index
    split_index = int(test_size * len(data))
    
    # Split the data
    train_data = data[split_index:]
    test_data = data[:split_index]
    
    X_train = np.array(train_data.iloc[:,:-1])
    X_test = np.array(test_data.iloc[:,:-1])
    
    Y_train = np.array(train_data.iloc[:,-1]).reshape(len(train_data.iloc[:,-1]),1)
    Y_test = np.array(test_data.iloc[:,-1]).reshape(len(test_data.iloc[:,-1]),1)
    
    
    return X_train, X_test, Y_train, Y_test


# In[7]:


def reading_spam_data():
    spam_column = []

    with open("spambase.names","r") as f:
        lines = f.readlines()
        for line in lines :
            if line[0] != "|":
                if line[0] != "\n":
                    if line[0] != "1":
                        spam_column.append(line.split(":")[0])

    spam_column.append("target")

    spam_df = pd.read_csv("spambase.data", names=spam_column)

    return spam_df


# In[8]:


spam_dataset = reading_spam_data()
spam_dataset.head()


# In[10]:


housing_train.head()


# In[11]:


housing_test.head()


# In[ ]:


# Normalizing the target column can be useful in regression tasks where the target variable has a large range of values, 
# as it can help the algorithm converge faster and improve its performance. 

# However, in classification tasks, normalizing the target column may not be necessary or even harmful, 
# as it could change the interpretation of the class labels. 


# In[ ]:


# vis_distribution_housing(housing_train_df)
# vis_distribution_housing(housing_test)


# In[178]:


vis_distribution_housing(housing_test)


# In[12]:


def accuracy_score(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    res = correct / len(y_true)
    return res


# In[132]:


# RMSE
def root_of_mean_squared_error(y_true, y_pred):
    val = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(val)

# MSE
def mean_squared_error(y_true, y_pred):
    val = np.mean((y_true - y_pred) ** 2)
    return val


# In[256]:


def k_fold_cv(dataset, k, model):
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    
    n = X.shape[0] # number of rows
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    X, y = X.loc[indices], y[indices]
    fold_size = n // k
    
    scores = []
    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size
        
        X_test = np.array(X[start:end])
        y_test = np.array(y[start:end]).reshape(-1,1)
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]]).reshape(-1,1)
        
        model.fit(X_train, y_train)
        
        
        y_pred = model.predict(X_test) 
        scores.append(accuracy_score(y_test, y_pred))
    
    print(np.mean(scores))
    return np.mean(scores)


# In[119]:





# In[20]:





# In[21]:





# In[ ]:





# In[ ]:


# Reference: https://www.youtube.com/watch?v=sgQAhG5Q7iY&ab_channel=NormalizedNerd


# In[ ]:


# Classifier


# In[257]:




class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value
        
        
        
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node (once the stopping condition has been met)
        Y = list(Y)
        leaf_value = max(Y, key = Y.count)
        
        # return leaf node
        return Node(value=leaf_value)
    
    
    
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
                dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
                
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    
    def fit(self, X, Y):       
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        all_predictions = []
        for row in X:
            all_predictions.append(self.make_prediction(row, self.root))
        return all_predictions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value != None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)  
            


# In[258]:


classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=3)


# In[ ]:





# In[259]:


# Using Train Test Split
# X_train, X_test, Y_train, Y_test = train_test_split(spam_dataset, test_size=.2)
# classifier.fit(X_train,Y_train)
# Y_pred = classifier.predict(X_test) 
# accuracy_score(Y_test, Y_pred)


# In[262]:


# Using k-Fold CV

k_fold_cv(spam_dataset, 2, classifier)


# In[ ]:





# In[ ]:





# In[ ]:


"""
Shift-and-scale normalization: substract the minimum, then divide by new maximum. Now all values are between 0-1
Zero mean, unit variance : substract the mean, divide by the appropriate value to get variance=1.

The difference is that: 
in scaling, you're changing the range of your data, while. 
in normalization, you're changing the shape of the distribution of your data. (more radical) (normal distribution)

if there is an outlier, diving by max is not a good idea

ZN and CHAS are kind of like categorical variables.

"""


# In[ ]:


"""
Calculate the information gain for each feature.
See how many unique values are present. (Outlook:sunny/overcast/rain, Temp:hot/mild/cool, etc)

Entropy.
See how many unique values are present in the target. (No, Yes)
Total Entropy calculated using proportion of each unique value. (using # of Yes and # of No)
After that, for each feature, calculate the entropy of each of their unique values. 
        (using # of sunny - how many yes how many no, # of overcast..., separately for each feature)
Entropy formula = -∑p(proportion_of_unique_target_for_that_feature_item)*log2(p(proportion_of_unique_target_for_that_feature_item))
If we have only yes or only no for a unique value of a feature, the entropy will be 0.
If we have equal number of yes and no for that feature_item, the entropy will be 1.

Now we calculate the gain.
Gain(feature) = Total Entropy - Summation of ( (#of item_within_feature)/(#of total_items) * entropy of item_within_feature )

Compare all the gains and see which feature has the maximum gain.

So we will consider that feature as the root.

Now we will draw out branches for each of the items within that feature. Each of those are now own trees. 
We filter out only those data and repeat the process.

If all the values within a particular branch is the same, we mark that as a leaf node.

Else, continue the process. Now number of features is one less from before.

We calculate the entropies again.

Find the gains.

Whichever one has the highest gain becomes the node at that level.

If all the items correspond to a particular value, we mark that as such. Else, we continue.

All the features may not be used always. I like keeping it till 2 branches, or else it might get too specific.
"""


# In[315]:


dummy = spam_dataset.sample(frac = 0.004)


# In[317]:


dummy = dummy[["word_freq_address", "word_freq_internet", "word_freq_all", "char_freq_(", "target"]]


# In[318]:


dummy


# In[322]:


X = dummy.iloc[:,:-1].values


# In[323]:


y = dummy.iloc[:,-1].values


# In[395]:


y


# In[406]:


def total_entropy(any_y):
    tot_entropy = 0
    for item in np.unique(any_y):
        total = (len(any_y))
        prop = (any_y == item).sum()
        curr_val = -(prop / total) * np.log2(prop / total)
        tot_entropy += curr_val
    return tot_entropy


# In[407]:


total_entropy(y)


# In[437]:


def calc_gain_feat(givenX, giveny):
    list_of_col_gain = []
    for col_val in range(X.shape[1]):
#         print("\tColumn: ", col_val)
        feat = (X[:,col_val])
        y = giveny

        vals = []
        unq_feat = np.unique(feat)
        unq_y = np.unique(y)
        
        gain = total_entropy(y)

        for i in range(len(feat)):
            tup = (feat[i], y[i])
            vals.append(tup)

        for unq_ft_item in unq_feat:
#             print("Unq Feat:", unq_ft_item)
            y_temp = []
            for each_val in vals:
                if (each_val[0] == unq_ft_item):
                    y_temp.append(each_val[1])

#             print("Entropy: ",total_entropy(np.array(y_temp)))
            gain -= (len(y_temp) / len(feat)) * (total_entropy(np.array(y_temp)))
            
        print("gain:", gain)
        print()
        tup = (col_val, gain)
        list_of_col_gain.append(tup)
    return list_of_col_gain


# In[439]:


res = calc_gain_feat(X,y)


# In[455]:


col_with_max_gain = max(res, key = lambda i : i[1])[0]
col_with_max_gain


# In[500]:


selected_feat = (X[:,col_with_max_gain])
unq_feat = np.unique(selected_feat)

# turn this feature into the node
for unq_ft_item in unq_feat:
    print("-- branch --")
    y_checker = []
    for i in range(len(X)):
        if X[i][col_with_max_gain] == unq_ft_item:
            print(X[i], y[i])
            y_checker.append(y[i])
    if (len(np.unique(y_checker))) <= 1:
        print("this is a leaf")
    else:
        print("new node forms here | repeat the whole process using just these data")
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[124]:


# Q1 part ii (Decision Tree Regressor)


# In[246]:


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' same as the classifier other than var_red ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value


# In[263]:


class DecisionTreeRegressor():
    # No changes in constructor
    def __init__(self, min_samples_split=2, max_depth=2):
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            
            # check if information gain is positive
            if best_split["var_red"]>0: ## instead of info gain, we just use variance reduction
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        leaf_value = np.mean(Y)
        return Node(value=leaf_value)

    
    def get_best_split(self, dataset, num_samples, num_features):
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    ## instead of info gain using gini, we just variance reduction        
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    
    # no change, same as above
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    # no change, same as above
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
      
    # no change, same as above
    def make_prediction(self, x, tree):
        if tree.value != None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    # no change, same as above
    def predict(self, X):
        all_predictions = []
        for row in X:
            all_predictions.append(self.make_prediction(row, self.root))
        return all_predictions


# In[264]:


def read_housing_data():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    housing_train_df = pd.read_csv('housing_train.txt', header=None, delimiter=r"\s+", names=column_names)
    housing_test_df = pd.read_csv('housing_test.txt', header=None, delimiter=r"\s+", names=column_names)
    
#     scale_col = ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT'] # avoided ones with outliers
#     norm_col = ['CRIM', 'ZN', 'AGE', 'DIS', 'RAD', 'B']

    scale_col = [] # avoided ones with outliers
    norm_col = []
    
    for col in scale_col:
        housing_train_df[col] = shift_scale(housing_train_df[col])
        housing_test_df[col] = shift_scale(housing_test_df[col])
      
    for col in norm_col:
        housing_train_df[col] = zero_mean_unit_variance(housing_train_df[col])
        housing_test_df[col] = zero_mean_unit_variance(housing_test_df[col])     
    
    return housing_train_df, housing_test_df
    


# In[265]:





# In[269]:


def housing_classification():
    housing_train, housing_test = read_housing_data()
    
#     print(len(housing_train))
#     print(len(housing_test))
    
    train_val = (housing_train).values
    test_val = (housing_test).values

    X_train = (train_val[:,:-1])
    y_train = (train_val[:,-1]).reshape(-1,1)

    X_test = (test_val[:,:-1])
    y_test = (test_val[:,-1]).reshape(-1,1)
    
#     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    regressor = DecisionTreeRegressor(min_samples_split=2, max_depth=2)
    regressor.fit(X_train,y_train)
    
    y_pred = regressor.predict(X_test) 
    
    return mean_squared_error(y_test, y_pred)
    
    
    


# In[270]:


housing_classification()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




