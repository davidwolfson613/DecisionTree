import numpy as np
import math
from collections import Counter
import time

class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        # print(feature)
        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """
    dt_root = None
    # TODO: finish this.

    l2_right = DecisionNode(None,None,lambda feature:feature[0] <= -0.16305)
    l2_right.left = DecisionNode(None,None,None,0)
    l2_right.right = DecisionNode(None,None,None,1)

    l1_right = DecisionNode(None,l2_right,lambda feature:feature[3]<= -0.46855)
    l1_right.left = DecisionNode(None,None,None,0)

    l1_left = DecisionNode(None,None,lambda feature:feature[0] <= -1.4249,None)
    l1_left.left = DecisionNode(None,None,None,0)
    l1_left.right = DecisionNode(None,None,None,2)

    dt_root = DecisionNode(l1_left,l1_right,lambda feature:feature[2] <= -0.7009,None)

    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.

    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.

    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|

    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]

    'count' function is expressed as 'count(actual label, predicted label)'.

    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    c_matrix = None
    # TODO: finish this.

    c_matrix = np.zeros((n_classes,n_classes),dtype=np.uint64)

    for i in range(len(true_labels)):
        c_matrix[true_labels[i],classifier_output[i]] += 1

    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output.
        So if the classifier is (0,1,2,...,n), the output should be in the below format:
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    # TODO: finish this.

    prec = np.zeros(n_classes)

    if pe_matrix:
        for i in range(n_classes):
            prec[i] = pe_matrix[i,i]/np.sum(pe_matrix[:,i])

    else:
        c_matrix = np.zeros((n_classes,n_classes),dtype=np.uint64)

        for i in range(len(true_labels)):
            c_matrix[true_labels[i],classifier_output[i]] += 1

        for i in range(n_classes):
            prec[i] = c_matrix[i,i]/np.sum(c_matrix[:,i])

    return prec


def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format:
        [recall (0), recall (1), recall (2), ... recall (n)].
    """
    # TODO: finish this.

    rec = np.zeros(n_classes)

    if pe_matrix is not None:
        for i in range(n_classes):
            rec[i] = pe_matrix[i,i]/np.sum(pe_matrix[i,:])

    else:
        c_matrix = np.zeros((n_classes,n_classes),dtype=np.uint64)

        for i in range(len(true_labels)):
            c_matrix[true_labels[i],classifier_output[i]] += 1

        for i in range(n_classes):
            rec[i] = c_matrix[i,i]/np.sum(c_matrix[i,:])

    return rec


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    # TODO: finish this.
    true_labels = np.array(true_labels)
    classifier_output = np.array(classifier_output)

    if pe_matrix is not None:

        pe_matrix = np.array(pe_matrix)
        acc = np.trace(pe_matrix)/np.sum(pe_matrix)

    else:
        c_matrix = np.zeros((n_classes,n_classes),dtype=np.uint64)

        for i in range(len(true_labels)):
            c_matrix[int(true_labels[i]),int(classifier_output[i])] += 1

        acc = np.trace(c_matrix)/np.sum(c_matrix)

    return acc


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    # TODO: finish this.
    # print(class_vector)
    class_vector = np.array(class_vector)
    # return 1-np.sum(np.array([(len(class_vector[class_vector==i])/len(class_vector))**2 for i in np.unique(class_vector)]))
    _,counts = np.unique(class_vector,return_counts=True)
    return 1 - np.sum((counts/np.sum(counts))**2)


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    # TODO: finish this.

    rem = sum(len(c)/len(previous_classes)*gini_impurity(c) for c in current_classes)
    return gini_impurity(previous_classes) - rem


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes, depth=0, rf_prob=0):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes, None, depth, rf_prob)

    def __build_tree__(self, features, classes, p_classes = None, depth=0, rf_prob=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        # TODO: finish this.
        features = np.array(features)
        classes = np.array(classes)

        # base case
        if depth == self.depth_limit or len(np.unique(classes)) <= 1 or len(classes) <= 1:

            # print(np.unique(classes,return_counts=True))
            if len(classes) == 1 or len(np.unique(classes)) == 1:
                return DecisionNode(None,None,None,classes[0])

            elif len(classes) == 0:
                unique,counts = np.unique(p_classes,return_counts=True)
                return DecisionNode(None,None,None,unique[counts.argmax()])

            else: # if depth == self.depth_limit
                if len(classes) > 0:
                    unique,counts = np.unique(classes,return_counts=True)
                else: # when class vector empty
                    unique,counts = np.unique(p_classes,return_counts=True)

                return DecisionNode(None,None,None,unique[counts.argmax()])


        alpha_best, thresh, left, right = self.get_split(features,classes,rf_prob)

        left_tree = self.__build_tree__(left[:,:-1],left[:,-1],p_classes=classes,depth=depth+1)
        right_tree = self.__build_tree__(right[:,:-1],right[:,-1],p_classes=classes,depth=depth+1)

        # print(features.shape)
        return DecisionNode(left_tree,right_tree,lambda features:features[alpha_best]<=thresh,None)

    def get_split(self,features,classes,rf_prob):

        if len(classes.shape) < 2:
            classes = classes[:,None] # add dimension to allow for hstack

        data = np.hstack((features,classes))
        max_gini = -1000 # initialize
        thresh = None
        idx = None
        left_data = None
        right_data = None

        if rf_prob != 0:
            vals = np.random.choice(features.shape[1],size=round(features.shape[1]*rf_prob),replace=False)

        else:
            vals = range(features.shape[1])

        # COULD VECTORIZE THIS FOR LOOP. IF HAVE TIME, COME BACK TO THIS
        for i in vals:

            mean = features[:,i].mean()
            left = data[data[:,i]<=mean]
            right = data[data[:,i]>mean]

            if len(left) == 0 or len(right) == 0:
                continue

            curr_classes = [left[:,-1],right[:,-1]]
            gini = gini_gain(classes.squeeze(),curr_classes)

            if gini > max_gini:

                max_gini = gini
                thresh = mean
                idx = i
                left_data = left
                right_data = right

        return idx,thresh,left_data,right_data


    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        # TODO: finish this.

        class_labels = [self.root.decide(f) for f in features]

        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    # TODO: finish this.

    features, classes = dataset
    classes = classes[:,None]

    dataset1 = np.hstack((features,classes))
    idx = len(dataset1)//k
    folds = []

    for i in range(k):

        if i == 0:
            train_f,train_c = dataset1[idx:,:-1],dataset1[idx:,-1]
            test_f,test_c = dataset1[:idx,:-1],dataset1[:idx,-1]

        elif i == k:
            train_f,train_c = dataset1[:idx*i,:-1],dataset1[:idx*i,-1]
            test_f,test_c = dataset1[idx*i:,:-1],dataset1[idx*i:,-1]

        else:
            train = np.vstack((dataset1[:idx*i,:],dataset1[idx*(i+1):,:]))
            train_f,train_c = train[:,:-1],train[:,-1]
            test = dataset1[idx*i:idx*(i+1),:]
            test_f,test_c = test[:,:-1],test[:,-1]

        folds.append(((train_f,train_c),(test_f,test_c)))

    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.

        features = np.array(features)
        classes = np.array(classes)

        for _ in range(self.num_trees):

            tree = DecisionTree(depth_limit=self.depth_limit)
            idx = np.random.choice(len(features),size=round(len(features)*self.example_subsample_rate))
            subsample_f = features[idx,:]
            subsample_c = classes[idx]
            tree.fit(subsample_f,subsample_c,rf_prob=self.attr_subsample_rate)
            self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        Returns:
            votes (list(int)): m votes for each element
        """
        # TODO: finish this.

        # print(features[:10].shape)
        votes = np.zeros(len(features))
        # print(votes.shape)
        # print(len(self.trees))
        all_votes = np.array([t.classify(features) for t in self.trees])
        # print(all_votes.shape)
        all_votes = all_votes.T
        # print(all_votes.shape)
        # print(all_votes[0])
        # print(np.unique(all_votes[0],return_counts=True))
        # print(votes.shape,all_votes.shape)
        # print(all_votes,'\n--------------------\n')
        for i,v in enumerate(all_votes):
            unique,counts = np.unique(v,return_counts=True)
            # print(all_votes[i])
            # print(unique,counts)
            votes[i] = unique[counts.argmax()]
        # print(votes)
        # exit()
        return list(votes)


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        # TODO: finish this.

        data = np.array(data)

        vectorized = data*data + data

        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        # TODO: finish this.

        data = np.array(data)

        tmp = np.sum(data[:100],axis=1)

        return np.max(tmp),np.argmax(tmp)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        # TODO: finish this.
        vals,count = np.unique(data[data>0],return_counts=True)
        return dict(zip(vals,count)).items()

    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        vectorized = None

        if dimension == 'c':
            vectorized = np.hstack((data,vector[:,None]))
        elif dimension == 'r':
            vectorized = np.vstack((data,vector[None,:]))

        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = np.copy(data)

        vectorized[data<threshold] = data[data<threshold]**2

        return vectorized


def return_your_name():
    # return your name
    # TODO: finish this

    return 'David Wolfson'
