
# coding: utf-8

# In[67]:

# Name: Peyman Hesami
# Email: phesami@eng.ucsd.edu
# PID: A53112379
from pyspark import SparkContext
sc = SparkContext()


# In[ ]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel, RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


# ### Cover Type
# 
# Classify geographical locations according to their predicted tree cover:
# 
# * **URL:** http://archive.ics.uci.edu/ml/datasets/Covertype
# * **Abstract:** Forest CoverType dataset
# * **Data Set Description:** http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info

# In[7]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)
inputRDD.first()


# In[22]:

# Transform the text RDD into an RDD of LabeledPoints
Data=inputRDD.map(lambda line: [float(strip(x)) for x in line.split(',')])     .map(lambda x:LabeledPoint(x[len(x)-1],x[:-1]))
Data.first()


# ### Making the problem binary
# 
# The implementation of BoostedGradientTrees in MLLib supports only binary problems. the `CovTYpe` problem has
# 7 classes. To make the problem binary we choose the `Lodgepole Pine` (label = 2.0). We therefor transform the dataset to a new dataset where the label is `1.0` is the class is `Lodgepole Pine` and is `0.0` otherwise.

# In[34]:

Label=2.0
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')])    .map(lambda V:LabeledPoint(1,V[:-1]) if V[len(V)-1]==Label else LabeledPoint(0,V[:-1]))


# In[39]:

(trainingData,testData)=Data.randomSplit([0.7,0.3],seed=255)
print 'Sizes: Data1=%d, trainingData=%d, testData=%d'%(Data.count(),trainingData.cache().count(),testData.cache().count())


# In[41]:

counts=testData.map(lambda lp:(lp.label,1)).reduceByKey(lambda x,y:x+y).collect()
counts.sort(key=lambda x:x[1],reverse=True)
counts


# ### Gradient Boosted Trees
# 
# * Following [this example](http://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts) from the mllib documentation
# 
# * [pyspark.mllib.tree.GradientBoostedTrees documentation](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.tree.GradientBoostedTrees)
# 
# #### Main classes and methods
# 
# * `GradientBoostedTrees` is the class that implements the learning trainClassifier,
#    * It's main method is `trainClassifier(trainingData)` which takes as input a training set and generates an instance of `GradientBoostedTreesModel`
#    * The main parameter from train Classifier are:
#       * **data** – Training dataset: RDD of LabeledPoint. Labels should take values {0, 1}.
#       * categoricalFeaturesInfo – Map storing arity of categorical features. E.g., an entry (n -> k) indicates that feature n is categorical with k categories indexed from 0: {0, 1, ..., k-1}.
#       * **loss** – Loss function used for minimization during gradient boosting. Supported: {“logLoss” (default), “leastSquaresError”, “leastAbsoluteError”}.
#       * **numIterations** – Number of iterations of boosting. (default: 100)
#       * **learningRate** – Learning rate for shrinking the contribution of each estimator. The learning rate should be between in the interval (0, 1]. (default: 0.1)
#       * **maxDepth** – Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (default: 3)
#       * **maxBins** – maximum number of bins used for splitting features (default: 32) DecisionTree requires maxBins >= max categories
#       
#       
# * `GradientBoostedTreesModel` represents the output of the boosting process: a linear combination of classification trees. The methods supported by this class are:
#    * `save(sc, path)` : save the tree to a given filename, sc is the Spark Context.
#    * `load(sc,path)` : The counterpart to save - load classifier from file.
#    * `predict(X)` : predict on a single datapoint (the `.features` field of a `LabeledPont`) or an RDD of datapoints.
#    * `toDebugString()` : print the classifier in a human readable format.

# In[63]:

from time import time
errors={}
for depth in [10]:
    start=time()
    model=GradientBoostedTrees.trainClassifier(trainingData,categoricalFeaturesInfo={}, numIterations=10, maxDepth=depth,learningRate=0.5)
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda x:x.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth],int(time()-start),'seconds'


# ### Random Forests
# 
# * Following [this example](http://spark.apache.org/docs/latest/mllib-ensembles.html#classification) from the mllib documentation
# 
# * [pyspark.mllib.trees.RandomForest documentation](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.tree.RandomForest)
# 
# **trainClassifier**`(data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy='auto', impurity='gini', maxDepth=4, maxBins=32, seed=None)`   
# Method to train a decision tree model for binary or multiclass classification.
# 
# **Parameters:**  
# * *data* – Training dataset: RDD of LabeledPoint. Labels should take values {0, 1, ..., numClasses-1}.  
# * *numClasses* – number of classes for classification.  
# * *categoricalFeaturesInfo* – Map storing arity of categorical features. E.g., an entry (n -> k) indicates that feature n is categorical with k categories indexed from 0: {0, 1, ..., k-1}.  
# * *numTrees* – Number of trees in the random forest.  
# * *featureSubsetStrategy* – Number of features to consider for splits at each node. Supported: “auto” (default), “all”, “sqrt”, “log2”, “onethird”. If “auto” is set, this parameter is set based on numTrees: if numTrees == 1, set to “all”; if numTrees > 1 (forest) set to “sqrt”.
# * *impurity* – Criterion used for information gain calculation. Supported values: “gini” (recommended) or “entropy”.  
# * *maxDepth* – Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (default: 4)  
# * *maxBins* – maximum number of bins used for splitting features (default: 32)
# * *seed* – Random seed for bootstrapping and choosing feature subsets.  
# 
# **Returns:**	
# RandomForestModel that can be used for prediction

# In[69]:

from time import time
errors={}
for depth in [15]:
    start=time()
    model = RandomForest.trainClassifier(trainingData,categoricalFeaturesInfo={}, numClasses=2, numTrees=10, maxDepth=depth, impurity='gini')
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda x:x.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth],int(time()-start),'seconds'

