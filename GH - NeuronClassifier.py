# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:25:36 2018

@author: nicknobad
"""

# Inputs
DeadPath = r'D:\stuff\00_Dead.zip'
AlivePath = r'D:\stuff\00_Aliive.zip'

from PIL import Image; import io, zipfile, numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import train_test_split

# to plot ML models' accuracy
import matplotlib.pyplot as plt; import seaborn as sns; from sklearn import metrics

'import data'
# function to read TIF images of Neurons from archive to pd df
def read_TIFs_from_archive_to_np( zip_path, label ):
    # prepare the list which will be appended to the pandas df
    # LEAVE IT THIS WAY: Neuron_numpyArrayToAdd = []; NeuronID = []; NeuronStatus = []
    Neuron_numpyArrayToAdd = []; NeuronID = []; NeuronStatus = []
    
    # read the archive
    archive = zipfile.ZipFile(zip_path, 'r')    
    for data in archive.namelist():    
        # read a file from Archive
        image_data = archive.read( data )        
        # open the file with io.BytesIO
        fh = io.BytesIO( image_data )        
        # save image to convert to numpy array # show the image
        img = Image.open(fh)        
        # save img to numpy array
        imgArray = np.array( img )        
        # from numpy array to PIL image
        # Image.fromarray( imgArray )        
        # append the numpy array to list
        Neuron_numpyArrayToAdd.append( imgArray )        
        # append the Neuron ID
        NeuronID.append( data )        
        # have '0' for Dead Neurons and '1' for Alive Neurons
        if label == 'Dead':
            NStatus = 0
        elif label == 'Alive':
            NStatus = 1
        NeuronStatus.append( NStatus )
        
    # get a list of: NeuronID AND NeuronStatus AND Neuron numpy arrays
    list_data = [ NeuronID, NeuronStatus, Neuron_numpyArrayToAdd ]
    return list_data

# Alive and Dead lists contain 3 lists:
# 1. file name of the Neuron image
# 2. Status of Neuron: Alive = 1, Dead = 0
# 3. numpy array of the Neuron image

# get df of Dead Neurons
listDeadNeurons = read_TIFs_from_archive_to_np( zip_path = DeadPath, label = 'Dead')

# get df of Alive Neurons
listAliveNeurons = read_TIFs_from_archive_to_np( zip_path = AlivePath , label = 'Alive')

# The resulting listAliveNeurons and listDeadNeurons are 3D arrays:
# 1st- and 2nd-Dimension arrays - NO IDEA WHAT THEY ARE!!
# 3rd Dimension array is the 40 x 40 numpy array of the image. Select this array to build the ML models on


'Check Image importing'
# 1. Check if the read images were imported correctly. If they aren't imported correctly then we wouldn't have
# good data for training the ML models.

# 2. Check that all arrays in the 3rd Dimension of each image-array are the same; 
# if set( list ) is True then there is not problem - all arrays in the 3rd Dimension of each image-array are the same

## %%timeit
# Alive
list_checkAlive_np_array = []
for wd in range( 0, len( listAliveNeurons[2] ) ):
    checkcheck = np.unique( listAliveNeurons[2][wd][:, :, 0] == listAliveNeurons[2][wd][:, :, 1] )[0]
    list_checkAlive_np_array.append( checkcheck )
    
    checkcheck = np.unique( listAliveNeurons[2][wd][:, :, 0] == listAliveNeurons[2][wd][:, :, 2] )[0]
    list_checkAlive_np_array.append( checkcheck )
# if len( list_checkAlive_np_array ) / len( listAliveNeurons[2] ) != 2 then there's a problem
set( list_checkAlive_np_array )

# Dead
list_checkDead_np_array = []
for wd in range( 0, len( listDeadNeurons[2] ) ):
    checkcheck = np.unique( listDeadNeurons[2][wd][:, :, 0] == listDeadNeurons[2][wd][:, :, 1] )[0]
    list_checkDead_np_array.append( checkcheck )
    
    checkcheck = np.unique( listDeadNeurons[2][wd][:, :, 0] == listDeadNeurons[2][wd][:, :, 2] )[0]
    list_checkDead_np_array.append( checkcheck )
# if len( list_checkDead_np_array ) / len( listDeadNeurons[2] ) != 2 then there's a problem
set( list_checkDead_np_array )
'/Check Image importing'


'check if len is 1600 in Dead and in Alive sets'
# Dead
for i in range( 0, len( listDeadNeurons[2] ) ):
    if len( listDeadNeurons[2][i][:, :, 0].flatten() ) != 1600:
        print( " numpy array in the Dead dataset that doesn't have the len of 1600: " % (i) )
else:
    print( " all numpy arrays in the Dead dataset have the len of 1600 " )

# Alive
for i in range( 0, len( listAliveNeurons[2] ) ):
    if len( listAliveNeurons[2][i][:, :, 0].flatten() ) != 1600:
        print( " numpy array in the Alive dataset that doesn't have the len of 1600: " % (i) )
else:
    print( " all numpy arrays in the Alive dataset have the len of 1600 " )

'/check if len is 1600'

'/import data'

'prepare data'

# get all the 40x40 image arrays into a tidy 2D array together with NeuronID and NeuronStatus
# Dead
nparrayFlatDeadNeurons = np.empty( (0, len( listDeadNeurons[2][0][:, :, 0].flatten() ) ), dtype = 'uint8' )

for line in range( 0, len( listDeadNeurons[2] ) ):
    result = listDeadNeurons[2][line][:, :, 0].flatten()
    nparrayFlatDeadNeurons = np.append( nparrayFlatDeadNeurons, [result], axis = 0 )

# Alive
nparrayFlatAliveNeurons = np.empty( (0, len( listDeadNeurons[2][0][:, :, 0].flatten() ) ), dtype = 'uint8' )

for line in range( 0, len( listAliveNeurons[2] ) ):
    result = listAliveNeurons[2][line][:, :, 0].flatten()
    nparrayFlatAliveNeurons = np.append( nparrayFlatAliveNeurons, [result], axis = 0 )


# combine Neuron ID, Neuron status and its numpy array
# Dead
npArrayAll_Dead = np.column_stack( ( listDeadNeurons[0], listDeadNeurons[1], nparrayFlatDeadNeurons ) )
# Alive
npArrayAll_Alive = np.column_stack( ( listAliveNeurons[0], listAliveNeurons[1], nparrayFlatAliveNeurons ) )

# combine npArrayAll_Dead and npArrayAll_Alive in one np array
npArrayAll = np.vstack( (npArrayAll_Dead, npArrayAll_Alive) )

# np.savetxt( 'test', nparrayFlatDeadNeurons[0:1, :], newline = ' ', delimiter = '', fmt = '%d')

'/prepare data'


'build the ML models'
''' compare classifiers
https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet
https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
https://www.kaggle.com/nirajvermafcb/comparing-various-ml-models-roc-curve-comparison

http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

'''

'/build the ML models'




'Check performance of ML models'
# -----------------------------------------------------------------------------
'benchmark different ML models'
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


'prepare models'
models = []
models.append( ( 'LR', LogisticRegression() ) )
models.append( ( 'LDA', LinearDiscriminantAnalysis() ) )
models.append( ( 'KNN', KNeighborsClassifier() ) )
models.append( ( 'CART', DecisionTreeClassifier() ) )
models.append( ( 'GNB', GaussianNB() ) )
models.append( ( 'SVM', SVC() ) )

X = npArrayAll[:, 2:].astype(np.uint8)
y = npArrayAll[:, 1].astype(np.uint8)
# split in train and test sets
X_train, X_test, y_train, y_test = train_test_split( npArrayAll[:, 2:].astype(np.uint8), npArrayAll[:, 1].astype(np.uint8)
                                                    , test_size = 0.33, random_state = 42 )
#----------------------------------------
# store metrics in a df
# all scoring metrics available here: http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
# List of scoring metrics:
# ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
list_metrics = [ 'accuracy', 'r2', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision', 'recall' ]

'''Get algorithms' short names, their mean accuracies and their standard deviation accuracies.'''
t0 = time.time() # time the for loop
sscore_result = []; mmetric = []; mmodel = []
for metric in list_metrics:
    for name, model in models:
        mmodel.append( name )
        mmetric.append( metric )
        kfold = model_selection.KFold( n_splits = 13 )
        sscore_result.append( model_selection.cross_val_score( model, X, y, cv = kfold, scoring = metric ).mean() )

t1 = time.time()
print( '\n CV of models ran in %.4fs or %.4fm ' % ( (t1 - t0), ((t1 - t0) / 60) ) )

# convert dict to pd DF
scoresLongDF = pd.DataFrame( { 'model' : mmodel, 'metric' : mmetric, 'value' : sscore_result } )

# conver tthe Long DF to wide DF
scoresWideDF = scoresLongDF.pivot( index = 'model', columns = 'metric', values = 'value' )

# convert Index column to column and re-order it
scoresWideDF['model'] = scoresWideDF.index
# delete Index
scoresWideDF = scoresWideDF.reset_index( drop =  True )

# reorder columns in the DF; model name column to be the first column
scoresWideDF = scoresWideDF[ ['model'] + [i for i in scoresWideDF.columns.tolist() if i != 'model'] ]

#  sort by column descending to get the most accurate model
scoresWideDF = scoresWideDF.sort_values( [ 'accuracy' ], ascending = [ False ] )

'/benchmark different ML models'


'tune the chosen model'

# http://scikit-learn.org/stable/modules/svm.html
# https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
# https://www.pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/

''' functions plots the Confusion Matrix for train and test sets.
It also returns a list with elements: 0. trainPreds, 1. trainScore, 2. train confusion matrix,
3. train_AUC_ROC, 4. trainPreds, 5. testScore, 6. test confusion matrix, 7. test_AUC_ROC '''
def makePredsScores( model, model_name ):
    'train'
    # make predictions on train
    trainPreds = model.predict( X_train )
    # Use score method to get accuracy of model
    trainScore = model.score( X_train, y_train )
    # build a confusion matrix
    trainCM = metrics.confusion_matrix( y_train, trainPreds )
    # calculate auc_roc
    train_AUC_ROC = metrics.roc_auc_score( y_train, trainPreds )
    
    # graph; confusion matrix with Seaborn
    plt.figure( figsize = (9, 9) )
    sns.heatmap( trainCM, annot = True, fmt = '.0f', linewidths = .5, square = True, cmap = 'Blues_r' )
    plt.ylabel( 'Actual label' )
    plt.xlabel( 'Predicted label' )
    all_sample_title = model_name + ' Train. ' + 'Accuracy Score: {0}'.format( round(trainScore, 6) )
    plt.title( all_sample_title, size = 15 )
    '/train'
    
    'test'
    # make predictions on test
    testPreds = model.predict( X_test )
    # Use score method to get accuracy of model
    testScore = model.score( X_test, y_test )
    # build a confusion matrix
    testCM = metrics.confusion_matrix( y_test, testPreds )
    # calculate auc_roc
    test_AUC_ROC = metrics.roc_auc_score( y_test, testPreds )
    
    # graph; confusion matrix with Seaborn
    plt.figure( figsize = (9, 9) )
    sns.heatmap( testCM, annot = True, fmt = '.0f', linewidths = .5, square = True, cmap = 'Blues_r' )
    plt.ylabel( 'Actual label' )
    plt.xlabel( 'Predicted label' )
    all_sample_title = model_name + ' Test. ' + 'Accuracy Score: {0}'.format( round(testScore, 6) )
    plt.title( all_sample_title, size = 15 )
    '/test'    
    return [ trainPreds, trainScore, trainCM, train_AUC_ROC, testPreds, testScore, testCM, test_AUC_ROC ]


SVM = SVC()
SVM.fit( X_train, y_train )

makePredsScores( model = SVM, model_name = 'SVM' )
'/tune the chosen model'




'/Check performance of ML models'



