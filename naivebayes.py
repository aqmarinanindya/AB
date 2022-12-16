import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import label_binarize

dataframe = pandas.read_csv("mergedaunbaseline-noheader-kanan.csv", header=None)
dataset = dataframe.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#-----------
# create model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
clf = model.fit(X_train, y_train)
#------------
y_pred = model.predict(X_test)

# Use score method to get accuracy of the model
score_te = model.score(X_test, y_test)
print('Accuracy Score: ', score_te)
cm = confusion_matrix(y_test, y_pred)
print (cm)
# Use accuracy_score to get accuracy of the model
acc = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', acc)
print(classification_report(y_test, y_pred))

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=0)

classes = ['daunsatu', 'daundua', 'dauntiga', 'daunempat', 'daunlima', 'daunenam']
# Binarize the output
y_bin = label_binarize(y, classes=classes)
n_classes = y_bin.shape[1]
#We define the model as an SVC in OneVsRestClassifier setting.
y_score = clf.predict_proba(X_test)
y_bin_test = label_binarize(y_test, classes=classes)

# Plotting and estimation of FPR, TPR
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
 fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
 roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'purple', 'orange'])
for i, color in zip(range(n_classes), colors):
 plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
