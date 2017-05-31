# decision-tree
# Was ist ein Entscheidungsbaum

Der Entscheidungsbaum soll die Entscheidungsfindung erleichtern und bildlich darstellen.

### Definition:

Ein hierarchisch angeordneter Such-Baum zur Darstellung von Entscheidungsregeln. Die Pfade des Baumes stellen Regeln dar. Entscheidungsbäume können binäre und numerische Daten behandeln

#### Beispiel

Entscheidung: Steht eine Frau oder ein Mann vor mir?

**Kriterien:

•	Trägt ein Rock (S)
•	Hat langes Haar (H)
•	Hat hohe Stimme (V)

Folgendes konnte Beobachtet werden:

S	 H	 V	   sex
j	 j	 j	    F
j	 j	 n	    F
j	 n	 j	    F
j	 n	 n	    M
n	 j	 j	    M
n	 n	 j	    M

Mittels dieser Daten kann gelernt werden

#### Aufbau eines Baums

Der Entscheidungsbaum besteht aus
•	Knoten mit Attributen
•	Fäden mit Bedeutung der Attribute
•	Blättern mit Antworten 

#### Optimaler Baum:
**Was ist Optimal?** 
→ Bereits mit erster Frage  großen Schritt zur Lösung des Problems machen. Somit sollte man nicht mit unwichtigen Fragen beginnen. Als erstes Kernfragen!
Diese Vorgehensweise wird uns schneller zur Lösung führen, da wir weniger Fragen stellen müssen. Die Information geht dabei nicht verloren und wird sogar erhöht. 

**Entropie:**

Wir besitzen die Menge aus N Elementen. Es gibt eine Eigenschaft S, diese kann zwei Zustände annehmen (Deutung, kann auch mehr als zwei sein). 
Die Menge M dieser Elemente besitz die Eigenschaft S=’x’. Die restlichen N-M besitzen die Eigenschaft ‚y’. Dann gilt:

**H(S)= - ∑pi log2(pi) = -(m/n + log2(m/n) + (m-n)/n * log2((m-n)/n))**

Pi → weißt drauf hin dass  S alle Eigenschaften annehmen kann




```python

```


```python
from IPython.display import YouTubeVideo
YouTubeVideo('R4OlXb9aTvQ')
```





        <iframe
            width="400"
            height="300"
            src="https://www.youtube.com/embed/R4OlXb9aTvQ"
            frameborder="0"
            allowfullscreen
        ></iframe>
        



#### Beispiel:

Das deutsche Alphabet hat 26 Buchstaben. Dabei kommt jeder Buchstabe mit gleicher Wahrscheinlichkeit vor (1/26). Die Eigenschaft S ist z.B. Vokal des Buchstabens
Somit gilt:
H(S) = - (5/26 * log2(5/26) + 23/33*log2(23/33))=

Bedeutet auch das um ein Buchstabe aufzunehmen bei so einer Wahrscheinlichkeit und Reihenfolge werden          Bit gebraucht.



**Hier kann die Frage Aufkommen**
→ Was ist wenn die Buchstaben nicht bei einer gleichen Wahrscheinlichkeit vorkommen?

Diese frage kann mit Bais Klassifikation beantwortet werden.

Informationsfluss des Attributs

Beispiel:

Angenommen wir haben ein Atrribut Q dieser kann zwei Bedeutungen annehmen.
Dann wir die Information-gain unterschiedlich entrotopisch ausgeprägt.

G(Q) =H(S) – p1H1(S) – p2H2(S)

P1= Anteil der Daten mit erster Bedeutung
Qp2 = Anteil mit zweiter Bedeutung
Hi entropie


Jetzt können wir den Informationsfluss bewerten, das kann mit dem Gini-Index (information gain) berechnet werden: 

Aus dem vorherigen Beispiel:

Eigenschaft		Info-gain

Rock	(S)		  0.553
Lange Haare	(H)	  0.057
Hohe Stimme	(V)   0

Das bedeutet, dass die Eigenschaft Rock beinhaltet die meiste Information um die Frage welches Geschlecht die Person hat zu beantworten.
Die Eigenschaft Stimme dagegen liefert überhaupt kein Information. Das liegt daran, dass wir genau zwei Männer und zwei Frauen die hohe Stimme besitzen und einen Mann und eine frau mit tiefer stimme. Somit kann hier nur Aufgrund dieser Beobachtungen noch keine Aussage getroffen werden. 

Da das Attribut Rock die größte Entropie hat werden wir damit beginnen unseren baum aufzubauen.
Wenn wir alle Daten mit Attribut Rock uns anschauen, 

S	H	V	sex
Ja	ja	ja	F
Ja	ja	nein	F
Ja	nein	ja	F
Ja	nein 	nein	M

Hier können wir vier Fälle betrachten. Um das nächste Attribut auszuwählen muss wieder die gain berechnet werden. In diesem Fall ist es aber auch so schon ersichtlich, dass die gain gleich ist (0.216). Somit ist es egal für welches Attribut wir uns entscheiden.
Wir wählen Haare

S=y, H=y					S=y, H=n
S	H	V	sex			S	H	V	sex
Ja	ja	ja	F			ja	nein	ja	F
Ja	ja	nein	F			ja	nein	nein	M

Das heisst, im Fall von H=y → Frau
	        Im Fall von H=n→ neue Analyse
S	H	V	sex
Nein	ja	ja	M
Nein	nein	ja	M

Leider ist es in diesem Fall nicht eindeutig → keine Unterscheidung mehr möglich.	




```python
# Klassifikation des Baums
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
 


```


```python
# einlesen der iris Daten
dataset = datasets.load_iris()



```


```python
# Prozess der Einstellung von CART zur Auswahl der Daten
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)



```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')



```python
# Ermittlung
expected = dataset.target
predicted = model.predict(dataset.data)



```


```python
# Wie gut wird ermittelt 
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00        50
              1       1.00      1.00      1.00        50
              2       1.00      1.00      1.00        50
    
    avg / total       1.00      1.00      1.00       150
    
    [[50  0  0]
     [ 0 50  0]
     [ 0  0 50]]



```python
from math import log
import operator

def createDataSet():
    dataSet = [[0, 1, 1, 'yes'],
               [0, 1, 0, 'no'],
               [1, 0, 1, 'no'],
               [1, 1, 1, 'no'],
               [0, 1, 0, 'no'],
               [0, 0, 1, 'no'],
               [1, 0, 1, 'no'],
               [1, 1, 0, 'no']]
    labels = ['Karte', 'Winter', 'mehr als 1 person']
    # diskrete Werte
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)


        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
   
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # extracting data
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # use Information Gain
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    #build a tree recursively
    myTree = {bestFeatLabel: {}}
    #print("myTree : "+labels[bestFeat])
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    #print("featValues: "+str(featValues))
    uniqueVals = set(featValues)
    #print("uniqueVals: " + str(uniqueVals))
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        #print("subLabels"+str(subLabels))
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #print("myTree : " + str(myTree))
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    #print("fistStr : "+firstStr)
    secondDict = inputTree[firstStr]
    #print("secondDict : " + str(secondDict))
    featIndex = featLabels.index(firstStr)
    #print("featIndex : " + str(featIndex))
    key = testVec[featIndex]
    #print("key : " + str(key))
    valueOfFeat = secondDict[key]
    #print("valueOfFeat : " + str(valueOfFeat))
    if isinstance(valueOfFeat, dict):
        #print("is instance: "+str(valueOfFeat))
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        #print("is Not instance: " + valueOfFeat)
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# collect data
myDat, labels = createDataSet()

#build a tree
mytree = createTree(myDat, labels)
print(mytree)
```

    {'Karte': {0: {'mehr als 1 person': {0: 'no', 1: {'Winter': {0: 'no', 1: 'yes'}}}}, 1: 'no'}}


Quellen: Softcomputing in der Informatik (Jurgen Paetz) Springer
         
