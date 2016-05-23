#!/usr/env python

import pandas as pd
import os

def removeExtension (arrayFiles):
    for i in range(len(arrayFiles)):
        arrayFiles[i] = os.path.splitext(arrayFiles[i])[0]
    return arrayFiles

def concatenateFiles(listOfFiles,pathOutFile):
    with open(pathOutFile, 'w') as outfile:
        for fname in listOfFiles:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


folder0 = "C:\\Users\\MarcosFelipe\\Documents\\RetinaDataSet\\0\\"
folder1 = "C:\\Users\\MarcosFelipe\\Documents\\RetinaDataSet\\1\\"
folder2 = "C:\\Users\\MarcosFelipe\\Documents\\RetinaDataSet\\2\\"
folder3 = "C:\\Users\\MarcosFelipe\\Documents\\RetinaDataSet\\3\\"
folder4 = "C:\\Users\\MarcosFelipe\\Documents\\RetinaDataSet\\4\\"

filesArray0 = [x for x in os.listdir(folder0) if os.path.isfile(os.path.join(folder0,x))]
filesArray1 = [x for x in os.listdir(folder1) if os.path.isfile(os.path.join(folder1,x))]
filesArray2 = [x for x in os.listdir(folder2) if os.path.isfile(os.path.join(folder2,x))]
filesArray3 = [x for x in os.listdir(folder3) if os.path.isfile(os.path.join(folder3,x))]
filesArray4 = [x for x in os.listdir(folder4) if os.path.isfile(os.path.join(folder4,x))]

filesArray0=removeExtension(filesArray0)
filesArray1=removeExtension(filesArray1)
filesArray2=removeExtension(filesArray2)
filesArray3=removeExtension(filesArray3)
filesArray4=removeExtension(filesArray4)

dictOfLabels0 = {"image":filesArray0,"level":0}
dictOfLabels1 = {"image":filesArray1,"level":1}
dictOfLabels2 = {"image":filesArray2,"level":2}
dictOfLabels3 = {"image":filesArray3,"level":3}
dictOfLabels4 = {"image":filesArray4,"level":4}

df0 = pd.DataFrame(dictOfLabels0)
df1 = pd.DataFrame(dictOfLabels1)
df2 = pd.DataFrame(dictOfLabels2)
df3 = pd.DataFrame(dictOfLabels3)
df4 = pd.DataFrame(dictOfLabels4)


df0.to_csv("trainLabels0.csv")
df1.to_csv("trainLabels1.csv")
df2.to_csv("trainLabels2.csv")
df3.to_csv("trainLabels3.csv")
df4.to_csv("trainLabels4.csv")

listOfFiles = ["trainLabels0.csv","trainLabels1.csv","trainLabels2.csv","trainLabels3.csv","trainLabels4.csv"]

concatenateFiles(listOfFiles,"trainLabel.csv")