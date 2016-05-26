#!/usr/env python
import os
import math
import pandas as pd

def linearSearch(inputArray,value):
    for i in inputArray:
        if i == value:
            return True
    return False


def generateDataSet(directoryName, numberFiles,labelFilesName):
    filesArray = [x for x in os.listdir(directoryName) if os.path.isfile(os.path.join(directoryName,x))]
    labelsFile = pd.read_csv(labelFileName)
    labelsArray = labelsFile.values
    newFolderData = directoryName+"dataset\\"
    if not os.path.exists(newFolderData):
        os.mkdir(newFolderData)

    for n in range(numberFiles):
        for i in range(len(labelsArray)):
            if labelsArray[i][1] == 0:
                if linearSearch(filesArray,labelsArray[i][0]+".jpeg"):
                    os.rename(directoryName+filesArray[i],newFolderData+filesArray[i])
                    break

        for i in range(len(labelsArray)):
            if labelsArray[i][1] == 1:
                if linearSearch(filesArray, labelsArray[i][0]+".jpeg"):
                    print("11")
                    os.rename(directoryName + filesArray[i], newFolderData + filesArray[i])
                    break

        for i in range(len(labelsArray)):
            if labelsArray[i][1] == 2:
                if linearSearch(filesArray, labelsArray[i][0]+".jpeg"):
                    print("21")
                    os.rename(directoryName + filesArray[i], newFolderData + filesArray[i])
                    break

        for i in range(len(labelsArray)):
            if labelsArray[i][1] == 3:
                if linearSearch(filesArray, labelsArray[i][0]+".jpeg"):
                    os.rename(directoryName + filesArray[i], newFolderData + filesArray[i])
                    break

        for i in range(len(labelsArray)):
            if labelsArray[i][1] == 4:
                if linearSearch(filesArray, labelsArray[i][0]+".jpeg"):
                    os.rename(directoryName + filesArray[i], newFolderData + filesArray[i])
                    break

if __name__ == "__main__":
    #directoryName = input("Enter the path for a directory ")
    #numberFiles = input("Enter the number of files by type ")
    #labelFileName = input("Enter the file with labels ")
    directoryName = "C:\\Users\\MarcosFelipe\\Documents\\train\\"
    numberFiles = 10
    labelFileName = "C:\\Users\\MarcosFelipe\\Documents\\trainLabels.csv"
    generateDataSet(directoryName,numberFiles,labelFileName)