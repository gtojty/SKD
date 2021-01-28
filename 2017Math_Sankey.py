# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:23:41 2020

@author: TGONG
"""

# Thi scode uses plotly to draw Sankey diagrams for 2017 NAEP Math items

# set current working directory
import os
#wd = "C:\\GongTao\\Academic Activities\\ETS\\NAEP\\2017 Math\\rawdata"
wd = '.'
#wd = os.getcwd()
os.chdir(wd)

# import required libraries/packages
import pandas as pd
import numpy as np

import plotly

def flattenNestedList(nestedList):
    ''' 
    Converts a nested list to a flat list
    :param nestedList a nested list
    :return a flatten list
    '''
    flatList = []
    # Iterate over all the elements in given list
    for elem in nestedList:
        # Check if type of element is list
        if isinstance(elem, list):
            # Extend the flat list by adding contents of this element (list)
            flatList.extend(flattenNestedList(elem))
        else:
            # Append the elemengt to the list
            flatList.append(elem)    
 
    return(flatList)

def find_Ngrams(input_list, n):
    """
    generate n-gram from the input list
    :param input_list list of words (paths)
    :param n size of n-gram
    :retur a list of n-gram components 
    """
    return list(map(list, zip(*[input_list[i:] for i in range(n)])))

def getTotST(df):
    """
    get all the sources and targets
    :param df data frame of a block
    :return listS, listT
    """
    allseq = list(df.Seq)
    clnseq = [x.replace("Add_", "").replace("Rem_", "").replace("Clear Answer;", "").replace(";;",";") for x in allseq]
    
    listS, listT = [], [] 
    for seq in clnseq:
        seqList = seq.split(";")
        listS = list(set(listS + [x.split("_")[0] for x in seqList]))
        listT = list(set(listT + [x.split("_")[1] if len(x.split("_"))==2 else '' for x in seqList]))
    
    listS.sort()
    if '' in listS: listS.remove('')
    listT.sort()
    if '' in listT: listT.remove('')
    
    return(listS, listT)

def addState(df, listS, listT, switch = 'off'):
    """
    add state column based on Seq column
    :param df data frame of a student
    :param listS all source items
    :param listT all target items
    :param switch on or off
    """
    if switch == 'on': print(df.BookletNumber.iloc[0])
    
    seqList = df.Seq.iloc[0].split(";")
    clrState = ['NA'] * len(listT)
    
    state = [['NA'] * len(listT)] # starting state
    for seq in seqList:
        if seq.startswith("Add_"): # add a source to a target
            addS, addT = seq.split("Add_")[1].split("_")[0], seq.split("Add_")[1].split("_")[1]
            Tcode = np.int64(addT.split("t")[1]) - 1
            newState = state[-1].copy()
            if newState[Tcode] == 'NA': newState[Tcode] = addS # the target is empty, add the source directly
            else: # the target is not empty, add the source and sort the target 
                existS = newState[Tcode].split(",") # get existing items
                if addS not in existS: existS.append(addS) # add new source (check first if it is already there, just incase wrong data capture)
                existS.sort() # sort
                newS = ','.join(existS)                
                newState[Tcode] = newS 
        if seq.startswith("Rem_"):
            remS, remT = seq.split("Rem_")[1].split("_")[0], seq.split("Rem_")[1].split("_")[1]
            Tcode = np.int64(remT.split("t")[1]) - 1
            newState = state[-1].copy()
            if newState[Tcode] == remS: newState[Tcode] = 'NA'
            else:
                existS = newState[Tcode].split(",")
                if remS in existS: existS.remove(remS) # remove remS (check first if it is already there, just in case wrong data capture)
                newS = ','.join(existS)
                newState[Tcode] = newS
        if seq.startswith("Clear"): newState = clrState
        
        state = state + [newState]
        
    stateSeq = ';'.join(["|".join(x) for x in state])    
    
    newdf = df.reset_index(drop=True)
    newdf.loc[:,'State'] = stateSeq   
    return(newdf)

def getLabels(df):
    """
    get labels for node for Sankey diagram
    :param df data from of a block
    :return labels
    """
    nodes = []    
    for i in range(len(df)):
        nodes = list(set(nodes + df.loc[i, 'State'].split(";")))
        
    nodes.sort()
    return(nodes)
 
def addpath(linkDF, bigramList, labels, listS, listT):
    """
    add bigram path to linkDF
    :param linkDF linkDF object for Sankey diagram
    :param bigramList list of bigrams
    :param labels node labels
    :param listS all source items
    :param listT all target items
    :return linkDF
    """
    nodecode = list(range(len(labels)))
    
    for i in range(len(bigramList)):
        source, target = bigramList[i][0], bigramList[i][1]
        sourcecode, targetcode = nodecode[labels.index(source)], nodecode[labels.index(target)]
        
        sourcenumberList = flattenNestedList([x.split(",") if "," in x else x for x in source.split("|")])
        sourcenumberNo = sum([1 if x !='NA' else 0 for x in sourcenumberList])
        if sourcenumberNo > len(listS): sourcenumberNo = len(listS)
        targetnumberList = flattenNestedList([x.split(",") if "," in x else x for x in target.split("|")])
        targetnumberNo = sum([1 if x !='NA' else 0 for x in targetnumberList])
        if targetnumberNo > len(listS): targetnumberNo = len(listS)
        
        if sourcenumberNo < targetnumberNo: linklabel, col = 'Add', 'brown' # add a source
        elif sourcenumberNo == targetnumberNo: linklabel, col = 'Same', 'gray' # same state movement
        else: 
            if targetnumberNo == 0: linklabel, col = 'Rem/Clr', 'black' # clear all or equivalent
            else: linklabel, col = 'Rem', 'pink' # remove a source
                
        if len(linkDF) == 0: linkDF = linkDF.append({'source': sourcecode, 'target': targetcode, 'value': 1, 'label': linklabel, 'color': col}, ignore_index = True)     
        else:
            ind = None
            for j in range(len(linkDF)):
                if linkDF.source[j] == sourcecode and linkDF.target[j] == targetcode:
                    ind = j; break
                    
            if not pd.isnull(ind): linkDF.loc[ind, 'value'] = linkDF.loc[ind, 'value'] + 1
            else: linkDF = linkDF.append({'source': sourcecode, 'target': targetcode, 'value': 1, 'label': linklabel, 'color': col}, ignore_index = True)  
    
    return(linkDF)
               
def getLink(df, labels, listS, listT, switch = 'off'):
    """
    get the linkDF for link for SankeyWidget
    :param df dataframe of one student
    :param labels node labels
    :param listS all source items
    :param listT all target items
    :return linkDF a data frame for Sankey diagram
    """
    if switch == 'on': print(df.BookletNumber.iloc[0])
    
    linkDF = pd.DataFrame()
    
    stateList = df.State.iloc[0].split(";")
    bigramList = find_Ngrams(stateList, 2)
    linkDF = addpath(linkDF, bigramList, labels, listS, listT)  
    
    return(linkDF)

def mergeValue(linkdf, switch = 'off'):
    """
    merge value of linkdf with same source and target
    :param linkdf one piece of linkDF
    :param switch whether or not show source and target
    """
    if switch == 'on': print("source: ", linkdf.source.iloc[0], "; target: ", linkdf.target.iloc[0])
    newdf = pd.DataFrame()
    newdf = newdf.append(linkdf.iloc[0,])
    newdf.loc[:,'value'] = sum(linkdf.value)
    
    return(newdf)

def getOccurKeep(df, linkDF, keepPerc, sourcetarget, switch = 'on'):
    """
    add occur and Keep column of stateDF
    :param df data frame of stateDF that has the same color labels (same stage states)
    :param linkDF linkDF for connections
    :param keepPerc how many percentage of states are kept at each stage; if keepPerc > 1, how many states are kept in each stage
    :param sourcetarget whether keep states based source ('source') or target ('target')
    :param switch = 'on' whether show process
    """
    if switch == 'on': print(df.color.iloc[0])
    
    newdf = df.copy().reset_index(drop=True)
    
    for stateInd in newdf.ind:
        if sourcetarget == 'source': sublinkDF = linkDF.loc[linkDF.source == stateInd,]
        if sourcetarget == 'target': sublinkDF = linkDF.loc[linkDF.target == stateInd,]
        newdf.loc[newdf.ind==stateInd, 'occur'] = sum(sublinkDF.value)
    
    newdf = newdf.sort_values(by=['occur'], ascending=False).reset_index(drop=True)
    newdf.loc[:,'perc'] = newdf.occur/sum(newdf.occur)
    
    if keepPerc > 1.0:
        if keepPerc <= len(newdf): 
            newdf.loc[0:keepPerc, 'keep'] = True
            newdf.loc[keepPerc:len(newdf), 'keep'] = False
        else: newdf.loc[:,'keep'] = True    
    else:
        newdf.loc[newdf.perc >= keepPerc, 'keep'] = True
        newdf.loc[newdf.perc < keepPerc, 'keep'] = False
    
    return(newdf)
    
def getSubLink(linkDF, keepPerc, minValue, labels, colors, listS, listT, coldic, sourcetarget):
    """
    reduce linkDF by keeping keepNo states at each stage
    :param linkDF original links
    :param keepPerc how many percentage of states are kept at each stage; if keepPerc > 1, how many states are kept in each stage
    :param minValue minimum value for keeping a link
    :param labels state labels
    :param colors colors of each label
    :param listS all source items
    :param listT all target items
    :param sourcetarget whether keep states based source ('source') or target ('target')
    :return sublinkDF
    """
    sublinkDF = pd.DataFrame()
    
    stateDF = pd.DataFrame({'label': labels, 'color': colors}); stateDF.loc[:,'ind'] = range(len(stateDF))
    stateDF.loc[:,'occur'] = None; stateDF.loc[:,'perc'] = None; stateDF.loc[:,'keep'] = None
    
    stateDF = (stateDF.groupby("color", as_index = False).apply(getOccurKeep, linkDF = linkDF, keepPerc = keepPerc, sourcetarget = sourcetarget, switch = 'off')).reset_index(drop=True)
    substateDF = stateDF.loc[stateDF.keep==True,].reset_index(drop=True)
    
    # further check the states in the last stage, if there is no previous state goes into that state at the last stage, also delete those states at the last stage
    # make sure states in later stages have flows from states of previous stages
    for i in range(len(listT), 0, -1):
        cur = substateDF.ind[(substateDF.color==coldic[i])&(substateDF.keep==True)] # make sure this state is still kept
        prev = substateDF.ind[(substateDF.color==coldic[i-1])&(substateDF.keep==True)] # make sure this state is still kept
        for ind in cur:
            idx = ((linkDF.source.isin(prev))&(linkDF.target.isin([ind])))|((linkDF.source.isin(cur))&(linkDF.target.isin([ind])))
            if sum(idx) == 0: substateDF.loc[substateDF.ind==ind, 'keep'] = False
    newsubstateDF = substateDF.loc[substateDF.keep==True,].reset_index(drop=True)
    
    idx = (linkDF.source.isin(newsubstateDF.ind))&(linkDF.target.isin(newsubstateDF.ind))&(linkDF.value > minValue)
    sublinkDF = linkDF.loc[idx,]
    
    return(sublinkDF)   

def getvisuLinkDF(sublinkDF, visuType):
    """
    get visual linkDF from sublinkDF
    :param sublinkDF original linkDF for visualization
    :param visuType type of visualization
    """
    assert(visuType in ['All', 'Add', 'Same', 'Rem'])
    
    if visuType == 'All': visulinkDF = sublinkDF
    if visuType == 'Add': visulinkDF = sublinkDF.loc[sublinkDF.label=='Add',]
    if visuType == 'Same': visulinkDF = sublinkDF.loc[sublinkDF.label=='Same',]
    if visuType == 'Rem': visulinkDF = sublinkDF.loc[(sublinkDF.label=='Rem')|(sublinkDF.label=='Rem/Clr'),]
    
    return(visulinkDF)

def drawSankey(wd, fName, acc, visuType, visulinkDF, labels, colors):
    """
    draw Sankey diagram
    :param wd current working directory
    :param fName folder name containing raw data
    :param acc accession number of the item
    :param visuType type of actions to be shown
    :param visulinkDF linkDF for showing
    :param labels labels for nodes
    :param colors colors for nodes
    """
    data = dict(type = 'sankey',
            node = {'pad': 30, 'thickness': 20, 'line': dict(color = "gray", width = 0.5), 'label': labels, 'color': colors},
            link = {'source': visulinkDF['source'], 'target': visulinkDF['target'], 'value': visulinkDF['value'], 'label': visulinkDF['label'], 'color': visulinkDF['color']}
          )
    layout = dict(title = acc + "; " + visuType + " actions", font = dict(size = 20))
    # draw
    fig = dict(data=[data], layout=layout)
    plotly.offline.plot(fig, validate=False, filename=os.path.join(wd, fName, "Sankey_" + acc + "_" + visuType + ".html"))
    

fName = 'clean'
AccNumList = ['VH139047', 'VH269376', 'VH302862'] # these are the drag and drop items

sourceDF = pd.DataFrame({'AccNum': ['VH139047', 'VH139047', 'VH139047', 'VH139047', 
                                    'VH269376', 'VH269376', 'VH269376', 'VH269376', 'VH269376', 'VH269376', 
                                    'VH302862', 'VH302862', 'VH302862', 'VH302862', 'VH302862'], 
                       'Source': ['s1', 's2', 's3', 's4', 
                                  's1', 's2', 's3', 's4', 's5', 's6',
                                  's1', 's2', 's3', 's4', 's5'],
                       'Label': ['1', '2', '6', '7', 
                                 '1/3', '2/3', '2/6', '4/6', '2/8', '4/8',
                                 '0.02', '0.20', '0.25', '2.0', '2.5']})

coldic = ['gray', 'yellow', 'blue', 'red', 'cyan', 'green', 'black'] # make sure cover the maximum number of sources (6)

keepPerc = 10; minValue = 5

# create result directory
os.makedirs(os.path.join(wd, "SankeyDiagrams"), exist_ok=True)

for acc in AccNumList:
    print(acc)
    
    dat = pd.read_csv(os.path.join(wd, fName, acc, acc + "_Seq.csv"))
    print(len(dat.BookletNumber.unique()), " students (", len(dat.BookletNumber[dat.DSEX==2]), " females)")
    # replace source as s and target as t
    dat.loc[:,'Seq'] = dat.Seq.apply(lambda x: x.replace("Rem_NaN;", "").replace("Add_;", "").replace("__", "_").replace("source_", "s").replace("target_", "t"))    
    dat.loc[:,'SeqLen'] = dat.Seq.apply(lambda x: len(x.split(";")))
    dat.to_csv(os.path.join(wd, fName, acc + "_all.csv")) # for R to draw distribution of sequence length
    
    listS, listT = getTotST(dat) # get source and target 
    
    # 1) add state column
    print("Get states!")
    dat = (dat.groupby("BookletNumber", as_index = False).apply(addState, listS = listS, listT = listT, switch = 'off')).reset_index(drop=True)
    
    # 2) get labels for node
    print("Get nodes!")
    labels = getLabels(dat)
    colors = []
    for label in labels:
        # calculate how many numbers in each label
        numberList = flattenNestedList([x.split(",") if "," in x else x for x in label.split("|")])
        numberNo = sum([1 if x !='NA' else 0 for x in numberList])
        if numberNo > len(listS): numberNo = len(listS)
        colors.append(coldic[numberNo])
        
    # 3) get linkDF for link
    print("Get links!")
    linkDF = (dat.groupby("BookletNumber", as_index = False).apply(getLink, labels = labels, listS = listS, listT = listT, switch = 'off')).reset_index(drop=True)   
    linkDF = (linkDF.groupby(["source", "target"], as_index = False).apply(mergeValue, switch = 'off')).reset_index(drop=True)
    linkDF = linkDF.sort_values(by=['color','value'], ascending=False).reset_index(drop=True)
    
    # 4) keep the most frequent state in each step
    print("Get subLinkDF!")
    sourcetarget = 'target'; sublinkDF = getSubLink(linkDF, keepPerc, minValue, labels, colors, listS, listT, coldic, sourcetarget)
    
    # 5) replace s1,s2,... to real label in the question
    for source in sourceDF.Source[sourceDF.AccNum==acc]:
        labels = [x.replace(source, sourceDF.loc[(sourceDF.AccNum==acc)&(sourceDF.Source==source), 'Label'].iloc[0]) for x in labels]
    
    # 6) draw Sankey diagram
    print("Draw diagrams!")
    resdir = os.path.join(wd, "SankeyDiagrams")
    visuType = 'All'; visulinkDF = getvisuLinkDF(sublinkDF, visuType); drawSankey(resdir, fName, acc, visuType, visulinkDF, labels, colors)
    visuType = 'Add'; visulinkDF = getvisuLinkDF(sublinkDF, visuType); drawSankey(resdir, fName, acc, visuType, visulinkDF, labels, colors)
    visuType = 'Same'; visulinkDF = getvisuLinkDF(sublinkDF, visuType); drawSankey(resdir, fName, acc, visuType, visulinkDF, labels, colors)
    visuType = 'Rem'; visulinkDF = getvisuLinkDF(sublinkDF, visuType); drawSankey(resdir, fName, acc, visuType, visulinkDF, labels, colors)
    
    
    
    
    
    