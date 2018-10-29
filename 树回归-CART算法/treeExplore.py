# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:51:21 2018

@author: 12100
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:51:23 2018

@author: 12100
"""
import numpy as np
import tkinter as tk
import matplotlib as mplt
mplt.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


import Tree_Cart

def reDraw(tolS, tolN):
    pass


def drawNewTree():
    pass


def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = Tree_Cart.createTree(reDraw.rawDat, Tree_Cart.modelLeaf, Tree_Cart.modelErr,(tolS, tolN))
        yHat = Tree_Cart.createForeCast(myTree, reDraw.testData, Tree_Cart.modelTreeEval)
    else:
        myTree = Tree_Cart.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = Tree_Cart.createForeCast(myTree, reDraw.testData)
    reDraw.a.scatter(reDraw.rawDat[:,0], reDraw.rawDat[:,1],s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()
    

    
    
def getInput():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0,mplt.END)
        tolNentry.insert(0,'10')
    try:
        tolS = float(mplt.tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tols")
        mplt.tolSentry.delete(0,mplt.END)
        tolNentry.insert(0,'1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = mplt.getInputs()
    reDraw(tolS,tolN)


root = tk.Tk()

tk.Label(root,text="Plot Please Holder").grid(row=0, columnspan=3)

tk.Label(root, text="tolN").grid(row=1,column=0)
tolNentry = tk.Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')

tk.Label(root, text="tolS").grid(row=2,column=0)
tolNentry = tk.Entry(root)
tolNentry.grid(row=2,column=1)
tolNentry.insert(0,'1.0')

tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2,rowspan=3)

chkBtnVar = tk.IntVar()
chkBtnVar = tk.Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtnVar.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = np.mat(Tree_Cart.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]),0.01)

reDraw(1.0, 10)

root.mainloop()
































