# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:11:43 2018

@author: Lung
"""

class TreeNode:
    def __init__(self, name, parent=None, count=1):
        self.name = name
        self.parent = parent
        self.count = count
        self.children = []
        self.next = None
    
    def increment(self):
        self.count +=1
    
    def appendNode(self, node):
        self.children.append(node)
        
    def setNext(self, nextnode):
        self.next = nextnode
        
    def show_info(self, level = 1):
        with open('tree.txt','a') as f:
            f.write("{}{}-{}\n".format("  "*level,self.name,self.count))
        for child in self.children:
            child.show_info(level + 1)
