# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 22:24:00 2018

@author: Lung
"""
import sys
from treeNode import TreeNode

def checkItemExist(item, treeNode_list):
    for node in treeNode_list:
        if item == node.name:
            return node
    return None

def main(min_sup, in_file_name = "sample2.in", out_file_name = "result.out" ):
    min_sup = float(min_sup)
    transaction_list = []
    
    with open(in_file_name,"r") as in_file:
        for line in in_file:
            item_list = line.strip().split(',')
            transaction_list.append(item_list)  # str list
            #transactions.append(list(map(int, item_list)))  # int list
    min_sup_num = min_sup*len(transaction_list)
    
    #build F-list
    F_list = {}
    for item_list in transaction_list:
        for item in item_list:
            if item in F_list:
                F_list[item] += 1
            else:
                F_list.update({item:1})
    F_list = {key:val for key, val in F_list.items() if val>=min_sup_num}
    F_list = sorted(F_list.items() ,key= lambda x:x[1], reverse=True)
    F_list = [list(x) for x in F_list]
    
    #generate ordered frequent item
    freq_item_list = []
    for item_list in transaction_list:
        order_item_list=[]
        for freq_pattern in F_list:   
            if freq_pattern[0] in item_list:
                order_item_list.append(freq_pattern[0])
        freq_item_list.append(order_item_list)

    #delete database
    del transaction_list
        
    #construct FPtree and header table
    header_table = {}
    head = TreeNode("top")
    for order_item_list in freq_item_list:
        prev = head
        for item in order_item_list:
            #item 在前一個node的children沒有被找到,所以要增加一個node,再往下繼續長 
            finded_treeNode = checkItemExist(item, prev.children)
            if finded_treeNode is None:
                newChild = TreeNode(item,prev)
                prev.appendNode(newChild)
                prev = newChild
                #在header table中已經有過同個key的tree node,找到此tree node的最後並接在後面
                if newChild.name in header_table:
                    node_ptr = header_table[newChild.name]
                    while node_ptr.next is not None:
                        node_ptr = node_ptr.next
                    node_ptr.setNext(newChild)
                #在header table中未曾有出現過此tree node的key,所以新增一個
                else:
                    header_table[newChild.name] = newChild
            #item 已在前一個的children 所以不需要增加node而是增加count,再往下繼續長
            else:
                existChild = prev.children[prev.children.index(finded_treeNode)]
                existChild.increment()
                prev = existChild

    #head.show_info()
    
if __name__ == "__main__":
    min_sup = sys.argv[1]
    main(min_sup, sys.argv[2], sys.argv[3])
