# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:54:08 2018

@author: Lung
"""
import sys

def eclat(prefix, item_set, min_support, freq_items):

    while item_set:
        # trace所有的item
        key, item_tids = item_set.pop()
        item_support = len(item_tids)
        if item_support >= min_support:
            # print frozenset(sorted(prefix+[key]))
            freq_items[frozenset(prefix+[key])] = item_support
            suffix = []  # 存儲當前長度的list,對於每個item往下找
            for other_key, other_item_tids in item_set: #再取一個在item_set中的item
                new_item_tids = item_tids & other_item_tids  #與該item的transaction_ids的交集
                if len(new_item_tids) >= min_support:
                    suffix.append((other_key, new_item_tids))
                    next_item_set = sorted(suffix, key=lambda item_tids: len(item_tids[1]), reverse=True)
                    eclat(prefix+[key], next_item_set, min_support, freq_items)
    
    return freq_items

def seperateToLevels(freq_items):
    freq_list=[]
    freq_item_list = list(freq_items)
    for item_list in freq_item_list:
        freq_list.append((sorted(list(item_list)),freq_items[item_list]))
    freq_item_list = sorted(freq_list, key=lambda item: len(item[0]))
    
    index_list=[]
    tmp = 0
    for i in range(len(freq_item_list)):
        if(tmp != len(freq_item_list[i][0])):
            tmp = len(freq_item_list[i][0])
            index_list.append(i)
    index_list.append(len(freq_item_list))
    
    level_items_list=[]
    for i in range(len(index_list)-1):
        sub_list = freq_item_list[index_list[i]:index_list[i+1]]
        sub_list.sort()
        level_items_list.append(sub_list)
    
    return level_items_list

def roundSelf(number, totalnum):
    num_string = format(number/totalnum,'0.5f')
    if(num_string[-1]=='5' or num_string[-1]=='6' or num_string[-1]=='7' or num_string[-1]=='8' or num_string[-1]=='9'):
        num_string = num_string[0:-2]+str(int(num_string[-2])+1)
    else:
        num_string = num_string[0:-1]
    return num_string
    
def main(min_sup, in_file_name, out_file_name ):
    min_sup = float(min_sup)
    transaction_list = []
    
    with open(in_file_name,"r") as in_file:
        for line in in_file:
            item_list = line.strip().split(',')
            #transaction_list.append(item_list)  # str list
            transaction_list.append(list(map(int, item_list)))  # int list
    min_sup_num = min_sup*len(transaction_list)

    #convert to vertical format
    data = {}
    transaction_id = 0
    for transaction in transaction_list:
        for item in transaction:
            if item not in data:
                data[item]=set()
            data[item].add(transaction_id)
        transaction_id += 1
        
    freq_items = {}
    #由support高到低去排列item
    sorted_item_set = sorted(data.items(), key=lambda item: len(item[1]), reverse=True)
    #calculate frequent item sets
    freq_items = eclat([], sorted_item_set, min_sup_num, freq_items)
    #split freq_items by their size
    levels_list = seperateToLevels(freq_items)
    
    with open(out_file_name,'w') as out_file:
        for one_level_list in levels_list:
            for freq_tuple in one_level_list:
                length = len(freq_tuple[0])
                for i in range(length-1):
                    out_file.write("{},".format(freq_tuple[0][i]))
                out_file.write("{}:{}\n".format(freq_tuple[0][-1],roundSelf(freq_tuple[-1],len(transaction_list))))
                
if __name__ == "__main__":
    min_sup = sys.argv[1]
    main(min_sup, sys.argv[2], sys.argv[3])