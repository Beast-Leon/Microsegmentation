#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:46:05 2022

@author: leon
"""
class Node:
    
    nodeid = None 
    feature = None
    depth = None
    threshold = None
    
    ## Links info 
    left_child = None
    right_child = None
    parent_node = None
    node_branch = None
    node_type = None
    
    # Node data type 
    dtype = None
    
    ## Node type info 
    is_leaf = False
    is_root = False
    
    ## Samples,IG  info 
    samples = None
    values = {}
    classes = None
    class_distribution_abs = None #Hold absolute  values
    class_distribution_perc = None # hold percentage values
    
    ## Node rules
    leftop = None
    rightop = None
    
    
    def __init__(self,nodeid, feature, threshold, parent_node, node_type,
                 dtype, depth, node_branch):
        self.nodeid = nodeid
        self.feature = feature
        self.parent_node = parent_node
        self.node_type = node_type
        self.dtype = dtype
        self.leftop = '<=' #if self.dtype!='object' else '=='
        self.rightop = '>' #if self.dtype!='object' else '!='
        self.threshold = threshold
        self.node_branch = node_branch
        self.depth = depth
        