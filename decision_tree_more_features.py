# -*- coding: utf-8 -*-
"""
Author: Leon Lu
Date: 20-June-2022
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from collections import namedtuple
from collections import UserList
from sklearn import tree

# new packages
from collections import defaultdict
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, MinMaxScaler
import os

# customized class
from Node import Node
from helper import convert_onehot_v1, preprocess
from load_config import load_yaml, dump_yaml, safe_load_yaml

## Initiaze logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger('microsegmenter')

# Set rule_outcome_map to be global
# This is a mapping from each rule to its other properties
# including total sample size, interested class sample size
rule_outcome_map = {}

class DecisionPath(UserList):
    def __init__(self, type):
        self.type = type
        self.data = []

    def append(self, object):
        UserList.append(self, object)
        
    def append(self, item):
        if not isinstance(item, self.type):
            raise TypeError('item is not of type %s' %(self.type))
        logger.info(f"Appending Rule {item.feature} {item.operator} {item.threshold}")
        #print("item", item)
        UserList.append(self, item)
        self.data = list(set(self.data))
        #print("data", self.data)
        logger.info("List of rules added to decision path so far")
        for feature,threshold,operator,dtype in self.data:
            logger.info(f"{feature} {threshold} {operator} , node type : {dtype}") # changed the order of operator and threshold, don't know the reason yet

     
class Track_DT():
    
    def __init__(self, clf, column_names, target_names, dtypes, interested_class, min_samples,
                                lift, base_rate, full_path = False, narrowed_condition = True, 
                                rule_outcome_map = rule_outcome_map):
        """
        

        Parameters
        ----------
        clf : classifier
            decision tree classifier
        column_names : np.array
            input feature column names
        target_names : np.array/list
            target category names
        dtypes : np.array/list
            data types for each column
        interested_class : int
            the class we are interested in
        min_samples : int
            minimum sample size we want for the interested class
        lift : int/float
            the target lift we want from the base rate
        base_rate : int/float
            the base conversion rate
        full_path : bool, optional
            Whether we want the whole tree paths. The default is False.
        narrowed_condition : bool, optional
            whether we want each decision path to filter narrowed condition. The default is True.
        rule_outcome_map : dict, optional
            mapping between Rule and its other attributes including current lift and total sample size for this node.
            The default is rule_outcome_map.

        Returns
        -------
        None.

        """
        
        self.n_nodes = clf.tree_.node_count # number of nodes in the tree
        self.children_left = clf.tree_.children_left # id of the left child of node i or -1 if leaf node
        self.children_right = clf.tree_.children_right # id of the right child of node i or -1 if leaf node
        self.feature = clf.tree_.feature # split on what features (feature on the leaf is not useful)
        self.feature_thresholds = clf.tree_.threshold # the value used to split
        self.samples = clf.tree_.n_node_samples # sample in each node
        self.feat_cols = column_names # features
        self.values = clf.tree_.value # number of sample in each node in vectors
        self.values = self.values.reshape(self.values.shape[0],self.values.shape[2]) # Reshaping to access in easy way 
        self.classes = clf.classes_ # target classes
        self.dtypes = dtypes # data types for each column
        self.target_names = target_names # each category name in the target column
        
        # four input values that can decide the final decision paths
        self.interested_class = interested_class 
        self.min_samples = min_samples
        self.lift = lift
        self.base_rate = base_rate
        
        # initialize the tree structure
        self.node_depth = np.zeros(shape = self.n_nodes, dtype = np.int64)
        self.is_leaves = np.zeros(shape = self.n_nodes, dtype = bool)
        self.parent = np.zeros(shape = self.n_nodes, dtype = np.int64)
        self.rules_left = [''] * self.n_nodes
        self.rules_right = [''] * self.n_nodes
        self.stack = [(0, 0, 0, 'root')]
        self.node_list = set()
        self.root_node = None
        self.nodeid_counter = 0
        self.decision_paths = [] # record decision paths
        self.visited_nodes = set()
        
        
        self.decision_features = set() # record features names in each path, used for removing duplicated feature sets while appending to the decision path list.
        self.rule_outcome_map = rule_outcome_map # mapping between rule to other attributes
        self.full_path = full_path
        self.narrowed_condition = narrowed_condition
        
        # call the following functions for building the tree, recording the paths, and show the output.
        self.initialize_nodes()
        self.visit_node(self.root_node, self.visited_nodes)
        self.summary_df = self.summary(self.decision_paths)
        
    def initialize_nodes(self):
        while len(self.stack) > 0:
            node_id, depth, parent_node, node_branch = self.stack.pop()
            self.node_depth[node_id] = depth
            feature_name = self.feat_cols[self.feature[node_id]]
            threshold = self.feature_thresholds[node_id]
            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = self.children_left[node_id] != self.children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            node_type = 'root' if node_id == 0 else ('leaf', 'inter')[is_split_node]
            dtype = self.dtypes[feature_name]
            
            node = Node(self.nodeid_counter, feature_name, threshold, parent_node,
                        node_type, dtype, depth, node_branch)
            node.samples = self.samples[node_id]
            node.classes = self.classes
            node.class_distribution_abs = self.values[node_id]
            node.class_distribution_perc = self.values[node_id] / np.sum(self.values[node_id])
            
            if node_type == 'root':
                self.root_node = node
                node.parent_node = node
            if node_type != 'root':
                if node_branch == 'left':
                    parent_node.left_child = node
                else:
                    parent_node.right_child = node
            
            if is_split_node:
                self.stack.append((self.children_left[node_id], depth + 1, node,'left'))
                self.stack.append((self.children_right[node_id], depth + 1, node,'right'))
            self.nodeid_counter += 1
            self.node_list.add(node.nodeid)
            
    def evaluate_microsegment(self, parent_node, current_node, interested_class,
                              min_samples, lift, base_rate):
        node_samples = current_node.samples
        classes = current_node.classes
        class_distribution_perc = current_node.class_distribution_perc
        index = np.where(classes == interested_class)
        class_dist = class_distribution_perc[index[0]]
        
        logger.info(f"Distribution(%) of classes in the node {classes} and total samples: {class_distribution_perc} , {node_samples}")
        logger.info(f"Distribution(%) of interested class in the node : {current_node.feature} -  {interested_class} : {class_dist}")
        
        flag = False
        if node_samples >= min_samples:
            if (class_dist * 100 / base_rate) > lift:
                flag = True
        logger.info(f"Evaluation of node to meet the microsegmentation criteria : {flag}")
        return flag
    
    def checkpoint_decisionpath(self, current_node):
    
        node = current_node.parent_node # Rule to be traced from parent node of current node all the way till root
        depth = node.depth
        node_branch = node.node_branch
        Rule = namedtuple('Rule',["feature","operator","threshold","feature_type"])
        Outcome = namedtuple('Outcome', ['interested_class', 'base_rate', 'total_sample_size', 'class_sample_size', 'target_lift', 'actual_lift'])
        
        dp = DecisionPath(Rule) 
        
        while(depth >= 0):
            feature = node.feature
            threshold = node.threshold
            operator = node.leftop if current_node.node_branch=='left' else node.rightop # changed from note to current_node
            feature_type = node.dtype
            rule = Rule(feature,operator,threshold,feature_type)
            dp.append(rule)
            
            interested_class = self.interested_class
            base_rate = self.base_rate
            total_sample_size = current_node.samples
            
            # record info about class_sample_size, class distribution
            # and put the info inside the rule_outcome_map dictionary.
            classes = current_node.classes
            index = np.where(classes == interested_class)
            class_sample_size = int(current_node.class_distribution_abs[index[0]][0])
            class_dist = (class_sample_size / total_sample_size) * 100 / base_rate
            actual_lift = round(class_dist, 2)
            target_lift = self.lift
            self.rule_outcome_map[rule] = Outcome(interested_class, base_rate, total_sample_size, class_sample_size, target_lift, actual_lift)
            depth -= 1
            node = node.parent_node # update parent node
            current_node = current_node.parent_node # update current node
        
        return dp
    
    # Loop through the current decision path
    # add all the feature names in the current decision path to feature_name set
    # set (feature name, operator) as keys of feature_dict, set (the index of the corresponding rule, current threshold) as values
    # use feature_dict to check duplicated rules
    def _loop_dp(self, current_dp):
        feature_name = set()
        feature_dict = defaultdict(list)
        for i, rule in enumerate(current_dp):
            cur_feature, cur_threshold, cur_operator = rule.feature, rule.threshold, rule.operator
            feature_name.add(cur_feature)
            feature_dict[(cur_feature, cur_operator)].append([i, cur_threshold])
        return tuple(feature_name), feature_dict
                

    # check if there are duplicated rules
    # e.g. a < 0.8 and a < 1.0 we choose a more narrowed condition: a < 0.8
    # if narrowed_condition = True, delete duplications
    def _check_duplicates(self, current_dp, feature_dict):
        print("current_dp: ", current_dp)
        new_dp = []
        # Loop through feature_dict
    
        for key, item in feature_dict.items():
            if len(item) == 1:
                new_dp.append(current_dp[item[0][0]])
            else:
                # Still do a check of the min/max constraint threshold for safety
                # in case the order get messed up but waste some computing time.
                cur_operator = key[1]
                if cur_operator == "<=":
                    append_index = min(item, key = lambda x: x[1])[0]
                else:
                    append_index = max(item, key = lambda x: x[1])[0] 
                new_dp.append(current_dp[append_index])
    
        return new_dp
    
    # append current decision path to the decision_paths list
    # full_path = True indicates we only want a path from root to leaf
    # full_path  = False indicates we accept subpath.
    # narrowed_condition = True indicates we want the path with
    # the most narrowed condition.
    def _append(self, decision_paths, current_dp, current_node, full_path = False, narrowed_condition = True):
        feature_name, feature_dict = self._loop_dp(current_dp)
        if full_path == True:
            node_type = current_node.node_type
            # No need to check duplications of full paths in the decision_paths list.
            if node_type == 'leaf':
                # the following if statement can be omitted
                if feature_name not in self.decision_features:
                    decision_paths.append(current_dp)
                    # to record the feature set we used this time
                    self.decision_features.add(feature_name)
        else:
            if narrowed_condition:
                # only keep the condition that has the narrowest range/threshold
                current_dp = self._check_duplicates(current_dp, feature_dict)
            if feature_name not in self.decision_features: # check if the current list contain the decision path
                decision_paths.append(current_dp)
                # record the feature set we used this time
                self.decision_features.add(feature_name)
        
    
    # print the decision by outputing a pandas dataframe from nested dictionary
    def summary(self, decision_paths):
        decision_dict = {}
        for i, decision in enumerate(decision_paths):
            constraint_dict = {}
            for j, constraint in enumerate(decision):
                rule = decision[j]
                cur_feature, cur_threshold, cur_operator, data_type = rule.feature, rule.threshold, rule.operator, rule.feature_type
                cur_outcome = self.rule_outcome_map[rule]
                interested_class, base_rate, total_sample_size,\
                    class_sample_size, target_lift, actual_lift = cur_outcome.interested_class,\
                        cur_outcome.base_rate, cur_outcome.total_sample_size,\
                            cur_outcome.class_sample_size, cur_outcome.target_lift, cur_outcome.actual_lift
                #interval = f"{cur_feature} {cur_operator} {cur_threshold}"
                constraint_dict[f"constraint {j}"] = {
                    "interested_class": interested_class,
                    "base_rate": base_rate,
                    "feature": cur_feature,
                    "operator": cur_operator,
                    "threshold": cur_threshold,
                    "total_sample_size": total_sample_size,
                    "interested_class_sample_size": class_sample_size,
                    'target_lift': target_lift,
                    "actual_lift": actual_lift,
                    "data type": data_type
                }
            decision_dict[f"decision {i}"] = constraint_dict
                
        summary_df = pd.DataFrame.from_dict(
            {
                (i, j): decision_dict[i][j]
                for i in decision_dict.keys()
                for j in decision_dict[i].keys()
            },
            orient = 'index'
        )   
        return summary_df
    
    def get_next_node(self, current_node, visited_nodes):
        if current_node.left_child!=None and current_node.left_child.nodeid not in visited_nodes:
            return current_node.left_child
        elif current_node.right_child!=None and current_node.right_child.nodeid not in visited_nodes:
            return current_node.right_child
        elif len(visited_nodes.intersection(self.node_list))==len(self.node_list):
            return -1
        else:
            print('Back Tracking!!')
            return current_node.parent_node  # backtrack
        
    def visit_node(self, current_node, visited_nodes):
        if current_node.node_type=='leaf':
            print(f"Branch: {current_node.node_branch}, Node type : {current_node.node_type} , Feature : {current_node.feature} , Parent : {current_node.parent_node.feature} , NodeID : {current_node.nodeid}") # change node to current_node
        else:
            print(f"Branch: {current_node.node_branch}, Node type : {current_node.node_type} , Feature : {current_node.feature} ,  Parent : {current_node.parent_node.feature} , Left Child : {current_node.left_child.feature} , Right Child : {current_node.right_child.feature} , NodeID : {current_node.nodeid}") # change node to current_node and add feature after child
    
        visited_nodes.add(current_node.nodeid)
        next_node = self.get_next_node(current_node,visited_nodes)
        
        
        ## Code to evalute if this node satisfies the microsegment criteria
        flag = self.evaluate_microsegment(current_node.parent_node,current_node,self.interested_class, self.min_samples, self.lift, self.base_rate) # Evaluate if node can be 
        if flag and current_node.node_type!='root':
            dp = self.checkpoint_decisionpath(current_node)
            #decision_paths.append(dp) # change it to an customized append function
            self._append(self.decision_paths, dp, current_node, self.full_path, self.narrowed_condition)
    
    #         for feature,threshold,operator,dtype in dp:
    #             logger.info(f"{feature} {operator} {threshold} , node type : {dtype}") 
            
        if next_node==-1:
            print('All nodes visited')
        else:
            visited_nodes.add(current_node.nodeid)
            self.visit_node(next_node,visited_nodes)
            
if __name__ == "__main__": 
     general_config = safe_load_yaml('general_config.yaml')
     sample_data = pd.read_csv(general_config['dataset']['data_path'])
     ordinal_encoder = OrdinalEncoder()
     onehot_encoder = OneHotEncoder()
     X, y, column_names, dtypes, target_names = preprocess(sample_data, ordinal_encoder,
                                             onehot_encoder)
     
     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

     clf = DecisionTreeClassifier(max_leaf_nodes=general_config['model']['decision_tree']['max_leaf_nodes'],
                                  random_state=general_config['model']['decision_tree']['random_state'])
     clf.fit(X_train, y_train)
     
     summary_dic = {}
     for i, (criteria_key, criteria_value) in enumerate(general_config['criteria'].items()):
         interested_class = criteria_value['interested_class']
         min_samples = criteria_value['min_samples']
         lift = criteria_value['lift']
         base_rate = criteria_value['base_rate']
         tree = Track_DT(clf, column_names, target_names, dtypes, interested_class, min_samples,
                     lift, base_rate, rule_outcome_map = {})
         summary_dic[f"summary {i}"] = tree.summary_df
         
         
         
         
         
         
         