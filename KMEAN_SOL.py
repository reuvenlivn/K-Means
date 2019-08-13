# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:28:14 2019

@author: reuve
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data

#plot the next iteration
def iteretion_plot(data, labels, centers, iter):
    figure(figsize=(8, 6))
    plt.title('Iteretion {}'.format(iter))
    plt.scatter(data[:, 2], data[:, 1], c=labels, marker ="*", s=50);
    plt.scatter(centers[:, 2], centers[:, 1], c='red', s=100, marker ="o",alpha=0.8);

#did we convergenced? 
#is there any difference between old and new centers??  
def is_convergenced(old_centers, new_centers):
  return (True == np.all(old_centers == new_centers))

# check an instance distanc from all the centers. pick the best one.
def single_label(instance, centers):
#    NO_centers = np.size(centers,0)
    min_dist = float('Infinity')
    for i in range(np.size(centers,0)):
        #calculate distance from p to the current center
        dist = np.linalg.norm(instance-centers[i])  
        if dist<min_dist:
            min_dist = dist
            label = i
    return label
   
#set the labels of all the instances   
def set_labels(data,centers):
#    data_size = np.size(data,0)  
    labels = []
    for i in range(np.size(data,0)):
        label = single_label(data[i],centers)
        labels.append(label)
    return np.asarray(labels)

#find the clusters   
def find_clusters(X, k):
    # init centers as random points
    centers = np.array(random.sample(list(X), k))
    iter=0
    
    while True:

       # Assign labels based on closest center
        labels = set_labels(X, centers)
               
        # Find new centers from means of points for every label
        new_centers = np.array([X[labels == i].mean(0) for i in range(k)])
        
        iter += 1
        iteretion_plot(data, labels, new_centers, iter)
        # Check for convergence
        if is_convergenced(centers, new_centers):
            break
        
        centers = new_centers
     
    return centers, labels

centers, cluster_labels = find_clusters(data, 3)
#print_clusters(data,cluster_labels,centers)

