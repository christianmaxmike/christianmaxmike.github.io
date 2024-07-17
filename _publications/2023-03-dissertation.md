---
title: "[Dissertation] Learning from Complex Networks"
collection: publications
permalink: /publications/2023-03-dissertation
venue: 'LMU Munich'
excerpt: "Graph Theory has proven to be a universal language for describing modern complex systems. The elegant theoretical framework of graphs drew the researchers' attention over decades. Therefore, graphs have emerged as a ubiquitous data structure in various applications where a relational characteristic is evident. Graph-driven applications are found, e.g., in social network analysis, telecommunication networks, logistic processes, recommendation systems, modeling kinetic interactions in protein networks, or the 'Internet of Things' (IoT) where modeling billions of interconnected web-enabled devices is of paramount importance."
date: 2023-11-10
paperurl: https://edoc.ub.uni-muenchen.de/32636/
citation: "Learning from Complex Networks, Christian Maximilian Michael Frey, Dissertation, 2023<br/>"
---

## Abstract
Graph Theory has proven to be a universal language for describing modern complex systems. The elegant theoretical framework of graphs drew the researchers' attention over decades. Therefore, graphs have emerged as a ubiquitous data structure in various applications where a relational characteristic is evident. Graph-driven applications are found, e.g., in social network analysis, telecommunication networks, logistic processes, recommendation systems, modeling kinetic interactions in protein networks, or the 'Internet of Things' (IoT) where modeling billions of interconnected web-enabled devices is of paramount importance. 



This thesis dives deep into the challenges of modern graph applications. It proposes a robustified and accelerated spectral clustering model in homogeneous graphs and novel transformer-driven graph shell models for attributed graphs.

A new data structure is introduced for probabilistic graphs to compute the information flow efficiently. Moreover, a metaheuristic algorithm is designed to find a good solution to an optimization problem composed of an extended vehicle routing problem. The thesis closes with an analysis of trend flows in social media data.



Detecting communities within a graph is a fundamental data mining task of interest in virtually all areas and also serves as an unsupervised preprocessing step for many downstream tasks. One most the most well-established clustering methods is Spectral Clustering. However, standard spectral clustering is highly sensitive to noisy input data, and the eigendecomposition has a high, cubic runtime complexity O(n^3). Tackling one of these problems often exacerbates the other. This thesis presents a new model which accelerates the eigendecomposition step by replacing it with a Nyström approximation. Robustness is achieved by iteratively separating the data into a cleansed and noisy part of the data. In this process, representing the input data as a graph is vital to identify parts of the data being well connected by analyzing the vertices' distances in the eigenspace.



With the advances in deep learning architectures, we also observe a surge in research on graph representation learning. 

The message-passing paradigm in Graph Neural Networks (GNNs) formalizes a predominant heuristic for multi-relational and attributed graph data to learn node representations. In downstream applications, we can use the representations to tackle theoretical problems known as node classification, graph classification/regression, and relation prediction. However, a common issue in GNNs is known as over-smoothing. By increasing the number of iterations within the message-passing, the nodes' representations of the input graph align and become indiscernible. 

This thesis shows an efficient way of relaxing the GNN architecture by employing a routing heuristic in the general workflow. Specifically, an additional layer routes the nodes' representations to dedicated experts. Each expert calculates the representations according to their respective GNN workflow. The definitions of distinguishable GNNs result from k-localized views starting from a central node. This procedure is referred to as Graph Shell Attention (SEA), where experts process different subgraphs in a transformer-motivated fashion.



Reliable propagation of information through large communication networks, social networks, or sensor networks is relevant to applications concerning marketing, social analysis, or monitoring physical or environmental conditions. However, social ties of friendship may be obsolete, and communication links may fail, inducing the notion of uncertainty in such networks. This thesis addresses the problem of optimizing information propagation in uncertain networks given a constrained budget of edges. 

A specialized data structure, called F-tree, addresses two NP-hard subproblems: the computation of the expected information flow and the optimal choice of edges. The F-tree identifies independent components of a probabilistic input graph for which the information flow can either be computed analytically and efficiently or for which traditional Monte-Carlo sampling can be applied independently of the remaining network.



The next part of the thesis covers a graph problem from the Operations Research point of view. A new variant of the well-known vehicle routing problem (VRP) is introduced, where customers are served within a specific time window (TW), as well as flexible delivery locations (FL) including capacity constraints. The latter implies that each customer is scheduled in one out of a set of capacitated delivery service locations. Practically, the VRPTW-FL problem is relevant for applications in parcel delivery, routing with limited parking space, or, for example, in the scope of hospital-wide scheduling of physical therapists. This thesis presents a metaheuristic built upon a hybrid Adaptive Large Neighborhood Search (ALNS). Moreover, a backtracking mechanism in the construction phase is introduced to alter unsatisfactory decisions at early stages. In the computational study, hospital data is used to evaluate the utility of flexible delivery locations and various cost functions. 



In the last part of the thesis, social media trends are analyzed, which yields insights into user sentiment and newsworthy topics. Such trends consist of bursts of messages concerning a particular topic within a time frame, significantly deviating from the average appearance frequency of the same subject. This thesis presents a method to classify trend archetypes to predict future dissemination by investigating the dissemination of such trends in space and time.



Generally, with the ever-increasing scale and complexity of graph-structured datasets and artificial intelligence advances, AI-backed models will inevitably play an important role in analyzing, modeling, and enhancing knowledge extraction from graph data.
