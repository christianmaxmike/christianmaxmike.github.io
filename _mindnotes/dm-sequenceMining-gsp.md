---
title: "GSP"
topic: dm-sequenceMining
collection: dm-sequenceMining
permalink: /mindnotes/dm-sequenceMining-gsp
---


<img src="logo_cmmf.png"
     alt="Markdown Monster icon"
     style="float: right" />
# MindNotes - Data Mining - Sequence Mining

**Author: Christian M.M. Frey**  
**E-Mail: <christianmaxmike@gmail.com>**

---

## Generalized Sequential Pattern
---

### Implementation of the Generalized Sequential Pattern (GSP) Algorithm

In this tutorial, we will implement the generalized sequential pattern (GSP) algorithm. The motivation is that in many application the order matters, e.g. because the ordering encodes spatial or temporal aspects (bioinformatics (DNA/protein sequences), Web mining, text mining, sensor data mining).

In sequential pattern mining the task is to find all frequent subsequences occuring in our transactions dataset. Note that the number of possible patterns is even larger than for frequent itemset mining (for an alphabet $\Sigma$ and sequences of length $k$, there are $\|\Sigma\|^k$ different $k-sequences$).


#### Load dependencies


```python
from collections import Counter
from itertools import chain, product
import numpy as np
```

#### Generate data
Let's consider the following transaction dataset:


```python
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diaper', 'Beer', 'Eggs'],
    ['Milk', 'Diaper', 'Beer', 'Coke'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk', 'Diaper', 'Coke']
]
```

#### Implementation of the GSP algorithm
The steps of the GSP algorithm are as follows:

* Breadth-first search: Generate frequent sequences ascending by length
* Given the set of frequent sequences at level $k$, generate all possible sequence extensions or candidates at level $k+1$
* Uses the  APriori principle (anti-monotonicity)
* Next compute  the  support of each candidate and prune the ones with $supp(c)<minSup$
* Stop the search when no more frequent extensions are possible


```python
class GSP (object):
    """ Class for the generalized sequential pattern algorithm
    
    In sequential pattern mining the trask is to find all frequent 
    subsequences occuring in our transactions dataset.
    
    Arguments:
      trans: the transactions being mined 
      
    Properties:
      freq_patterns : variable storing the frequent patterns being found
      transactions: variable containing the transactions
      max_size: maximal length ot a transacction in the set of transactions
      unique_candiates: set of candidates
    """    
    def __init__ (self, trans):
        self.freq_patterns = []
        self._preprocess(trans)
    
    def _preprocess(self, trans):
        """
        Method used for preprocessing the dataset containing the set of 
        transactions. 
        
        Sets the values for the properties max_size, transactions, and unique candidates.
        - max_size: will be set to the length of the longest transaction.
        - transactions: we will convert the transactions to tuples as tuples are immutable
        - unique_candidates: a list containing the unique candidates (as tuples) ,e.g., tuple([item]) 
            such that 'item' is not splitted up into 'i', 't', 'e', 'm'
        
        Arguments:
          trans: the transaction database being used
        """
        self.max_size = max([len(i) for i in trans])
        self.transactions = [tuple(list(i)) for i in trans]
        counts = Counter(chain.from_iterable(trans))
        self.unique_candidates = [tuple([k]) for k, c in counts.items()]
        
    def _is_subseq_in_list(self, slice_, transaction) -> bool:
        """
        The method _is_subseq_in_list(.) yields a boolean value indicating whether
        a sequence of items - being attached as the parameter 'slice_' - can be 
        found in the attached transaction.
        
        Arguments:
          slice_: the slice consists of a sequence of items which we try to find
            in the transaction database
          transaction: transaction in which we try to find the slice_
        
        Returns:
          Boolean indicating if the slice_ could be found at any consecutive sequence in the transaction or not
        """
        len_s = len(slice_)
        return any(slice_ == transaction[i:len_s+i] for i in range(len(transaction) - len_s + 1))
        
    def _get_frequency(self, results, item, minsup) -> dict:
        """
        Computes the frequency of an item and yields it in case of the frequency
        being greater than the minimum support.
        
        Arguments:
          results: dictionary containing the frequency of each item above the minimum support
          item: item for which we will compute the frequency
          minsup: minimum support being used for the frequent itemsset mining
        
        Returns:
          dictionary containing the frequency of the item being attached as parameter
        """
        
        # compute frequency of the attached 'item' in the set of 'self.transactions'
        f = len([t for t in self.transactions if self._is_subseq_in_list(item, t)])
        
        # check if the frequency is greater than the minSupport. If that's the case store the result in the 
        # dictionary 'results'
        if f >= minsup:
            results[item] = f
    
    def _get_support(self, items, minsup=0) -> dict:
        """
        yields the support of each item in the variable items being attached as a 
        parameter. 
        
        Arguments:
          items: items whose support are computed
          minsup: minimum supprt of the frequent itemset mining procedure
        
        Returns:
          dictionary containing the frequencies of all items (probably sequence
          of items) being above the minimum support
        """
        results = dict()
        
        # compute the frequency for each item being stored in the list 'items'
        for i in items:
            self._get_frequency(results, i, minsup)
            
        # return the dictionary containing the results, i.e., the items with their specific frequency
        return results
        
    def search(self, minsup=.4):
        """
        search procedure executing the GSP algorithm 
        
        Arguments:
          minsup: minimum support being used in the frequent itemset procedure
          
        Returns: 
          frequent itemset being identified by the generalized sequential pattern
          mining procedure
        """
        minsup = len(self.transactions) * minsup
        candidates = self.unique_candidates
        
        # append candidates for which the support has been calculated to be greater than minSupport 
        # (call _get_support()) to the list of frequent patterns
        self.freq_patterns.append(self._get_support(candidates, minsup))
        
        # while we encouter a frequent pattern in the list storing frequent pattern and that
        # we haven't reached the maximal size, we produce new candidates according to the apriori 
        # algorithm and calculate the new set of frequent pattern and append it to the class variable.
        # In order to get the candidates, one possiblity coud be to first identify and then take the cartesian
        # product of those unique elements (hint: itertools.product(...))
        k_items = 1
        while len(self.freq_patterns[k_items - 1]) and (k_items + 1 <= self.max_size):
            k_items += 1
            items = np.unique(list(set(self.freq_patterns[k_items - 2].keys())))
            candidates = list(product(items, repeat=k_items))
            self.freq_patterns.append(self._get_support(candidates, minsup))
        return self.freq_patterns[:-1]
```


```python
GSP(transactions).search()
```




    [{('Bread',): 4, ('Milk',): 4, ('Diaper',): 4, ('Beer',): 3, ('Coke',): 2},
     {('Bread', 'Milk'): 3, ('Diaper', 'Beer'): 3, ('Milk', 'Diaper'): 3},
     {('Bread', 'Milk', 'Diaper'): 2, ('Milk', 'Diaper', 'Beer'): 2}]



#### Additional notes:
The GSP algorithm has been proposed in :
 * Sirkant & Aggarwal: Mining sequential patterns: Generalizations and performance improvements. EDBT 1996

# End of this MindNote
