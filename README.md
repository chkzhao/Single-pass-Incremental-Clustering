# Final Project For INFSCI 2160 Course
Reproducing the orignal paper - <em>Learning Similarity Metrics for Event Identification in Social Media</em>.

* Team members: Chenkai Zhao, Ziqian Wu, Ming Gao, Youyang Feng
* Project paper: https://github.com/chkzhao/Single-pass-Incremental-Clustering/blob/main/Reports/2160%20Final%20Project_404NotFound.pdf
* Original paper: https://github.com/chkzhao/Single-pass-Incremental-Clustering/blob/main/original%20paper.pdf

To view the dataset, click [here](https://github.com/chkzhao/Single-pass-Incremental-Clustering/blob/main/data.zip)

## Description

{A summary of what the project is and does, the technology it employs, and the purpose behind the project.}
{Provide a summary for the major directories and files and what they do. You can also provide a table of content here and link to seperate readme files.}

  In this paper, we aim to reproduce the work from <em>Learning Similarity Metrics for Event Identification in Social Media </em>, by Hila Becker et al. In order to reproduce the original result as close as possible, we implemented single pass incremental clustering, TF-IDF, classification-based similarities matrix, logistic regression, NMI, B-cubed score, and many other technologies.
  The purpose behind the project comes from both academic life and reality. Academically, this paper is an extension from our topics covered at class, not totally unreachable, but slightly challenging, which provides us a great chance to better understand data mining and observe how to apply these technologies in academic research. On the other hand, this paper does point out an real-life issue that demands a quick solution. Skyrocketing development of social media is accompanied by an exponential increase of number of texts, pcitures, videos, and other medium that carries information. Traditionally, people manually archive these posts by re-reading them, however, such approach is now less and less feasible, when our life keeps generates tons of works to be re-read. A more automatic and modern solution is in great need, pushing the author to research and eventually to find out one solution as she explained in the paper.

  classification-based-similarities.R, computes the TF-IDF and other analysis result to measure the similarities between documents, train logistic regression model, and use NMI and B-cubed to evaluate the performance of it.

**single_pass_incremental_clustering.py :**

Provide the algorithm need to cluster the data for each feature, like the single pass incremental cluster algorihtm and it's help function.

**single_pass_incremental_clustering.Rmd  :**

Data Preprocessing and complete the overall computation flow.

**Ensemble_based_similarity_clustering.py :**

Provide the key component of the ensemble cluster based algorithm, including single pass cluster centroid ensemble voting algorithm and document pairs similarity voting algorithm

**ensemble_based_similarity_clustering.Rmd :**

Data Preprocessing and complete the overall computation flow.


## Prerequisites
    Below are the R packages necessary for this project:
    
    dplyr, tidytext, ROCR, e1071, rpart, ada, text2vec, tidyr, tm, lsa, 
    data.table, magrittr, textstem, stringr, e1071, reticulate, aricode, DPBBM
    
    Below are the Python packages necessary for this project:
    Numpy
    
    

## Authors  
    Chenkai Zhao chz97@pitt.edu
    
    Ming Gao mig82@pitt.edu
    
    Ziqian Wu ziw37@pitt.edu
    
    Youyang Feng yof15@pitt.edu  
