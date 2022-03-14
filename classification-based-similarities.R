# library(plyr) # for recoding data
library(dplyr)
library(tidytext)
library(ROCR) # for plotting roc
library(e1071) # for NB and SVM
library(rpart) # for decision tree
library(ada) # for adaboost
library(text2vec)
library(tidyr)

library(tm)
library(lsa)

library(data.table)
library(magrittr)
library(glmnet)
library(remotes)
library(stylo)

set.seed(12345) # set the seed so you can get exactly the same results whenever you run the code

data_test <- read.table("upcoming-wsdm10.txt", head=FALSE, comment.char = "", quote = "", sep ="\t", fill=T, encoding = "UTF-8")
colnames(data_test)<-c("author","title","description","filmTime","uploadTime",
                       "tags","location","GeoData","EventID")

temp <- data_test[data_test$title != "", ]
temp <- temp[temp$tags != "", ]
temp <- temp[temp$description != "", ]

corpus = Corpus(VectorSource(temp$title))
# corpus = tm_map(corpus, tolower) ## convert text to lower case
corpus = tm_map(corpus, removePunctuation) ## remove punctuations
corpus = tm_map(corpus, removeNumbers) ## remove numbers
corpus = tm_map(corpus, function(x) removeWords(x, stopwords("english"))) ## remove stopwords
corpus = tm_map(corpus, stemDocument, language = "english") ## stemming
tokens = word_tokenizer(tolower(corpus))
# tokens = word_tokenizer(tolower(iconv(corpus,"WINDOWS-1252","UTF-8")))

dtm = create_dtm(itoken(tokens), hash_vectorizer())
model_tfidf = TfIdf$new()
dtm_tfidf = model_tfidf$fit_transform(dtm)
dtm_tfidf <- as.matrix(dtm_tfidf)



# begin run here.

temp$id = 1:nrow(temp)

setDT(temp)
setkey(temp, id)
all_ids = temp$id
train_ids = sample(all_ids, 90000)
test_ids = setdiff(all_ids, train_ids)
train = temp[J(train_ids)]
test = temp[J(test_ids)]

prep_fun = tolower
tok_fun = word_tokenizer
it_train = itoken(train$title, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$id, 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)




train_tokens = tok_fun(prep_fun(train$title))

it_train = itoken(train_tokens, 
                  ids = train$id,
                  # turn off progressbar because it won't look nice in rmd
                  progressbar = FALSE)

vocab = create_vocabulary(it_train)
vocab


vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)


it_test = tok_fun(prep_fun(test$title))
it_test = itoken(it_test, ids = test$id, progressbar = FALSE)


dtm_test = create_dtm(it_test, vectorizer)

stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours")
vocab = create_vocabulary(it_train, stopwords = stop_words)


pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
# create dtm_train with new pruned vocabulary vectorizer
dtm_train  = create_dtm(it_train, vectorizer)
dtm_test = create_dtm(it_test, vectorizer)





vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# define tfidf model
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf = create_dtm(it_test, vectorizer)
dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)

cosine_similarity_matrix <- function(m){
  ret <- m %*% t(m) / (sqrt(rowSums(m^2) %*% t(rowSums(m^2))))
  return(ret)
}
cos_sim <- cosine_similarity_matrix(as.matrix(dtm_test_tfidf)[1:10000,])

cosine(as.matrix(dtm_test_tfidf)[1, ], as.matrix(dtm_test_tfidf)[2, ])

dist.cosine(as.matrix(dtm_test_tfidf)[1:10000, ])

val <- cosine(dtm_test_tfidf[1, ], dtm_test_tfidf[2, ])
mat <- cbind(dtm_test_tfidf[1, ], dtm_test_tfidf[2, ])
cosine(mat)


cos_sim_two <- function(line1, line2) {
  # line1 = dtm_test_tfidf[1, ]
  # line2 = dtm_test_tfidf[2, ]
  simtemp = cbind(c(1, 0), c(0, 1))
  if (sum(line1) != 0 && sum(line2) != 0) {
    simtemp = cosine(line1, line2)
  }
  return (simtemp)  
}
  
tempmat <- as.matrix(dtm_test_tfidf[1:10,])