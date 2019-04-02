options(java.parameters = "-Xmx64g")
#.libPaths("C:/Users/installer/Documents/R/win-library/3.5")
library(RJDBC)
library(data.table)
library(tm)
library(reticulate)
library(scales)
library(tidyverse)
library(readxl)
library(tidytext)
library(text2vec)
library(Matrix.utils)
library(keras)
library(caret)
library(bit64)
library(arules)
library(text2vec)
DIR = getwd()
filenames <- list.files(paste0(DIR,"/"), pattern="*.csv", full.names=TRUE)
ldf <- lapply(filenames, function(x) fread(x, stringsAsFactors = TRUE ,check.names = TRUE))
# dt <- rbindlist(ldf,fill = TRUE)#####
# setnames(dt,names(dt),tolower(names(dt)))
dt_nyt <- ldf[[1]]
dt_chg <- ldf[[2]]
dt_lv <- ldf[[3]]
## columns to join 1) DBA , DBA.NAME, Restaurant Name ,
###### summary statistics
summary(dt)
### column names
setnames(dt_nyt,names(dt_nyt), tolower(names(dt_nyt)))
setnames(dt_chg,names(dt_chg), tolower(names(dt_chg)))
setnames(dt_lv,names(dt_lv), tolower(names(dt_lv)))
#### fixing the column names of each dataset . In order to append them 
setnames(dt_chg,"dba.name","dba")
setnames(dt_lv,"restaurant.name","restaurant")
setnames(dt_chg,"dba.name","dba")
setnames(dt_lv,"inspection.result","result")
setnames(dt_nyt,"action","result")
# appending dataset
ldf <- list(dt_nyt,dt_chg,dt_lv)
dt <- rbindlist(ldf,fill = TRUE)#####
####DBA,INSPECTION.DATE,GRADE,inspection.type
### ML pipeline to predict inspection result based on restaurant name's letters as an input
## predictior == "dba"
## target == "result
### drop rows with missing value
### Total number of records 739K data points
dt = na.omit(dt[,.(dba,result)])
### remove missing and blank values from result
dt = dt[!(is.na(dt$result) | dt$result==""),]
####
dt[, inspection_passed := ifelse(grepl("No violations",result),"pass","fail")]
####
## number of failed inspection == 378K 
### number of pass inspection == 4.5K
dt[,.N,by ="inspection_passed"]
## to lower dba names
dt[, dba := tolower(dba)]
###
train_tokens = dt$dba %>% word_tokenizer()

it_train = itoken(train_tokens, progressbar = T)
vocab = create_vocabulary(it_train, ngram = c(ngram_min = 1L,ngram_max = 3L))
# ###View(vocab) Top 10 words used in a restaurant name

# ### filter fault codes that has >= 100 jobs
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 10
                                #doc_proportion_max = 0.5,
                                #doc_proportion_min = 0.00001,
                                #vocab_term_max = 500000
)
#vectorizer = vocab_vectorizer(pruned_vocab)
#saveRDS(vectorizer,paste(DIR,paste0(x,"vectorizer.RDS"),sep = "/"))
#dtm_trans = create_dtm(it_train, vectorizer)
#### 
### text toeknization and embedding
tokenize_seq <- function(dt_featurize,x , char_level = FALSE) {
  print(x) 
  x = "dba"
  train_tokens = dt[,x,with = FALSE][[1]] %>% space_tokenizer(sep = ",")
  
  it_train = itoken(train_tokens, progressbar = T)
  vocab = create_vocabulary(it_train, ngram = c(ngram_min = 1L,ngram_max = 1L))
  # ###View(vocab)
  # ### filter fault codes that has >= 100 jobs
  pruned_vocab = prune_vocabulary(vocab,
                                  term_count_min = 10
                                  #doc_proportion_max = 0.5,
                                  #doc_proportion_min = 0.00001,
                                  #vocab_term_max = 500000
  )
  #print(dt_featurize[,x,with = FALSE])
  tokenizer <- text_tokenizer(oov_token = "unknown",filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",split = " ",char_level = char_level) %>%
    fit_text_tokenizer(dt_featurize[,x,with = FALSE][[1]])
  DIR = "/application/ramvora/Kaggle/tokenizer/"
  save_text_tokenizer(tokenizer,paste(DIR,paste0(x,"tokenizer"),sep = "/"))
  print(tokenizer)
  sequences <- texts_to_sequences(tokenizer , dt_featurize[,x,with = FALSE][[1]])
  word_index = tokenizer$word_index
  print(word_index)
  View(word_index)
  print(tokenizer$num_words)
  max_len = quantile(map_int(sequences, length), probs = .99)
  
  data <- pad_sequences(sequences ,
                        padding = "pre",
                        maxlen = max_len,
                        truncating = "post")
  return(data)
}
##
dba_embedding <- tokenize_seq(dt[,.(dba)],x ="dba")
#
dba_char_embedding <- tokenize_seq(dt[,.(dba)],x ="dba", char_level = TRUE)
### Target labelling
classes = unique(factor(dt$inspection_passed))
#classes = unique(factor(target_items$com_model_name))
i_inx = 1:nrow(dt) #these are just row indexes
j_inx = c() # I see what is the class index of each observation
for (j in 1:nrow(dt)) {
  j_inx[j] = which(classes %in% dt$inspection_passed[j])
  #j_inx[j] = which(classes %in% target_items$com_model_name[j])
}
####
# i_inx, j_inx is all that's needed to build the sparse matrix
labels = sparseMatrix(x = 1, i = i_inx, j = j_inx)
colnames(labels) = classes
###########

result <- dt[,.(result_cnt = .N),by = .(inspection_passed)][order(-result_cnt)] #[vmrs_cnt >= 100,]
result[,cum_sum := cumsum(result_cnt)]
result[,cum_sum_perc := 100*(cum_sum/nrow(dt))]
result = result[order(-result_cnt)]
#vmrs_cnt = vmrs_cnt[y != "",]
#vmrs_cnt[,y := clean_text(y)]
View(result)
result[, class_weight := (result_cnt + 191375) / result_cnt]
#com_model_name[, class_weight := (model_cnt + 98) / model_cnt]
y_train_weight <- data.table(inspection_passed = classes[j_inx], id = j_inx)
#y_train_weight <- data.table(com_model_name = classes[j_inx], id = j_inx)
y_train_weight = merge(result, y_train_weight , by = "inspection_passed")
#y_train_weight = merge(com_model_name, y_train_weight , by = "com_model_name")
y_train_weight = unique(y_train_weight[, .(inspection_passed, class_weight, id)])
#y_train_weight = unique(y_train_weight[, .(com_model_name, class_weight, id)])
#
lkp = data.table(result = classes , id = unique(j_inx))
lkp = lkp[, id := unique(j_inx)]
fwrite(lkp,paste0(DIR,"lkp.csv"))
##### y_train_weight[,id := seq(0, nrow(y_train_weight), 1)]
#I substract 1 to the class indeces so it works in python that is zero indexed (otherwise it throws an error)
class_weight = setNames(as.list(y_train_weight$class_weight), y_train_weight$id -1)
######## 

