#### Data spliting
#####
#setnames(dt,"vmrs_33" , "vmrs_cd")
inx_train = as.vector(createDataPartition(
  y = factor(dt$inspection_passed),
  #y = factor(target_items$com_model_name),
  p = 0.8,
  list = FALSE
)[, 1])
inx_test = setdiff(1:nrow(dt), inx_train)

dba_embedding_train <- dba_embedding[inx_train,]
dba_embedding_val <-  dba_embedding[inx_test,]
  
dba_char_embedding_train <- dba_char_embedding[inx_train,]
dba_char_embedding_val <-   dba_char_embedding[inx_test,]
  

# one_hot_encode_train = one_hot_encode[inx_train, ]
# one_hot_encode_val = one_hot_encode[inx_test, ]
y_train = labels[inx_train, ]
y_val = labels[inx_test, ]
#####
