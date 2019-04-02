#classes_unit <- 10018 #length(classes) #685
classes_unit <- length(classes) #2427
#char_index_len <- length(tokenizer_char$word_index)
################ categorical layer
#declare
#########

dba_embedding_featurize_cd <- 
  layer_input(shape = 6,
              dtype = 'float32',
              name = 'dba_embedding_featurize_cd')
dba_embedding_input <- dba_embedding_featurize_cd  %>%
  layer_embedding(input_dim = 13724 #2456 
                ,output_dim = 100,mask_zero = FALSE,name = "dba_embedding_embedding_layer")%>%
  #layer_global_average_pooling_1d() %>%
  layer_average_pooling_1d(pool_size = 6)  %>%
  layer_flatten()

dba_char_embedding_featurize_cd <- layer_input(shape = #7
                                           39,
                                           dtype = 'float32',
                                           name = 'dba_char_embedding_featurize_cd')
dba_char_embedding_input <- dba_char_embedding_featurize_cd %>%
  layer_embedding(input_dim = 64 #7281
                  , output_dim = 16,mask_zero = FALSE,name = "dba_char_embedding_embedding_layer") %>%
  layer_average_pooling_1d(pool_size = 6) %>%
  layer_flatten()


main_output <-
  layer_concatenate(c(
                      dba_embedding_input,
                      
                      dba_char_embedding_input
                    ))   %>%
  # layer_dense(units = 8192,activation = "relu" ,regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001)) %>%
  # layer_dropout(0.2) %>%
  # layer_dense(units = 4096,activation = "relu" ,regularizer_l1_l2(l1 = 0.001, l2 = 0.001)) %>%
  # layer_dropout(0.2) %>%
  # #layer_dense(units = 1024,activation = "relu" ,regularizer_l1_l2(l1 = 0.0001, l2 = 0.0001)) %>%
  # layer_dropout(0.2) %>%
  layer_dense(units = classes_unit,
              activation = 'softmax',
              name = 'main_output')

### model list
######
model <-
  keras_model(list(
                   dba_embedding_featurize_cd,
            
                   dba_char_embedding_featurize_cd), main_output)
##### optimizer
#####
optimizer = optimizer_adamax(
  lr = 0.0001,
  beta_1 = 0.9,
  beta_2 = 0.999,
  epsilon = 1e-08,
  decay = 0,
  clipnorm = NULL,
  clipvalue = NULL
)
#####
### Compile the model
#####
model %>% compile(
  loss = focal_loss(gamma=5., alpha=.25),
  metrics = c(top_5_categorical_accuracy = metric_top_5_categorical_accuracy,
              "accuracy"),
  optimizer = optimizer
  
)
summary(model)
#
#batch_size = 512
### train fit generator
###### run 
######
### Train 
batch_size = 128
#####
############
#load_model_weights_hdf5(model,"/application/ramvora/Kaggle/checkpoint/model.h5")
#save_model_weights_hdf5(model,"/application/ramvora/Kaggle/checkpoint/model.h5")
###train fit
#####

hist <-
  model %>% fit(
    x = list(
             dba_embedding_train,
          
             dba_char_embedding_train),
    y = list(y_train),
    batch_size = 64,
    epochs = 50,
    validation_data = list(list(
                                dba_embedding_val,
                                
                                dba_char_embedding_val),y_val),
    class_weight = class_weight ,
#    callbacks = callbacks,
    verbose = 1
  )
