######### predict 
predict_multi_input = model %>% predict(
  train_tokens,
  batch_size = 1,
  verbose = 1
)


##### embeddings
summary(model)

#predict(model,)
# save_model_weights_hdf5(model,paste0(DIR,"model.h5"))
# 
# load_model_weights_hdf5(paste0(DIR,"model.h5"))
emb_vals <- (model$get_layer("dba_char_embedding_embedding_layer") %>% get_weights())[[1]]
emb_vals %>% dim() # 10x128
emb_vals <- (model$get_layer("dba_embedding_embedding_layer") %>% get_weights())[[1]]
emb_vals %>% dim() # 10x128

library(keras)


embedding_matrix <- emb_vals
dba_char_embeddingtokenizer <- load_text_tokenizer("/application/ramvora/Kaggle/tokenizer/dba_char_embeddingtokenizer")
words <- data_frame(
  word = names(dba_char_embeddingtokenizer$word_index), 
  id = as.integer(unlist(dba_char_embeddingtokenizer$word_index))
)

View(words)
#train_levels_dt = train_levels %>% map(as.character) %>% unlist() %>% data.table() %>% setnames("values") 
# 
# words <- data_frame(
#   word = words, 
#   id = seq(1,586,1))


words <- words %>%
  filter(id <= dba_char_embeddingtokenizer$num_words) %>%
  arrange(id)

row.names(embedding_matrix) <- c(words$word)
####
library(text2vec)

find_similar_words <- function(word, embedding_matrix, n = 10) {
  similarities <- embedding_matrix[word, , drop = FALSE] %>%
    sim2(embedding_matrix, y = ., method = "cosine")
  
  similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}


#####
sim_items <- find_similar_words(tolower("jrdn aj1 mid-bk/gy/wh"), embedding_matrix)
###
library(Rtsne)
library(ggplot2)
library(plotly)

tsne <- Rtsne(embedding_matrix[2:7281,], perplexity = 5, pca = FALSE)

tsne_plot <- (tsne$Y) %>% as.data.table()
tsne_plot[, word := words$word[2:nrow(words)]]

# mutate(word = row.names(embedding_matrix)[2:7281]) %>%
# slice(1:500) %>%
ggplot(aes(x = V1, y = V2, label = word)) + 
  geom_text(size = 3)
tsne_plot

