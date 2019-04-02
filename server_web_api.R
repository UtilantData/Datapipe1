source("/application/ramvora/Kaggle_Restaurant_Inspection/global_web_api.R")
#* @get /Model

function(dba)
{
  
  df <- data.table(dba = dba)
  
  rslt_vmrs <- predict(model, newdata = df$dba 
                       )
  
  rslt_vmrs <- data.table(rslt_vmrs)
  
  return(rslt_vmrs)
  
}