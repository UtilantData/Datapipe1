

suppressWarnings(suppressMessages(library(DBI)))
suppressWarnings(suppressMessages(library(RSQLite)))

suppressWarnings(suppressMessages(library(stringr)))

Sys.setenv(TZ="	America/Indiana/Indianapolis")

#sqlite_path <- file.path("database/Main.sqlite")

mydb <- dbConnect(RSQLite::SQLite(), "my-db.sqlite")
  
  

dd <- list.files(path = ".", pattern = "csv$", all.files = FALSE,
                   full.names = FALSE, recursive = FALSE,
                   ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)



for(item in dd){
  
  
  gg <- read.csv(item,fill = TRUE )
  
 
  gg<-as.data.frame(gg) 
  dbWriteTable(conn = mydb,
               name = item,
               value = gg,
               row.names = FALSE,
               header = TRUE,
               overwrite = TRUE
  )
  
}



