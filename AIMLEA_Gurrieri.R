################## Descrizione, Obiettivo e Dataset #####################

# il dataset contiene dati riguardo le specifiche di CPU e GPU rilasciate sul mercato nel corso degli anni.

# date le caratteristiche numeriche di un nuovo processore, determinarne il produttore.

# il dataset è composto dalle seguenti colonne:
# . Product: il prodotto
# . Type: la tipologia del prodotto (CPU, GPU)
# . Release Date: data di rilascio del prodotto
# . Process Size (nm): dimensione dei transistor
# . TDP (W): consumo del processore
# . Die Size (mm^2): dimensione del processore
# . Transistors (million): numero di transistor nel processore
# . Freq (MHz): frequenza del processore
# . Foundry: da chi è stato prodotto
# . Vendor: da chi è commercializzato
# . FP16 GFLOPS: miliardi di operazioni in virgola mobile per secondo su 16 bit
# . FP32 GFLOPS: miliardi di operazioni in virgola mobile per secondo su 32 bit
# . FP64 GFLOPS: miliardi di operazioni in virgola mobile per secondo su 64 bit

################## Caricamento e Pulizia Dataset #####################

# la funzione "set.seed" imposta una seme specifico per garantire la riproducibilità del progetto.
set.seed(123)

# dopo aver inserito il dataset all'interno dell'environment...
# (cliccando su "Import Dataset" e selezionando "From Text (base)..." è possibile inserire il file .csv)
# ...creo un dataframe chiamato "df".
df <- produttore.nuovo.processore
df

# stampo una descrizione del dataframe utilizzando la funzione str().
str(df)

# creo una colonna chiamata "Year" estraendo l'anno dalla colonna "Release.Date".
df$Release.Date <- as.Date(df$Release.Date)
df$Year <- as.numeric(format(df$Release.Date, "%Y"))

# vedo se la colonna è stata creata correttamente.
head(df)

# stampo le frequenze assolute per la variabile "Foundry".
table(df$Foundry)

# importo la libreria "plyr".
library(plyr)
# modifico la colonna "Foundry", raggruppando i valori con una frequenza minore di 100 sotto la categoria "Other".
df$Foundry <- revalue(x = as.factor(df$Foundry),
                          replace = c("UMC" = "Other",
                                      "Samsung" = "Other",
                                      "Sony" = "Other",
                                      "IBM" = "Other", 
                                      "NEC" = "Other", 
                                      "Renesas" = "Other"))

# importo la libreria "Hmisc".
library(Hmisc)
# utilizzo la libreria "Hmisc" per sostituire i valori mancanti di ciascuna colonna con la media dei valori presenti nella medesima.
cols_to_impute = c("Process.Size..nm.", "TDP..W.", "Die.Size..mm.2.", "Transistors..million.", "Release.Date", "Year")
df[cols_to_impute] <- lapply(df[cols_to_impute], function(x) impute(x, mean))

# stampo il numero di valori mancanti nel dataframe (il numero corretto è 10772 e corrisponde ai valori mancanti nelle ultime 3 colonne del dataset).
sum(is.na(df))

################## EDA #####################

# importo la libreria "plotly".
library(plotly)
# creo una tabella della distribuzione dei valori della colonna "Type" del dataframe "df"...
values <- table(df$Type)
names <- names(values)
# ...e un grafico a torta (pie chart) per visualizzare questi valori.
plot_ly(labels = names, values = values, type = 'pie', hole = 0.2) %>%
  layout(title = list(text = 'Type Distribution', x = 0.5))

# importo la libreria "ggplot2".
library(ggplot2)
# creo grafici a dispersione grazie alla libreria "ggplot2".
# tutti gli scatterplot utilizzano la colonna "Release.Date" del dataframe "df" come x.
ggplot(df, aes(x = Release.Date, y = Process.Size..nm.)) +
  geom_point(aes(color = factor(Type)))

ggplot(df, aes(x = Release.Date, y = TDP..W.)) +
  geom_point(aes(color = factor(Type)))

ggplot(df, aes(x = Release.Date, y = Die.Size..mm.2.)) +
  geom_point(aes(color = factor(Type)))

ggplot(df, aes(x = Release.Date, y = Transistors..million.)) +
  geom_point(aes(color = factor(Type)))

ggplot(df, aes(x = Release.Date, y = Freq..MHz.)) +
  geom_point(aes(color = factor(Type)))

ggplot(df, aes(x = Release.Date, y = Freq..MHz.)) +
  geom_point(aes(color = factor(Foundry)))

# separo le features e la variabile target.
X = df[c('Process.Size..nm.', 'TDP..W.', 'Die.Size..mm.2.', 'Transistors..million.', 'Freq..MHz.', "Year")]
y = df["Foundry"]

# importo la libreria "ggcorrplot".
library(ggcorrplot)
# creo una matrice di correlazione tra le features...
corr <- round(cor(X), 2)
# ...e un grafico di correlazione utilizzando ggcorrplot.
ggcorrplot(corr, hc.order = TRUE, 
           type = "full", 
           lab = TRUE, 
           lab_size = 3, 
           method="square", 
           colors = c("#B3000C", "white", "#00B32C"), 
           ggtheme=theme_bw)

table(df$Foundry)
# creo una tabella della distribuzione dei valori della colonna "Foundry" del dataframe "df"...
values <- table(df$Foundry)
names <- names(values)
# ...e un grafico a torta (pie chart) per visualizzare questi valori.
plot_ly(labels = names, values = values, type = 'pie', hole = 0.2) %>%
  layout(title = list(text = 'Foundry Distribution', x = 0.5))

################## Scaling & Splitting #####################

set.seed(123)

# importo le librerie caret e caTools
library(caret)
library(caTools)
# utilizzo la funzione "scale" per scalare il dati in X
X_scaled <- scale(X)

# utilizzo la funzione "data.frame" per riassemblare il dataset
df_scaled <- data.frame(X_scaled, y)

# utilizzo la funzione "sample.split" per dividere il set di dati "df_scaled" in un set di addestramento 
# e un set di test, utilizzando un rapporto di divisione specificato del 75%.
split<- sample.split(df_scaled, SplitRatio = 0.75)
# utilizzo la funzione "subset" per creare due nuovi set di dati: "train_scaled" e "test_scaled",
# che corrispondono rispettivamente al set di addestramento e al set di test derivati dalla divisione iniziale.
train_scaled <- subset(df_scaled, split == "TRUE")
test_scaled <- subset(df_scaled, split == "FALSE")

table(train_scaled$Foundry)

################## Random Under-sampling #####################

# creo un nuovo dataframe chiamato "df_undersampled".
df_undersampled <- data.frame()
# il dataframe di training "train_scaled" è utilizzato come input.
# il codice utilizza un ciclo "for" per iterare attraverso i valori unici della colonna "Foundry" del dataframe originale.
for(foundry in unique(train_scaled$Foundry)) {
  # per ogni valore unico di "Foundry", viene creato un nuovo dataframe chiamato "foundry_data" che contiene solo le righe del dataframe originale in cui la colonna "Foundry" ha il valore corrente.
  foundry_data <- train_scaled[train_scaled$Foundry == foundry, ]
  # "foundry_data_undersampled" viene creato utilizzando una funzione di campionamento casuale: 
  # tot righe casuali vengono estratte dal dataframe originale "foundry_data" e combinate in un nuovo dataframe "foundry_data_undersampled" senza sostituzione.
  foundry_data_undersampled <- foundry_data[sample(1:nrow(foundry_data), min(table(train_scaled$Foundry)), replace = FALSE), ]
  # infine, ogni "foundry_data_undersampled" viene aggiunto al dataframe "df_undersampled" utilizzando la funzione rbind. 
  df_undersampled <- rbind(df_undersampled, foundry_data_undersampled)
}

table(df_undersampled$Foundry)
# creo una tabella della distribuzione dei valori della colonna "Foundry" del dataframe "df_undersampled"...
values_under <- table(df_undersampled$Foundry)
names_under <- names(values_under)
# ...e un grafico a torta (pie chart) per visualizzare questi valori.
plot_ly(labels = names_under, values = values_under, type = 'pie', hole = 0.2) %>%
  layout(title = list(text = 'Under-sampling', x = 0.5))

################## Decision Tree Under-sampling #####################

# importo la libreria partykit
library(partykit)

# addestro un modello di classificazione Decision Tree usando la funzione "ctree"
# la funzione prende come imput:
# la formula "Foundry ~ .", la quale specifica che la colonna "Foundry" verrà usata come variabile target e tutte le altre colonne come predittori.
# "df_undersampled" è il dataframe che verrà usato per addestrare il modello.
classifier_dt_under <- ctree(Foundry ~ ., data = df_undersampled)
#classifier_dt_under

# utilizzo la funzione "predict" per effettuare la previsione utilizzando il modello addestrato e i dati di testing "test_scaled".
model_dt_under = predict(classifier_dt_under, newdata = test_scaled)

# utilizzo la funzione "table" per creare una matrice di confusione tra le etichette previste e le etichette reali dei dati di test.
cmtx_dt_under = table(test_scaled$Foundry, model_dt_under)

# "confusionMatrix" viene utilizzata per generare una rappresentazione visiva della matrice di confusione.
confusionMatrix(cmtx_dt_under)

# calcolo l'accuracy per confermare il precedente output
ac_dt_under <- sum(diag(cmtx_dt_under))/sum(cmtx_dt_under)
print(paste("Accuracy:", ac_dt_under))
# calcolo la precision
pr_dt_under <- diag(cmtx_dt_under) / colSums(cmtx_dt_under)
pr_dt_under
# calcolo la recall
rc_dt_under <- diag(cmtx_dt_under) / rowSums(cmtx_dt_under)
rc_dt_under
# calcolo l'f1 score
f1_dt_under <- 2 * pr_dt_under * rc_dt_under / (rc_dt_under + rc_dt_under)
f1_dt_under

################## Naive Bayes Under-sampling #####################

# importo la libreria "e1071"
library(e1071)

# addestro un modello di classificazione Naive Bayes usando la funzione "naiveBayes"
# la funzione prende come imput:
# la formula "Foundry ~ .", la quale specifica che la colonna "Foundry" verrà usata come variabile target e tutte le altre colonne come predittori.
# "df_undersampled" è il dataframe che verrà usato per addestrare il modello.
classifier_nb_under <- naiveBayes(Foundry ~ ., data = df_undersampled)
#classifier_nb_under

# utilizzo la funzione "predict" per effettuare la previsione utilizzando il modello addestrato e i dati di testing "test_scaled".
model_nb_under = predict(classifier_nb_under, newdata = test_scaled)

# utilizzo la funzione "table" per creare una matrice di confusione tra le etichette previste e le etichette reali dei dati di test.
cmtx_nb_under = table(test_scaled$Foundry, model_nb_under)

# "confusionMatrix" viene utilizzata per generare una rappresentazione visiva della matrice di confusione.
confusionMatrix(cmtx_nb_under)

# calcolo l'accuracy per confermare il precedente output
ac_nb_under <- sum(diag(cmtx_nb_under))/sum(cmtx_nb_under)
print(paste("Accuracy:", ac_nb_under))
# calcolo la precision
pr_nb_under <- diag(cmtx_nb_under) / colSums(cmtx_nb_under)
pr_nb_under
# calcolo la recall
rc_nb_under <- diag(cmtx_nb_under) / rowSums(cmtx_nb_under)
rc_nb_under
# calcolo l'f1 score
f1_nb_under <- 2 * pr_nb_under * rc_nb_under / (rc_nb_under + rc_nb_under)
f1_nb_under

################## Random Forest Under-sampling #####################

# importo la libreria randomForest
library(randomForest)

# addestro un modello di classificazione Random Forest usando la funzione "randomForest"
# la funzione prende come imput:
# la formula "Foundry ~ .", la quale specifica che la colonna "Foundry" verrà usata come variabile target e tutte le altre colonne come predittori.
# "df_undersampled" è il dataframe che verrà usato per addestrare il modello.
classifier_rf_under <- randomForest(Foundry ~ ., data = df_undersampled)
#classifier_rf_under

# utilizzo la funzione "predict" per effettuare la previsione utilizzando il modello addestrato e i dati di testing "test_scaled".
model_rf_under = predict(classifier_rf_under, newdata = test_scaled)

# utilizzo la funzione "table" per creare una matrice di confusione tra le etichette previste e le etichette reali dei dati di test.
cmtx_rf_under = table(test_scaled$Foundry, model_rf_under)

# "confusionMatrix" viene utilizzata per generare una rappresentazione visiva della matrice di confusione.
confusionMatrix(cmtx_rf_under)

# calcolo l'accuracy per confermare il precedente output
ac_rf_under <- sum(diag(cmtx_rf_under))/sum(cmtx_rf_under)
print(paste("Accuracy:", ac_rf_under))
# calcolo la precision
pr_rf_under <- diag(cmtx_rf_under) / colSums(cmtx_rf_under)
pr_rf_under
# calcolo la recall
rc_rf_under <- diag(cmtx_rf_under) / rowSums(cmtx_rf_under)
rc_rf_under
# calcolo l'f1 score
f1_rf_under <- 2 * pr_rf_under * rc_rf_under / (rc_rf_under + rc_rf_under)
f1_rf_under

# creo un plot per vedere l'importanza delle features
varImpPlot(classifier_rf_under, main = "Features Importance Plot")

################## Random Over-sampling #####################

# creo un nuovo dataframe chiamato "df_oversampled".
df_oversampled <- data.frame()
# il dataframe di training "train_scaled" è utilizzato come input.
# il codice utilizza un ciclo "for" per iterare attraverso i valori unici della colonna "Foundry" del dataframe originale.
for(foundry in unique(train_scaled$Foundry)) {
  # per ogni valore unico di "Foundry", viene creato un nuovo dataframe chiamato "foundry_data" che contiene solo le righe del dataframe originale in cui la colonna "Foundry" ha il valore corrente.
  foundry_data <- train_scaled[train_scaled$Foundry == foundry, ]
  # "foundry_data_oversampled" viene creato utilizzando una funzione di campionamento casuale: 
  # tot righe casuali vengono estratte dal dataframe originale "foundry_data" e combinate in un nuovo dataframe "foundry_data_oversampled" senza sostituzione.
  foundry_data_oversampled <- foundry_data[sample(1:nrow(foundry_data), max(table(train_scaled$Foundry)), replace = TRUE), ]
  # infine, ogni "foundry_data_oversampled" viene aggiunto al dataframe "df_oversampled" utilizzando la funzione rbind. 
  df_oversampled <- rbind(df_oversampled, foundry_data_oversampled)
}

table(df_oversampled$Foundry)
# creo una tabella della distribuzione dei valori della colonna "Foundry" del dataframe "df_oversampled"...
values_over <- table(df_oversampled$Foundry)
names_over <- names(values_over)
# ...e un grafico a torta (pie chart) per visualizzare questi valori.
plot_ly(labels = names_over, values = values_over, type = 'pie', hole = 0.2) %>%
  layout(title = list(text = 'Over-sampling', x = 0.5))

################## Decision Tree Over-sampling #####################

# addestro un modello di classificazione Decision Tree usando la funzione "ctree"
# la funzione prende come imput:
# la formula "Foundry ~ .", la quale specifica che la colonna "Foundry" verrà usata come variabile target e tutte le altre colonne come predittori.
# "df_oversampled" è il dataframe che verrà usato per addestrare il modello.
classifier_dt_over <- ctree(Foundry ~ ., data = df_oversampled)
#classifier_dt_over

# utilizzo la funzione "predict" per effettuare la previsione utilizzando il modello addestrato e i dati di testing "test_scaled".
model_dt_over = predict(classifier_dt_over, newdata = test_scaled)

# utilizzo la funzione "table" per creare una matrice di confusione tra le etichette previste e le etichette reali dei dati di test.
cmtx_dt_over = table(test_scaled$Foundry, model_dt_over)

# "confusionMatrix" viene utilizzata per generare una rappresentazione visiva della matrice di confusione.
confusionMatrix(cmtx_dt_over)

# calcolo dell'accuracy per confermare il precedente output
ac_dt_over <- sum(diag(cmtx_dt_over))/sum(cmtx_dt_over)
print(paste("Accuracy:", ac_dt_over))
# calcolo la precision
pr_dt_over <- diag(cmtx_dt_over) / colSums(cmtx_dt_over)
pr_dt_over
# calcolo la recall
rc_dt_over <- diag(cmtx_dt_over) / rowSums(cmtx_dt_over)
rc_dt_over
# calcolo l'f1 score
f1_dt_over <- 2 * pr_dt_over * rc_dt_over / (rc_dt_over + rc_dt_over)
f1_dt_over

################## Naive Bayes Over-sampling #####################

# addestro un modello di classificazione Naive Bayes usando la funzione "naiveBayes"
# la funzione prende come imput:
# la formula "Foundry ~ .", la quale specifica che la colonna "Foundry" verrà usata come variabile target e tutte le altre colonne come predittori.
# "df_oversampled" è il dataframe che verrà usato per addestrare il modello.
classifier_nb_over <- naiveBayes(Foundry ~ ., data = df_oversampled)
#classifier_nb_over

# utilizzo la funzione "predict" per effettuare la previsione utilizzando il modello addestrato e i dati di testing "test_scaled".
model_nb_over = predict(classifier_nb_over, newdata = test_scaled)

# utilizzo la funzione "table" per creare una matrice di confusione tra le etichette previste e le etichette reali dei dati di test.
cmtx_nb_over = table(test_scaled$Foundry, model_nb_over)

# "confusionMatrix" viene utilizzata per generare una rappresentazione visiva della matrice di confusione.
confusionMatrix(cmtx_nb_over)

# calcolo l'accuracy per confermare il precedente output
ac_nb_over <- sum(diag(cmtx_nb_over))/sum(cmtx_nb_over)
print(paste("Accuracy:", ac_nb_over))
# calcolo la precision
pr_nb_over <- diag(cmtx_nb_over) / colSums(cmtx_nb_over)
pr_nb_over
# calcolo la recall
rc_nb_over <- diag(cmtx_nb_over) / rowSums(cmtx_nb_over)
rc_nb_over
# calcolo l'f1 score
f1_nb_over <- 2 * pr_nb_over * rc_nb_over / (rc_nb_over + rc_nb_over)
f1_nb_over

################## Random Forest Over-sampling #####################

# addestro un modello di classificazione Random Forest usando la funzione "randomForest"
# la funzione prende come imput:
# la formula "Foundry ~ .", la quale specifica che la colonna "Foundry" verrà usata come variabile target e tutte le altre colonne come predittori.
# "df_oversampled" è il dataframe che verrà usato per addestrare il modello.
classifier_rf_over <- randomForest(Foundry ~ ., data = df_oversampled)
#classifier_rf_over

# utilizzo la funzione "predict" per effettuare la previsione utilizzando il modello addestrato e i dati di testing "test_scaled".
model_rf_over = predict(classifier_rf_over, newdata = test_scaled)

# utilizzo la funzione "table" per creare una matrice di confusione tra le etichette previste e le etichette reali dei dati di test.
cmtx_rf_over = table(test_scaled$Foundry, model_rf_over)

# "confusionMatrix" viene utilizzata per generare una rappresentazione visiva della matrice di confusione.
confusionMatrix(cmtx_rf_over)

# calcolo dell'accuracy per confermare il precedente output
ac_rf_over <- sum(diag(cmtx_rf_over))/sum(cmtx_rf_over)
print(paste("Accuracy:", ac_rf_over))
# calcolo la precision
pr_rf_over <- diag(cmtx_rf_over) / colSums(cmtx_rf_over)
pr_rf_over
# calcolo la recall
rc_rf_over <- diag(cmtx_rf_over) / rowSums(cmtx_rf_over)
rc_rf_over
# calcolo l'f1 score
f1_rf_over <- 2 * precision * recall / (precision + recall)
f1_rf_over

# creo un plot per vedere l'importanza delle features
varImpPlot(classifier_rf_over, main = "Features Importance Plot")


