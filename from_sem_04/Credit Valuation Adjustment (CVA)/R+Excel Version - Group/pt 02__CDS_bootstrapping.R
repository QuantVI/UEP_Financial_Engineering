# This script is devoted to bootrstrapping implied probabilities of default from CDS term structure.
# If you are now in "Taras_bootstrapping" R Project, the functions must be already read into your memory. The functions are defined in the end of the script (in section "List of functions").
# In order to run functions as they are, please open "Taras_bootstrapping" R Project. File paths are defined as file names.
# This script requires following packages:
  require(ggplot2)
  require(readxl)
  require(reshape)
  require(tidyverse)
# Please install them before running the functions.

# Data preparation ====
### Description of "get_data" function:
# Function Parameters:
  # file_path - file path of the original "CVA_data.xlsx" file. No file modification is required.
# OUTPUT: This function returns a two-element list. Both elements are data frames. Those elements are:
  # CDS_quotes <- term structure of CDS spreads
  # Quotes_data <- term structure of EUR & PLN discount factors, and of EUR/PLN FX rate
example_of_data_extraction <- get_data(file_path = "CVA_data.xlsx")
  CDS_quotes <- example_of_data_extraction$CDS_quotes
  Quotes_data <- example_of_data_extraction$Quotes_data
  
# Actual extraction of implied probabilities ====
### Description of "probabilities" function:
# Function Parameters:
  # CDS_quotes    -  term structure of the CDS from the original "CVA_data.xlsx". Use get_data("CVA_data.xlsx")$Quotes_data
  # Quotes_data   - term structure of of both discount factors and XF rate from the original "CVA_data.xlsx". Use get_data("CVA_data.xlsx")$Quotes_data
  # RR            - recovery rate (assumed to be 0.4)
  # t_s           - time step (assumed to be 1/12, i.e. monthly time step)
  # end           - Simulation horizon (end corresponds to the last tenor used to interpolate). E.g. if you want to see plots with max tenor 10Y, change variable "end" to 10
  # spline_method - spline method e.g. "natural"
# 
# OUTPUT: This function returns data frame with probabilities of default and survival probabilties over specified tenors
probabilities <- implied_probab(CDS_quotes = get_data("CVA_data.xlsx")$CDS_quotes, 
                                Quotes_data = get_data("CVA_data.xlsx")$Quotes_data, 
                                RR = .4, t_s = 1/12, end = 3, spline_method = "natural")

# Visualization of probabilites ====
# Probability of survival
ggplot(probabilities, aes(x=t_h0,y=PS)) +
  geom_line() + geom_point() +
  ylab("Probability of Survival, %") +
  xlab("Tenor, years") +
  ggtitle("Bootstrapped Survival Probabilities") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Probability of default
ggplot(probabilities, aes(x=t_h0,y=PD)) +
  geom_line() + geom_point() +
  ylab("Probability of Default, %") +
  xlab("Tenor, years") +
  ggtitle("Bootstrapped Default Probabilities") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

require(reshape)
mProbs <- melt(probabilities, id.vars = "t_h0")
ggplot(mProbs, aes(x=t_h0,y=value,colour=variable)) +
  geom_line() +
  geom_point() +
  ylab("Probability, %") +
  xlab("Tenor, years") +
  ggtitle("Bootstrapped Probabilities") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


# List of functions ====

get_data <- function(file_path) {
  library(readxl)
  # raw_data <- as.data.frame(read_xlsx("CVA_data.xlsx", col_names = F, trim_ws = T, col_types = NULL))
  raw_data <- as.data.frame(read_xlsx(file_path, col_names = F, trim_ws = T, col_types = NULL))
  
  # Get term sructure of CDS ("CDS_data")
  CDS_col <- grep("CDS DB", raw_data)
  CDS_row <- grep("CDS DB", raw_data[,CDS_col])
  CDS_data <- raw_data[(CDS_row+1):(CDS_row+8),c(CDS_col,CDS_col+1)]
  colnames(CDS_data) <- raw_data[CDS_row,c(CDS_col,CDS_col+1)]
  CDS_data[,2] <- as.numeric(CDS_data[,2])
  
  # Convert CDS tenors to year fractions
  tenor <- c(substr(CDS_data[-8,1], 3, 4), substr(CDS_data[8,1], 3, 5))
  y <- c()
  for (i in tenor) {
    if (grepl("M",i)) {
      y <- c(y, as.numeric(gsub("([0-9]+).*$", "\\1", i))/12)
    }
    else {
      y <- c(y, as.numeric(gsub("([0-9]+).*$", "\\1", i)))
    }
  }
  tenors <- y
  CDS_quotes <- data.frame(tenors, spread = CDS_data[,2])
  
  # Get term structure of discount factors and FX rate ("Quotes_data")
  Quotes_col <- grep("Month", raw_data)
  Quotes_row <- grep("Month", raw_data[,Quotes_col])
  Quotes_data <- raw_data[(Quotes_row+1):(Quotes_row+length(raw_data[,Quotes_col])-Quotes_row),Quotes_col:(Quotes_col+3)]
  colnames(Quotes_data) <- raw_data[Quotes_row,Quotes_col:(Quotes_col+3)]
  Quotes_data <- apply(Quotes_data, MARGIN = 2, FUN = as.numeric)
  
  tenors <- Quotes_data[,1]/12
  Quotes_data <- Quotes_data[,-1]
  Quotes_data <- as.data.frame(cbind(tenors, Quotes_data))
  
  out <- list("CDS_quotes" = CDS_quotes, "Quotes_data" = Quotes_data)
  return(out)
}

implied_probab <- function(CDS_quotes, Quotes_data, RR, t_s, end, spline_method) {
  LGD <- 1-RR
  # Time horizon (fraction of year tenors from t_s to end). Moment zero will be added in the end of the function.
  t_h <- seq(from = t_s, to = end, by = t_s)
  # Using natural cubic spline method to interpolate CDS quotes over a specific time horizon ("t_h") with a specific time step
  # CDS spreads denominated in basis points
  Sbp <- spline(x = CDS_quotes$tenors, y = CDS_quotes$spread, xout = t_h, method = spline_method)$y
  # CDS spreas denominated in percentage points
  S <- Sbp/10000
  
  # Get discount factors
  # I am assuming that our counterparty has EUR discount factors
  raw_df <- Quotes_data
  require(tidyverse)
  # We are omitting discount factor at moment zero and filtering for tenors that will be used for bootstrapping
  df <- raw_df %>%
    filter(tenors != 0 & tenors <= end) %>%
    select(EUR_DF)
  # Making it  a vector
  DF <- df[,1]
  
  # Delta t (time step is evenly spaced in our case)
  dt <- diff(c(0, t_h))
  # Analogous to the "Input" matrix in the spreadsheet
  input <- data.frame(S, DF, dt)
  
  # Number of tenors to bootstrap
  N <- dim(input)[1]
  
  # Triangular matrix from the spreadsheet. It's row wise sum is equal to "Licznik" column from the spreadsheet.
  num <- matrix(rep(0, N*N), ncol = N)
  # Corresponds to "Mianownik" column from the spreadsheet.
  den <- rep(0, N)
  # Corresponds to "Pierwszy element" column from the spreadsheet.
  fst <- rep(0, N)
  # Corresponds to "Drugi element" column from the spreadsheet.
  snd <- rep(0, N)
  # Survival probability
  PS <- rep(0, N)
  # Probability of survival at moment 0. Will be concatenated to the "PS" vector in the end
  PS0 <- 1
  
  # Calcualtion of tenor 1 of "Drugi element"
  snd[1] <- PS0*LGD/(dt[1]*S[1]+LGD)
  # Probability of survival at moment 1
  PS[1] <- snd[1]
  
  # Calculating "Mianownik" denominator but leaving tenor 1 to be zero
  den <- c(0, DF[-1]*(LGD+S[-1]*dt[-1]))
  
  # Calculation of probability of survival at tenor 2. Forced to do this in order to stick to the notation from the formula. In python it would not be required (as indexation starts from 0 there).
  num[2,1] <- DF[1]*(LGD*PS0-PS[1]*(LGD+S[2]*dt[2]))
  fst[2] <- sum(num[2,])/den[2]
  snd[2] <- PS[1]*LGD/(dt[2]*S[2]+LGD)
  PS[2] <-  fst[2] + snd[2]
  
  # Calculation of the general formula for probability of survival from tenor 3 to tenor N
  for (i in 3:N) {
    num[i, 1] <- DF[1]*(LGD*PS0-PS[1]*(LGD+S[i]*dt[1]))
    for (n in 2:(i-1)) {
      num[i, n] <- DF[n]*(LGD*PS[n-1]-PS[n]*(LGD+S[i]*dt[n]))
    }
    fst[i] <- sum(num[i,])/den[i]
    snd[i] <- PS[i-1]*LGD/(dt[i]*S[i]+LGD)
    PS[i] <- fst[i] + snd[i]
  }
  
  # Adding almost sure probability of survival at tenor 0
  PS <- c(PS0, PS)
  # Converting probability of survival into percentage points
  PS <- PS*100
  # Adding tenor z
  t_h0 <- c(0, t_h)
  SurProb <- data.frame(t_h0, PS)
  
  # Calculating probability of default
  PD <- 100-PS
  DefProb <- data.frame(t_h0, PD)
  
  # Data frame containing tenors from 0 to N and corresponding probabilities of survival and probabilies of default
  Probs <- cbind(DefProb,PS)
  return(Probs)
}
