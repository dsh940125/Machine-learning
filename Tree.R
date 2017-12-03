
############# Input the June 2017 SCA Substantial Dataset ############
setwd("C:\\Users\\shaohua\\Documents\\Advanced modeling\\")
library(sas7bdat)

# Input the substantial dataset.
sca2017 <- read.sas7bdat("smp1706_witheduc_matchid.sas7bdat")

# Subset the merged dataset. Only include the variables we want to use (Sample ID, race, age, gender, education, family income, the number of call attempt).
sca <- sca2017[, c("SAMPID", "RACE", "AGE", "SEX", "EDUC", "INCOME", "CALLNU",
               "PHCELL", "USEWEB", "NUMADT", "NUMKID", "V1626", "V1627", "V1628",
               "ATTIW", "TIMETOT", "IWDA2", "IWDA1")]


################## Recode the Missing Data of Predictors ###################

### Recode all the missing data categories (DK, RF, and NA) into real NA. Let the tree handle the missing data by itself. 

# Race. The missing data categories are 8 (DK) and 9 (RF).
sca$RACE[sca$RACE == 8 | sca$RACE == 9] <- NA

# Age. The missing data categories 99 (NA) and 00 (inapplicable).
sca$AGE[sca$AGE == 99 | sca$AGE == 00] <- NA

# Education. The missing data category is 9 (NA).
sca$EDUC[sca$EDUC == 9] <- NA

# The number of working cell phones. The missing data categories are 98 (DK) and 99 (NA).
sca$PHCELL[sca$PHCELL == 98 | sca$PHCELL == 99] <- NA

# Whether use email or the Internet. The missing data categories are 8 (DK) and 9 (NA).
sca$USEWEB[sca$USEWEB == 8 | sca$USEWEB == 9] <- NA

# The number of adults (18 years and older) in the household. The missing data categories are 8 (DK) and 9 (NA).
sca$NUMADT[sca$NUMADT == 8 | sca$NUMADT == 9] <- NA

# The number of kids (17 years and younger) in the household. The missing data categories are 98 (DK) and 99 (NA).
sca$NUMKID[sca$NUMKID == 98 | sca$NUMKID == 99] <- NA

# Whether the questions were asked about respondents only or respondents and their family. The missing data categories are 8 (DK) and 9 (NA).
sca$V1626[sca$V1626 == 8 | sca$V1626 == 9] <- NA

# Interviewing language (English or Spanish). The missing data category is 9 (NA).
sca$V1627[sca$V1627 == 9] <- NA

# Respondents' understanding of the questions. The missing data categories are 8 (DK) and 9 (NA).
sca$V1628[sca$V1628 == 8 | sca$V1628 == 9] <- NA

# Respondent's attitudes toward the interview. The missing data category is 9 (NA).
sca$ATTIW[sca$ATTIW == 9] <- NA

# Note: Income (INCOME), gender (SEX), length of interview (TIMETOT), date interview begin (IWDA1) and date interview concluded (IWDA2) have no missing data category.


########### Descriptive Analysis for Continuous Variables ###############

# The number of call attempt
summary(sca$CALLNU) # mean 3.453 Min. 1 Max. 17
sd(sca$CALLNU) # sd 2.459576

# Age
summary(sca$AGE) # mean 49.55 min 18 max 92

# The number of cell phones in the household
summary(sca$PHCELL) # mean 2.461 min 1 max 8

# The number of the children in the household
# The category 96 of this variable means None. So we have to recode this variable first.
sca$NUMKID[sca$NUMKID == 96] <- 0
summary(sca$NUMKID) # mean 0.557 min 0 max 8

# The length of interview
summary(sca$TIMETOT) # mean 36.04 min 19.73 max 82.70


################ Recode all the Predictors as Factor Variables ##############

# Recode the race as factor variables. The factor labels are: 1. White or Caucasian except Hispanic; 2. Black or African American except Hispanic; 3. Hispanic or Latino (including interviews in Spanish); 4. American Indian or Alaskan native; 5. Asian or Pacific Islander.
sca2017$RACE <- factor(sca2017$RACE)

# Recode the education as factor variables. The factor labels are: 1. Grades 0-8 and no high school diploma; 2. Grades 9-12 and no high school diploma; 3. Grades 0-12 with high school diploma; 4. Grades 13-17 with some college; 5. Grades 13-16 with bachelors degree; 6. Grade 17 with college degree.
sca2017$EDUC <- factor(sca2017$EDUC)

# Recode the gender as factor variables. The factor labels are: 1. Male; 2. Female.
sca2017$SEX <- factor(sca2017$SEX)

# Recode the variable of whether the respondent use email or the Internet or not as factor variables. The factor labels are: 1. Yes; 5. No.
sca2017$USEWEB <- factor(sca2017$USEWEB)

# Recode the number of adults as factor variables. The factor labels are: 1. 1; 2. 2; 3. 3; 4. 4; 5. 5; 6. 6; 7. 7 or more.
sca2017$NUMADT <- factor(sca2017$NUMADT)

# Recode the variable of whether the questions were asked about respondents or respondents and family as factor variables. The factor labels are: 1. respondents only; 2. respondents and family only.
sca2017$V1626 <- factor(sca2017$V1626)

# Recode the interviewing language as factor variables. The factor labels are: 1. English; 2. Spanish.
sca2017$V1627 <- factor(sca2017$V1627)

# Recode the respondents' understanding as factor variables. The factor labels are: 1. excellent; 2. Good; 3. Fair; 4. Poor.
sca2017$V1628 <- factor(sca2017$V1628)

# Recode the respondents' attitude as factor variables. The factor labels are: 1. friendly & interested; 2. cooperative but not particularly interested; 3. impartient; 4. hostile.
sca2017$ATTIW <- factor(sca2017$ATTIW)


####################### Make the Regression Tree ########################
library(rpart)
library(rpart.plot)

# Set a seed of 500 to ensure that the results are consistent each time I does the same tree.
set.seed(500)
# Try the regression tree with default control parameters
tree1 <- rpart(CALLNU ~ RACE + AGE + EDUC + INCOME + SEX + PHCELL + USEWEB + 
                NUMADT + NUMKID + V1626 + V1627 + V1628 + ATTIW +
                 TIMETOT + IWDA2 + IWDA1, data = sca, method = "anova")
# Plot the tree
prp(tree1, extra =1, digits = 3)
# Examine the quality of the solution
printcp(tree1)
# Calculate the apparent error rate
0.55311*3653.9/605 # 3.34051
# Obtain the detailed summary of the splits and solution
summary(tree1)
# Make the three-way plot. Identifying the CP parameters that result in useful size by error rate compromises.
plotcp(tree1)
# According to the three-way plot, the relative error rate of the test datasets goes down as the CP parameter decreases, indicating the tree does a good job here. When the size of the tree is 6 (corresponding to a CP of 0.02019078), the X-val relative error rate is the lowest. Thus, we should prune the tree by setting up the CP as 0.02019078.

# Prune the tree according to the result with default settings. Set up the CP as 0.02019078.
set.seed(500)
# Set up the CP as 0.02019078. Use the default setting for other arguments.
n.control = rpart.control(cp=0.02019078)
# Make the pruned tree
tree2 <- rpart(CALLNU ~ RACE + AGE + EDUC + INCOME + SEX + PHCELL + USEWEB + 
                 NUMADT + NUMKID + V1626 + V1627 + V1628 + ATTIW + TIMETOT + 
                 IWDA2 + IWDA1, data = sca, method = "anova", control = n.control)
# Plot the tree
prp(tree2, extra =1, digits = 3)
# Examine the quality of the solution
printcp(tree2)
# Calculate the apparent error rate
0.70044*3653.9/605 # 4.23031
# Make the three-way plot. See if X-val relative error rate goes down as expected.
plotcp(tree2)
