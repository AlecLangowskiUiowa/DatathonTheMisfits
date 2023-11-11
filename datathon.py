import pandas as pd

#Read in neccesary datasets
core18 = pd.read_sas("Data/hepBCA18.xpt")
core16 = pd.read_sas("Data/hepBCA16.xpt")

surface18 = pd.read_sas("Data/hepBSA18.xpt")
surface16 = pd.read_sas("Data/hepBSA16.xpt")

demographics18 = pd.read_sas("Data/DEMO_J.xpt")
demographics16 = pd.read_sas("Data/DEMO_I.xpt")

#Combine 2015-2016 & 2017-2018 datasets
core = pd.concat([core18, core16], ignore_index=True)
core = core.drop("LBDHD", axis=1)
surface = pd.concat([surface18, surface16], ignore_index=True)
demographics = pd.concat([demographics18, demographics16], ignore_index=True)
demographics = demographics[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDHHIN2", "WTINT2YR"]]

#Create a combined dataset with all variables necessary to categorize hepB status
hepB = pd.merge(core, surface, on="SEQN")

#4 classes important, first we are interested in vaccinated & unvaccinated
#Within the unvaccinated class we are interested in those who have immunity thru infection, infection, or neither leaving the person non immune
def hepStatus(row):
    if row["LBXHBS"] == 1:
        return "vaccinated"
    else:
        #Unvaccinated classes
        if row["LBDHBG"] == 1:
            return "previously infected"
        elif row["LBXHBC"] == 1:
            return "currently infected"
        else:
            return "no immunity"

#Apply the above function
hepB["status"] = hepB.apply(hepStatus, axis=1)

analysisData = pd.merge(hepB, demographics, on='SEQN', how='inner')
analysisData["ageDecade"] = round(analysisData["RIDAGEYR"]/10)
analysisData = analysisData.drop("RIDAGEYR", axis=1)

#Function to classify income
def incomeClassifier(value):
    if value in [1, 2, 3, 4, 13]:
        return 1
    elif value in [5, 6, 7]:
        return 2
    elif value in [8, 9, 10]:
        return 3
    elif value in [14, 15]:
        return 4
    else:
        return 0

#creation of income classification variable
analysisData["income"] = analysisData["INDHHIN2"].apply(incomeClassifier)
analysisData = analysisData.drop("INDHHIN2", axis = 1)
weightedData = analysisData.loc[analysisData.index.repeat(round(analysisData['WTINT2YR'])/100)]

#Metrics and Data Analysis

#Method to print metric dictionaries
def printMetrics(dictionary):
    for key in dictionary:
        print("Metrics for: " + key)
        print("Infection Rate: " + str(round(dictionary[key][0] * 100, 2)) + "%")
        print("Vaccination Rate: " + str(round(dictionary[key][1] * 100, 2)) + "%")
        print("Immunity Rate: " + str(round(dictionary[key][2] * 100, 2)) + "%")
        print("Infection Rate of Non Vaccinated People: " + str(round(dictionary[key][3]* 100, 2)) + "%")
        print()

#Frequency Table of each predictor level for each response class
genderFT = weightedData.groupby('RIAGENDR')['status'].value_counts().unstack(fill_value=0)

#Creation of the metrics dictionary
genderMetrics = {"male" : [genderFT.at[1, "currently infected"]/(weightedData['RIAGENDR'] == 1).sum(),
                           genderFT.at[1, "vaccinated"]/(weightedData['RIAGENDR'] == 1).sum(),
                           ((weightedData['RIAGENDR'] == 1).sum() - genderFT.at[1, "no immunity"])/(weightedData['RIAGENDR'] == 1).sum(),
                           genderFT.at[1, "currently infected"]/((weightedData['RIAGENDR'] == 1).sum() - genderFT.at[1, "vaccinated"])],
                 "female" : [genderFT.at[2, "currently infected"]/(weightedData['RIAGENDR'] == 2).sum(),
                             genderFT.at[2, "vaccinated"]/(weightedData['RIAGENDR'] == 2).sum(),
                             ((weightedData['RIAGENDR'] == 2).sum() - genderFT.at[2, "no immunity"])/(weightedData['RIAGENDR'] == 2).sum(),
                             genderFT.at[2, "currently infected"]/((weightedData['RIAGENDR'] == 2).sum() - genderFT.at[2, "vaccinated"])]}

#Frequency table and metrics for race
raceFT = weightedData.groupby("RIDRETH3")["status"].value_counts().unstack(fill_value=0)

raceMetrics = {"mexican american" : [raceFT.at[1, "currently infected"] / (weightedData["RIDRETH3"] == 1).sum(),
                                    raceFT.at[1, "vaccinated"] / (weightedData["RIDRETH3"] == 1).sum(),
                                    ((weightedData["RIDRETH3"] == 1).sum() - raceFT.at[1, "no immunity"]) / (weightedData["RIDRETH3"] == 1).sum(),
                                    raceFT.at[1, "currently infected"] / ((weightedData["RIDRETH3"] == 1).sum() - raceFT.at[1, "vaccinated"])],
              "hispanic": [raceFT.at[2, "currently infected"] / (weightedData["RIDRETH3"] == 2).sum(),
                           raceFT.at[2, "vaccinated"] / (weightedData["RIDRETH3"] == 2).sum(),
                           ((weightedData["RIDRETH3"] == 2).sum() - raceFT.at[2, "no immunity"]) / (weightedData["RIDRETH3"] == 2).sum(),
                           raceFT.at[2, "currently infected"] / ((weightedData["RIDRETH3"] == 2).sum() - raceFT.at[2, "vaccinated"])],
              "white" : [raceFT.at[3, "currently infected"] / (weightedData["RIDRETH3"] == 3).sum(),
                         raceFT.at[3, "vaccinated"] / (weightedData["RIDRETH3"] == 3).sum(),
                         ((weightedData["RIDRETH3"] == 3).sum() - raceFT.at[3, "no immunity"]) / (weightedData["RIDRETH3"] == 3).sum(),
                         raceFT.at[3, "currently infected"] / ((weightedData["RIDRETH3"] == 3).sum() - raceFT.at[3, "vaccinated"])],
              "black" : [raceFT.at[4, "currently infected"] / (weightedData["RIDRETH3"] == 4).sum(),
                         raceFT.at[4, "vaccinated"] / (weightedData["RIDRETH3"] == 4).sum(),
                         ((weightedData["RIDRETH3"] == 4).sum() - raceFT.at[4, "no immunity"]) / (weightedData["RIDRETH3"] == 4).sum(),
                         raceFT.at[4, "currently infected"] / ((weightedData["RIDRETH3"] == 4).sum() - raceFT.at[4, "vaccinated"])],
              "asian" : [raceFT.at[6, "currently infected"] / (weightedData["RIDRETH3"] == 6).sum(),
                         raceFT.at[6, "vaccinated"] / (weightedData["RIDRETH3"] == 6).sum(),
                         ((weightedData["RIDRETH3"] == 6).sum() - raceFT.at[6, "no immunity"]) / (weightedData["RIDRETH3"] == 6).sum(),
                         raceFT.at[6, "currently infected"] / ((weightedData["RIDRETH3"] == 6).sum() - raceFT.at[6, "vaccinated"])],
              "other" : [raceFT.at[7, "currently infected"] / (weightedData["RIDRETH3"] == 7).sum(),
                         raceFT.at[7, "vaccinated"] / (weightedData["RIDRETH3"] == 7).sum(),
                         ((weightedData["RIDRETH3"] == 7).sum() - raceFT.at[7, "no immunity"]) / (weightedData["RIDRETH3"] == 7).sum(),
                         raceFT.at[7, "currently infected"] / ((weightedData["RIDRETH3"] == 7).sum() - raceFT.at[7, "vaccinated"])]}

#Age frequency table and metrics
ageFT = weightedData.groupby("ageDecade")["status"].value_counts().unstack(fill_value=0)

ageMetrics = {"0-9" : [ageFT.at[1, "currently infected"] / (weightedData["ageDecade"] == 1).sum(),
                       ageFT.at[1, "vaccinated"] / (weightedData["ageDecade"] == 1).sum(),
                       ((weightedData["ageDecade"] == 1).sum() - ageFT.at[1, "no immunity"]) / (weightedData["ageDecade"] == 1).sum(),
                       ageFT.at[1, "currently infected"] / ((weightedData["ageDecade"] == 1).sum() - ageFT.at[1, "vaccinated"])],
             "10-19" : [ageFT.at[2, "currently infected"] / (weightedData["ageDecade"] == 2).sum(),
                       ageFT.at[2, "vaccinated"] / (weightedData["ageDecade"] == 2).sum(),
                       ((weightedData["ageDecade"] == 2).sum() - ageFT.at[2, "no immunity"]) / (weightedData["ageDecade"] == 2).sum(),
                       ageFT.at[2, "currently infected"] / ((weightedData["ageDecade"] == 2).sum() - ageFT.at[2, "vaccinated"])],
             "20-29" : [ageFT.at[3, "currently infected"] / (weightedData["ageDecade"] == 3).sum(),
                       ageFT.at[3, "vaccinated"] / (weightedData["ageDecade"] == 3).sum(),
                       ((weightedData["ageDecade"] == 3).sum() - ageFT.at[3, "no immunity"]) / (weightedData["ageDecade"] == 3).sum(),
                       ageFT.at[3, "currently infected"] / ((weightedData["ageDecade"] == 3).sum() - ageFT.at[3, "vaccinated"])],
             "30-39" : [ageFT.at[4, "currently infected"] / (weightedData["ageDecade"] == 4).sum(),
                       ageFT.at[4, "vaccinated"] / (weightedData["ageDecade"] == 4).sum(),
                       ((weightedData["ageDecade"] == 4).sum() - ageFT.at[4, "no immunity"]) / (weightedData["ageDecade"] == 4).sum(),
                       ageFT.at[4, "currently infected"] / ((weightedData["ageDecade"] == 4).sum() - ageFT.at[4, "vaccinated"])],
             "40-49" : [ageFT.at[5, "currently infected"] / (weightedData["ageDecade"] == 5).sum(),
                       ageFT.at[5, "vaccinated"] / (weightedData["ageDecade"] == 5).sum(),
                       ((weightedData["ageDecade"] == 5).sum() - ageFT.at[5, "no immunity"]) / (weightedData["ageDecade"] == 5).sum(),
                       ageFT.at[5, "currently infected"] / ((weightedData["ageDecade"] == 5).sum() - ageFT.at[5, "vaccinated"])],
             "50-59" : [ageFT.at[6, "currently infected"] / (weightedData["ageDecade"] == 6).sum(),
                       ageFT.at[6, "vaccinated"] / (weightedData["ageDecade"] == 6).sum(),
                       ((weightedData["ageDecade"] == 6).sum() - ageFT.at[6, "no immunity"]) / (weightedData["ageDecade"] == 6).sum(),
                       ageFT.at[6, "currently infected"] / ((weightedData["ageDecade"] == 6).sum() - ageFT.at[6, "vaccinated"])],
             "60-69" : [ageFT.at[7, "currently infected"] / (weightedData["ageDecade"] == 7).sum(),
                       ageFT.at[7, "vaccinated"] / (weightedData["ageDecade"] == 7).sum(),
                       ((weightedData["ageDecade"] == 7).sum() - ageFT.at[7, "no immunity"]) / (weightedData["ageDecade"] == 7).sum(),
                       ageFT.at[7, "currently infected"] / ((weightedData["ageDecade"] == 7).sum() - ageFT.at[7, "vaccinated"])],
             "70+" : [ageFT.at[8, "currently infected"] / (weightedData["ageDecade"] == 8).sum(),
                       ageFT.at[8, "vaccinated"] / (weightedData["ageDecade"] == 8).sum(),
                       ((weightedData["ageDecade"] == 8).sum() - ageFT.at[8, "no immunity"]) / (weightedData["ageDecade"] == 8).sum(),
                       ageFT.at[8, "currently infected"] / ((weightedData["ageDecade"] == 8).sum() - ageFT.at[8, "vaccinated"])]}

#income frequency table and metrics
incomeFT = weightedData.groupby("income")["status"].value_counts().unstack(fill_value=0)

incomeMetrics = {"0-24,999" : [incomeFT.at[1, "currently infected"] / (weightedData["income"] == 1).sum(),
                               incomeFT.at[1, "vaccinated"] / (weightedData["income"] == 1).sum(),
                               ((weightedData["income"] == 1).sum() - incomeFT.at[1, "no immunity"]) / (weightedData["income"] == 1).sum(),
                               incomeFT.at[1, "currently infected"] / ((weightedData["income"] == 1).sum() - incomeFT.at[1, "vaccinated"])],
                "25,000-44,999": [incomeFT.at[2, "currently infected"] / (weightedData["income"] == 2).sum(),
                               incomeFT.at[2, "vaccinated"] / (weightedData["income"] == 2).sum(),
                               ((weightedData["income"] == 2).sum() - incomeFT.at[2, "no immunity"]) / (weightedData["income"] == 2).sum(),
                               incomeFT.at[2, "currently infected"] / ((weightedData["income"] == 2).sum() - incomeFT.at[2, "vaccinated"])],
                "45,000-74,999": [incomeFT.at[3, "currently infected"] / (weightedData["income"] == 3).sum(),
                               incomeFT.at[3, "vaccinated"] / (weightedData["income"] == 3).sum(),
                               ((weightedData["income"] == 3).sum() - incomeFT.at[3, "no immunity"]) / (weightedData["income"] == 3).sum(),
                               incomeFT.at[3, "currently infected"] / ((weightedData["income"] == 3).sum() - incomeFT.at[3, "vaccinated"])],
                "75,000+" : [incomeFT.at[4, "currently infected"] / (weightedData["income"] == 4).sum(),
                               incomeFT.at[4, "vaccinated"] / (weightedData["income"] == 4).sum(),
                               ((weightedData["income"] == 4).sum() - incomeFT.at[4, "no immunity"]) / (weightedData["income"] == 4).sum(),
                               incomeFT.at[4, "currently infected"] / ((weightedData["income"] == 4).sum() - incomeFT.at[4, "vaccinated"])]}

#print metrics
print(genderFT)
printMetrics(genderMetrics)
print(raceFT)
printMetrics(raceMetrics)
print(ageFT)
printMetrics(ageMetrics)
print(incomeFT)
printMetrics(incomeMetrics)