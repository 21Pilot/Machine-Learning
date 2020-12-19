# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/11/13
# Frequentist Machine Learning_Project6

# Data 
# Data from Carngie-Mellon SataLib repositry : as well as from skitlearn datasets.
# "Titanic.txt

# Set Up
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
# Apriori
from mlxtend.frequent_patterns import apriori, association_rules

# Data load
df=pd.read_csv("titanic.csv",index_col=0)
# Data Type Check
# 891 passengers
df.info()
# Check any missing value on each feature
# Feature 'Cabin' shows significant missing values of 687 cases
df.isnull().sum()
# Feature 'Cabin' is dropped -> df2
df2 =df.drop(columns='Cabin')

# Feature 'Sex','Ticket' and 'Embarked' are object type.
# Check the range of those feature
# Feature 'Sex' : Male / Female
# Feature 'Ticekt' : Ticket with different serial number and State Initial
# Feature 'Embarked' :  S,Q,C - three different embarktation sites
print(df2.Sex.unique())
print(df2.Ticket.unique())
print(df2.Embarked.unique())

# Feature 'Survived' 'Pclass', 'Age', 'SibSp', 'Parch', and 'Fare'  are Numeric type.
# Check the range of those feature Automatically checks Count, Mean, STD, MIN, MAX, and Quartely Percentage.
# df2 includes 'Survived' but the values calcaultes for 'Survived' on this cell can be ignored. 
df2.describe()

# Feature Analysis
# 'Survived'
# Assigned : '0' for Dead / '1' for Survived 
change = {0: "Dead", 1: "Survived"}
df2.replace({'Survived' : change}, inplace=True)

# 'Age' 
# Assigned 
# Young : Age under 20 years old
# Adult : Age over 20 years and under 60 years old
# Old : Age over 60 years old 
def change_age(x):
    if x==pd.NaT:
        return 'None'
    elif x<20:
        return 'young'
    elif x<60:
        return 'adult'
    else:
        return 'older'    
df2['age_cate'] =df2['Age'].apply(change_age)

# 'Pclass'
# Catgorical 
df2['Pclass'] =df2['Pclass'].astype("str")

# 'SibSp'
# Sibling or Spouses. 
# Assigned 
# Many: SibSp same or more than 2 
# Low : SibSp lower than 2
def split_data_sib(x):
    if x>=2:
        return 'many_sib'
    else:
        return 'low_sib' 
df2['SibSp2'] =df2['SibSp'].apply(split_data_sib)

# 'Parch'
# Parents or children
# Assigned 
# Many: Parch same or more than 1 
# Low : Parch lower than 1
df2['Pclass'] =df2['Pclass'].astype("str")   
def split_data_parch(x):
    if x>=1:
        return 'many_par'
    else:
        return 'low_par'
df2['Parch2'] =df2['Parch'].apply(split_data_parch)

# 'Fare'
print(df2.Fare.describe())
def change_Fare(x):
    if x<7.1:
        return 'row'
    elif x<14.45:
        return 'middle'
    elif x<31.00:
        return 'high'
    else:
        return 'max'
df2['Fare_cate'] =df2['Fare'].apply(change_Fare)

# Test data, only with categorical data
# 'Survived','Pclass','Sex','Embarked','age_cate','Fare_cate','SibSp2','Parch2'
df_test =df2[['Survived','Pclass','Sex','Embarked','age_cate','Fare_cate','SibSp2','Parch2']]
# Show new data set with all processed categorical data
df_test

# FML.HW.MarketBasket.Seo

Market Basket Analsis

Selected Data Set - Titanic raw data set

Requirements - Market basket analysis/ Finding frequent itemsets/ Association rules.

Instruction - Carefully tuned the select frequent itemset and association rules with high support, confidence and life

A Prioiri algorithme reference : http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

The 'Titanic data set' is selected to be analyzed. The Titanic data set does not appear to be a typical data set for a market basket analysis. Still, I found it more interesting and challenging rather than the standard merchandise-purchase history data set. I expected that I will not be able to define a clear, definitive rule but will get a grasp of the trend.

The data set contains a series of information about 891 passengers aboard on Titanic on its drowning day. The data primary shows if a passenger is dead or survived during the drowning. Then it shows the passenger’s relevant information that includes the ticket class, name, sex, age, siblings, and spouses' number, parent and children's number, ticket serial number, fare price, cabin number, and the site of his or her embarkment. Each feature was attempted to be categorized as an interpretable value. The measurement ( for example, is two siblings/spouse a 'many' or 'low' ) is inevitably arbitrarily decided based on common sense.

The study attempts to show the most repeating, frequent info of a 'dead' passenger. However, the study only achieved a segment of such a goal, given that there are many values to be analyzed with numerous resulting combinations. Also, there are 177 missing 'Age' values, which may have contributed to some error or disparity.

Set Up

[ ]
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

# Apriori
from mlxtend.frequent_patterns import apriori, association_rules

Data Processing

Data Load : titanic.csv

Feature/ Description/ Data Type

Survived: Survival of Passenger: integer ( 0 NO, 1 YES)
Pclass: Ticket Class : interger ( 1st,2nd and 3rd)
Name: obejct
Sex: object
Age: float
SibSp: # of siblings or spouses aboard the Titanic together: integer
Parch : # of parents or children aboard the Titatnic together: integer
Ticket: Ticket serial number: object
Fare: Passenger fare : float
Cabin: Cabin number : object
Embarcked : Port of Embarkation: object ( C = Cherbourg, Q = Queenstown, S = Southampton)
[ ]
# Data load
df=pd.read_csv("titanic.csv",index_col=0)
[ ]
# Data Type Check
# 891 passengers
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Survived  891 non-null    int64  
 1   Pclass    891 non-null    int64  
 2   Name      891 non-null    object 
 3   Sex       891 non-null    object 
 4   Age       714 non-null    float64
 5   SibSp     891 non-null    int64  
 6   Parch     891 non-null    int64  
 7   Ticket    891 non-null    object 
 8   Fare      891 non-null    float64
 9   Cabin     204 non-null    object 
 10  Embarked  889 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 83.5+ KB
[ ]
# Check any missing value on each feature
# Feature 'Cabin' shows significant missing values of 687 cases
df.isnull().sum()
Survived      0
Pclass        0
Name          0
Sex           0
Age         177
SibSp         0
Parch         0
Ticket        0
Fare          0
Cabin       687
Embarked      2
dtype: int64
[ ]
# Feature 'Cabin' is dropped -> df2
df2 =df.drop(columns='Cabin')
[ ]
# Feature 'Sex','Ticket' and 'Embarked' are object type.
# Check the range of those feature

# Feature 'Sex' : Male / Female
# Feature 'Ticekt' : Ticket with different serial number and State Initial
# Feature 'Embarked' :  S,Q,C - three different embarktation sites

print(df2.Sex.unique())
print(df2.Ticket.unique())
print(df2.Embarked.unique())
['male' 'female']
['A/5 21171' 'PC 17599' 'STON/O2. 3101282' '113803' '373450' '330877'
 '17463' '349909' '347742' '237736' 'PP 9549' '113783' 'A/5. 2151'
 '347082' '350406' '248706' '382652' '244373' '345763' '2649' '239865'
 '248698' '330923' '113788' '347077' '2631' '19950' '330959' '349216'
 'PC 17601' 'PC 17569' '335677' 'C.A. 24579' 'PC 17604' '113789' '2677'
 'A./5. 2152' '345764' '2651' '7546' '11668' '349253' 'SC/Paris 2123'
 '330958' 'S.C./A.4. 23567' '370371' '14311' '2662' '349237' '3101295'
 'A/4. 39886' 'PC 17572' '2926' '113509' '19947' 'C.A. 31026' '2697'
 'C.A. 34651' 'CA 2144' '2669' '113572' '36973' '347088' 'PC 17605' '2661'
 'C.A. 29395' 'S.P. 3464' '3101281' '315151' 'C.A. 33111' 'S.O.C. 14879'
 '2680' '1601' '348123' '349208' '374746' '248738' '364516' '345767'
 '345779' '330932' '113059' 'SO/C 14885' '3101278' 'W./C. 6608'
 'SOTON/OQ 392086' '343275' '343276' '347466' 'W.E.P. 5734' 'C.A. 2315'
 '364500' '374910' 'PC 17754' 'PC 17759' '231919' '244367' '349245'
 '349215' '35281' '7540' '3101276' '349207' '343120' '312991' '349249'
 '371110' '110465' '2665' '324669' '4136' '2627' 'STON/O 2. 3101294'
 '370369' 'PC 17558' 'A4. 54510' '27267' '370372' 'C 17369' '2668'
 '347061' '349241' 'SOTON/O.Q. 3101307' 'A/5. 3337' '228414' 'C.A. 29178'
 'SC/PARIS 2133' '11752' '7534' 'PC 17593' '2678' '347081'
 'STON/O2. 3101279' '365222' '231945' 'C.A. 33112' '350043' '230080'
 '244310' 'S.O.P. 1166' '113776' 'A.5. 11206' 'A/5. 851' 'Fa 265302'
 'PC 17597' '35851' 'SOTON/OQ 392090' '315037' 'CA. 2343' '371362'
 'C.A. 33595' '347068' '315093' '363291' '113505' 'PC 17318' '111240'
 'STON/O 2. 3101280' '17764' '350404' '4133' 'PC 17595' '250653' 'LINE'
 'SC/PARIS 2131' '230136' '315153' '113767' '370365' '111428' '364849'
 '349247' '234604' '28424' '350046' 'PC 17610' '368703' '4579' '370370'
 '248747' '345770' '3101264' '2628' 'A/5 3540' '347054' '2699' '367231'
 '112277' 'SOTON/O.Q. 3101311' 'F.C.C. 13528' 'A/5 21174' '250646'
 '367229' '35273' 'STON/O2. 3101283' '243847' '11813' 'W/C 14208'
 'SOTON/OQ 392089' '220367' '21440' '349234' '19943' 'PP 4348' 'SW/PP 751'
 'A/5 21173' '236171' '347067' '237442' 'C.A. 29566' 'W./C. 6609' '26707'
 'C.A. 31921' '28665' 'SCO/W 1585' '367230' 'W./C. 14263'
 'STON/O 2. 3101275' '2694' '19928' '347071' '250649' '11751' '244252'
 '362316' '113514' 'A/5. 3336' '370129' '2650' 'PC 17585' '110152'
 'PC 17755' '230433' '384461' '110413' '112059' '382649' 'C.A. 17248'
 '347083' 'PC 17582' 'PC 17760' '113798' '250644' 'PC 17596' '370375'
 '13502' '347073' '239853' 'C.A. 2673' '336439' '347464' '345778'
 'A/5. 10482' '113056' '349239' '345774' '349206' '237798' '370373'
 '19877' '11967' 'SC/Paris 2163' '349236' '349233' 'PC 17612' '2693'
 '113781' '19988' '9234' '367226' '226593' 'A/5 2466' '17421' 'PC 17758'
 'P/PP 3381' 'PC 17485' '11767' 'PC 17608' '250651' '349243'
 'F.C.C. 13529' '347470' '29011' '36928' '16966' 'A/5 21172' '349219'
 '234818' '345364' '28551' '111361' '113043' 'PC 17611' '349225' '7598'
 '113784' '248740' '244361' '229236' '248733' '31418' '386525'
 'C.A. 37671' '315088' '7267' '113510' '2695' '2647' '345783' '237671'
 '330931' '330980' 'SC/PARIS 2167' '2691' 'SOTON/O.Q. 3101310' 'C 7076'
 '110813' '2626' '14313' 'PC 17477' '11765' '3101267' '323951' 'C 7077'
 '113503' '2648' '347069' 'PC 17757' '2653' 'STON/O 2. 3101293' '349227'
 '27849' '367655' 'SC 1748' '113760' '350034' '3101277' '350052' '350407'
 '28403' '244278' '240929' 'STON/O 2. 3101289' '341826' '4137' '315096'
 '28664' '347064' '29106' '312992' '349222' '394140' 'STON/O 2. 3101269'
 '343095' '28220' '250652' '28228' '345773' '349254' 'A/5. 13032' '315082'
 '347080' 'A/4. 34244' '2003' '250655' '364851' 'SOTON/O.Q. 392078'
 '110564' '376564' 'SC/AH 3085' 'STON/O 2. 3101274' '13507' 'C.A. 18723'
 '345769' '347076' '230434' '65306' '33638' '113794' '2666' '113786'
 '65303' '113051' '17453' 'A/5 2817' '349240' '13509' '17464'
 'F.C.C. 13531' '371060' '19952' '364506' '111320' '234360' 'A/S 2816'
 'SOTON/O.Q. 3101306' '113792' '36209' '323592' '315089' 'SC/AH Basle 541'
 '7553' '31027' '3460' '350060' '3101298' '239854' 'A/5 3594' '4134'
 '11771' 'A.5. 18509' '65304' 'SOTON/OQ 3101317' '113787' 'PC 17609'
 'A/4 45380' '36947' 'C.A. 6212' '350035' '315086' '364846' '330909'
 '4135' '26360' '111427' 'C 4001' '382651' 'SOTON/OQ 3101316' 'PC 17473'
 'PC 17603' '349209' '36967' 'C.A. 34260' '226875' '349242' '12749'
 '349252' '2624' '2700' '367232' 'W./C. 14258' 'PC 17483' '3101296'
 '29104' '2641' '2690' '315084' '113050' 'PC 17761' '364498' '13568'
 'WE/P 5735' '2908' '693' 'SC/PARIS 2146' '244358' '330979' '2620'
 '347085' '113807' '11755' '345572' '372622' '349251' '218629'
 'SOTON/OQ 392082' 'SOTON/O.Q. 392087' 'A/4 48871' '349205' '2686'
 '350417' 'S.W./PP 752' '11769' 'PC 17474' '14312' 'A/4. 20589' '358585'
 '243880' '2689' 'STON/O 2. 3101286' '237789' '13049' '3411' '237565'
 '13567' '14973' 'A./5. 3235' 'STON/O 2. 3101273' 'A/5 3902' '364848'
 'SC/AH 29037' '248727' '2664' '349214' '113796' '364511' '111426'
 '349910' '349246' '113804' 'SOTON/O.Q. 3101305' '370377' '364512'
 '220845' '31028' '2659' '11753' '350029' '54636' '36963' '219533'
 '349224' '334912' '27042' '347743' '13214' '112052' '237668'
 'STON/O 2. 3101292' '350050' '349231' '13213' 'S.O./P.P. 751' 'CA. 2314'
 '349221' '8475' '330919' '365226' '349223' '29751' '2623' '5727' '349210'
 'STON/O 2. 3101285' '234686' '312993' 'A/5 3536' '19996' '29750'
 'F.C. 12750' 'C.A. 24580' '244270' '239856' '349912' '342826' '4138'
 '330935' '6563' '349228' '350036' '24160' '17474' '349256' '2672'
 '113800' '248731' '363592' '35852' '348121' 'PC 17475' '36864' '350025'
 '223596' 'PC 17476' 'PC 17482' '113028' '7545' '250647' '348124' '34218'
 '36568' '347062' '350048' '12233' '250643' '113806' '315094' '36866'
 '236853' 'STON/O2. 3101271' '239855' '28425' '233639' '349201' '349218'
 '16988' '376566' 'STON/O 2. 3101288' '250648' '113773' '335097' '29103'
 '392096' '345780' '349204' '350042' '29108' '363294' 'SOTON/O2 3101272'
 '2663' '347074' '112379' '364850' '8471' '345781' '350047' 'S.O./P.P. 3'
 '2674' '29105' '347078' '383121' '36865' '2687' '113501' 'W./C. 6607'
 'SOTON/O.Q. 3101312' '374887' '3101265' '12460' 'PC 17600' '349203'
 '28213' '17465' '349244' '2685' '2625' '347089' '347063' '112050'
 '347087' '248723' '3474' '28206' '364499' '112058' 'STON/O2. 3101290'
 'S.C./PARIS 2079' 'C 7075' '315098' '19972' '368323' '367228' '2671'
 '347468' '2223' 'PC 17756' '315097' '392092' '11774' 'SOTON/O2 3101287'
 '2683' '315090' 'C.A. 5547' '349213' '347060' 'PC 17592' '392091'
 '113055' '2629' '350026' '28134' '17466' '233866' '236852'
 'SC/PARIS 2149' 'PC 17590' '345777' '349248' '695' '345765' '2667'
 '349212' '349217' '349257' '7552' 'C.A./SOTON 34068' 'SOTON/OQ 392076'
 '211536' '112053' '111369' '370376']
['S' 'C' 'Q' nan]
[ ]
# Feature 'Survived' 'Pclass', 'Age', 'SibSp', 'Parch', and 'Fare'  are Numeric type.
# Check the range of those feature
# Automatically checks Count, Mean, STD, MIN, MAX, and Quartely Percentage.

# df2 includes 'Survived' but the values calcaultes for 'Survived' on this cell
# can be ignored. 

df2.describe()

Analysis

[ ]
def change_age(x):
    if x==pd.NaT:
        return 'None'
    elif x<20:
        return 'young'
    elif x<60:
        return 'adult'
    else:
        return 'older'
    

[ ]
# Check on Quarterly value of 'Fare' - 25%,50%,75%
# 25% Fare = 7.91
# 50% Fare = 14.454
# 75% Fare = 31

print(df2.Fare.describe())
count    891.000000
mean      32.204208
std       49.693429
min        0.000000
25%        7.910400
50%       14.454200
75%       31.000000
max      512.329200
Name: Fare, dtype: float64
[ ]
# 'Fare'
# Assigned 
# Low : Fare lower than 25%. 7.1
# Middle : Fare higher than 25%, lower than 50%. 14.45
# High : Fare higher than 50%, lower than 75%. 31
# Max : Fare higher than 75%. 31

def change_Fare(x):
    if x<7.1:
        return 'row'

[ ]
# Test data, only with categorical data
# 'Survived','Pclass','Sex','Embarked','age_cate','Fare_cate','SibSp2','Parch2'

df_test =df2[['Survived','Pclass','Sex','Embarked','age_cate','Fare_cate','SibSp2','Parch2']]
[ ]
# Show new data set with all processed categorical data
df_test

# Finding Frequent Itemsets/Market Basket Analysis
dataset = []
for i in range(0, df_test.shape[0]-1):
    dataset.append([str(df_test.values[i,j]) for j in range(0, df_test.shape[1])])

# TransectionEncoder
encoder = TransactionEncoder()
encoder_array = encoder.fit(dataset).transform(dataset)
df = pd.DataFrame(encoder_array, columns=encoder.columns_)
df.head()
dataset

# Apriori
# min_support of 0.01
output = apriori(df, min_support=0.01, use_colnames=encoder.columns_)
output

# List : Market Basket Analysis on Dead Passengers
# The combinations of features for a Dead passengers 
# for 1 ~ 9 features 
def change_to_list(x):
    return list(x)[-1]
output['list'] = output.itemsets.apply(change_to_list)

# Configuration 
config = [
    ('antecedent support', 0.8),
    ('support', 0.4),
    ('confidence', 0.8),
    ('conviction', 3)
]
for metric_type, th in config:
    rules = association_rules(output, metric=metric_type, min_threshold=th)
output.loc[output.list=='Dead']

# Define Minimum rule 
# Minimum rule :
# The lowest confidence  = 0.6875 with support = 0.01236
rules[rules['confidence']==rules['confidence'].min()]
# Define Maximum rule
# Maximum confidence = 1
rules[rules['confidence']==rules['confidence'].max()]

# Graph
output = apriori(df, min_support=0.01, use_colnames=encoder.columns_)
output.sort_values("support",ascending=False)
def change_to_list_Dead(x):
  if 'Dead' in list(x):
    return True
  else:
    return False
output['list'] = output.itemsets.apply(change_to_list_Dead)
dead_output = output.loc[output.list==True]

# Show all the combination of feature associated to Dead passenger 
dead_output.sort_values('support',ascending=False)
plt.hist(dead_output['support'])
plt.title('Dead Include-Support Histogram')
plt.show()

# Association rules on graph 
# Support and Confidence 
rules = association_rules(output, metric='antecedent support', min_threshold=0.7)
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title(metric_type+' : '+'0.7')
plt.show()

# Association rules on graph 
# Antecedent Support vs Density 
rules = association_rules(output, metric='antecedent support', min_threshold=0.7)
plt.hist(rules['antecedent support'])
plt.xlabel('Antecedent Support')
plt.ylabel('Density')
plt.title('Antecedent Support:IF')
plt.show()
