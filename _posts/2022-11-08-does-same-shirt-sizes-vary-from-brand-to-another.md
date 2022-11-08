# What is Going on?


#### Since I was a child, I was curious to know does brand offer the same standards when they say this shirt's size is XL? 

#### Well, guess I am finding out today.


# Getting Familiar with the Dataset



```python
import pandas as pd
import missingno as msno
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import  numpy as np
import matplotlib.pyplot as plt
import warnings
```


```python
df = pd.read_csv('../input/shirt-size-recommendation/Shirt Size Recommendation.csv')
```


```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 836 entries, 0 to 835
Data columns (total 7 columns):
 #   Column               Non-Null Count  Dtype 
---  ------               --------------  ----- 
 0   Brand Name           835 non-null    object
 1   Type                 835 non-null    object
 2   Size                 835 non-null    object
 3   Brand Size           835 non-null    object
 4   Chest(cm)            835 non-null    object
 5   Front Length(cm)     835 non-null    object
 6   Across Shoulder(cm)  830 non-null    object
dtypes: object(7)
memory usage: 45.8+ KB
</pre>
Who on Earth extracts number values as objects?



```python
df['Size'] = pd.to_numeric(df['Size'],errors = 'coerce')
df['Chest(cm)'] = pd.to_numeric(df['Chest(cm)'],errors = 'coerce')
df['Front Length(cm)'] = pd.to_numeric(df['Front Length(cm)'],errors = 'coerce')
df['Across Shoulder(cm)'] = pd.to_numeric(df['Across Shoulder(cm)'],errors = 'coerce')
```


```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 836 entries, 0 to 835
Data columns (total 7 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   Brand Name           835 non-null    object 
 1   Type                 835 non-null    object 
 2   Size                 834 non-null    float64
 3   Brand Size           835 non-null    object 
 4   Chest(cm)            834 non-null    float64
 5   Front Length(cm)     834 non-null    float64
 6   Across Shoulder(cm)  829 non-null    float64
dtypes: float64(4), object(3)
memory usage: 45.8+ KB
</pre>

```python
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Brand Name</th>
      <th>Type</th>
      <th>Size</th>
      <th>Brand Size</th>
      <th>Chest(cm)</th>
      <th>Front Length(cm)</th>
      <th>Across Shoulder(cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Roadster</td>
      <td>Cotton</td>
      <td>38.0</td>
      <td>S</td>
      <td>100.3</td>
      <td>73.7</td>
      <td>43.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Roadster</td>
      <td>Cotton</td>
      <td>40.0</td>
      <td>M</td>
      <td>107.4</td>
      <td>74.7</td>
      <td>45.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Roadster</td>
      <td>Cotton</td>
      <td>42.0</td>
      <td>L</td>
      <td>115.1</td>
      <td>74.7</td>
      <td>45.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Roadster</td>
      <td>Cotton</td>
      <td>44.0</td>
      <td>XL</td>
      <td>122.7</td>
      <td>76.5</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Roadster</td>
      <td>Cotton</td>
      <td>46.0</td>
      <td>XXL</td>
      <td>130.3</td>
      <td>82.0</td>
      <td>50.8</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>Chest(cm)</th>
      <th>Front Length(cm)</th>
      <th>Across Shoulder(cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>834.000000</td>
      <td>834.000000</td>
      <td>834.000000</td>
      <td>829.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>42.104317</td>
      <td>113.946763</td>
      <td>76.115228</td>
      <td>47.326538</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.736348</td>
      <td>35.658632</td>
      <td>5.146851</td>
      <td>3.770188</td>
    </tr>
    <tr>
      <th>min</th>
      <td>33.000000</td>
      <td>11.800000</td>
      <td>64.000000</td>
      <td>33.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.000000</td>
      <td>104.900000</td>
      <td>73.700000</td>
      <td>44.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>42.000000</td>
      <td>111.800000</td>
      <td>76.200000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>44.000000</td>
      <td>119.400000</td>
      <td>78.700000</td>
      <td>49.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1067.000000</td>
      <td>119.400000</td>
      <td>62.200000</td>
    </tr>
  </tbody>
</table>
</div>



```python
for col in df:
    print("The Percentage in missing data is" ,df[col].isnull().sum()/len(df)*100)
```

<pre>
The Percentage in missing data is 0.11961722488038277
The Percentage in missing data is 0.11961722488038277
The Percentage in missing data is 0.23923444976076555
The Percentage in missing data is 0.11961722488038277
The Percentage in missing data is 0.23923444976076555
The Percentage in missing data is 0.23923444976076555
The Percentage in missing data is 0.8373205741626795
</pre>

```python
df = df.dropna()
```


```python
df.isna().sum()
```

<pre>
Brand Name             0
Type                   0
Size                   0
Brand Size             0
Chest(cm)              0
Front Length(cm)       0
Across Shoulder(cm)    0
dtype: int64
</pre>
# Question 1: Does the Shirt Attributes Vary from Size to Another?



```python
fig = px.bar(data_frame = df,
             x='Brand Size')
fig.show()
```

![newplot](https://user-images.githubusercontent.com/115538847/200497042-ae85deb1-4108-4045-8353-32399eb5bd21.png)


```python
fig = px.box(df, 
             x="Chest(cm)" )
fig.show()
```

![newplot](https://user-images.githubusercontent.com/115538847/200497203-2138031a-6917-47bf-8131-fb579cdd09f1.png)

```python
plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y="Chest(cm)", jitter = True, size = 6)
```

![image](https://user-images.githubusercontent.com/115538847/200497289-39b6a11c-dc3a-477c-8788-b6295935d717.png)

```python
df['Chest(cm)'].describe()
```

<pre>
count     829.000000
mean      114.031001
std        35.738908
min        11.800000
25%       104.900000
50%       111.800000
75%       119.400000
max      1067.000000
Name: Chest(cm), dtype: float64
</pre>

```python
df.loc[df['Chest(cm)'] == 1067]
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Brand Name</th>
      <th>Type</th>
      <th>Size</th>
      <th>Brand Size</th>
      <th>Chest(cm)</th>
      <th>Front Length(cm)</th>
      <th>Across Shoulder(cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>554</th>
      <td>D Kumar</td>
      <td>Cotton</td>
      <td>38.0</td>
      <td>M</td>
      <td>1067.0</td>
      <td>76.2</td>
      <td>45.7</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.drop([554], axis = 0, inplace = True)
```


```python
plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y="Chest(cm)", jitter = True, size = 6)
```

![image](https://user-images.githubusercontent.com/115538847/200497486-71e65833-754a-4089-81d7-75c396792b18.png)

```python
plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y='Across Shoulder(cm)', jitter = True, size = 6)
```

![image](https://user-images.githubusercontent.com/115538847/200497553-ee49555f-3223-424b-ae43-346fc68f3574.png)


```python
plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y='Front Length(cm)', jitter = True, size = 6)
```

![image](https://user-images.githubusercontent.com/115538847/200497615-d49ee9a8-4a0b-46e8-a6ec-07235e1b27f6.png)


```python
fig = sns.pairplot(df, hue = 'Size')
```

![image](https://user-images.githubusercontent.com/115538847/200497787-1b615340-c5b9-4732-9af4-ff3991258131.png)


```python
_fig = sns.pairplot(df, hue = 'Brand Size')
```

![image](https://user-images.githubusercontent.com/115538847/200498072-b89ad37c-8005-4673-9b83-01efd27a7a5d.png)


# Question 1: Answered! 


Here is what I found out so far:



1. All the attributes that belongs to a shirt (things like: Front Length - Across Shoulders) can be the same across different sizes. That is obvious in the previous graphs, they are very close in sizes **though the brand sizes remain the same**. 



2. There is a significant increase in the shirt's attributes (Front length, for instance) as you go further in the size,  **though it's likely you find a small shirt that has the same chest size as an XL shirt.**


# Question Two: Does the shirt Material affect the shirt's attributes?



```python
fig = px.bar(data_frame = df, x='Type')
fig.show()
```

![newplot](https://user-images.githubusercontent.com/115538847/200498224-13f3053a-cb4f-4834-bbd0-62b8a84f8b0c.png)


```python
df['Type'] = df['Type'].str.replace('cotton','Cotton')
```


```python
Shirt_Material = df['Type'].value_counts().plot.pie(autopct='%23f%%', radius = 2)
```

![image](https://user-images.githubusercontent.com/115538847/200498328-4f841ba8-9d61-44d2-9a6e-0c32e9ba2da6.png)

# Question 2: One Day with larger data I will Answer it!


To Answer a such a question, it would be hard - if it's not impossible to give an accurate answer- within the data I have. It's not accurate to compare a mean of five shirts to a mean of nearly 750 shirts. Needless to say that those five shirts does not cover all the sizes of the shirt. **So We Won't be able to give an answer to such question.**


# Qeustion 3: Does Sizes Vary from Brand X to Brand Y?



```python
fig = px.bar(data_frame = df, x = 'Brand Name')
fig
```

<div>                            <div id="1cc8c824-786b-427c-9e95-e5f33f1c83ea" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("1cc8c824-786b-427c-9e95-e5f33f1c83ea")) {                    Plotly.newPlot(                        "1cc8c824-786b-427c-9e95-e5f33f1c83ea",                        [{"alignmentgroup":"True","hovertemplate":"Brand Name=%{x}<br>count=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["Roadster","Roadster","Roadster","Roadster","Roadster","Roadster","WROGN","WROGN","WROGN","WROGN","WROGN","WROGN","WROGN","WROGN","WROGN","WROGN","Dennis Lingo","Dennis Lingo","Dennis Lingo","Dennis Lingo","Dennis Lingo","Highlander","Highlander","Highlander","Highlander","Highlander","Campus Sutra","Campus Sutra","Campus Sutra","Campus Sutra"," The Indian Garage"," The Indian Garage"," The Indian Garage"," The Indian Garage"," The Indian Garage","INVICTUS","INVICTUS","INVICTUS","INVICTUS","INVICTUS","The Bear House","The Bear House","The Bear House","The Bear House","The Bear House","United Colors of Benetto","United Colors of Benetto","United Colors of Benetto","United Colors of Benetto","United Colors of Benetto","United Colors of Benetto","US POLO","US POLO","US POLO","US POLO","US POLO","US POLO","LOCOMOTIVE","LOCOMOTIVE","LOCOMOTIVE","LOCOMOTIVE","HERE & NOW","HERE & NOW","HERE & NOW","HERE & NOW","HERE & NOW","Mast & Harbour","Mast & Harbour","Mast & Harbour","Mast & Harbour","Peter England","Peter England","Peter England","Peter England","Peter England","Peter England","Peter England","Arrow Sport","Arrow Sport","Arrow Sport","Arrow Sport","Arrow Sport","Arrow Sport","Arrow Sport","Wild West","Wild West","Wild West","Wild West","Wild West","Wild West","Van Heusen Denim Labs","Van Heusen Denim Labs","Van Heusen Denim Labs","Van Heusen Denim Labs","Van Heusen Denim Labs","Van Heusen Denim Labs","Bene Kleed","Bene Kleed","Bene Kleed","Bene Kleed","Bene Kleed","R&B","R&B","R&B","R&B","R&B","JANISH","JANISH","JANISH","JANISH","JANISH","Van Heusen Sport","Van Heusen Sport","Van Heusen Sport","Van Heusen Sport","Ether","Ether","Ether","Ether","Nautica","Nautica","Nautica","Nautica","Nautica","IVOC","IVOC","IVOC","IVOC","IVOC","V Mart","V Mart","V Mart","V Mart","V Mart","V Mart","Harvard","Harvard","Harvard","Harvard","Allen Solly","Allen Solly","Allen Solly","Allen Solly","Allen Solly","Allen Solly","Allen Solly","Tisbtane","Tisbtane","Tisbtane","Tisbtane","Tisbtane","Tisbtane","Levis","Levis","Levis","Levis","Levis","Sztori","Sztori","Sztori","Sztori","Sztori","Hancock","Hancock","Hancock","Hancock","Park Avenue","Park Avenue","Park Avenue","Park Avenue","Park Avenue","Black Berry","Black Berry","Black Berry","Black Berry","Black Berry","Black Berry","Forever ","Forever ","Forever ","Forever ","Forever ","Forever ","Tommy Hilfiger","Tommy Hilfiger","Tommy Hilfiger","Tommy Hilfiger","Tommy Hilfiger","Tommy Hilfiger","Raymond","Raymond","Raymond","Raymond","Raymond","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Shaftesbury London","Dennison","Dennison","Dennison","Dennison","Sojanya","Sojanya","Sojanya","Sojanya","Sojanya","SELECTED","SELECTED","SELECTED","SELECTED","SELECTED","English Navy","English Navy","English Navy","English Navy","Nirvaan","Nirvaan","Nirvaan","Nirvaan","Code By Lifestyle","Code By Lifestyle","Code By Lifestyle","Code By Lifestyle","Code By Lifestyle","Code By Lifestyle","Identiti","Identiti","Identiti","Identiti","Identiti"," Harvard"," Harvard"," Harvard"," Harvard","Soratia","Soratia","Soratia","Soratia","Soratia","Parx","Parx","Parx","Parx","Green Fibre","Green Fibre","Green Fibre","Green Fibre","Green Fibre","Basics","Basics","Basics","Basics","Basics","Basics","Basics","Color Plus","Color Plus","Color Plus","Color Plus","AD by Aravimdh","AD by Aravimdh","AD by Aravimdh","AD by Aravimdh","Arrow Newyork","Arrow Newyork","Arrow Newyork","Arrow Newyork","Arrow Newyork","Arrow Newyork","Arrow Newyork","Arrow Newyork","Arrow Newyork","Andamen","Andamen","Andamen","Andamen","Andamen","American Eagle","American Eagle","American Eagle","American Eagle","American Eagle","American Eagle","abof","abof","abof","abof","Antony Morato","Antony Morato","Antony Morato","Antony Morato","Antony Morato","Anuok","Anuok","Anuok","Anuok","Anuok","Amrut Varsha Creation","Amrut Varsha Creation","Amrut Varsha Creation","Amrut Varsha Creation","Amrut Varsha Creation","Amrut Varsha Creation","Amrut Varsha Creation","ANFOLD","ANFOLD","ANFOLD","ANFOLD","ANFOLD","ANFOLD","ANFOLD","ANFOLD","ANFOLD"," American Bull"," American Bull"," American Bull"," American Bull"," American Bull","Allen Solly Tribe","Allen Solly Tribe","Allen Solly Tribe","Allen Solly Tribe","A okay","A okay","A okay","Armisto","Armisto","Armisto","Armisto","Armisto","American Swan","American Swan","American Swan","American Swan","American Swan","Aazing London","Aazing London","Aazing London","Aazing London","Aazing London","Aazing London","Aesthetic Bodies","Aesthetic Bodies","Aesthetic Bodies","Aesthetic Bodies","Aesthetic Bodies","Beat London","Beat London","Beat London","Beat London","Beat London","Beat London","Being Human","Being Human","Being Human","Being Human","Being Human","Byford By Pantaloons","Byford By Pantaloons","Byford By Pantaloons","Byford By Pantaloons","Byford By Pantaloons","Brunn & Stengade","Brunn & Stengade","Brunn & Stengade","Brunn & Stengade","Brunn & Stengade","Black Coffe","Black Coffe","Black Coffe","Black Coffe","Bawerin","Bawerin","Bawerin","Bawerin","BigBanana","BigBanana","BigBanana","BigBanana","BigBanana","BigBanana","Bushirt","Bushirt","Bushirt","Bushirt","Bossini","Bossini","Bossini","Bossini","Bossini","Bossini","Bene Kleed Plus","Bene Kleed Plus","Bene Kleed Plus","Bene Kleed Plus","Bedgasm","Bedgasm","Bedgasm","Bedgasm","Blamblack","Blamblack","Blamblack","Blamblack","Blamblack","Break Bounce","Break Bounce","Break Bounce","Break Bounce","Break Bounce","Balista","Balista","Balista","Balista","Bewakoof","Bewakoof","Bewakoof","Bewakoof","Bewakoof","Bewakoof","British club","British club","British club","British club","British club","Braclo","Braclo","Braclo","Braclo","Braclo","Cavallo By LInen","Cavallo By LInen","Cavallo By LInen","Cavallo By LInen","Canary Lodon","Canary Lodon","Canary Lodon","Canary Lodon","Canary Lodon","Cantabil","Cantabil","Cantabil","Cantabil","Celio","Celio","Celio","Celio","Crocodile","Crocodile","Crocodile","Crocodile","Crocodile","Crimsoune Cloub","Crimsoune Cloub","Crimsoune Cloub","Crimsoune Cloub","Crimsoune Cloub","Cape Canary","Cape Canary","Cape Canary","Cape Canary","Chennis","Chennis","Chennis","Chennis","Chennis","Calvin Klein Jeans","Calvin Klein Jeans","Calvin Klein Jeans","Calvin Klein Jeans","Calvin Klein Jeans","Chkokko","Chkokko","Chkokko","Chkokko","Cobb","Cobb","Cobb","Cobb","Cobb","Club York","Club York","Club York","Club York","Club York","Cherokke","Cherokke","Cherokke","Cherokke","Cherokke","Copperline","Copperline","Copperline","Copperline","Classic Polo ","Classic Polo ","Classic Polo ","Classic Polo ","Classic Polo ","Classic Polo ","Columbia","Columbia","Columbia","Columbia","Columbia","Colvynharris","Colvynharris","Colvynharris","Colt","Colt","Colt","Colt","Colt","Croydon","Croydon","Croydon","Croydon","Canoe","Canoe","Canoe","Canoe","Canoe","Croydon Uk","Croydon Uk","Croydon Uk","Croydon Uk"," Camla"," Camla"," Camla"," Camla"," Camla"," Cross Court"," Cross Court"," Cross Court"," Cross Court","Callno Lodon","Callno Lodon","Callno Lodon","Callno Lodon","Callno Lodon","Callno Lodon","D Kumar","D Kumar","D Kumar","Dillinger","Dillinger","Dillinger","Dillinger","Dillinger","Dejano","Dejano","Dejano","Dejano","Dejano","Don Vino","Don Vino","Don Vino","Don Vino","Ducati","Ducati","Ducati","Ducati","Ducati","Double Two","Double Two","Double Two","Double Two","Double Two","Donzell","Donzell","Donzell","Donzell","Donzell","Dcot By Donear","Dcot By Donear","Disrupt","Disrupt","Disrupt","Disrupt","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Drean of Glory INC ","Denimize by Fame Forever","Denimize by Fame Forever","Denimize by Fame Forever","Denimize by Fame Forever","Denimize by Fame Forever","English Navy","English Navy","English Navy","English Navy","excalibur","excalibur","excalibur","excalibur","excalibur","Encore By Invictus","Encore By Invictus","Encore By Invictus","Encore By Invictus","Encore By Invictus","Estela","Estela","Estela","Estela","Ecentric","Ecentric","Ecentric","Ecentric","Ecentric","Even","Even","Even","Even","Even","Evooq","Evooq","Evooq","Evooq","Emerals","Emerals","Emerals","Emerals","Ethnix By Raymond","Ethnix By Raymond","Ethnix By Raymond","Ethnix By Raymond","Eppe ","Eppe ","Eppe ","Eppe "," ElaBaroda"," ElaBaroda"," ElaBaroda"," ElaBaroda"," ElaBaroda","Flying Machine","Flying Machine","Flying Machine","Flying Machine","Flying Machine","Flying Machine","Fame Forever By Lifestyle","Fame Forever By Lifestyle","Fame Forever By Lifestyle","Fame Forever By Lifestyle","Fame Forever By Lifestyle","Forca","Forca","Forca","Forca","Forca","Forca","Fashion Fricks","Fashion Fricks","Fashion Fricks","French Connection","French Connection","French Connection","French Connection","French Connection","Fubar","Fubar","Fubar","Forcaz By Decathlon","Forcaz By Decathlon","Forcaz By Decathlon","Forcaz By Decathlon","Forcaz By Decathlon","Forcaz By Decathlon","Finnoy","Finnoy","Finnoy","Finnoy","Finnoy","FreeSoul","FreeSoul","FreeSoul","FreeSoul","Fabindia","Fabindia","Fabindia","Fabindia","Fabindia","Firangi Yarn","Firangi Yarn","Firangi Yarn","Firangi Yarn","Forca By Lifestyle","Forca By Lifestyle","Forca By Lifestyle","Forca By Lifestyle","Forca By Lifestyle","Forca By Lifestyle","Foga","Foga","Foga","Foga","Foga","French Crown","French Crown","French Crown","French Crown","French Crown","French Crown","French Crown","French Crown","French Crown","Four One Oh","Four One Oh","Four One Oh","Four One Oh","Fugazze","Fugazze","Fugazze","Fugazze","Fugazze","GreenFibre","GreenFibre","GreenFibre","GreenFibre","GreenFibre","Globus","Globus","Globus","Globus","Globus","Gant","Gant","Gant","Gant","Gant","Gant","Gant","Gespo","Gespo","Gespo","Gespo","Gristones","Gristones","Gristones","Gristones","Glordano","Glordano","Glordano","Glordano","Glordano","Hangup","Hangup","Hangup","Hangup","High Star","High Star","High Star","High Star","HubberHolme","HubberHolme","HubberHolme","HubberHolme","Huetrap","Huetrap","Huetrap","Huetrap","Huetrap","Harsam","Harsam","Harsam","Harsam","Harsam","Hatheli ","Hatheli ","Hatheli ","Hatheli ","Hatheli ","Hatheli "," Hypernation"," Hypernation"," Hypernation"," Hypernation"," Hypernation","Huggun","Huggun","Huggun","Huggun","InstaFab Plus","InstaFab Plus","InstaFab Plus","InstaFab Plus","Indian Terrian","Indian Terrian","Indian Terrian","Indian Terrian","Indian Terrian","Ivoc Plus","Ivoc Plus","Ivoc Plus","Ivoc Plus","Indo Era","Indo Era","Indo Era","Indo Era","Indo Era","Imyoung","Imyoung","Imyoung","Imyoung","Imyoung","Indus Route By Pantaloons","Indus Route By Pantaloons","Indus Route By Pantaloons","Indus Route By Pantaloons","Indus Route By Pantaloons"],"xaxis":"x","y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Brand Name"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('1cc8c824-786b-427c-9e95-e5f33f1c83ea');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Alright, We have a pretty balanced range of brands in the dataset. But we need to see which brand offer which sizes before diving deep into it.



```python
print(df['Brand Size'].unique())
```

<pre>
[' S' 'M' 'L' 'XL' 'XXL' '3XL' 'S' 'EL' 'KL' ' M' 'TXL' 'XS' 'FXL' ' L'
 ' XS' 'TWOXL' 'FOXL' 'SXL' ' XXL' 'FIXL' 'XXS' 'SEXL' ' XL' 'SIXL' 'EXL'
 'NXL']
</pre>

```python
len(df['Brand Name'].unique())
```

<pre>
164
</pre>
After going on through multiple technique of trying to represent 164 brand into a clear visualization, I couldn't. Maybe this isn't impossible but sure it's beyond my skill. So I decided to visit the good old sampling techniques.

So, We will choose 4 random brands and compare them to see what the results look like.



```python
fig = px.bar(data_frame= df[df.Size == 40 ],
             x='Brand Size')
fig.show()
```

<div>                            <div id="5014bf0d-af27-4cf8-ad5b-1b8cb153f54f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("5014bf0d-af27-4cf8-ad5b-1b8cb153f54f")) {                    Plotly.newPlot(                        "5014bf0d-af27-4cf8-ad5b-1b8cb153f54f",                        [{"alignmentgroup":"True","hovertemplate":"Brand Size=%{x}<br>count=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["M","M","M","M","M","M","M","L","M","M"," M","M","M","M","M","L","M","M","L","M","M","M"," M"," M"," M","M","M","M","M","M","M","M","L","M","M","M","M","M","M","M","M","M","M","M","M","L","L","L","M","M","M","M","M","L","S","M","M","L","M","L","M","M","M","M","L","M","M","L","M","M","M","M","M","M","M","L","M"," M","L","M"," M","M","L","L","M","M","M","M","M","M","L","S","L","M","M","M","M","M","M","M","L","M","M","M"," M","M","M","M","M","M","M","M","M","M","L","L","M","M","M","M","M","S","M","L","M","L","M","M","M","S","M","M","M","M"," M","M","M","M","L","L","M","M"," M","M","M","L"," L","L","M","M","M","M","M","M"],"xaxis":"x","y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Brand Size"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('5014bf0d-af27-4cf8-ad5b-1b8cb153f54f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
fig = px.bar(data_frame= df[df.Size == 39 ],
             x='Brand Size')
fig.show()
```

<div>                            <div id="270cbdb2-4457-4ea8-a80a-602e631892ae" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("270cbdb2-4457-4ea8-a80a-602e631892ae")) {                    Plotly.newPlot(                        "270cbdb2-4457-4ea8-a80a-602e631892ae",                        [{"alignmentgroup":"True","hovertemplate":"Brand Size=%{x}<br>count=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["S","S","S","S","M","S","S","M","S","M","S","S","S"," S"," S","M","S","S","S","S","M","M","S","S","S","S","S","M","S","S","M","L","M"," S","S","S"," S","M","M","S","S","M","S","S"," S","S","S","S","S","S","XS","S","S"," S","S","M","S","S"],"xaxis":"x","y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Brand Size"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('270cbdb2-4457-4ea8-a80a-602e631892ae');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
fig = px.bar(data_frame= df[df.Size == 42 ], x='Brand Size')
fig.show()
```

<div>                            <div id="115f407a-96b5-42b1-a225-9b724c497535" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("115f407a-96b5-42b1-a225-9b724c497535")) {                    Plotly.newPlot(                        "115f407a-96b5-42b1-a225-9b724c497535",                        [{"alignmentgroup":"True","hovertemplate":"Brand Size=%{x}<br>count=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["L","L","L","L","L","L","L","XL","L","L","L","L","L","L","L","XL","L","L","XL","L","L","L"," L"," L","L","L","L","L","L","L","L","L","XL","XL","L","L","L","L","L","L","L","L","L","L","L","XL","XL","XL","L","L","L","L","L","XL","M","L","M","M","L","XL","L","XL","L","L","L","L","XL","L","L","XL","L","L","L","L","L","L","L","XL","L"," L","XL","L"," L","L","XL","XL","L","L","L","L","L","L","XL","M","XL","L","L","L","L","L","L","L","XL","L","L","L"," L","L","L","L","L","L","L","L","L"," XL","XL","L","L","L","L","L","M","L","XL","L","XL","L","L","L","M","L","L","L","L"," L","L","L","L","XL","XL","L","L","L","L","L","XL","XL","XL","L","L","L","L","L"],"xaxis":"x","y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Brand Size"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('115f407a-96b5-42b1-a225-9b724c497535');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
df.rename(columns = {'Brand Name': 'Brand_Name'}, inplace = True)
```


```python
sns.set(style="darkgrid")
sns.set_palette("Paired")
fig, axs = plt.subplots(3, 3, figsize=(25, 25))

sns.barplot(data= df[df.Brand_Name == 'Roadster'], x = 'Brand Size', y = 'Size', ax=axs[0, 0]).set(title = 'Roadster')
sns.barplot(data= df[df.Brand_Name == 'WROGN'], x = 'Brand Size', y = 'Size', ax=axs[0, 1]).set(title = 'WROGN')
sns.barplot(data= df[df.Brand_Name == 'R&B'], x = 'Brand Size', y = 'Size', ax=axs[0, 2]).set(title = 'R&B')
sns.barplot(data= df[df.Brand_Name == 'Campus Sutra'], x = 'Brand Size', y = 'Size', ax=axs[1, 0]).set(title = 'Campus Sutra')
sns.barplot(data= df[df.Brand_Name == 'Black Coffe'], x = 'Brand Size', y = 'Size', ax=axs[1, 1]).set(title = 'Black Coffe')
sns.barplot(data= df[df.Brand_Name == 'Arrow Newyork'], x = 'Brand Size', y = 'Size', ax=axs[1, 2]).set(title = 'Arrow Newyork')
sns.barplot(data= df[df.Brand_Name == 'Black Berry'], x = 'Brand Size', y = 'Size', ax=axs[2, 0]).set(title = 'Black Berry')
sns.barplot(data= df[df.Brand_Name == 'Cantabil'], x = 'Brand Size', y = 'Size', ax=axs[2, 1]).set(title = 'Cantabil')
sns.barplot(data= df[df.Brand_Name == 'Forca'], x = 'Brand Size', y = 'Size', ax=axs[2, 2]).set(title = 'Forca')
```

<pre>
[Text(0.5, 1.0, 'Forca')]
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABakAAAWRCAYAAACbkxsKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAAsTAAALEwEAmpwYAAC31klEQVR4nOzde5zVdZ0/8NcMMCCCjCAIoqt5Cdlc4zLe8kJBKRre2koWFzN1jSy1fkHiDRDzwmUrM11ydS1bzaiU4qJuBlmaS6KQS1YSqakgIDAqyM2Z+f3Ro9GR24DDfGfg+Xw85vGY8/1+z/e8z+ecmff5vs73fE5JTU1NTQAAAAAAoAClRRcAAAAAAMCuS0gNAAAAAEBhhNQAAAAAABRGSA0AAAAAQGGE1AAAAAAAFEZIDQAAAABAYYTUsAt56aWX0qNHj7z11ltFlwIAAAAASYTU0GT0798/hx9+eHr37p1jjz02I0eOzOrVqwup5b777su//Mu/FHLbANBcfOc738kFF1xQZ9mJJ564yWXTp09Pjx490qtXr/Tu3TvHH398brjhhlRVVdXZdtasWfnkJz+ZXr165aijjspXvvKVvPLKK3W2Wbp0aa666qocd9xx6d27dwYMGJCRI0dm4cKFSd5+U/rf/u3f6lxv+PDhufnmmxvq7gPATmlrx+aPPvpoBg4cmN69e+e0007LM888U+f6s2fPzqGHHprevXvX9vxvfetbjX03oNkRUkMTMmnSpMydOzdTpkzJM888k9tuu63okraLM7UB2BVUVFRk7ty5tUHz0qVL89Zbb+UPf/hDnWUvvPBCKioqkiQ//elPM3fu3Pz3f/93ZsyYkZ/85Ce1+3vwwQfzla98JZ/5zGfyv//7v5k2bVrKysoyZMiQvPbaa0mSlStXZvDgwVmzZk3uueeePPXUU7n//vtzxBFH5De/+U2d+p5++uk89dRTjTEUALBT2dKx+ciRI3Peeeflqaeeyr//+79njz322Oj6Xbp0ydy5czN37tzcc889+fGPf5yHH364Me8CNDtCamiCOnfunOOOOy5/+MMfkiS/+MUv8vGPfzwVFRUZOnRo7ZlSSXLbbbflox/9aHr37p1TTjklP//5z2vXVVVVZdy4cTnqqKMyYMCAPPLII3Vu57777suAAQPSu3fv9O/fPz/72c+ycOHCjB49OvPmzUvv3r1rD6rXr1+fcePG5cMf/nA+9KEPZdSoUVm7dm2Sv71TfMIJJ+S2227Lsccem8svv3xHDxEAFO6f/umfakPpJJkzZ06OOuqovO9976uz7B/+4R+y995717nu/vvvnz59+tRuV1NTk3HjxuXzn/98Tj311LRp0yadO3fOddddl7Zt2+a73/1ukuS73/1u2rVrlwkTJuQf/uEfUlJSkj322CP//M//nKFDh9a5jfPPPz/f+MY3dvAoAMDO693H5knSsmXLdO/ePSUlJTnkkEOy7777bnEf++23X3r37p0///nPO7pcaNaE1NAEvfLKK/n1r3+df/iHf8hzzz2Xr3zlK7niiivy+OOP54QTTsiwYcOyfv36JH9reHfffXeefPLJfPGLX8yIESOydOnSJMnkyZMza9asTJkyJT/5yU/y4IMP1t7Gm2++ma997Wv5z//8z8ydOzf33ntvevbsmYMOOijXXHNNevXqlblz52bOnDlJkokTJ+a5557LlClT8j//8z9ZunRpbrnlltr9vfrqq3nttdcya9asXHvttY04WgBQjLKyshx++OG1vXLOnDnp27dv+vbtW2fZ39/wfaeFCxfmySefzP77758k+ctf/pJFixZl4MCBdbYrLS3NiSeeWHuW9OOPP56PfexjKS3d+sv4IUOG5Pnnn9/oDGsAoH7eeWye/O1N5cMPPzxXXXVVXnrppXrt4/nnn89TTz2VD37wgzuyVGj2hNTQhHzhC19I7969069fv3Ts2DGXXHJJZsyYkX79+uXYY49Nq1atcv7552ft2rWZO3dukuTkk0/O3nvvndLS0pxyyinZf//98/TTTydJHnjggXzmM59Jt27dUl5ens997nN1bq+0tDQLFizI2rVr06VLlxxyyCGbrKumpiaTJ0/OFVdckfLy8rRr1y6f+9znMn369Dr7uuSSS1JWVpY2bdrsoBECgKblyCOPzBNPPJHk7UC6b9++dZYdeeSRtdufeeaZ6dWrV0455ZQceeSRGTJkSJK/TeOR/O3jwe/WuXPn2vUrV67MXnvtVbvuF7/4RSoqKtK7d++cd955da7Xpk2bDBs2LN/85jcb7g4DwC5gU8fmSfKf//mfWbNmTb785S/n3HPPrQ2qf/SjH+Xiiy+uvf7SpUtTUVGRPn365KSTTsoHP/jB9O3bt5D7As2FkBqakFtuuSVz587N97///fzlL3/JypUrs3Tp0uyzzz6125SWlqZbt25ZsmRJkmTKlCk5/fTTU1FRkYqKiixYsKD2QHbp0qXp1q1b7XXfuZ+2bdvmG9/4Ru69994cd9xxufDCC+tMI/JOK1asyJo1a/KJT3yi9nYuuOCC2ttJkj333DOtW7du0PEAgKauoqIiTz75ZCorK7NixYoccMAB6dOnT+bOnZvKysosWLCgzpnU999/f+bOnZtvfOMb+d3vfpc333wzyd/6aJLaT0O907Jly2rXl5eXZ9myZbXrBgwYkDlz5uSKK67Ihg0bNrrupz71qbz66quZOXNmg95vANiZberYPEnuuuuuXHTRRTnttNNy/vnn55xzzslLL72Up556KkcffXTt9bt06ZI5c+bkqaeeypw5c9K6deuMHDmyqLsDzYKQGpqgI488Mp/4xCcybty4dOnSJYsWLapdV1NTk8WLF2fvvffOyy+/nKuuuipXX311Zs+enTlz5tQ5G7pz585ZvHhx7eV3/p4kxx9/fO688848+uijOfDAA3P11VcnSUpKSupst+eee6ZNmzaZPn165syZkzlz5uTJJ5+sPZt7U9cBgF1B7969s2rVqkyePDl9+vRJkrRr1y5dunTJ5MmT06VLl+y33351rlNSUpJTTjklvXr1qp0668ADD0zXrl3rTM2VJNXV1fmf//mf2gPfY445Jg8//HCqq6vrVV9ZWVm++MUv5qabbkpNTc17vbsAsEt557F5krz11lt56623kiT/8i//kk9/+tM555xzMnv27Jx++umb3Ef79u1z6qmnZtasWY1WNzRHQmpooj7zmc/kN7/5Te0XHj7++OPZsGFD/uu//itlZWXp3bt31qxZk5KSknTs2DFJ8pOf/CQLFiyo3cfJJ5+c73//+3nllVfy2muv1flG4ldffTUPP/xw3nzzzZSVlaVt27a181t26tQpS5YsqZ33urS0NJ/61Kdy/fXXZ/ny5UmSJUuW5Ne//nVjDQcANElt2rTJYYcdlu9+97t1zpju27fvRsve7cILL8yPfvSjLFu2LCUlJbnsssvyH//xH5k6dWrWrVuXZcuW5corr8yqVaty7rnnJknOPffcvP766xkxYkT++te/pqamJqtWrarzhU7vdvrpp2fdunV59NFHG+x+A8Cu4u/H5n/84x8zcODAjB8/Pi+++GLeeuutHH744amsrEyrVq02+wby6tWrM3369Bx88MGNXDk0L0JqaKI6duyY008/PbfccksmTJiQa6+9NkcffXRmzZqVSZMmpaysLAcffHDOO++8DB48OB/60Ify7LPP1p7FlSSf/vSnc9xxx+X000/PmWeemRNPPLF2XXV1db773e/m+OOPr51Pc8yYMUmSo48+OgcffHCOO+64HHXUUUmSESNGZP/998+nP/3p9OnTJ+eee26ee+65Rh0TAGiKjjjiiCxfvrzOXJN9+/bN8uXLc8QRR2z2ej169EhFRUXuuOOOJMkpp5yS8ePH57vf/W6OOuqofPzjH8+6devygx/8oHa6j44dO+aHP/xhWrdunSFDhqRPnz4544wzsnr16to+/m4tWrTIJZdcksrKyga7zwCwq3jnsfnIkSNTUVGRs88+O0cccURuvvnm3HLLLTn00EPzxS9+sXbqraVLl6Z3797p3bt3+vfvn9deey0TJ04s+J5A01ZS43N/AAAAAAAUxJnUAAAAAAAURkgNAAAAAEBhhNQAAAAAABRGSA0AAAAAQGFaNtYN9e/fP2VlZWndunWSZPjw4Tn++OMzb968jBo1KuvWrUv37t0zYcKEdOrUqbHKAgAAAACgQCU1NTU1jXFD/fv3z6RJk/L+97+/dll1dXVOOumk3HDDDamoqMitt96aF198MTfccMM27XvlytWprm6UuwHALqS0tCR77rl70WXsNPRrAHYE/bph6dcA7Chb6tmNdib1psyfPz+tW7dORUVFkmTw4MEZMGDANofU1dU1migANHH6NQA0ffo1AEVo1JB6+PDhqampSd++ffP//t//y+LFi7PPPvvUru/YsWOqq6tTWVmZ8vLyeu+3U6d2O6BaAAAAAAB2tEYLqe++++5069Yt69evz3XXXZexY8fmYx/7WIPse/nyVd7pBaDBlZaWeCMUAAAAdrDSxrqhbt26JUnKysoyZMiQPPXUU+nWrVsWLVpUu82KFStSWlq6TWdRAwAAAADQfDVKSP3mm2/mjTfeSJLU1NRkxowZ6dmzZw477LCsXbs2c+bMSZLce++9GThwYGOUBAAAAABAE9Ao030sX748F198caqqqlJdXZ2DDjooo0ePTmlpacaPH5/Ro0dn3bp16d69eyZMmNAYJQEAAAAA0ASU1NTUNPvJnM1JDcCOYE7qhqVfA7Aj6NcNS78GYEfZ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E/KBD3ygwMoAgL/TrwGgedCzAWiOmsR0HwAAAAAA7JoKD6m7deuWJUuWpKqqKklSVVWVpUuXpttO/JFyAGhu9GsAaB70bACao8JD6k6dOqVnz56ZNm1akmTatGnp2bNnvee3BAB2PP0aAJoHPRuA5qjwOamTZOHChRk5cmRef/317LHHHhk3blwOPPDAossCAN5BvwaA5kHPBqC5aRIhNQAAAAAAu6bCp/sAAAAAAGDXJaQGAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBC6gbw2muv5f/9v/+XQYMG5dRTT83pp5+exx9/vOiyGl3//v1z3HHHpaqqqnbZfffdlx49euS///u/C6ysWP3798+zzz5bdBmFqayszAknnJCnn366dtmkSZNy8cUX57777ssll1xSYHU73pbu/6RJk/L5z3++zrYf/vCH8/vf/z7Jzv3cueiii3LaaafljDPOyJAhQ/KHP/xhi2OVZJd4vlCMBx54IGeccUZOP/30DBw4MF/5yleKLqnZ8Rqg4ezM//t3pF399cb28jqF5kbPfm/064bl/+C206+3z67Sr4XUDeCb3/xm9t5770ydOjVTp07Nd7/73ey///5Fl1WILl265NFHH629fP/99+cDH/hAgRVRtPLy8owaNSqXX3551q9fnz/96U+5++67M2bMmKJLaxRbuv8XXHBBli5dmilTpiRJxo4dmzPPPHOX+JsZN25cfvazn2XKlCk577zzcsUVV+zyzxWKsXTp0lxzzTX5j//4j/z0pz/NAw88kPPPP7/ospolrwEokh6yfbxOoTnRsxuGfk2R9Ovts6v065ZFF7AzeOWVV3LUUUelpKQkSbLnnntmzz33LLiqYpx55pm577770q9fv7z44ot588038/73v7/osijYRz/60Tz44IOZOHFinnjiiVx++eXp1KlT0WU1mi3d/xtvvDHnnntuXn/99SxcuDDjxo0ruNrG0b59+9rfV61aVfv/c1d/rtD4Xn311bRs2TLl5eVJkpKSkvzjP/5jsUU1U14DUDQ9ZPt4nUJzoWc3DP2aounX22dX6NfOpG4A55xzTm655ZZ88pOfzHXXXbdLTvXxd0ceeWSeffbZvPbaa7n//vtzxhlnFF0STcTVV1+dH//4x+nevXtOOeWUostpdJu7/4ccckg++clP5vrrr8+NN96YVq1aFVhl47ryyivz4Q9/ON/4xjfqNNFd/blC4zr00ENz+OGH58Mf/nAuueSSfPe7383KlSuLLqtZ8hqApkAP2T5ep9Ac6NkNQ7+mKdCvt8/O3q+F1A3gmGOOyaxZs/L5z38+rVq1ype+9KXcdtttRZdViJKSkpx88smZPn16pk+fnkGDBhVdEk3E448/nnbt2uUvf/lL1q9fX3Q5jW5z93/Dhg351a9+lb333jt/+tOfCqyw8V133XX55S9/mS9/+csZP3587fJd/blC4yotLc2tt96a73//+znqqKPyyCOP5LTTTktlZWXRpTU7XgPQFOgh28frFJoDPbth6Nc0Bfr19tnZ+7WQuoG0a9cuAwYMyFe/+tWMHj06U6dOLbqkwpx55pn51re+lfe///277LQn1LVixYpcf/31ue2223LYYYflW9/6VtElNaot3f/vfOc72X///fNf//VfmThxYpYuXVpgpcU444wzMnv27KxcuXKXf65QnPe///05++yzc+edd6Z9+/b57W9/W3RJzZLXABRJD9k+XqfQ3OjZ751+TZH06+2zK/RrIXUDeOyxx7Jq1aokSU1NTZ555pnsu+++BVdVnP322y9f/vKXc9FFFxVdCk3ENddck09/+tM59NBDc+WVV2batGn5v//7v6LLajSbu/9/+MMfMnny5IwaNSoHHXRQzjnnnIwePbrocne41atXZ/HixbWXZ86cmQ4dOqS8vHyXf67Q+JYsWZK5c+fWXn7llVeyYsWKXbqPvxdeA1AkPWT7eJ1Cc6FnNxz9miLp19tnV+jXvjixAfzpT3/KjTfemJqamiTJ/vvvn1GjRhVcVbHOOuusoktoUj772c+mRYsWtZenTp2aDh06FFhR45kxY0aef/75TJw4MUnSoUOHjBo1KldccUX+9V//NY888khOOOGE2u0/8YlP5Etf+lJB1Ta8zd3/r371qykpKcnll1+ejh07JknOP//8nHXWWfnZz36W0047LcnO+dxZs2ZNLr300qxZsyalpaXp0KFDJk2alAceeGCzz5Wf/OQnSbLTP19ofG+99VZuvvnmvPzyy2nTpk2qq6vzpS99yRcxvQdeA7x3O+P//h1tV3+9sb28TqE50bMbln7dMPwf3Db69fbZVfp1Sc3fk1UAAAAAAGhkpvsAAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBCagAAAAAACiOkBgAAAACgMEJq2AXNnj07J5xwwnZdd9KkSbnyyisbuCIA4N30awBo+vRraBgtiy4AdkX9+/fPq6++mhYtWqRly5bp3bt3rrnmmnTr1q3o0pIkDz/8cG6++ea8+OKLadWqVXr06JHrrrsu++23X4YNG1Z0eQDQKPRrAGj69GvYOTiTGgoyadKkzJ07N48++mg6deqUa6+9drPbVlVVNVpdL7zwQi677LKMHDkyTz75ZH7xi1/k7LPPTosWLRqtBgBoKvRrAGj69Gto/oTUULDWrVtn4MCBWbhwYe2ykSNHZvTo0fm3f/u39OrVK7Nnz84vf/nLnHHGGenTp0/69euXm2++uXb7l156KT169Mj999+fD3/4wznqqKPyH//xH7Xr165dm5EjR+aII47IKaeckv/7v//bbD1/+MMfsu++++aYY45JSUlJ2rVrl5NOOin77LNPkuTmm2/O8OHDkyRjx45N7969a3/+8R//sbauJUuW5OKLL87RRx+d/v3756677mrQcQOAxqRfA0DTp19D8yWkhoKtWbMmM2bMyAc/+ME6y6dNm5Zhw4blqaeeSt++fbPbbrtl3LhxmTNnTr7zne/kBz/4QR5++OE613nyySfz4IMP5nvf+15uueWW2sb87W9/O3/961/z85//PHfccUemTJmy2Xo+8IEP5C9/+Uuuv/76/O///m9Wr1692W1HjRqVuXPnZu7cubnnnnuyxx57ZMCAAamurs7nP//59OjRI7/61a/yve99L9/73vfy61//evsHCgAKpF8DQNOnX0PzJaSGgnzhC19IRUVFKioq8thjj+X888+vs37AgAHp27dvSktL07p16xx11FHp0aNHSktLc+ihh+bjH/94fvvb39a5zhe/+MW0adMmhx56aA499ND88Y9/TJI88MADGTZsWMrLy9OtW7cMHTp0s3Xtt99++f73v58lS5bkS1/6Uo4++uiMHDlyi810xYoV+cIXvpCrr746//iP/5j/+7//y4oVK/LFL34xZWVl2W+//fLpT386M2bMeA8jBgCNT78GgKZPv4bmzxcnQkFuueWWfOhDH0pVVVV+8YtfZOjQoZk+fXo6d+6cJBt9ycPvfve7TJw4MQsWLMiGDRuyfv36DBw4sM42e+21V+3vu+22W958880kydKlS+vs7+8fLdqcXr165aabbkqSPP300/nyl7+cSZMm5Stf+cpG227YsCGXXHJJBg0alI9//ONJkpdffjlLly5NRUVF7XZVVVV1LgNAc6BfA0DTp19D8+dMaihYixYtcuKJJ6a0tDRPPvnkZrf7yle+kgEDBuSRRx7Jk08+mcGDB6empqZet9G5c+csXry49vI7f9+aww8/PCeeeGIWLFiwyfXXXntt2rVrly996Uu1y7p165Z99903c+bMqf2ZO3du/vM//7PetwsATYl+DQBNn34NzZeQGgpWU1OThx9+OK+//noOOuigzW63evXqdOjQIa1bt87TTz+dadOm1fs2Tj755Nx222157bXX8sorr+T73//+ZredM2dOJk+enOXLlydJFi5cmJkzZ240p1eS3HvvvXniiScyceLElJa+/e/k8MMPz+67757bbrsta9euTVVVVZ599tk8/fTT9a4ZAJoS/RoAmj79Gpov031AQYYNG5YWLVokSbp3754bb7wxhxxyyGa3Hz16dMaNG5exY8fmyCOPzMknn5zXX3+9Xrf1xS9+MaNHj86AAQPSpUuXfOITn9jstwHvsccemTlzZr75zW9mzZo12XPPPXPyySfnggsu2Gjb6dOn58UXX8zxxx9fu+xzn/tchg0blkmTJmXcuHEZMGBA1q9fn/e973113g0GgOZAvwaApk+/huavpKa+n2cAAAAAAIAGZroPAAAAAAAKI6QGAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBCagAAAAAACiOkBgAAAACgMEJqAAAAAAAKI6QGAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBCagAAAAAACiOkBgAAAACgMEJqAAAAAAAKI6QGAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBCagAAAAAACiOkBgAAAACgMEJqAAAAAAAKI6QGAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBCagAAAAAACiOkBgAAAACgMEJqAAAAAAAKI6QGAAAAAKAwQmoAAAAAAAojpAYAAAAAoDBCagAAAAAACiOkBgAAAACgMEJqAADYBiNHjsw3vvGN97SP2bNn54QTTmigirbs1Vdfzdlnn53evXvnxhtvTE1NTS6//PIcccQR+eQnP9koNQAAja8hXrNAY2lZdAGwM5k6dWruvPPOPPfcc9l9991z6KGHZtiwYamoqCi6tO2yYMGC3HDDDZk/f36qq6vzD//wD7n00kvTr1+/rV535MiR2XvvvfPlL3+5ESoFgIbTv3//vPrqq2nRokVatmyZ3r1755prrkm3bt0Kq2np0qX55je/mV/96ldZvXp19t5775xyyim54IIL0rZt2y1e94c//GH23HPPPPXUUykpKcmcOXPy2GOP5ZFHHtnqdQGgCEOHDs0f//jHPPbYYykrKyu6nLz00ksZMGBATjjhhPznf/5n7fLhw4dn//33z8UXX1xgdbBzcCY1NJA777wz119/fYYNG5bHHnsss2bNypAhQ/KLX/yi6NK227Bhw/KhD30ojz76aH7zm9/kyiuvzO67794g+37rrbcaZD8AsCNMmjQpc+fOzaOPPppOnTrl2muvLayWysrKDB48OOvWrcu9996buXPn5s4778zrr7+ev/71r1u9/qJFi3LQQQelpKQkSfLyyy+ne/fuAmoAmqSXXnopc+bMSUlJyVaPp6uqqupc3tHHmU8//XSeeuqpHXobDeXdYwNNnZAaGsAbb7yRb33rWxk1alROPPHEtG3bNq1atUr//v1z2WWXJflbMzvrrLNSUVGR4447LmPHjs369etr99GjR4/cfffdOfHEE9O7d+9885vfzF//+tcMHjw4ffr0yaWXXlq7/d8/Ijxp0qQcddRR6d+/f372s5/V7mvo0KH50Y9+VHv5vvvuy7/8y78kSWpqanL99dfnmGOOSZ8+fXLqqafm2Wef3eg+rVixIi+99FI+/elPp6ysLGVlZenbt2/tWeHv3Oc778MLL7yQH/7wh5k6dWruuOOO9O7dO8OGDUvytzPTbrvttpx66qnp1atX3nrrrdx222356Ec/mt69e+eUU07Jz3/+84Z4SACgQbRu3ToDBw7MwoULN7n+tddey+c+97kcffTROeKII/K5z30ur7zySu36ysrKXH755TnuuONyxBFH5KKLLtrkfu66666ccsopda77d3feeWd23333TJgwIfvuu2+SpFu3brnqqqty6KGHJkmeeuqp/PM//3P69u2bf/7nf649gB45cmSmTJlS25PvvffeXHXVVZk3b1569+6db33rW0mSWbNm5fTTT09FRUUGDx6cP/7xj9s/aADwHkyZMiUf/OAHc+aZZ2bKlCl11o0cOTKjR4/Ov/3bv6VXr16ZPXv2Jo8zf/GLX+TjH/94KioqMnTo0No+/pOf/KT2+DRJTjzxxFxyySW1l/v165c//OEPm63t/PPP3+L0GZvrp/W93WuuuSY33nhjnX0OGzYs3/3ud5MkCxcuzNChQ1NRUZGPf/zjdUL8TY3NO61atSpDhw7N1772tdTU1Gz2PkBRhNTQAObOnZt169blYx/72Ga3KS0tzeWXX57//d//zb333pvHH38899xzT51tHn300dx3332ZPHlybr/99lx99dWZMGFCHnnkkSxYsCDTp0+v3fbVV1/NypUr8+tf/zo33nhjRo0alb/85S9brfXRRx/NnDlz8tBDD+XJJ5/MN7/5zZSXl2+03Z577pn9998/I0aMyMMPP5xXX3213uNx1lln5dRTT83555+fuXPnZtKkSbXrpk+fnttuuy1z5sxJy5Yts99+++Xuu+/Ok08+mS9+8YsZMWJEli5dWu/bAoAdac2aNZkxY0Y++MEPbnJ9dXV1PvGJT2TWrFmZNWtWWrdunbFjx9au/+pXv5o1a9Zk+vTp+c1vfpNzzz13o318+9vfzv3335///u//TteuXTda//jjj+djH/tYSks3/dK9srIyn/vc5zJ06NDMnj07n/3sZ/O5z30uK1euzI033linJw8ePDjXXHNNevXqlblz5+aSSy7JM888kyuuuCJjx47N7Nmzc9ZZZ+Wiiy6q82Y6ADSWn/70pzn11FNz6qmn5tFHH93oWHTatGkZNmxYnnrqqfTt2zdJ3ePMF198MV/5yldyxRVX5PHHH88JJ5yQYcOGZf369TnyyCMzZ86cVFdXZ8mSJdmwYUPmzZuXJHnxxRfz5ptvpkePHputbciQIXn++efzm9/8ZqN1W+qn9b3dM888M9OmTUt1dXWSv5089vjjj2fQoEHZsGFDhg0blmOPPTa/+c1vctVVV2X48OF1coBNjU2SrFy5Mueee2769OmTq666qvbTVdCUCKmhAVRWVmbPPfdMy5abn+b9sMMOS69evdKyZcvsu+++Oeuss/LEE0/U2eaCCy5Iu3btcsghh+T9739/jj322Oy3335p3759TjjhhDzzzDN1tr/00ktTVlaWI488Mv369csDDzyw1VpbtmyZ1atX5y9/+Utqampy0EEHpUuXLhttV1JSkrvuuivdu3fPjTfemOOOOy5nn312nn/++foNymYMHTo03bp1S5s2bZIkJ598cvbee++UlpbmlFNOyf7775+nn376Pd0GALxXX/jCF1JRUZGKioo89thjOf/88ze53Z577pmTTjopu+22W9q1a5fPf/7ztf196dKl+dWvfpVrrrkmHTp0SKtWrXLkkUfWXrempiY33HBDHnvssdx1113p2LHjJm+jsrIynTt33mytv/zlL7P//vvnjDPOSMuWLTNo0KAceOCBmTVrVr3u6w9/+MOcddZZ+eAHP5gWLVrkzDPPTKtWrWoPngGgscyZMyeLFi3KySefnMMOOyz77bdfpk2bVmebAQMGpG/fviktLU3r1q2T1D3OnDFjRvr165djjz02rVq1yvnnn5+1a9dm7ty52W+//bL77rvnD3/4Q+bMmZPjjjsuXbp0ycKFC/Pb3/62dr+b06ZNmwwbNizf/OY3N1q3pX5a39s9/PDD0759+zz++ONJkhkzZuTII4/MXnvtld/97nd58803c+GFF6asrCzHHHNMPvKRj9Q5mW1TY7N06dIMHTo0AwcO9J1RNGm+OBEaQHl5eVauXJm33nprs0H1c889lxtvvDHz58/PmjVrUlVVlQ984AN1ttlrr71qf2/duvVGl9/5DvIee+xRZy7JffbZp15nIB9zzDE5++yzM3bs2Lz88ss58cQTc9lll6Vdu3Ybbdu1a9eMGjUqSbJ48eJcffXVueyyy/LDH/5wq7ezOe/+0qkpU6bkzjvvzMsvv5wkefPNN7Ny5crt3j8ANIRbbrklH/rQh1JVVZVf/OIXGTp0aKZPn75RWLxmzZrccMMN+fWvf53XXnstSbJ69epUVVXllVdeSYcOHdKhQ4dN3sYbb7yRyZMn5xvf+Ebat2+/2VrKy8uzbNmyza5funRp9tlnnzrL9tlnnyxZsqRe93XRokWZMmVK/vu//7t22YYNG3yyCYBGN2XKlBx77LG1b9wOGjQo999/f51PIm3qi4zfuezdfbG0tDTdunWr7YtHHHFEfvvb3+aFF17IEUcckfbt2+eJJ57IvHnz6ryZvDmf+tSncscdd2TmzJl1lm+tn9b3ds8888z87Gc/y7HHHpuf/exnOeecc2rvV9euXeuE6O/u95sam79/UfLgwYO3et+gSM6khgbQu3fvlJWV5eGHH97sNmPGjMmBBx6Yhx56KE899VS+/OUvv6d5oF5//fW8+eabtZcXL15ce0b0brvtljVr1tSue/fHo84555zcd999mTFjRp5//vncfvvtW729bt265eyzz66dv3q33XbL2rVra9e/++B5cx8feufyl19+OVdddVWuvvrqzJ49O3PmzMkhhxyy1VoAoLG0aNEiJ554YkpLS/Pkk09utP6//uu/8txzz2Xy5Ml56qmncvfddyf521nSXbt2zWuvvZbXX399k/veY489MmnSpFx++eWb3PffHXPMMfn5z39e+9Hfd+vSpUsWLVpUZ9nixYuz99571+s+duvWLcOGDcucOXNqf373u99l0KBB9bo+ADSEtWvX5oEHHsgTTzyRY489Nscee2y+973v5Y9//ONWvyvhnceZ7+6LNTU1dfrikUcemdmzZ+fJJ5/MkUcemSOPPDJPPPFEfvvb3+aII47Yap1lZWX54he/mJtuuqnOMf3W+ml9b/e0007LL37xi/zxj3/MwoUL89GPfrT2fr3yyit1Xg/Up99/6lOfyvHHH58LL7ywToYATY2QGhpA+/btc8kll2Ts2LF5+OGHs2bNmmzYsCGPPPJIxo8fn+RvZ1Xtvvvu2X333bNw4cL84Ac/eM+3e/PNN2f9+vWZM2dOfvnLX2bgwIFJkp49e+bnP/951qxZkxdeeCE//vGPa6/z9NNP53e/+102bNiQ3XbbLWVlZZv8ONNrr72Wb33rW3nhhRdSXV2dFStW5Cc/+Ul69eqVJDn00EOzYMGC/OEPf8i6dety880317l+p06d8tJLL22x/jVr1qSkpKT2XfKf/OQnWbBgwXsZEgBoUDU1NXn44Yfz+uuv56CDDtpo/erVq9O6devsscceqayszLe//e3adV26dMkJJ5yQa665Jq+99lo2bNiw0VRfRx11VCZOnJiLL754s9Ndffazn83q1atz2WWX1X7yaMmSJbnhhhvyxz/+Mf369cvzzz+fqVOn5q233sqMGTPy5z//OR/+8IfrdR8/9alP5d57783vfve71NTU5M0338wvf/nLrFq1qp6jBADv3cMPP5wWLVpk+vTpmTJlSqZMmZIZM2akoqJioy9Q3JKTTz45jzzySB5//PFs2LAh//Vf/5WysrL07t07yd/OaJ49e3bWrl2brl27pqKiIr/+9a9TWVmZf/zHf6zXbZx++ulZt25dHn300dplW+un9b3drl275p/+6Z8yYsSInHjiibVTZR5++OFp06ZNbr/99mzYsCGzZ8/OzJkzc8opp2y13lGjRuV973tfhg0bVudkM2hKhNTQQM4777yMHDkyt956a4455ph8+MMfzt133137rudll12WadOmpU+fPrn66qvr1Ui2ZK+99soee+yR448/PsOHD8+YMWNqD54/85nPpFWrVvnQhz6Uyy67LKeeemrt9VavXp2rrroqRx55ZD7ykY+kvLx8k/NstmrVKi+//HI++9nPpm/fvjn11FNTVlZW+03D73vf+/KFL3wh5557bk488cQ6X8qQJJ/85Cfz5z//ORUVFbnooos2eR8OPvjgnHfeeRk8eHA+9KEP5dlnn02fPn3e07gAQEMYNmxYevfunT59+uSb3/xmbrzxxk1+2uczn/lM1q1bl6OPPjpnnXVWjj/++Drrx48fn5YtW+bkk0/Ohz70oXzve9/baB/HHntsrr/++gwbNiy///3vN1pfXl6eH/zgB2nZsmU+/elPp3fv3vnMZz6T9u3bZ//998+ee+6ZSZMm5c4778xRRx2V22+/PZMmTdrsHNfv9k//9E+59tprM3bs2BxxxBE58cQTc99999VzpACgYdx///35xCc+kX322SedO3eu/Tn77LNr34itjwMPPDATJkzItddem6OPPjqzZs3KpEmTUlZWluRvx7K77757KioqkiTt2rXLvvvumz59+qRFixb1uo0WLVrkkksuSWVlZe2yrfXTbbndM844I88++2xOP/302mVlZWWZNGlSfvWrX+Xoo4/ONddck/Hjx2/yTfR3KykpybXXXpuuXbvmoosuyrp16+p1P6ExldS8l/kGgELMnj07I0aMyK9+9auiSwEAAAAa0BNPPJERI0Zk1qxZm51KE3Y2zqQGAAAAgCZgw4YNueuuu/LJT35SQM0uRUgNAAAAAAVbuHBhjjjiiCxbtiznnntu0eVAozLdBwAAAAAAhXEmNQAAAAAAhRFSAwAAAABQmJZFF9AQVq5cnepqs5YA0LBKS0uy5567F13GTkO/BmBH0K8bln4NwI6ypZ69U4TU1dU1migANHH6NQA0ffo1AEUw3QcAAAAAAIURUgMAAAAAUBghNQAAAAAAhRFSAwAAAABQGCE1AAAAAACFEVIDAAAAAFAYITUAAAAAAIURUgMAAAAAUBghNQAAAAAAhRFSAwAAAABQGCE1AAAAAACFEVIDAAAAAFAYITUAAAAAAIURUgMAAAAAUBghNQAAAAAAhRFSAwAAAABQmJZF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Bw7M//zP/xSsjOSv1+eZZ54pXcZ6b+nSpdlnn33y5JNPNmybNm1avva1r+WWW27JySefXLA6aBl6dtuiX7d9enbboGfTWtb2WHvjjTfyjW98IyNGjMhBBx2UkSNH5qGHHipY7dq1t57fnntie+gV7eV5dG11Tps2Lccff3yjYz/3uc/lt7/9bZKy1+ELX/hCRo4cmQMOOCA77rhjRo4cmZEjR+bII4/M0KFD89ZbbzUce8wxx+S6665LkowbN67o43vo0KEZPnx4Q72TJ0/OueeemwsuuKDhmBdeeCF77713FixYkCQZOHBgli1b1mbqvfzyy1NTU5MkmTFjRg499NC89957SZIVK1bkoIMOyt13350kOfLII3Pvvfe2WH2dW+zMNMmiRYty/vnnZ/r06amqqkp9fX1+97vflS6LJJtuuml+9atfZciQIUmS6dOnZ6eddipcFbQdffr0yfjx43PWWWdl+vTpee6553L99dfn1ltvzX333Ve6PGh2enbbpF/Dh9OzaS1re6x997vfzWabbZZvf/vbqaioyOuvv57ly5eXLnm12mvP1xNbTnt5Hl1bnb17987o0aNz6623ZtSoUZk4cWIOOeSQNvEYufnmm5MkL730Uv7t3/4tt912W8O+yZMnZ/Lkybnoooty4403ZsWKFTnyyCNLlbqK733ve/n4xz/ecPvtt9/OwQcfnH333TfV1dUZN25cTj311FRVVRWs8m8+WO/ll1/e8P+DDjood955Z6ZNm5aTTjopl19+eQYNGpRhw4a1Sm1C6sJee+21dO7cOX369EmSVFRUZMcddyxbFEmSQw45JLfcckuGDBmSF198MX/5y18a/SADyec///nceeedmTJlSh599NGcddZZ6devX+myoEXo2W2Tfg1No2fTWtb0WHvllVcyePDgVFRUJEk23njjbLzxxoWrXb322vP1xJbVXp5H11bnxRdfnC9/+ct58803M3/+/IZ30LZl3/jGNzJy5Mhcf/31+f73v58bbrih4XmkLerRo0cmTZqUb33rWxk1alR69uyZww47rHRZTXb++edn1KhR6d+/f2bOnNnoFwYtzXIfhe2www7ZZZdd8rnPfS4nn3xy/vu//zuvv/566bJIsscee+SZZ57JG2+8kenTp2fUqFGlS4I26dxzz81Pf/rTDBgwIAcccEDpcqDF6Nltk34NTadn01pW91g76qijMnXq1Bx22GG58MIL2/RSH+215+uJLa+9PI+uqc7tt98+hx12WCZPnpyLL744Xbp0KVhl03Tv3j0TJkzIxIkTc9xxx2WrrbYqXVIjJ598csPyGQ888ECS5DOf+Ux23XXXXHPNNZk0aVLhChtbXb1/b5NNNskpp5yS8ePH55xzzkmvXr1arTYhdWGVlZW58sor8+Mf/ziDBw/Offfdl4MPPjhLly4tXdp6r6KiIvvvv39mzZqVWbNmZcSIEaVLgjbpoYceSo8ePfLHP/4xK1asKF0OtBg9u23Sr6Hp9Gxay+oea3vttVfuvffeHH/88enSpUtOPfXUXH311YUrXb322vP1xJbXXp5H11TnypUrc//992ezzTbLH/7wh4IVrpu77rorm2++eZtcdud73/tebrvtttx222357Gc/myR566238vjjj6dHjx554YUXClfY2Orq/aC77767yHgLqduIj3/84zniiCNy7bXXpmfPnnnkkUdKl0T++nGp99fraasfRYOSlixZksmTJ+fqq6/OzjvvnO9973ulS4IWp2e3Pfo1fDg9m9aytsdajx49MmzYsJx55pmZMGFCZsyYUbDSD9cee76e2HLay/Po2uq86qqrsvXWW+e//uu/MmXKlCxatKhgpU3z8MMP54EHHsj06dMzd+7c3H///aVL+lCTJ0/O/vvvn0suuSTnnHNOm11/f3WmT5+epUuX5uabb87//b//t1V/mSGkLmzhwoWZO3duw+1XXnklS5YsyZZbblmwKt631VZb5etf/3pOOOGE0qVAm3T++efni1/8YnbYYYecffbZmTlzZp566qnSZUGL0LPbLv0aPpyeTWtZ02PtwQcfzNtvv50kqa+vz7x589psD23PPV9PbDnt5Xl0TXX+7ne/y0033ZTx48dnu+22y1FHHZUJEyaULnetli1blrPPPjsXXHBB+vbtm8mTJ+e8885reC5pi375y1/mN7/5TU4++eTstdde2XPPPfPtb3+7dFlNsnDhwnz729/O5MmTs+mmm2bcuHE566yz8t5777XK9/eHEwt77733cvnll+fPf/5zunfvnrq6upx66qnt4o8yrC9Gjx5dugRW4+ijj06nTp0abs+YMSO9e/cuWNH654477sjzzz+fKVOmJEl69+6d8ePH51vf+la+9KUv5b777ss+++zTcPyhhx6aU089tVC18I/Ts9s2/brt0rPL07NpLWt7rB100EG5+OKLU19fnyTZeuutM378+JLlrlF77/ntsSe29V7RXp5H11TnmWeemYqKipx11lnp27dvkuSYY47J6NGjc/vtt+fggw9O0vauw6WXXprPfvazGTx4cJJk9913z7777puLL744F1xwQZLksssua7R00KRJkzJkyJAi9b755ps577zz8t3vfjfdunVLkpx55pkZOXJkhg8fnurq6iTJ8OHDG/744wYbbJDZs2cXqfeDzj333Hz5y1/OtttumyQ58MADc+edd+YHP/hBjj/++CTJuHHjGu5bklx99dXZYYcdmuX7V9S/3yEAAAAAAKCVWe4DAAAAAIBihNQAAAAAABQjpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgN66GHH344++yzz0f62mnTpuXss89u5ooAgA/SrwGg7dOvoXl0Ll0ArI+GDh2a1157LZ06dUrnzp2z22675fzzz09VVVXp0pIkd911Vy6//PK8+OKL6dKlSwYOHJgLL7wwW221VcaOHVu6PABoFfo1ALR9+jV0DN5JDYVMmzYtc+fOza9+9av069cvkyZNWuOxtbW1rVbXCy+8kG9+85sZN25cHnvssdx999054ogj0qlTp1arAQDaCv0aANo+/RraPyE1FNatW7cMHz488+fPb9g2bty4TJgwIf/xH/+RXXfdNQ8//HB++ctfZtSoUdl9990zZMiQXH755Q3Hv/TSSxk4cGCmT5+ez33ucxk8eHC+//3vN+x/5513Mm7cuHzqU5/KAQcckKeeemqN9fzud7/Llltumb322isVFRXp0aNH9ttvv2yxxRZJkssvvzynn356kmTixInZbbfdGv7tuOOODXUtXLgwX/va17Lnnntm6NChue6665p13ACgNenXAND26dfQfgmpobDly5fnjjvuyCc+8YlG22fOnJmxY8fm8ccfzyc/+clssMEGqampyZw5c3LVVVflhhtuyF133dXoax577LHceeed+dGPfpSpU6c2NOYrrrgif/rTn/KLX/wi11xzTW699dY11rPTTjvlj3/8YyZPnpxf//rXWbZs2RqPHT9+fObOnZu5c+fmJz/5SXr16pVhw4alrq4uxx9/fAYOHJj7778/P/rRj/KjH/0oDzzwwEcfKAAoSL8GgLZPv4b2S0gNhZx44omprq5OdXV1HnzwwRxzzDGN9g8bNiyf/OQnU1lZmW7dumXw4MEZOHBgKisrs8MOO+TAAw/MI4880uhrTjrppHTv3j077LBDdthhh/z+979PkvzsZz/L2LFj06dPn1RVVeXII49cY11bbbVVfvzjH2fhwoU59dRTs+eee2bcuHFrbaZLlizJiSeemHPPPTc77rhjnnrqqSxZsiQnnXRSunbtmq222ipf/OIXc8cdd/wDIwYArU+/BoC2T7+G9s8fToRCpk6dmk9/+tOpra3N3XffnSOPPDKzZs1K//79k2SVP/Lwm9/8JlOmTMmzzz6blStXZsWKFRk+fHijYzbZZJOG/2+wwQb5y1/+kiRZtGhRo/O9/9GiNdl1111z2WWXJUmefPLJfP3rX8+0adNy2mmnrXLsypUrc/LJJ2fEiBE58MADkyR//vOfs2jRolRXVzccV1tb2+g2ALQH+jUAtH36NbR/3kkNhXXq1Cn77rtvKisr89hjj63xuNNOOy3Dhg3Lfffdl8ceeyyHH3546uvrm/Q9+vfvnwULFjTc/vv/f5hddtkl++67b5599tnV7p80aVJ69OiRU089tWFbVVVVttxyy8yZM6fh39y5c/ODH/ygyd8XANoS/RoA2j79GtovITUUVl9fn7vuuitvvvlmtttuuzUet2zZsvTu3TvdunXLk08+mZkzZzb5e+y///65+uqr88Ybb+SVV17Jj3/84zUeO2fOnNx0001ZvHhxkmT+/Pm55557VlnTK0luvPHGPProo5kyZUoqK//2dLLLLrtko402ytVXX5133nkntbW1eeaZZ/Lkk082uWYAaEv0awBo+/RraL8s9wGFjB07Np06dUqSDBgwIBdffHG23377NR4/YcKE1NTUZOLEidljjz2y//77580332zS9zrppJMyYcKEDBs2LJtuumkOPfTQNf414F69euWee+7Jd7/73Sxfvjwbb7xx9t9//xx77LGrHDtr1qy8+OKL+exnP9uw7atf/WrGjh2badOmpaamJsOGDcuKFSvyT//0T41+GwwA7YF+DQBtn34N7V9FfVM/zwAAAAAAAM3Mch8AAAAAABQjpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgNAAAAAEAxQmoAAAAAAIoRUgMAAAAAUIyQGgAAAACAYoTUAAAAAAAUI6QGAAAAAKAYITUAAAAAAMUIqQEAAAAAKEZIDQAAAABAMUJqAAAAAACKEVIDAAAAAFCMkBoAAAAAgGKE1AAAAAAAFCOkBgAAAACgGCE1AAAAAADFCKkBAAAAAChGSA0AAAAAQDFCagAAAAAAihFSAwAAAABQjJAaAAAAAIBihNQAAAAAABQjpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgNAAAAAEAxQmpoI8aNG5f//M///IfO8fDDD2efffZppooAgPbgpZdeysCBA/Pee++tdv+0adNy9tlnN+lYAAAooXPpAmB9MXTo0Lz22mvp1KlTOnfunN122y3nn39+qqqqitRzyy235Oyzz0737t2TJH379s0xxxyTMWPGFKkHANqjGTNm5Nprr81zzz2XjTbaKDvssEPGjh2b6urqj3zOW265JTfffHNuuOGGZqlx7NixzXIeAOiI/n6u/r4777wzm222WcGqYP0jpIZWNG3atHz605/Ou+++m/POOy+TJk3KlVdeWayeXXfdtWECPG/evBxxxBHZdddds+OOO67Teerr61NfX5/Kyr99OOO9995L586eYgDouK699tpcffXVOf/887P33nunS5cueeCBB3L33Xf/QyE1ANC63p+rfxTmvtA8LPcBBXTr1i3Dhw/P/PnzV7v/jTfeyFe/+tXsueee+dSnPpWvfvWreeWVVxr2L126NGeddVb23nvvfOpTn8oJJ5yw2vNcd911OeCAAxp97ZrsuOOO2W677RrV9MQTT+Twww9PdXV1Dj744Dz88MMN+4488sj853/+Zw4//PB84hOfyIsvvpiBAwfm+uuvz7777pt99903559/fi6++OJG32fs2LH57//+7w+tBwDasrfeeivf+973Mn78+Oy7777ZcMMN06VLlwwdOjTf/OY38+STT2b06NGprq7O3nvvnYkTJ2bFihUNXz9w4MDccMMN2XfffVNdXZ3zzz8/9fX1mT9/fiZMmJAnnngiu+22W0PY/ctf/jKjRo3K7rvvniFDhuTyyy9fpab//d//zd577529994711xzTcP2yy+/PKeffnrLDwoAdBArVqzIhRde2NBXL7zwwoY+/v4ym1dffXU+85nP5KyzzkptbW2mTZuWz3/+89ltt91y6KGHZsGCBUmSCy64IEOGDMnuu++eQw89NHPmzCl516DNElJDAcuXL88dd9yRT3ziE6vdX1dXl0MPPTT33ntv7r333nTr1i0TJ05s2H/mmWdm+fLlmTVrVv7f//t/+fKXv7zKOa644opMnz49//M//5PNN9/8Q2t68skn8/zzz2fnnXdOkixcuDBf/epXc/zxx+eRRx7JN7/5zZx88slZsmRJw9fcdtttmTRpUh5//PFsscUWSZK77rorN910U+64444ccsghmTlzZurq6pIkS5YsyUMPPZQRI0Y0eawAoC2aO3du3n333fzrv/7ravdXVlbmrLPOyq9//evceOONeeihh/KTn/yk0TG//OUv89Of/jS33357fvazn+WBBx7Idtttl/PPPz+77rpr5s6d2zCR3WCDDVJTU5M5c+bkqquuyg033JC77rqr0fkefvjh/PznP88111yTH/zgB/l//+//tcydB4AO7vvf/35+85vf5Lbbbsvtt9+ep556qtGnoF977bW88cYbuffeezNp0qRce+21mTVrVq6++uo8/vjjmTx5csPSmv/8z/+cW2+9NY888khGjBiRU045Je+++26puwZtlpAaWtGJJ56Y6urqVFdX58EHH8wxxxyz2uM23njj7Lffftlggw3So0ePHH/88Xn00UeTJIsWLcr999+f888/P717906XLl2yxx57NHxtfX19Lrroojz44IO57rrr0rdv3zXW85vf/CbV1dXZbbfd8oUvfCEjR47MNttsk+SvAfQ+++yTIUOGpLKyMp/5zGey884757777mv4+kMOOSTbb799OnfunC5duiRJjjvuuPTp0yfdu3fPLrvskp49e+ahhx5Kktxxxx3ZY489sskmm/xD4wgApS1dujQbb7zxGj/eu/POO2fXXXdN586ds+WWW2b06NENvfx9//Ef/5FevXpliy22yODBg/P73/9+jd9v8ODBGThwYCorK7PDDjvkwAMPzCOPPNLomBNPPDEbbrhhBg4cmEMPPTQzZ878x+8oAKwH/n6ufsIJJ2TGjBk58cQT069fv/Tt2zcnnnhibr/99objKysrc/LJJ6dr167p3r17br755pxyyinZdtttU1FRkR122CEbb7xxkmTkyJENrxm+8pWvZMWKFXnuuedK3VVosyyaA61o6tSp+fSnP53a2trcfffdOfLIIzNr1qz079+/0XHLly/PRRddlAceeCBvvPFGkmTZsmWpra3NK6+8kt69e6d3796r/R5vvfVWbrrppvznf/5nevbsudZ6PvGJTzSsSf3aa6/lG9/4Rr7zne/ktNNOy8svv5w777wz9957b8Px7733XgYPHtxwe3V/9PGD2w455JDcfvvt+cxnPpPbb789Rx111FprAoD2oE+fPnn99dfXuA7lc889l4svvjhPP/10li9fntra2uy0006Njvn7/r/BBhtk2bJla/x+v/nNbzJlypQ8++yzWblyZVasWJHhw4c3Oubve/CAAQPyzDPPfNS7BwDrlffn6u/bZZddGj4tnCRbbLFFFi1a1HB74403Trdu3Rpuv/LKK/k//+f/rPbc11xzTX76059m0aJFqaioyNtvv53XX3+9Be4FtG/eSQ0FdOrUKfvuu28qKyvz2GOPrbL/v/7rv/Lcc8/lpptuyuOPP57rr78+yV/fJb355pvnjTfeyJtvvrnac/fq1SvTpk3LWWedtdpzr8kmm2yS/fbbryGUrqqqysiRIzNnzpyGf0888USOO+64hq+pqKhY5Twf3HbwwQfn7rvvzu9///vMnz8/n//855tcEwC0Vbvttlu6du26ypIb7zvvvPOy7bbbZvbs2Xn88cfz9a9/PfX19U069+r662mnnZZhw4blvvvuy2OPPZbDDz98lfO9v/Zlkrz88svZdNNN1+EeAQDv23TTTfPyyy833F6wYEGjvvrBXr355pvnT3/60yrnmTNnTn74wx/mu9/9bh599NHMmTMnPXv2bPJrAlifCKmhgPr6+tx111158803s912262yf9myZenWrVt69eqVpUuX5oorrmjYt+mmm2afffbJ+eefnzfeeCMrV65c5ePDgwcPzpQpU/K1r30tTz75ZJNqev311/OLX/wiH/vYx5L8NVy+995788ADD6S2tjbvvvtuHn744Sb9Eca/t/nmm+ef//mfc8YZZ2TfffdtWJcLANqznj175uSTT87EiRNz1113Zfny5Vm5cmXuu+++XHLJJVm2bFk22mijbLTRRpk/f37DJ5eaol+/flm4cGGjP7S4bNmy9O7dO926dcuTTz652qU8rrzyyixfvjzPPvtsbrnllhxwwAHNcl8BYH1z4IEH5vvf/36WLFmSJUuWZOrUqTnooIPWePwXvvCFXHbZZXn++edTX1+f3//+93n99dezbNmydOrUKX379s17772XK664Im+//XYr3hNoPyz3Aa1o7Nix6dSpU5K/fgz34osvzvbbb7/Kcf/+7/+e008/PXvuuWc23XTTHH300Y3eqXXJJZfkoosuyv7775+VK1dm8ODB+dSnPtXoHJ/5zGcyefLkjB07Nj/4wQ9W+YhxkjzxxBPZbbfdkiTdu3fPXnvtlbPPPjvJX99JfeWVV+bSSy/NaaedlsrKyuyyyy4577zz1vl+jxo1KmeeeWbDuQGgI/jKV76STTbZJFdeeWVOP/30bLTRRtlpp50yduzY/Mu//EvOPffcXHPNNRk0aFAOOOCA/PrXv27Seffcc8987GMfy957752Kioo8/PDDmTBhQmpqajJx4sTsscce2X///Vf5VNUee+yRf/3Xf019fX2+8pWvZO+9926Juw0AHd4JJ5yQZcuW5eCDD06SDB8+PCeccMIajz/66KOzYsWKfOUrX8nrr7+ebbfdNlOnTs3ee++dz372s9lvv/2y4YYb5t///d9Xu2wmkFTU+4wB0MIeffTRnHHGGbn33ntX+xFmAAAAANZflvsAWtTKlStz3XXX5bDDDhNQAwAAALAKITXQYubPn59PfepTefXVV/PlL3+5dDkAAAAAtEGW+wAAAAAAoBjvpAYAAAAAoBghNQAAAAAAxQipAQAAAAAopnPpAprD668vS12dpbUBaF6VlRXZeOONSpfRYejXALQE/bp56dcAtJS19ewOEVLX1dVrogDQxunXAND26dcAlGC5DwAAAAAAihFSAwAAAABQjJAaAAAAAIBihNQAAAAAABQjpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgNAAAAAEAxQmoAAAAAAIoRUgMAAAAAUIyQGgAAAACAYoTUAAAAAAAUI6QGAAAAAKAYITUAAAAAAMUIqQEAAAAAKEZIDQAAAABAMZ1LFwBA+7Bxjy7pvEH30mW0iPeWv5PX315ZugwAOpCevbuke9eO2Tdb0zsr3slbb+jRAHQ85tiNCakBaJLOG3TPo9V7lC6jRXxqziOJkBqAZtS9a/f8y+WfKV1Gu3fv1x7MW9GjAeh4zLEbs9wHAAAAAADFCKkBAAAAAChGSA0AAAAAQDFCagAAAAAAihFSAwAAAABQjJAaAAAAAIBihNQAAAAAABQjpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgNAAAAAEAxQmoAAAAAAIoRUgMAAAAAUIyQGgAAAACAYjqXLgCgrerbp3s6delSuowWUbtyZZYsfad0GQAAAKwnOuoc2/y6eQipAdagU5cuWXrDT0qX0SL6/P/GJNFEAdq7Pht3T5fOHW+yV8LK91Zm6et6IwC0lI46xza/bh5CagAAaKe6dO6Smb+9snQZHcKInU6ICSYAQBnWpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgNAAAAAEAxQmoAAAAAAIoRUgMAAAAAUIyQGgAAAACAYoTUAAAAAAAUI6QGAAAAAKAYITUAAAAAAMUIqQEAAAAAKEZIDQAAAABAMUJqAGAVV1xxRQYOHJhnnnkmSfLEE0/k4IMPzn777ZevfOUrWbx4ceEKAQAA6CiE1ABAI7/97W/zxBNPZMCAAUmSurq6nHHGGRk/fnxmz56d6urqTJkypXCVAAAAdBSdSxcAtB09e3dJ967dS5fR7N5Z8U7eemNl6TKgXVixYkUmTpyYb3/72znqqKOSJE8//XS6deuW6urqJMnhhx+eYcOG5aKLLipZKgAAtFkddX6dmGPTMoTUQIPuXbvnXy7/TOkymt29X3swb0UDhaa47LLLcvDBB2fLLbds2LZgwYJsscUWDbf79u2burq6LF26NH369Gnyufv169GcpdLK3l1Zm25dOpUuo90zjm1b//49S5fAarguQHvUUefXiTk2LUNIDQAkSebOnZunn346p59+eoucf/Hit1NXV98i56bl9e/fM58847rSZbR7j116VF599a1mO5/wrnm5Nm3Th12XysoKvwgFgHZOSA0AJEkeffTRzJ8/P8OGDUuSvPLKKznmmGNy5JFH5uWXX244bsmSJamsrFynd1EDAADAmvjDiQBAkuS4447Lr371q9xzzz255557svnmm+eaa67Jsccem3feeSdz5sxJktx4440ZPnx44WoBAADoKLyTGgBYq8rKylxyySWZMGFC3n333QwYMCCXXnpp6bIAAADoIITUAMBq3XPPPQ3/33333TNjxoyC1QAAANBRWe4DAAAAAIBihNQAAAAAABQjpAYAAAAAoBghNQAAAAAAxQipAQAAAAAoRkgNAAAAAEAxrR5SX3HFFRk4cGCeeeaZJMkTTzyRgw8+OPvtt1++8pWvZPHixa1dEgAAALRL5tgAdAStGlL/9re/zRNPPJEBAwYkSerq6nLGGWdk/PjxmT17dqqrqzNl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AAAAAKAYITUAAAAAAMUIqQEAAAAAKKZz6QIAgLbh9ddfz5lnnpk//elP6dq1a7beeutMnDgxffv2zRNPPJHx48fn3XffzYABA3LppZemX79+pUsGAACgA/BOagAgSVJRUZFjjz02s2fPzowZM7LVVltlypQpqauryxlnnJHx48dn9uzZqa6uzpQpU0qXCwAAQAchpAYAkiR9+vTJ4MGDG27vuuuuefnll/P000+nW7duqa6uTpIcfvjhufPOO0uVCQAAQAdjuQ8AYBV1dXW54YYbMnTo0CxYsCBbbLFFw76+ffumrq4uS5cuTZ8+fZp8zn79erRApaxJ//49S5fAGrg2bZdr0za5LgDQ8QmpAYBVTJo0KRtuuGG+9KUv5Re/+EWznHPx4rdTV1e/1mMEEc3n1VffatbzuTbNpzmvjevSvFybtunDrktlZYVfhAJAOyekBgAaqampyQsvvJBp06alsrIyVVVVefnllxv2L1myJJWVlev0LmoAAABYE2tSAwANvvOd7+Tpp5/O1KlT07Vr1yTJzjvvnHfeeSdz5sxJktx4440ZPnx4yTIBAADoQLyTGgBIkjz77LO56qqrss022+Twww9Pkmy55ZaZOnVqLrnkkkyYMCHvvvtuBgwYkEsvvbRwtQAAAHQUrRJSv/766znzzDPzpz/9KV27ds3WW2+diRMnpm/fvnniiScyfvz4RpPefv36tUZZAMDf2X777fOHP/xhtft23333zJgxo5UrAgBWxxwbgI6mVZb7qKioyLHHHpvZs2dnxowZ2WqrrTJlypTU1dXljDPOyPjx4zN79uxUV1dnypQprVESAAAAtEvm2AB0NK0SUvfp0yeDBw9uuL3rrrvm5ZdfztNPP51u3bqluro6SXL44YfnzjvvbI2SAAAAoF0yxwago2n1Nanr6upyww03ZOjQoVmwYEG22GKLhn19+/ZNXV1dli5dmj59+jT5nP369WiBStu2/v17li6hzTEmjRmPxozHqoxJY8YDAGiPmnuOvT7OrxOvBT/IeKzKmDRmPBozHqta1zFp9ZB60qRJ2XDDDfOlL30pv/jFL5rlnIsXv526uvpVtnfkB8irr761zl/TkccjWfcxMR6r6shjYjxW5WemsdWNR2VlxXo7UQMA2ofmnmOvj/PrxHzhg4zHqsyfGvMYacx4rGpd59itGlLX1NTkhRde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vve4Nd3/D8sh2hIhdTPp0aNHhg0bljPPPDMTJkzIjBkzSpdUzCGHHJLvfe97+fjHP77efRyL1VuyZEkmT56cq6++OjvvvHO+973vlS6piKuuuipbb711/uu//itTpkzJokWLGvYZo796+OGH88ADD2T69OmZO3du7r///tIlwSo+/vGP54gjjsi1116bnj175pFHHild0nrPaw/4cF5r0Fat7bFpnv1XXns0pu/zQet7j1vb/V9bDtHWCKmbwYMPPpi33347SVJfX5958+Zlyy23LFxVOVtttVW+/vWv54QTTihdCm3E+eefny9+8YvZYYcdcvbZZ2fmzJl56qmnSpfVqn73u9/lpptuyvjx47PddtvlqKOOyoQJExr2G6Nk2bJlOfvss3PBBRekb9++mTx5cs4777yG51cobeHChZk7d27D7VdeeSVLlixZr3t+W+G1B3w4rzVoq9b02DTP9tpjTfR9Pmh973Fruv8flkO0Nf5wYjP4wx/+kIsvvjj19fVJkq233jrjx48vXFVZo0ePLl1Cm3L00UenU6dODbdnzJiR3r17F6yo9dxxxx15/vnnM2XKlCRJ7969M378+HzrW9/Kl770pdx3333ZZ599Go4/9NBDc+qppxaqtmWsXLky48aNy1lnnZW+ffsmSY455piMHj06t99+ezp37rzGMfrf//3fJFkvxunSSy/NZz/72QwePDhJsvvuu2fffffNxRdfnAsuuCBJctlllzX6mOekSZMa/rI3tLT33nsvl19+ef785z+ne/fuqaury6mnnrre/wGjtsJrj7ZrfX4d1FZ4PUZbtbbH5kEHHbTez7O99lgzfb+x9bnXru89bk33/8wzz0xFRcUac4iDDz44Sdt67FTUv/+MDwAAAAAArcxyHwAAAAAAFCOkBgAAAACgGCE1AAAAAADFCKkBAAAAAChGSA0AAAAAQDFCalgPPfzww9lnn30+0tdOmzYtZ599djNXBAB8kH4NAG2ffg3No3PpAmB9NHTo0Lz22mvp1KlTOnfunN122y3nn39+qqqqSpeWJLnrrrty+eWX58UXX0yXLl0ycODAXHjhhdlqq60yduzY0uUBQKvQrwGg7dOvoWPwTmooZNq0aZk7d25+9atfpV+/fpk0adIaj62trW21ul544YV885vfzLhx4/LYY4/l7rvvzhFHHJFOnTq1Wg0A0Fbo1wDQ9unX0P4JqaGwbt26Zfjw4Zk/f37DtnHjxmXChAn5j//4j+y66655+OGH88tf/jKjRo3K7rvvniFDhuTyyy9vOP6ll17KwIEDM3369Hzuc5/L4MGD8/3vf79h/zvvvJNx48blU5/6VA444IA89dRTa6znd7/7XbbccsvstddeqaioSI8ePbLffvtliy22SJJcfvnlOf3005MkEydOzG677dbwb8cdd2yoa+HChfna176WPffcM0OHDs11113XrOMGAK1JvwaAtk+/hvZLSA2FLV++PHfccUc+8YlPNNo+c+bMjB07No8//ng++clPZoMNNkhNTU3mzJmTq666KjfccEPuuuuuRl/z2GOP5c4778yPfvSjTJ06taExX3HFFfnTn/6UX/ziF7nmmmty6623rrGenXbaKX/84x8zefLk/PrXv86yZcvWeOz48eMzd+7czJ07Nz/5yU/Sq1evDBs2LHV1dTn++OMzcODA3H///fnRj36UH/3oR3nggQc++kABQEH6NQC0ffo1tF9CaijkxBNPTHV1daqrq/Pggw/mmGOOabR/2LBh+eQnP5nKysp069YtgwcPzsCBA1NZWZkddtghBx54YB555JFGX3PSSSele/fu2WGHHbLDDjvk97//fZLkZz/7WcaOHZs+ffqkqqoqRx555Brr2mqrrfLjH/84CxcuzKmnnpo999wz48aNW2szXbJkSU488cSce+652XHHHfPUU09lyZIlOemkk9K1a9dstdVW+eIXv5g77rjjHxgxAGh9+jUAtH36NbR//nAiFDJ16tR8+tOfTm1tbe6+++4ceeSRmTVrVvr3758kq/yRh9/85jeZMmVKnn322axcuTIrVqzI8OHDGx2zySabNPx/gw02yF/+8pckyaJFixqd7/2PFq3JrrvumssuuyxJ8uSTT+brX/96pk2bltNOO22VY1euXJmTTz45I0aMyIEHHpgk+fOf/5xFixalurq64bja2tpGtwGgPdCvAaDt06+h/fNOaiisU6dO2XfffVNZWZnHHntsjceddtppGTZsWO6777489thjOfzww1NfX9+k79G/f/8sWLCg4fbf///D7LLLLtl3333z7LPPrnb/pEmT0qNHj5x66qkN26qqqrLllltmzpw5Df/mzp2bH/zgB03+vgDQlujXAND26dfQfgmpobD6+vrcddddefPNN7Pddtut8bhly5ald+/e6datW5588snMnDmzyd9j//33z9VXX5033ngjr7zySn784x+v8dg5c+bkpptuyuLFi5Mk8+fPzz333LPKml5JcuONN+bRRx/NlClTUln5t6eTXXbZJRtttFGuvvrqvPPOO6mtrc0zzzyTJ598ssk1A0Bbol8DQNunX0P7ZbkPKGTs2LHp1KlTkmTAgAG5+OKLs/3226/x+AkTJqSmpiYTJ07MHnvskf333z9vvvlmk77XSSedlAkTJmTYsGHZdNNNc+ihh67xrwH36tUr99xzT7773e9m+fLl2XjjjbP//vvn2GOPXeXYWbNm5cUXX8xnP/vZhm1f/epXM3bs2EybNi01NTUZNmxYVqxYkX/6p39q9NtgAGgP9GsAaPv0a2j/Kuqb+nkGAAAAAABoZpb7AAAAAACgGCE1AAAAAADFCKkBAAAAAChGSA0AAAAAQDFCagAAAAAAihFSAwAAAABQjJAaAAAAAIBihNQAAAAAABTz/wdPtYnzOgl46wAAAABJRU5ErkJggg=="/>

# Question 3: Answered!


Earlier, We have seen the attributes change within the same shirt size, meaning that size does not offer much details when it comes to it. We have also seen that the Size S can be anywhere from 36 to 39. Needless to say, when it comes to Brands, The whole size thing becomes a mess. Though they all relatively close but it shows flaws on the sizing techniques. They don't offer as much details and your favorite slim S shirt could be L at a different brand.


# Final Answer: Yes, They vary from one brand to another.


Thanks for making that far in my notebook, I would love it if you take the time to share your opinion on the notebook and correct me if you found any flaws in my work. 

Stay Safe. 

