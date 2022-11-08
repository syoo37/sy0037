# Does Same Shirt Sizes Vary from Brand to Another?

## 브랜드마다 같은 사이즈 셔츠의 크기가 다른가?

#### 예전부터 셔츠의 사이즈가 XL라고 하면 모든 브랜드가 같은 기준으로 제공하는가?

#### 그것에 대해 알아봅시다


# 데이터셋 준비



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

![newplot](https://user-images.githubusercontent.com/115538847/200498458-544c4386-2c54-44a5-871a-44da24decc6a.png)



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

![newplot](https://user-images.githubusercontent.com/115538847/200498597-b01d802f-418b-4dc1-9cd2-50d719cfeff3.png)



```python
fig = px.bar(data_frame= df[df.Size == 39 ],
             x='Brand Size')
fig.show()
```

![newplot](https://user-images.githubusercontent.com/115538847/200498684-53dc7525-0df6-469f-a41a-a163a95ec943.png)




```python
fig = px.bar(data_frame= df[df.Size == 42 ], x='Brand Size')
fig.show()
```


![newplot](https://user-images.githubusercontent.com/115538847/200498847-0e8452a4-b7a6-4deb-97bb-221924a36b70.png)



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

![image](https://user-images.githubusercontent.com/115538847/200498976-2e8b0f33-1ada-4a83-b1b2-815b0b601d5a.png)

# Question 3: Answered!


Earlier, We have seen the attributes change within the same shirt size, meaning that size does not offer much details when it comes to it. We have also seen that the Size S can be anywhere from 36 to 39. Needless to say, when it comes to Brands, The whole size thing becomes a mess. Though they all relatively close but it shows flaws on the sizing techniques. They don't offer as much details and your favorite slim S shirt could be L at a different brand.


# Final Answer: Yes, They vary from one brand to another.


Thanks for making that far in my notebook, I would love it if you take the time to share your opinion on the notebook and correct me if you found any flaws in my work. 

Stay Safe. 

