<h1 align="center"> Does Same Shirt Sizes Vary from Brand to Another? </h1>

<h2> Introduction </h2>
Since I was a child, I was curious to know does brand offer the same standards when they say this shirt's size is XL?
Well, guess I am finding out today.¶


```python
# Getting Familiar with the Dataset

import pandas as pd
import missingno as msno
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import  numpy as np
import matplotlib.pyplot as plt
import warnings

df = pd.read_csv('../input/shirt-size-recommendation/Shirt Size Recommendation.csv')
df.info()

df['Size'] = pd.to_numeric(df['Size'],errors = 'coerce')
df['Chest(cm)'] = pd.to_numeric(df['Chest(cm)'],errors = 'coerce')
df['Front Length(cm)'] = pd.to_numeric(df['Front Length(cm)'],errors = 'coerce')
df['Across Shoulder(cm)'] = pd.to_numeric(df['Across Shoulder(cm)'],errors = 'coerce')
df.info()

df.head()

df.describe()

for col in df:
    print("The Percentage in missing data is" ,df[col].isnull().sum()/len(df)*100)
    
df = df.dropna()
df.isna().sum()

# Question 1: Does the Shirt Attributes Vary from Size to Another?

fig = px.bar(data_frame = df,
             x='Brand Size')
fig.show()

fig = px.box(df, 
             x="Chest(cm)" )
fig.show()

plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y="Chest(cm)", jitter = True, size = 6)

df['Chest(cm)'].describe()

df.loc[df['Chest(cm)'] == 1067]

df.drop([554], axis = 0, inplace = True)
plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y="Chest(cm)", jitter = True, size = 6)

plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y='Across Shoulder(cm)', jitter = True, size = 6)

plt.figure(figsize=(15,7))
sns.stripplot(data=df, x="Brand Size", y='Front Length(cm)', jitter = True, size = 6)

fig = sns.pairplot(df, hue = 'Size')

_fig = sns.pairplot(df, hue = 'Brand Size')

# Question 1: Answered!
[## Here is what I found out so far:

All the attributes that belongs to a shirt (things like: Front Length - Across Shoulders) can be the same across different sizes. That is obvious in the previous graphs, they are very close in sizes though the brand sizes remain the same.

There is a significant increase in the shirt's attributes (Front length, for instance) as you go further in the size, though it's likely you find a small shirt that has the same chest size as an XL shirt.]

# Question Two: Does the shirt Material affect the shirt's attributes?

fig = px.bar(data_frame = df, x='Type')
fig.show()

df['Type'] = df['Type'].str.replace('cotton','Cotton')
Shirt_Material = df['Type'].value_counts().plot.pie(autopct='%23f%%', radius = 2)

# Question 2: One Day with larger data I will Answer it!
[## To Answer a such a question, it would be hard - if it's not impossible to give an accurate answer- within the data I have. It's not accurate to compare a mean of five shirts to a mean of nearly 750 shirts. Needless to say that those five shirts does not cover all the sizes of the shirt. So We Won't be able to give an answer to such question.]

# Qeustion 3: Does Sizes Vary from Brand X to Brand Y?

fig = px.bar(data_frame = df, x = 'Brand Name')
fig

[## Alright, We have a pretty balanced range of brands in the dataset. But we need to see which brand offer which sizes before diving deep into it.]

print(df['Brand Size'].unique())

len(df['Brand Name'].unique())

[## After going on through multiple technique of trying to represent 164 brand into a clear visualization, I couldn't. Maybe this isn't impossible but sure it's beyond my skill. So I decided to visit the good old sampling techniques. So, We will choose 4 random brands and compare them to see what the results look like.]

fig = px.bar(data_frame= df[df.Size == 40 ],
             x='Brand Size')
fig.show()

fig = px.bar(data_frame= df[df.Size == 39 ],
             x='Brand Size')
fig.show()

fig = px.bar(data_frame= df[df.Size == 42 ], x='Brand Size')
fig.show()

df.rename(columns = {'Brand Name': 'Brand_Name'}, inplace = True)

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

# Question 3: Answered!
[## Earlier, We have seen the attributes change within the same shirt size, meaning that size does not offer much details when it comes to it. We have also seen that the Size S can be anywhere from 36 to 39. Needless to say, when it comes to Brands, The whole size thing becomes a mess. Though they all relatively close but it shows flaws on the sizing techniques. They don't offer as much details and your favorite slim S shirt could be L at a different brand.]

# Final Answer: Yes, They vary from one brand to another.
[## Thanks for making that far in my notebook, I would love it if you take the time to share your opinion on the notebook and correct me if you found any flaws in my work. Stay Safe.]


