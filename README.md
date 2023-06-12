# Spotify Top 100 Songs of 2010-2019
<p align="center">
  <img src="spotify.png" width="350" title="hover text">
</p>

## Introduction
Spotify top songs data set contains information about top 100 songs on Spotify between 2010 and 2019. Descriptors of each songs are as follow:

- song's title
- artist
- genre
- year released
- day, when it was added to Spotify
- bpm - beats per minute, tempo
- nrgy - energy of the song
- dnc - danceability
- db - loudness
- live - whether it's live recording
- val - positivity of the mood of the song
- duration of the song
- whether the song is acoustic
- whether it's focused on spoken word
- popularity
- year, when it was on top
- artist type

Data set was aggregated by https://www.kaggle.com/muhmores on the kagle platform.

## Preparing the data
We start by first importing the libraries that will help us and then we save the dataset into a variable

~~~
!pip install wordcloud==1.8.1

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

df = pd.read_csv('/work/datasets/Spotify 2010 - 2019 Top 100.csv')
~~~

The data are presented. It contains all of the variables named in the introduction, has 1003 rows and 17 columns.

~~~
df.head(20)

     | title                                                            | artist             | top genre        |   year released | added      |   bpm |   nrgy |   dnce |   dB |   live |   val |   dur |   acous |   spch |   pop |   top year | artist type   |
|---:|:-----------------------------------------------------------------|:-------------------|:-----------------|----------------:|:-----------|------:|-------:|-------:|-----:|-------:|------:|------:|--------:|-------:|------:|-----------:|:--------------|
|  0 | STARSTRUKK (feat. Katy Perry)                                    | 3OH!3              | dance pop        |            2009 | 2022‑02‑17 |   140 |     81 |     61 |   -6 |     23 |    23 |   203 |       0 |      6 |    70 |       2010 | Duo           |
|  1 | My First Kiss (feat. Ke$ha)                                      | 3OH!3              | dance pop        |            2010 | 2022‑02‑17 |   138 |     89 |     68 |   -4 |     36 |    83 |   192 |       1 |      8 |    68 |       2010 | Duo           |
|  2 | I Need A Dollar                                                  | Aloe Blacc         | pop soul         |            2010 | 2022‑02‑17 |    95 |     48 |     84 |   -7 |      9 |    96 |   243 |      20 |      3 |    72 |       2010 | Solo          |
|  3 | Airplanes (feat. Hayley Williams of Paramore)                    | B.o.B              | atl hip hop      |            2010 | 2022‑02‑17 |    93 |     87 |     66 |   -4 |      4 |    38 |   180 |      11 |     12 |    80 |       2010 | Solo          |
|  4 | Nothin' on You (feat. Bruno Mars)                                | B.o.B              | atl hip hop      |            2010 | 2022‑02‑17 |   104 |     85 |     69 |   -6 |      9 |    74 |   268 |      39 |      5 |    79 |       2010 | Solo          |
|  5 | Magic (feat. Rivers Cuomo)                                       | B.o.B              | atl hip hop      |            2010 | 2022‑02‑17 |    82 |     93 |     55 |   -4 |     35 |    79 |   196 |       1 |     34 |    71 |       2010 | Solo          |
|  6 | The Time (Dirty Bit)                                             | Black Eyed Peas    | dance pop        |            2010 | 2022‑02‑17 |   128 |     81 |     82 |   -8 |     60 |    44 |   308 |       7 |      7 |    75 |       2010 | Band/Group    |
|  7 | Imma Be                                                          | Black Eyed Peas    | dance pop        |            2009 | 2022‑02‑17 |    92 |     52 |     60 |   -7 |     31 |    41 |   258 |      18 |     37 |    71 |       2010 | Band/Group    |
|  8 | Talking to the Moon                                              | Bruno Mars         | dance pop        |            2010 | 2022‑02‑17 |   146 |     59 |     50 |   -5 |     11 |     8 |   218 |      51 |      3 |    87 |       2010 | Solo          |
|  9 | Just the Way You Are                                             | Bruno Mars         | dance pop        |            2010 | 2022‑02‑17 |   109 |     84 |     64 |   -5 |      6 |    42 |   221 |       1 |      4 |    86 |       2010 | Solo          |
| 10 | Teach Me How to Dougie                                           | Cali Swag District | pop rap          |            2011 | 2022‑02‑17 |    85 |     44 |     85 |   -5 |      9 |    51 |   237 |      20 |     14 |    71 |       2010 | Band/Group    |
| 11 | Forget You                                                       | CeeLo Green        | atl hip hop      |            2010 | 2022‑02‑17 |   127 |     88 |     70 |   -4 |     16 |    77 |   223 |      13 |      6 |    69 |       2010 | Solo          |
| 12 | Deuces (feat. Tyga & Kevin McCall)                               | Chris Brown        | dance pop        |            2011 | 2022‑02‑17 |    74 |     74 |     69 |   -5 |      8 |    22 |   277 |       3 |     11 |    74 |       2010 | Solo          |
| 13 | Memories (feat. Kid Cudi)                                        | David Guetta       | big room         |            2009 | 2022‑02‑17 |   130 |     87 |     56 |   -6 |     25 |    50 |   210 |       0 |     34 |    74 |       2010 | Solo          |
| 14 | Gettin' Over You (feat. Fergie & LMFAO)                          | David Guetta       | big room         |            2009 | 2022‑02‑17 |   130 |     91 |     62 |   -5 |      8 |    45 |   188 |      18 |      8 |    55 |       2010 | Solo          |
| 15 | All I Do Is Win (feat. T-Pain, Ludacris, Snoop Dogg & Rick Ross) | DJ Khaled          | dance pop        |            2010 | 2022‑02‑17 |   150 |     78 |     54 |   -4 |     16 |    28 |   233 |       1 |     19 |    55 |       2010 | Solo          |
| 16 | Over                                                             | Drake              | canadian hip hop |            2010 | 2022‑02‑17 |   100 |     85 |     35 |   -6 |     12 |    45 |   234 |       1 |     20 |    72 |       2010 | Solo          |
| 17 | Find Your Love                                                   | Drake              | canadian hip hop |            2010 | 2022‑02‑17 |    96 |     61 |     63 |   -6 |      3 |    76 |   209 |       2 |     17 |    70 |       2010 | Solo          |
| 18 | Barbra Streisand (Radio Edit)                                    | Duck Sauce         | disco house      |            2010 | 2022‑02‑17 |   128 |     93 |     76 |   -2 |     22 |    46 |   195 |       0 |     10 |    61 |       2010 | Duo           |
| 19 | Stereo Love - Radio Edit                                         | Edward Maya        | romanian house   |            2010 | 2022‑02‑17 |   127 |     78 |     80 |   -4 |      8 |    59 |   185 |       3 |      3 |    71 |       2010 | Solo          |

~~~

Let's look at the type of data we have in each column and see if we have null data.

~~~
print(df.info())
total_nan_values = df.isna().sum()
print ("Total Number of NaN values:")
print(total_nan_values)

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1003 entries, 0 to 1002
Data columns (total 17 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   title          1000 non-null   object 
 1   artist         1000 non-null   object 
 2   top genre      1000 non-null   object 
 3   year released  1000 non-null   float64
 4   added          1000 non-null   object 
 5   bpm            1000 non-null   float64
 6   nrgy           1000 non-null   float64
 7   dnce           1000 non-null   float64
 8   dB             1000 non-null   float64
 9   live           1000 non-null   float64
 10  val            1000 non-null   float64
 11  dur            1000 non-null   float64
 12  acous          1000 non-null   float64
 13  spch           1000 non-null   float64
 14  pop            1000 non-null   float64
 15  top year       1000 non-null   float64
 16  artist type    1000 non-null   object 
dtypes: float64(12), object(5)
memory usage: 133.3+ KB
None
Total Number of NaN values:
title            3
artist           3
top genre        3
year released    3
added            3
bpm              3
nrgy             3
dnce             3
dB               3
live             3
val              3
dur              3
acous            3
spch             3
pop              3
top year         3
artist type      3
dtype: int64
~~~
As we can see in the table above we have at least three rows with null data, we must remove them.
~~~
df = df.dropna(axis=0)
~~~

## Let us first ask a few questions
To better understand the content of the data and what we can expect to find in it, let us first establish some questions that may help us to this end.
- What have been the most important genres of the decade?
- Are there songs within the top that are outside the decade, that is, songs released outside the range?
- What have been the most relevant artists of the decade?
- Have the songs become sadder?
- How has the energy variable evolved?
- How has the danceability variable evolved?
- Create a map of the most common words in song titles
- Is there a correlation between the variables of energy, danceability and mood?

## Hands on data

### What have been the most important genres of the decade?
~~~
df.groupby('top genre')['top genre'].count().sort_values(ascending=False)

top genre
dance pop        361
pop               57
atl hip hop       39
art pop           37
hip hop           21
                ... 
idol               1
indie folk         1
dark clubbing      1
basshall           1
acoustic pop       1
Name: top genre, Length: 132, dtype: int64
~~~

The above table tells us that the dance pop genre has been the most popular genre throughout the decade.

~~~
genres_year = df[['top genre', 'top year']]
genres_year = genres_year.rename(columns={'top year':'top_year'})
def cont_gen(ds):
    years = range(2010, 2020)
    y_g = pd.DataFrame()
    for i in years:
        col = ds[ds.top_year == i]['top genre'].value_counts()
        col = col[:3]
        col.to_frame()
        col = col.rename(str(i))
        y_g = y_g.append(col)

    return y_g

genres_years = cont_gen(genres_year)
genres_years

|      |   atl hip hop |   barbadian pop |   dance pop |   contemporary country |   pop |   art pop |   candy pop |   big room |   canadian contemporary r&b |   alt z |   canadian pop |   canadian hip hop |   latin |
|-----:|--------------:|----------------:|------------:|-----------------------:|------:|----------:|------------:|-----------:|----------------------------:|--------:|---------------:|-------------------:|--------:|
| 2010 |            11 |               5 |          42 |                    nan |   nan |       nan |         nan |        nan |                         nan |     nan |            nan |                nan |     nan |
| 2011 |           nan |             nan |          45 |                      5 |     4 |       nan |         nan |        nan |                         nan |     nan |            nan |                nan |     nan |
| 2012 |           nan |             nan |          37 |                    nan |     4 |         6 |         nan |        nan |                         nan |     nan |            nan |                nan |     nan |
| 2013 |           nan |             nan |          37 |                    nan |     4 |       nan |           3 |        nan |                         nan |     nan |            nan |                nan |     nan |
| 2014 |           nan |             nan |          39 |                    nan |     9 |       nan |         nan |          4 |                         nan |     nan |            nan |                nan |     nan |
| 2015 |           nan |             nan |          38 |                    nan |     5 |       nan |         nan |        nan |                           5 |     nan |            nan |                nan |     nan |
| 2016 |           nan |             nan |          48 |                    nan |   nan |       nan |         nan |        nan |                         nan |       6 |              4 |                nan |     nan |
| 2017 |             8 |             nan |          25 |                    nan |    10 |       nan |         nan |        nan |                         nan |     nan |            nan |                nan |     nan |
| 2018 |             7 |             nan |          28 |                    nan |   nan |       nan |         nan |        nan |                         nan |     nan |            nan |                  5 |     nan |
| 2019 |           nan |             nan |          22 |                    nan |    11 |       nan |         nan |        nan |                         nan |     nan |            nan |                nan |       6 |

~~~
~~~
f1 = plt.figure()
colors = ['tomato','lightseagreen', 'Aqua', 'Firebrick', 'DarkViolet', 'Gainsboro', 'Olive', 'Gold', 'Pink', 'Black', 'Blue', 'Red']
plt.style.use('seaborn-darkgrid')
genres_years.plot(kind='bar', ax=f1.gca(), stacked=True ,figsize=(10,9), color=colors, ylabel='Number of songs in the top', xlabel='Years', title='Most popular genres of the decade')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
~~~
<center><img src="graphs/Most popular genres of the decade.png"/></center>

It is indisputable that dance pop has dominated throughout the decade.
In addition to "dance pop", we can see that genres such as "pop" have also been relevant within the decade. The "alt hip hop" genre has been relevant at the beginning and end of the decade.
There is no dispute that globally the pop genre and all its variants have dominated the decade.

### Are there songs within the top that are outside the decade, that is, songs released outside the range?

~~~
artist = df.query('`year released` < 2010')
artist.head(20)

|    | title                                   | artist          | top genre          |   year released | added      |   bpm |   nrgy |   dnce |   dB |   live |   val |   dur |   acous |   spch |   pop |   top year | artist type   |
|---:|:----------------------------------------|:----------------|:-------------------|----------------:|:-----------|------:|-------:|-------:|-----:|-------:|------:|------:|--------:|-------:|------:|-----------:|:--------------|
|  0 | STARSTRUKK (feat. Katy Perry)           | 3OH!3           | dance pop          |            2009 | 2022‑02‑17 |   140 |     81 |     61 |   -6 |     23 |    23 |   203 |       0 |      6 |    70 |       2010 | Duo           |
|  7 | Imma Be                                 | Black Eyed Peas | dance pop          |            2009 | 2022‑02‑17 |    92 |     52 |     60 |   -7 |     31 |    41 |   258 |      18 |     37 |    71 |       2010 | Band/Group    |
| 13 | Memories (feat. Kid Cudi)               | David Guetta    | big room           |            2009 | 2022‑02‑17 |   130 |     87 |     56 |   -6 |     25 |    50 |   210 |       0 |     34 |    74 |       2010 | Solo          |
| 14 | Gettin' Over You (feat. Fergie & LMFAO) | David Guetta    | big room           |            2009 | 2022‑02‑17 |   130 |     91 |     62 |   -5 |      8 |    45 |   188 |      18 |      8 |    55 |       2010 | Solo          |
| 29 | Replay                                  | Iyaz            | dance pop          |            2009 | 2022‑02‑17 |    91 |     75 |     71 |   -6 |     17 |    20 |   182 |      17 |      7 |    78 |       2010 | Solo          |
| 33 | Down                                    | Jay Sean        | dance pop          |            2009 | 2022‑02‑17 |   132 |     68 |     73 |   -4 |      8 |    73 |   213 |       1 |      3 |    83 |       2010 | Solo          |
| 34 | Do You Remember                         | Jay Sean        | dance pop          |            2009 | 2022‑02‑17 |   126 |     67 |     85 |   -5 |     10 |    82 |   211 |       2 |      7 |    74 |       2010 | Solo          |
| 35 | Young Forever                           | JAY-Z           | east coast hip hop |            2009 | 2022‑02‑17 |   140 |     69 |     64 |   -3 |     21 |    10 |   254 |      42 |      7 |    71 |       2010 | Solo          |
| 36 | Heartbreak Warfare                      | John Mayer      | neo mellow         |            2009 | 2022‑02‑17 |    97 |     55 |     62 |   -8 |     30 |    31 |   270 |      19 |      2 |    69 |       2010 | Solo          |
| 37 | Half of My Heart                        | John Mayer      | neo mellow         |            2009 | 2022‑02‑17 |   115 |     59 |     68 |   -9 |     11 |    73 |   250 |      44 |      3 |    68 |       2010 | Solo          |
| 46 | Wavin' Flag                             | K'NAAN          | reggae fusion      |            2009 | 2022‑02‑17 |    76 |     70 |     63 |   -6 |     24 |    72 |   221 |      13 |      7 |    63 |       2010 | Solo          |
| 47 | Live Like We're Dying                   | Kris Allen      | idol               |            2009 | 2022‑02‑17 |    92 |     89 |     59 |   -3 |     34 |    94 |   213 |       3 |      4 |    57 |       2010 | Solo          |
| 48 | Bad Romance                             | Lady Gaga       | art pop            |            2009 | 2022‑02‑17 |   119 |     92 |     70 |   -4 |      8 |    71 |   295 |       0 |      4 |    86 |       2010 | Solo          |
| 49 | Telephone                               | Lady Gaga       | art pop            |            2009 | 2022‑02‑17 |   122 |     84 |     82 |   -6 |     11 |    72 |   221 |       1 |      4 |    75 |       2010 | Solo          |
| 50 | Alejandro                               | Lady Gaga       | art pop            |            2009 | 2022‑02‑17 |    99 |     79 |     62 |   -7 |     38 |    36 |   274 |       0 |      5 |    72 |       2010 | Solo          |
| 59 | Fireflies                               | Owl City        | indietronica       |            2009 | 2022‑02‑17 |   180 |     66 |     51 |   -7 |     12 |    47 |   228 |       3 |      4 |    84 |       2010 | Band/Group    |
| 62 | Rude Boy                                | Rihanna         | barbadian pop      |            2009 | 2022‑02‑17 |   174 |     75 |     56 |   -4 |      8 |    81 |   223 |      11 |     13 |    80 |       2010 | Solo          |
| 65 | Hard                                    | Rihanna         | barbadian pop      |            2009 | 2022‑02‑17 |   182 |     75 |     31 |   -4 |     65 |    16 |   251 |       1 |     11 |    60 |       2010 | Solo          |
| 70 | Riverside                               | Sidney Samson   | dutch house        |            2009 | 2022‑02‑17 |   126 |     98 |     80 |   -2 |     13 |    29 |   321 |       0 |      5 |    42 |       2010 | Solo          |
| 85 | Carry Out (Featuring Justin Timberlake) | Timbaland       | dance pop          |            2009 | 2022‑02‑17 |   116 |     57 |     53 |   -7 |     26 |    27 |   232 |      11 |     11 |    74 |       2010 | Solo          |

~~~

As we can see in the table above it is not easy to answer this question, since most of the songs that were released in 2009 reached the top in the following year, this may be due to the fact that the date when these songs were released was a date close to 2010.  
In order to get an answer that satisfies our initial question, let's remove those songs that reached the top in 2010.

~~~
artist.query('`top year` != 2010')

|     | title                               | artist      | top genre    |   year released | added      |   bpm |   nrgy |   dnce |   dB |   live |   val |   dur |   acous |   spch |   pop |   top year | artist type  |
|----:|:------------------------------------|:------------|:-------------|----------------:|:-----------|------:|-------:|-------:|-----:|-------:|------:|------:|--------:|-------:|------:|-----------:|:-------------|
| 178 | Good Life                           | OneRepublic | dance pop    |            2009 | 2020‑06‑16 |    95 |     69 |     63 |   -8 |     13 |    65 |   253 |       8 |      5 |    78 |       2011 | Band/Group   |
| 982 | Bohemian Rhapsody - Remastered 2011 | Queen       | classic rock |            1975 | 2020‑06‑22 |   144 |     40 |     39 |  -10 |     24 |    23 |   354 |      29 |      5 |    77 |       2019 | Band/Group    |
~~~

We obtain that only two songs do not seem to follow this relationship of year released equal to the year in which they reached the top or close to it.   
Case 1:  
- For the first case we have the song "Good Life" by "OneRepublic", which was initially released in 2009, but did not reach the top until 2011. Doing a quick search on Youtube we can find that the official video of the song was published in 2011, this may mean that maybe the song was not so popular when it was released but until most people could know it by the video on youtube. If we think more about this data and put things in context it can give us very relevant information about the impact of the internet on music, the song "Good life" was not relevant until it was uploaded to youtube indicates that we can not know the music of the artists until it is on the internet.  
Case 2: 
- Another very interesting case is the song "Bohemian Rhapsody" by "Queen", a song that was released in the year 1975. Undoubtedly, it is a song that has been popular since the date of its release, what stands out is that specifically in the year 2019 it reaches the top. Actually, the explanation is very simple, it was at the end of 2018 when the movie "Bohemian Rhapsody" was released, which was a biopic about the British singer Freddie Mercury and the rock band Queen. The movie was so popular that many of Queen's songs had a high amount of plays during the months following the movie's release.

### What have been the most relevant artists of the decade?
~~~
df.groupby('artist')['artist'].count().sort_values(ascending=False)

artist
Taylor Swift     21
Calvin Harris    18
Drake            18
Rihanna          14
Ariana Grande    14
                 ..
Milky Chance      1
Mr. Probz         1
Muse              1
Mustard           1
CeeLo Green       1
Name: artist, Length: 444, dtype: int64
~~~

~~~
artist_decade = df[['artist', 'top year']]
artist_decade = artist_decade.rename(columns={'top year':'top_year'})
def cont_art(ds):
    years = range(2010, 2020)
    a_d = pd.DataFrame()
    for i in years:
        col = ds[ds.top_year == i]['artist'].value_counts()
        col = col[:3]
        col.to_frame()
        col = col.rename(str(i))
        a_d = a_d.append(col)

    return a_d

artist_decades = cont_art(artist_decade)
artist_decades

|      |   Jason Derulo |   Kesha |   Rihanna |   Bruno Mars |   Chris Brown |   Katy Perry |   Adele |   Calvin Harris |   Lana Del Rey |   Macklemore & Ryan Lewis |   Taylor Swift |   Ariana Grande |   Maroon 5 |   Fetty Wap |   The Weeknd |   The Chainsmokers |   Imagine Dragons |   Kendrick Lamar |   Lorde |   Drake |   Marshmello |   Post Malone |   Billie Eilish |
|-----:|---------------:|--------:|----------:|-------------:|--------------:|-------------:|--------:|----------------:|---------------:|--------------------------:|---------------:|----------------:|-----------:|------------:|-------------:|-------------------:|------------------:|-----------------:|--------:|--------:|-------------:|--------------:|----------------:|
| 2010 |              3 |       4 |         5 |          nan |           nan |          nan |     nan |             nan |            nan |                       nan |            nan |             nan |        nan |         nan |          nan |                nan |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2011 |            nan |     nan |       nan |            3 |             3 |            3 |     nan |             nan |            nan |                       nan |            nan |             nan |        nan |         nan |          nan |                nan |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2012 |            nan |     nan |       nan |          nan |           nan |          nan |       2 |               3 |              3 |                       nan |            nan |             nan |        nan |         nan |          nan |                nan |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2013 |            nan |     nan |       nan |          nan |           nan |          nan |     nan |               4 |            nan |                         3 |              3 |             nan |        nan |         nan |          nan |                nan |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2014 |            nan |     nan |       nan |          nan |           nan |          nan |     nan |             nan |            nan |                       nan |              3 |               3 |          3 |         nan |          nan |                nan |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2015 |            nan |     nan |       nan |          nan |           nan |          nan |     nan |             nan |            nan |                       nan |              4 |             nan |        nan |           3 |            4 |                nan |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2016 |            nan |     nan |       nan |          nan |           nan |          nan |       3 |             nan |            nan |                       nan |            nan |               3 |        nan |         nan |          nan |                  3 |               nan |              nan |     nan |     nan |          nan |           nan |             nan |
| 2017 |            nan |     nan |       nan |          nan |           nan |          nan |     nan |             nan |            nan |                       nan |            nan |             nan |        nan |         nan |          nan |                nan |                 3 |                4 |       3 |     nan |          nan |           nan |             nan |
| 2018 |            nan |     nan |       nan |          nan |           nan |          nan |     nan |             nan |            nan |                       nan |            nan |             nan |        nan |         nan |          nan |                nan |               nan |              nan |     nan |       5 |            4 |             3 |             nan |
| 2019 |            nan |     nan |       nan |          nan |           nan |          nan |     nan |             nan |            nan |                       nan |            nan |               4 |        nan |         nan |          nan |                nan |               nan |              nan |     nan |     nan |          nan |             5 |               4 |'

~~~

Doing a count of the songs by artist that have been inside the top, we can see that the most popular artists have been: Taylor Swift, Calvin Harris, Drake, Rihanna and Ariana Grande.

~~~
f2 = plt.figure()
colors = ['tomato','lightseagreen', 'Aqua', 'Firebrick', 'DarkViolet', 'Gainsboro', 'Olive', 'Gold', 'Pink', 'Black', 'Blue', 'Red', 'Goldenrod', 'mistyrose', 'tan', 'azure', 'gray', 'yellow', 'white', 'greenyellow', 'rosybrown', 'royalblue', 'bisque']
plt.style.use('seaborn-darkgrid')
artist_decades.plot(kind='bar', ax=f2.gca(), stacked=True ,figsize=(10,9), color=colors, title='Most popular artists of the decade', ylabel='Number of songs in the top', xlabel='Years')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
~~~
<center><img src="graphs/Most popular artists of the decade.png"/></center>

From the graph above we can see that at the beginning of the decade it was dominated by Rihanna, in the middle of the decade the most popular artist was Calvin Harris and at the end of the decade new artists like Post Malone and Billie Eilish were starting to gain popularity.

### Have the songs become sadder?
To obtain the change of mood of the songs throughout the year, what was done was to obtain the average of the variable val in a year and then plot the value of each respective year.

~~~
sad = df[['val', 'top year']]
sad = sad.rename(columns={'top year':'top_year'})
def cont_sad(ds):
    years = range(2010, 2020)
    s = {}
    for i in years:
        sa = sad[sad.top_year == i]
        pro = sa.mean()
        s[i] = pro[0]
    return s
sad1 = cont_sad(sad)
sad1 = pd.DataFrame(sad1.items(), columns=['year', 'prom_sad'])
sad1

|    |   year |   prom_sad |
|---:|-------:|-----------:|
|  0 |   2010 |      56.74 |
|  1 |   2011 |      54.85 |
|  2 |   2012 |      53.29 |
|  3 |   2013 |      53.05 |
|  4 |   2014 |      50.95 |
|  5 |   2015 |      46.82 |
|  6 |   2016 |      46.92 |
|  7 |   2017 |      47.72 |
|  8 |   2018 |      47.56 |
|  9 |   2019 |      51.11 |
~~~

~~~
sns.set_theme(style="darkgrid")
f3 = sns.lineplot(data=sad1, x="year", y="prom_sad")
f3.set_title('Graph of the change in the mood of the songs in the decade 2010-2019')
f3.set_ylabel('Positivity of the songs')
f3.set_xlabel('Years')
plt.show()
~~~
<center><img src="graphs/Graph of the change in the mood of the songs in the decade 2010-2019.png"/></center>

In the middle of the decade there was a clear decrease in the mood of the songs and it continued until the end of the decade, being the year 2019, the year in which the songs would gain positivity again.

### How has the energy variable evolved?
The same methodology was used, as in the previous section.

~~~
nrgy = df[['nrgy', 'top year']]
nrgy = nrgy.rename(columns={'top year':'top_year'})
def cont_nrgy(ds):
    years = range(2010, 2020)
    s = {}
    for i in years:
        sa = nrgy[nrgy.top_year == i]
        pro = sa.mean()
        s[i] = pro[0]
    return s
energy = cont_nrgy(nrgy)
energy = pd.DataFrame(energy.items(), columns=['year', 'prom_energy'])
energy

|    |   year |   prom_energy |
|---:|-------:|--------------:|
|  0 |   2010 |         76.65 |
|  1 |   2011 |         76.1  |
|  2 |   2012 |         74.07 |
|  3 |   2013 |         72.44 |
|  4 |   2014 |         71.03 |
|  5 |   2015 |         67.53 |
|  6 |   2016 |         65.8  |
|  7 |   2017 |         63.16 |
|  8 |   2018 |         65.45 |
|  9 |   2019 |         62.79 |
~~~
~~~
sns.set_theme(style="darkgrid")
f4 = sns.lineplot(data=energy, x="year", y="prom_energy")
f4.set_title('Graph of the change of energy of the songs in the decade 2010-2019')
f4.set_ylabel('Energy of the songs')
f4.set_xlabel('Years')
plt.show()
~~~
<center><img src="graphs/Graph of the change of energy of the songs in the decade 2010-2019.png"/></center>

As with the mood variable, the energy of the songs also decreased over the decade.

### How has the danceability variable evolved?

~~~
dnce = df[['dnce', 'top year']]
dnce = dnce.rename(columns={'top year':'top_year'})
def cont_dnce(ds):
    years = range(2010, 2020)
    s = {}
    for i in years:
        sa = dnce[dnce.top_year == i]
        pro = sa.mean()
        s[i] = pro[0]
    return s
dance = cont_dnce(dnce)
dance = pd.DataFrame(dance.items(), columns=['year', 'prom_dance'])
dance

|    |   year |   prom_dance |
|---:|-------:|-------------:|
|  0 |   2010 |        65.29 |
|  1 |   2011 |        63.56 |
|  2 |   2012 |        64.01 |
|  3 |   2013 |        63.55 |
|  4 |   2014 |        65.88 |
|  5 |   2015 |        65.22 |
|  6 |   2016 |        64.53 |
|  7 |   2017 |        73.58 |
|  8 |   2018 |        70.8  |
|  9 |   2019 |        72.34 |
~~~

~~~
sns.set_theme(style="darkgrid")
f5 = sns.lineplot(data=dance, x="year", y="prom_dance")
f5.set_title('Graph of the change of danceability of the songs in the decade 2010-2019')
f5.set_ylabel('Danceability of the songs')
f5.set_xlabel('Years')
plt.show()
~~~
<center><img src="graphs/Graph of the change of danceability of the songs in the decade 2010-2019.png"/></center>

Oddly enough, although the change in energy and mood of the songs was negative, the danceability of the songs shows an increase at the end of the decade.

### Create a map of the most common words in song titles
~~~
text = " ".join(title for title in df.title)
text = "".join(re.split("\(|\)|\[|\]", text)[::2])
text = text.replace('Edit ', '')
text = text.replace('Radio ', '')
text = text.replace('Remix', '')

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.title('Map of the most common words in song titles')
plt.axis("off")
plt.show()
~~~
<center><img src="graphs/Map of the most common words in song titles.png"/></center>

### Is there a correlation between the variables of energy, danceability and mood?

~~~
sns.set_theme(style="darkgrid")
f5 = sns.lineplot(data=dance, x="year", y="prom_dance", label='prom_dance')
f4 = sns.lineplot(data=energy, x="year", y="prom_energy", label='prom_energy')
f3 = sns.lineplot(data=sad1, x="year", y="prom_sad", label='prom_sad')
f5.set_xlabel('Years')
f5.set_ylabel('Points')
f5.legend()
f5.set(title='Correlation between the variables of energy, danceability and mood')
plt.show()
~~~
<center><img src="graphs/Correlation between the variables of energy, danceability and mood.png"/></center>

It seems that there is a correlation between the variables of energy, danceability and mood, however this cannot be confirmed by the graph above.

~~~
expr = df[['dnce', 'nrgy', 'val']]
sns.heatmap(expr.corr(), annot = True)
~~~
<center><img src="graphs/Heat map.png"/></center>
With the heat map we can confirm that there is no relationship between these three variables.

### The most danceable songs of the decade
Now let's find out which were the most danceable songs of the decade.
~~~
df1 = df.rename(columns={'top year':'top_year'})
df1 = df1[df1.dnce > 66]
df1 = df1.reset_index()
df1['dnce'].sort_values(ascending=False)

228    96.0
449    96.0
496    95.0
348    94.0
380    94.0
       ... 
331    67.0
328    67.0
321    67.0
320    67.0
372    67.0
Name: dnce, Length: 549, dtype: float64
~~~
~~~
df1.iloc[[228, 449, 496, 348, 380]]

|     |   index | title      | artist      | top genre   |   year released | added      |   bpm |   nrgy |   dnce |   dB |   live |   val |   dur |   acous |   spch |   pop |   top_year | artist type   |
|----:|--------:|:-----------|:------------|:------------|----------------:|:-----------|------:|-------:|-------:|-----:|-------:|------:|------:|--------:|-------:|------:|-----------:|:--------------|
| 228 |     461 | Anaconda   | Nicki Minaj | dance pop   |            2014 | 2020‑06‑10 |   130 |     61 |     96 |   -6 |     21 |    65 |   260 |       7 |     18 |    70 |       2014 | Solo          |
| 449 |     856 | Yes Indeed | Lil Baby    | atl hip hop |            2018 | 2020‑06‑22 |   120 |     35 |     96 |   -9 |     11 |    56 |   142 |       4 |     53 |    84 |       2018 | Solo          |
| 496 |     923 | Money      | Cardi B     | dance pop   |            2018 | 2020‑06‑22 |   130 |     59 |     95 |   -7 |     11 |    22 |   184 |       1 |     29 |    78 |       2019 | Solo          |
| 348 |     705 | Caroline   | Aminé       | hip hop     |            2017 | 2021‑01‑28 |   120 |     34 |     94 |  -10 |     26 |    71 |   210 |      17 |     51 |    80 |       2017 | Solo          |
| 380 |     750 | Gucci Gang | Lil Pump    | emo rap     |            2017 | 2021‑01‑28 |   120 |     52 |     94 |   -7 |     12 |    70 |   124 |      24 |      6 |    69 |       2017 | Solo          |

~~~
The most danceable song of the decade was Anaconda by Nicki Minaj.

## Conclusions
In conclusion, we can highlight the following points:
- Dance pop was the most popular genre of the decade.
- The most relevant artists of the decade were Taylor Swift, Calvin Harris, Drake, Rihanna and Ariana Grande.
- The most danceable song of the decade was Anaconda by Nicki Minaj.
- There is not a correlation between the variables of energy, danceability and mood
- Although the song "Bohemian Rhapsody" by "Queen" was released in 1975, it got a big boost in popularity thanks to the release of the movie "Bohemian Rhapsody".

### Future work
- The data has some inconsistencies such as the 'dB' category, which has negative values.
- Other categories are not clear about the methodology by which they were obtained and therefore it is difficult to give them meaning.
