# News-Article-Analysis

This work is based on a dataset of front-page articles scraped from a number of popular US News publications over the course of several years. Analyses focus on only the date range for which there is a sufficient number of articles: The 18 month period from January 2016 - June 2017. The data contains title, author, date, publication, and content of each article. The analyses rely only on the publication, date, and content columns and are as follows:

- In [Plots Folder](https://github.com/hmtessier/News-Article-Analysis/tree/master/Plots)
  - Average sentiment across all articles by month
  - Average sentiment by publication by month 
  - Adjusted average sentiment by publication by month
  - Flesch Readability Score by publication
- In [Topic Models Folder](https://github.com/hmtessier/News-Article-Analysis/tree/master/Topic%20Models)  
  - Topic modeling by month for 2016
  - Topic modeling by Publication
  - Topic modeling using only named entities
- Project.py : Initial data cleaning and sentiment plot code
- DocumentComplexity.ipynb : Flesch Score and plot code

**Topic model interactive visualizations will not display on default github view. To view, click nbviewer link in top right of notebook.**

[Link to data (raw and cleaned .csv)](https://virginia.box.com/s/0xh4mw7x6khr5bdvuvsqq5qkzt9en8wd)
