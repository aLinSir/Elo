# Christmas : December 25 2017
data['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)
# Mothers Day: May 14 2017
data['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)
# fathers day: August 13 2017
data['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)
# Childrens day: October 12 2017
data['Children_day_2017'] = (pd.to_datetime('2017-10-12') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)
# Valentine's Day : 12th June, 2017
data['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)
# Black Friday : 24th November 2017
data['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)

# 2018
# Mothers Day: May 13 2018
data['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - data['purchase_date']).dt.days.apply(
    lambda x: x if x > 0 and x < 100 else 0)