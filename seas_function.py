# iso geography mapping

def iso_country(df, country_col):
     
    import pycountry
    
    df['iso_nm'] = df[country_col]
    
    df['iso_2'] = ''
    
    for country_name in sorted(df.iso_nm.unique()):
        
        try:
            df['iso_2'][df.iso_nm == country_name] = pycountry.countries.search_fuzzy(country_name)[0].alpha_2
            df['iso_nm'][df.iso_nm == country_name] = pycountry.countries.search_fuzzy(country_name)[0].name
            
        except:
            df = df.loc[df.iso_nm != country_name]
        
    return df

# takes a data set in wide format (dates in columns) and converts to long format (dates in rows)

def wide_to_long(df, pat1, var_nm, val_nm):

    import pandas as pd
    
    id_cols, value_cols = [],[]

    for cols in df.columns:
        try:        
            if cols.find(pat1) == -1:
                id_cols.append(cols)
            else:
                if cols.find(pat1) != -1:
                    value_cols.append(cols)                    
        except:            
            value_cols.append(cols)
                
    df=(pd.melt(df, 
             id_vars=id_cols, 
             value_vars=value_cols)
       .rename(columns = {
           'variable': var_nm, 
           'value': val_nm}))

    df=df.loc[df[val_nm] > 0].reset_index()
    
    return df

# prepares dataframe for seasonal adjustment

def seas_prep(df,
             date_col, 
             cat_col, 
             value_col):
    
    agg_date_cols=[cat_col] 

    agg_date_cols.append(date_col)

    df_sa_dates=(df
                 .groupby(cat_col)
                 .agg({date_col: ['min', 'max']})
                 .reset_index())

    df_sa_dates.columns = ['_'.join(tup).rstrip('_') for tup in df_sa_dates.columns.values]
    
    df_sa=(df
          .groupby(agg_date_cols)[value_col]
          .sum()
          .reset_index()
          .pivot_table(index=[date_col], columns=cat_col)[value_col]
          .fillna(0)
          .stack()
          .reset_index()
          .rename(columns = {0:value_col})
          .merge(df_sa_dates))
    
    df_sa=df_sa.loc[(df_sa[date_col] >= df_sa[date_col+"_min"]) &
                    (df_sa[date_col] <= df_sa[date_col+"_max"])
                   ]
       
    return df_sa

# uses seasonal_decompose to create seasonal adjustment factors

def seas_exec(df_source, date_col, range_size, agg_cols, val_col):
    
    from statsmodels.tsa.seasonal import seasonal_decompose

    agg_list= df_source[agg_cols].unique()

    i = 1


    for agg in agg_list:

        df_agg_x = df_source.loc[df_source[agg_cols] == agg]

        yr_limit=(df_agg_x[date_col]
                  .sort_values(ascending = False)
                  .values[len(df_agg_x[date_col])-1]
                  .astype('datetime64[Y]')
                  .astype(int)+ 1970+range_size)

        yr_start = (df_agg_x[date_col]
                    .sort_values(ascending = False)
                    .values[0]
                    .astype('datetime64[Y]')
                    .astype(int) + 1970)

        if yr_limit <= yr_start:

            for yr in range(yr_start, yr_limit, -1):

                df_x=(df_agg_x.
                      loc[(df_agg_x.date.dt.year <= yr) & (df_agg_x.date.dt.year >= yr-range_size)]
                      .sort_values(by = date_col))
                

                if df_x[val_col].min() > 0:

                    decomp=seasonal_decompose(
                        df_x[val_col], 'multiplicative', period = 12, two_sided = True)

                    df_x['seasonal'] = decomp.seasonal


                else: # this else function should be changed

                    decomp=seasonal_decompose(
                        df_x[val_col]+1, 'multiplicative', period = 12, two_sided = True)                
                    df_x['seasonal'] = decomp.seasonal
   

                if i == 1:

                    df_x_sa=df_x.loc[df_x.date.dt.year == yr]

                    i = 2

                else:

                    df_x_sa=df_x_sa.append(df_x.loc[df_x.date.dt.year == yr])
                    
    df_x_sa['volume_sa']=df_x_sa[val_col]/df_x_sa.seasonal
                    
    return df_x_sa.sort_values(by = date_col)

# working days

def working_days(df, date_col):
    
    from workalendar.registry import registry # library for working days by geography
    calendars = registry.get_calendars()
    
    from dateutil.relativedelta import relativedelta
    
    f = lambda x: cal().get_working_days_delta(x[date_col], x[date_col]+ relativedelta(months=1, days=-1))

    df['working_days'] =''

    for iso in sorted(df.iso_2.unique()):

        cal = registry.get(iso)
        df['working_days'].loc[df.iso_2==iso] = df.loc[df.iso_2==iso].apply(f, axis=1)

    df['working_days']=df.working_days.astype(str).astype(int)
    
    return df

# calendar_adjust

def calendar_adjust(df, cat_col, val_ca, val_col):
    
    import pandas as pd
    
    df=pd.merge(df,
            df.groupby([cat_col])['working_days'].mean().reset_index().rename(columns={'working_days':'mean_wd'})
            , left_on = cat_col, 
            right_on = cat_col)
    df['calendar_factor']=df.working_days/df.mean_wd
    df[val_ca]=df[val_col]/df.calendar_factor
    
    return df

# display results

def disp_results(df, c):
    
    import matplotlib.pyplot as plt # display results

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,12))
    
    x=df['date'].loc[df.country==c]
    y=df['volume'].loc[df.country==c]
    y2=df['volume_sa'].loc[df.country==c]
    y3=df['seasonal'].loc[df.country==c]
    y4=df['calendar_factor'].loc[df.country==c]
    y5=df.csf.loc[df.country==c]
    
    ax1.set_title('Raw Series vs. Adjusted Series')
    ax2.set_title('Combined Factor')
    ax3.set_title('seasonal factor and calendar factor')
    
    ax1.plot(x, y,
             x, y2)
    ax2.plot(x, y5)
    ax3.plot(x, y3,
             x, y4)