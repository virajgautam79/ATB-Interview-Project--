#Initializing the environment:
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.dates as mdates

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
CSV_PATH='office_data.csv'

# Set a clean, professional style using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Define ATB color palette
atb_blue = "#0033A0"  # Primary ATB Blue
atb_light_blue = "#00A3E0"  # Secondary ATB Light Blue
atb_gray = "#7D8A97"  # Neutral gray for contrast
atb_dark_gray = "#2D3E50"  # Dark gray for text
atb_teal = "#00B3A6"  # Teal for complementary colors
atb_green = "#78BE21"  # Green for positive trends

# Set up Seaborn global styles
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": atb_dark_gray,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": atb_dark_gray,
    "xtick.color": atb_dark_gray,
    "ytick.color": atb_dark_gray,
    "grid.color": atb_gray,
    "legend.frameon": False,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (12, 6),
    "axes.grid": False
})

# Custom color palette
atb_palette = [atb_blue, atb_light_blue, atb_teal, atb_gray, atb_green]

# Apply custom palette
sns.set_palette(atb_palette)

# Function to format charts for presentation
def format_chart(title, xlabel, ylabel, legend=True):
    plt.title(title, fontsize=18, weight='bold', color=atb_dark_gray)
    plt.xlabel(xlabel, fontsize=14, color=atb_dark_gray)
    plt.ylabel(ylabel, fontsize=14, color=atb_dark_gray)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if legend:
        plt.legend(frameon=False, fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color=atb_gray)
    plt.tight_layout()

# Now every plot will use these settings


# Optionally, further customize Seaborn parameters

# Now, all plots you generate will follow these global settings.

# Increase display width and limit number of rows/columns
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)



class Environment: 
    '''Loads and Cleans the Dataset.'''
    def __init__(self,csv_path):
        self.csv_path=csv_path
        self.data=None


    def load_data(self):
    # import the csv
        self.data=pd.read_csv(self.csv_path)
        return self.data
    
    #GETTERS: 

    def clean_data(self):
        '''cleans the dataset.'''
    #first i checked if there are any missing vals--> surprise surprise there are
    # now we remove those from the data
        #IMPORTANT METRICS:
        '''
        Week                    
        1   Building               
        2   Floor                   
        3   Space Type             
        4   Space Name             
        5   Capacity                
        6   Avg. Occupancy          
        7   Avg Utilization         
        8   Avg Hours Used (HH:MM) '''

        if self.data is None:
            raise ValueError('File not loaded; refer to load_data')
        
        self.data=self.data[['Week','Building','Floor','Space Type','Capacity','Avg. Occupancy','Avg Utilization','Avg Hours Used (HH:MM)']]
        #IN THE SpaceType column, i have detected 99 rows of #VALUE categorical val

        self.data['Space Type']=self.data['Space Type'].replace('#VALUE!',np.nan)

        self.data=self.data.dropna()
        #converting the week col to datetime obj
        self.data['Week']= pd.to_datetime(self.data['Week'])
        #arrange dataset in ascending order by weeks:
        self.data=self.data.sort_values(by='Week', ascending=True)

        #converting percentages to float
        self.data['Avg. Occupancy']=self.data['Avg. Occupancy'].astype(str).str.rstrip('%').astype(float)

        self.data['Avg Utilization']=self.data['Avg Utilization'].astype(str).str.rstrip('%').astype(float)


        #converting hh:mm to minutes:
        self.data['Avg Hours Used (Minutes)']=self.data['Avg Hours Used (HH:MM)'].str.split(':').apply(lambda x: int(x[0])*60+int(x[1]))
        self.data=self.data.drop(['Avg Hours Used (HH:MM)'],axis=1)

        #need to calculate weighted averages---> get weekly observation counts:
        



        #There are duplicates in the dataset to remove:
        self.data=self.data.drop_duplicates() \
        .reset_index(drop=True)
        counts= self.data.groupby('Week')['Week'].transform('count')
        self.data['Weekly Observations']=counts

        return self.data
    


    def get_unique_vals_(self):
        space_type=self.data['Space Type'].unique()
        avg_occupancy= self.data['Avg. Occupancy'].unique()
        avg_util= self.data['Avg Utilization'].unique()
        capacity_=self.data['Capacity'].unique()
        floor_=self.data['Floor'].unique()
        avg_hours_in_mins=self.data['Avg Hours Used (Minutes)'].unique()
        return f'Unique Space Types in Space Type col:{space_type}\n,Unique Avg. Occ. numbers: {avg_occupancy}\n,Unique avg util values: {avg_util}\n,Unique capacity values:{capacity_}\n,floors in the building:{floor_}\n,avg hours used in mins(int dtype):{avg_hours_in_mins}'


    '''def groupby__(self):
        numeric_data = self.data.select_dtypes(include=[np.number])
        group = numeric_data.groupby(self.data['Space Type'])['Avg Utilization']
        return group.describe()'''
    #desk data across both buildings:
    # def get_desk_space(self):
    #     '''Displays the table for desk spaces across both buildings'''
    #     desk_space= self.data[self.data['Space Type']=='Desks']
    #     return self.data
    #meeting room data across both buildings:
    # def get_meeting_room_Space(self):
    #     mr_space=self.data[self.data['Space Type']=='Meeting Room']
    #     return mr_space

    #BUILDING ONE DATA COMPREHENSIVE:
    def get_building_one_data_overall(self):
        building_1=self.data[self.data['Building']=='Calgary Building 1']

        return f'BUILDING ONE DATA BOTH SPACE TYPES: {building_1.head(3)}'
    
    def get_building_one_desk_only_data(self):
        building_1=Environment.get_building_one_data_overall()
        building_1_desk=building_1[building_1['Space Type']=='Desks']
        return f'BUILDING ONE DESK ONLY DATA{building_1_desk.head(3)}'
    
    def get_building_one_meeting_room_only_data(self):
        building_1=Environment.get_building_one_data_overall()
        meeting=building_1[building_1['Space Type']=='Meeting Room']
        return f'BUILDING ONE MEETING ROOM ONLY DATA{meeting.head(3)}'
    



    #BUILDING 2 DATA:
    def get_building_two_data_overall(self):
        building_2=self.data[self.data['Building']=='Calgary Building 2']
        return f'BUILDING 2 DATA OVERALL:{building_2.head(3)}'
    def get_building_two_desk_only_data(self):
        building_2=Environment.get_building_two_data_overall()
        building_2_desk=building_2[building_2['Space Type']=='Desks']
        return f'BUILDING 2 DESK ONLY DATA:{building_2.head(3)}'
    def get_building_2_meeting_room_only_data(self):
        building_2=Environment.get_building_two_data_overall()
        building_2_mr=building_2[building_2['Space Type']=='Meeting Room']
        return f'BUILDING 2 DATA MEETING ROOM ONLY: {building_2_mr.head(3)}'

    #Meeting Room Data:
    

    def get_pair_plot(self):
        sns.pairplot(self.data,
                     x_vars=['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)'],
                     hue='Space Type')
        plt.title('Pair Plot of Metrics Across Space Types')
        plt.xlabel('Metrics')
        plt.ylabel('Counts')
    
        plt.show()
    

    def get_correl_whole_datasest(self):
        correl=self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr()
        heatmap=sns.heatmap(self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr(),annot=True)
        plt.title('Correlation Matrix of Metrics Across the Data')
        #return f'correl coeffs across the whole datset:{correl}'
        plt.show()

    

    def get_correl_building_one(self):
       
        correl=self.get_building_one_data_overall()[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr()
        heatmap=sns.heatmap(self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr(),annot=True)
        plt.title('Correlation Matrix of Metrics Across Building 1')
        #return f'correl coeffs across the whole datset:{correl}'
        plt.show()

    def get_correl_building_1_(self):
        correl=self.get_building_one_desk_only_data()[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr()
        heatmap=sns.heatmap(self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr(),annot=True)
        plt.title('Correlation Matrix of Metrics Across the Data')
        #return f'correl coeffs across the whole datset:{correl}'
        plt.show()

    # CHECK WHAT THIS FUNCTION DOES PROPERLY:
    def get_full_occupancy(self):
        full_occupancy=self.data #[self.data['Avg. Occupancy']==100]
        space_type=full_occupancy[['Building','Space Type','Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)']]
        meeting= space_type[space_type['Space Type']=='Meeting Room']
        grouping=space_type.groupby(['Building','Space Type']).agg({
            'Avg Utilization':'mean',
            'Avg. Occupancy':'mean',
            'Avg Hours Used (Minutes)':'mean'
        }).reset_index()
        stats=grouping.plot(kind='hist',x='Building',y=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'])
        plt.show()

    
#UNIVARIATE ANALYSIS
#time to do some exploratory data analysis!


class UnivariateAnalysis:
    def __init__(self,data:pd.DataFrame):
        self.data=data

    #CHECK FOR UTILIZATION MERTICS ACROSS DIFFERENT CATEGORIES:

    def describe_avg_utilization_overall(self):
        util=self.data['Avg Utilization']
        return f'Metrics for Avg Utilization Across the data {util.describe()}'
    def hist_plot_avg_utilization_overall(self):
        util_overall=self.data['Avg Utilization']
        util_building_1=self.data[self.data['Building']=='Calgary Building 1']
        util_building_2=self.data[self.data['Building']=='Calgary Building 2']
        util_building_1_desks=util_building_1[util_building_1['Space Type']=='Desks']['Avg Utilization']
        util_building_1_mr=util_building_1[util_building_1['Space Type']=='Meeting Room']['Avg Utilization']
        util_building_2_desks=util_building_2[util_building_2['Space Type']=='Desks']['Avg Utilization']
        util_building_2_mr=util_building_2[util_building_2['Space Type']=='Meeting Room']['Avg Utilization']

        #lets use a facetgrid:
        facet=sns.FacetGrid(self.data[self.data['Space Type']=='Meeting Room'],row='Building',col='Space Type',margin_titles=True, height=4, aspect=2)
        facet.map_dataframe(sns.histplot,'Avg Utilization',bins=30,binwidth=10,kde=True)
       

        #ns.barplot(data=self.data,x='Space Type')
    
    # Adjust spacing to give more room for the row labels on the right
        facet.fig.subplots_adjust(top=0.9, right=0.9, wspace=0.1, hspace=0.2)
        facet.fig.suptitle("Distribution of Avg Utilization by Building and Space Type")
        facet.set_ylabels('Count of Observations')
        
        plt.legend()
        

        plt.show()


        # fig,ax=plt.subplots(figsize=(10,6))

        # sns.histplot(util_overall,bins=100,binwidth=5,color='navy',kde=True,legend=True,alpha=1.0,label='Overall')
        # plt.xlabel('Average Utilization of Space Type')
        # plt.ylabel('Counts')

        # sns.histplot(util_building_1['Avg Utilization'],bins=100,binwidth=5,color='green',kde=True,legend=True,alpha=0.8,ax=ax,label='Building 1 space types')
        # plt.xlabel('Average Utilization of Space Type in Building 1')
        # plt.ylabel('Counts')

        # sns.histplot(util_building_1_desks,bins=100,binwidth=5,color='red',kde=True,legend=True,alpha=0.3,ax=ax,label='building 1 desks')
        # plt.xlabel('Average Utilization of DESK Space Type in Building 1')
        # plt.ylabel('Counts')

        # sns.histplot(util_building_1_mr,bins=100,binwidth=5,color='purple',kde=True,legend=True,alpha=0.5,ax=ax,label='building 1 meeting room')
        # plt.xlabel('Average Utilization of MEETING ROOM Space Type in Building 1')
        # plt.ylabel('Counts')

        

        # plt.ylabel('Counts')

        # sns.histplot
        # ax.legend()
        # plt.show()


    
    def box_plot_avg_utilization_overall(self):
        util=self.data
        plt.figure(figsize=(8, 9))

        sns.boxplot(self.data[self.data['Space Type']=='Meeting Room'],hue='Building',x='Space Type',y='Avg Utilization',palette='viridis')
        plt.xlabel('Space Type')
        plt.ylabel('Avg Utilization (in %)')
        
        plt.legend(loc='upper center',fontsize=8, markerscale=0.7)
        plt.show()

        # sns.boxplot(self.data,hue='Building',x='Space Type',y='Avg Utilization',palette='viridis')
        # plt.xlabel('Average Utilization')
        # plt.ylabel('Counts')
        # plt.show()

    #LETS SEE DATA FOR AVERAGE OCCUPANCY NOW:
    def decribe_avg_occupancy_overall(self):
        occupancy=self.data['Avg. Occupancy']
        return f'DESCRIPTION OF AVG OCCUPANCY OVER THE WHOLE DATASET:{occupancy.describe()}'
    def hist_plot_avg_occupancy_overall(self):
        occupancy=self.data
        facet=sns.FacetGrid(self.data[self.data['Space Type']=='Meeting Room'],row='Building',col='Space Type',margin_titles=True, height=4, aspect=2)
        facet.map(sns.histplot,'Avg. Occupancy',binwidth=10,kde=True)
        facet.set_axis_labels('Avg Occupancy (%)','Count')
        facet.set_titles(col_template="{col_name}", row_template="{row_name}", size=20)
        facet.fig.subplots_adjust(top=0.9, right=0.9, wspace=0.1, hspace=0.2)
        facet.fig.suptitle("Distribution of Avg Weekly Occupancy by Building and Space Type")
        # sns.histplot(occupancy,bins=100,binwidth=20,color='blue',kde=True,legend=False)
        # plt.xlabel('Avg Occupancy of Spaces in both buildings')
        # plt.ylabel('Counts')
        plt.show()
    def box_plot_avg_occupancy_overall(self):
        sns.boxplot(self.data,hue='Building',x='Space Type',y='Avg. Occupancy',palette='viridis')
        plt.xlabel('Average Occupancy of Space Type')
        plt.ylabel('Counts of Observations:')
        plt.legend(loc='lower center',fontsize=8, markerscale=0.7)
        plt.show()
        


    #LETS SEE DATA FOR AVERAGE HOURS IN MINUTES NOW:
    def describe_avg_hours_overall(self):
        return f'DESCRIPTION OF AVG HOURS USED OVERALL:{self.data["Avg Hours Used (Minutes)"].describe()}'
    
    def hist_plot_avg_hours_used_overall(self):
        hours=self.data['Avg Hours Used (Minutes)']
        facet=sns.FacetGrid(self.data,row='Building',col='Space Type',margin_titles=True, height=4, aspect=2)
        facet.map(sns.histplot,'Avg Hours Used (Minutes)',binwidth=10,kde=True)
        facet.set_axis_labels('Avg hours Used (mins)','Count')
        plt.show()
    def box_plot_avg_hours_used_overall(self):
        sns.boxplot(self.data,hue='Building',x='Space Type',y='Avg Hours Used (Minutes)',palette='viridis')
        plt.xlabel('Counts')
        plt.ylabel('Average hours used of spaces in both buildings')
        plt.legend(loc='upper center',fontsize=8, markerscale=0.7)
        plt.show()

    
    #LETS SUBSET THE DATA NOW AND THEN DO UNIVARIATE ANALYSIS:

    def scatterplot(self):
        pass
    def return_(self):
        return self.data



class BivariateAnalysis:
    
    def __init__(self,data:pd.DataFrame):
        self.data=data

#     def plot_corr(self,**kwargs):
#         corr = self.data[['Avg Utilization', 'Avg. Occupancy', 'Avg Hours Used (Minutes)', 'Capacity', 'Floor']].corr()
#     # Create the heatmap
#         sns.heatmap(corr, annot=True, cmap='viridis', **kwargs)

        
    def get_correl_whole_datasest(self):
        building_one_whole=self.data[self.data['Building']=='Calgary Building 1']
        building_one_desks_only=building_one_whole[building_one_whole['Space Type']=='Desks']
        building_one_meeting_room_only=building_one_whole[building_one_whole['Space Type']=='Meeting Room']
        

# # Create a FacetGrid that splits the data by Building (row) and Space Type (col)
#         facet = sns.FacetGrid(self.data, row='Building', col='Space Type', margin_titles=True, height=4, aspect=2)
# # Map the plot_corr function to each facet
#         facet.map_dataframe(self.plot_corr())
#         # correl=self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr()
#         # heatmap=sns.heatmap(self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr(),annot=True)
#         # #return f'correl coeffs across the whole datset:{correl}'
#         plt.show()

    

    def get_correl_building_one(self):
       # building_1=self.data[[Space T]]
       
        correl=self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr()
        sns.heatmap(correl,annot=True)
        #return f'correl coeffs across the whole datset:{correl}'
        plt.show()

    def get_correl_building_1__desks(self):
        correl=self.data[(self.data['Building']=='Building1') & (self.data['Space Type']=='Desks')][['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr()
        heatmap=sns.heatmap(self.data[['Avg. Occupancy','Avg Utilization','Avg Hours Used (Minutes)','Capacity','Floor']].corr(),annot=True)
        #return f'correl coeffs across the whole datset:{correl}'
        plt.show()


    def get_full_occupancy(self):
        full_occupancy=self.data #[self.data['Avg. Occupancy']==100]
        space_type=full_occupancy[['Building','Space Type','Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)']]
        meeting= space_type[space_type['Space Type']=='Meeting Room']
        grouping=space_type.groupby(['Building','Space Type']).agg({
            'Avg Utilization':'mean',
            'Avg. Occupancy':'mean',
            'Avg Hours Used (Minutes)':'mean'
        }).reset_index()
        stats=grouping.plot(kind='bar',x='Building',y=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'])
        plt.show()


class BuildingAnalysis:
    def __init__(self,data: pd.DataFrame):
        self.data=data

    


    
    def grouped_metrics_by_building(self):
        grouped_metrics=self.data.groupby(['Building','Space Type'])[['Avg Utilization','Avg. Occupancy']].agg(
            weighted_Avg_Util=('Avg Utilization',lambda x: np.average(x,weights=self.data.loc[x.index,'Weekly Observations'].values)),
            weighted_Avg_Occupancy=('Avg. Occupancy',lambda x: np.average(x,weights=self.data.loc[x.index,'Weekly Observations'].values))
        ).reset_index()
        return grouped_metrics
    

    
    def plot_grouped_metrics_avg_util(self):

        
        sns.barplot(data=self.grouped_metrics_by_building(),x='Space Type',y='weighted_Avg_Util',hue='Building',palette='viridis')
        plt.title("Average Utilization by Space Type and Building")
        plt.xlabel("Space Type")
        plt.ylim(0,100)
        plt.grid(True)
        plt.ylabel("Average Utilization (%)")
        plt.show()
    


    def plot_avg_occupancy(self):
        sns.barplot(data=self.grouped_metrics_by_building(),x='Space Type',y='weighted_Avg_Occupancy',hue='Building',palette='viridis')
        plt.title("Average Occupancy by Space Type and Building")
        plt.xlabel("Space Type")
        plt.ylim(0,100)
        plt.grid(True)
        plt.ylabel("Average Occupancy (%)")
        plt.show()


    def Building_Medians(self):
        grouped_buildings=self.data.groupby(['Building','Space Type'])[['Avg Utilization','Avg. Occupancy']].agg(np.median)

        return grouped_buildings
    def plot_median_util(self):
        sns.barplot(data=self.Building_Medians(),x='Space Type',y='Avg Utilization',palette='viridis',hue='Building')
        plt.title("Median Utilization by Space Type and Building")
        plt.xlabel("Space Type")
        plt.ylim(0,100)
        plt.grid(True)
        plt.ylabel("Avg Utilization (%)")
        plt.show()
    
    def plot_median_avg_occupancy(self):
        sns.barplot(data=self.Building_Medians(),x='Space Type',y='Avg. Occupancy',palette='viridis',hue='Building')
        plt.title("Median Occupancy by Space Type and Building")
        plt.xlabel("Space Type")
        plt.ylim(0,100)
        plt.grid(True)
        plt.ylabel("Median Occupancy (%)")
        plt.show()


    def identify_outliers_avg_hours_used(self):
        outlier=self.data.sort_values(by='Avg Hours Used (Minutes)',ascending=True).reset_index()
        outlier=outlier['Avg Hours Used (Minutes)']
        q1=outlier.iloc[round((0.25)*(int(outlier.size)))]
        q3=outlier.iloc[round((0.75)*(int(outlier.size)))]
        iqr=q3-q1
        lower_bound=int(q1-1.5*iqr)
        upper_bound=int(q3+1.5*(iqr))
        upper_iqr=self.data[self.data['Avg Hours Used (Minutes)']>=upper_bound]
        lower_iqr=self.data[self.data['Avg Hours Used (Minutes)']<=lower_bound]
        #hours=self.data[(self.data['Avg Hours Used (Minutes)']>250)] #[['Avg Utilization','Capacity','Avg. Occupancy']]
        return upper_iqr

    def identify_outliers_avg_util(self):
        outlier=self.data.sort_values(by='Avg Utilization',ascending=True).reset_index()
        outlier=outlier['Avg Utilization']
        q1=outlier.iloc[round((0.25)*(int(outlier.size)))]
        q3=outlier.iloc[round((0.75)*(int(outlier.size)))]
        iqr=q3-q1
        lower_bound=int(q1-1.5*iqr)
        upper_bound=int(q3+1.5*(iqr))
        upper_iqr=self.data[self.data['Avg Utilization']>=upper_bound]
        lower_iqr=self.data[self.data['Avg Utilization']<=lower_bound]
        #hours=self.data[(self.data['Avg Hours Used (Minutes)']>250)] #[['Avg Utilization','Capacity','Avg. Occupancy']]
        return upper_iqr

    def identify_outliers_avg_occupancy(self):
        outlier=self.data.sort_values(by='Avg. Occupancy',ascending=True).reset_index()
        outlier=outlier['Avg. Occupancy']
        q1=outlier.iloc[round((0.25)*(int(outlier.size)))]
        q3=outlier.iloc[round((0.75)*(int(outlier.size)))]
        iqr=q3-q1
        lower_bound=int(q1-1.5*iqr)
        upper_bound=int(q3+1.5*(iqr))
        upper_iqr=self.data[self.data['Avg. Occupancy']>=upper_bound]
        lower_iqr=self.data[self.data['Avg. Occupancy']<=lower_bound]
        #hours=self.data[(self.data['Avg Hours Used (Minutes)']>250)] #[['Avg Utilization','Capacity','Avg. Occupancy']]
        return upper_iqr
    # def plotting_outliers(self):
    #   sns.histplot(data=self.identify_outliers(),x='Avg. Occupancy',hue='Space Type',kde=True)
    #   plt.show()
    def check_common_outliers(self):
        cols=['Avg Hours Used (Minutes)','Avg Utilization']
        common=self.identify_outliers_avg_hours_used()[cols].isin(self.identify_outliers_avg_util()[cols].to_dict(orient='list')).all(axis=1)
        common_rows=self.identify_outliers_avg_hours_used()[common]
        return common_rows
    
    def plot_common_outliers(self):
        desk=self.check_common_outliers()[self.check_common_outliers()['Space Type']=='Desks']
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=desk, x="Capacity", y="Avg Utilization", hue="Building", palette="viridis", )
        plt.title("Outlier Analysis: Capacity vs. Utilization")
        plt.xlabel("Capacity of Space")
        plt.ylabel("Avg Utilization (%)")
        plt.grid(True)
        plt.show()

    def outlier_heatmap(self):
        pivot_outliers = self.check_common_outliers().pivot_table(values="Avg Utilization", index="Floor", columns="Space Type", aggfunc="count")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_outliers, cmap="Blues", annot=True, fmt=".0f", linewidths=0.5)
        plt.title("Heatmap of Outlier Distribution Across Floors")
        plt.xlabel("Building")
        plt.ylabel("Floor Level")
        plt.show()

    def meeting_room_analysis_outliers(self):
        outliers=self.check_common_outliers()[self.check_common_outliers()['Space Type']=='Desks']
        meeting_room_outliers = outliers
        capacity_outliers = meeting_room_outliers.groupby(["Capacity",'Space Type'])["Avg Utilization"].count().reset_index()
        capacity_outliers.columns = ["Capacity", "Space Type","Outlier Count"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=capacity_outliers, x="Capacity", y="Outlier Count",hue='Space Type', palette="Blues_r",ci=None)

        plt.title("Count of Outliers by Desk Capacity", fontsize=14, fontweight='bold')
        plt.xlabel("Desk Capacity", fontsize=12)
        plt.ylabel("Number of Outliers", fontsize=12)
        plt.xticks(rotation=45)
        
        plt.yticks(range(0, int(capacity_outliers["Outlier Count"].max()) + 1)) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    #there are a lot of common outliers within the datatse
    #this is a cool pattern
    # i wanna see how this is divided into buildings,
    #space type and desks

    def outlier_analysis(self):
        grouping=self.check_common_outliers().groupby(['Building','Space Type','Capacity','Floor']).agg(
            {'Avg Hours Used (Minutes)':['median'],
             'Avg Utilization':['median'],
             'Avg. Occupancy':['median']}
        )
        grouping=grouping.reset_index()
        grouping.columns = ['_'.join(filter(None, col)).strip() for col in grouping.columns.values]

        return grouping
    
    def outlier_analysis_plot(self):
        melted_grouping=self.outlier_analysis().melt(id_vars=['Building','Space Type','Capacity','Floor'],
        value_vars=['Avg Hours Used (Minutes)_median','Avg. Occupancy_median','Avg Utilization_median'],var_name='Metric',value_name='Value')
        sns.catplot(data=melted_grouping,x='Space Type',y='Value',hue='Building',col='Metric',kind='bar',palette='viridis',height=6,aspect=4)
        plt.subplots_adjust(top=0.85)
        plt.suptitle("Median Metrics by Space Type and Building")
        plt.show()

    def outlier_analysis_plot(self):
        melted_grouping=self.outlier_analysis().melt(id_vars=['Building','Space Type','Capacity','Floor'],
        value_vars=['Avg Hours Used (Minutes)_median','Avg. Occupancy_median','Avg Utilization_median'],var_name='Metric',value_name='Value')
        sns.catplot(data=melted_grouping,x='Capacity',y='Value',hue='Building',col='Metric',kind='bar',palette='viridis',height=6,aspect=4)
        plt.subplots_adjust(top=0.85)
        plt.suptitle("Median Metrics by Space Type and Building")
        plt.show()

  
       

class FloorAnalysis:
  def __init__(self,data:pd.DataFrame):
    self.data=data

  def floor_analysis(self):
    desk=self.data #[self.data['Space Type']=='Desks']
    floors=desk.groupby(['Floor','Building','Space Type']).agg(
        Avg_Util=('Avg Utilization','median'),
        Avg_Occupancy=('Avg. Occupancy','median'),
        Avg_Hours_Used=('Avg Hours Used (Minutes)','median')
    )
    return self.data['Floor'].unique()


class MeetingRoom:
    def __init__(self,data=pd.DataFrame):
        self.data=data

    def meeting_room_analysis_by_capacity(self):
        data=self.data[self.data['Space Type']=='Meeting Room']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Space Type','Capacity','Building'],aggfunc='median')

        sns.scatterplot(data=pivot,x='Capacity',y='Avg Utilization',hue='Building',palette='viridis')
        plt.title("Median Avg. Utilzation Across Capacities")
        plt.xlabel('Capacity')
        plt.ylabel('Avg Utilization (in %)')
        plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
        plt.show()


    def meeting_room_analysis_by_floor(self):
      data=self.data[self.data['Space Type']=='Meeting Room']
      pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Floor','Space Type','Capacity','Building'],aggfunc='median')

      sns.barplot(data=pivot,x='Floor',y='Avg Utilization',hue='Building',palette='viridis')
      plt.title("Median Avg. Utilzation Across Floors")
      plt.xlabel('Floors')
      plt.ylabel('Avg Utilization (in %)')
      plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
      plt.show()
      
    
    def meeting_room_analysis_avg_hours(self):
        data=self.data[self.data['Space Type']=='Meeting Room']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Space Type','Capacity','Building'])

        sns.barplot(data=pivot,x='Capacity',y='Avg Hours Used (Minutes)',hue='Building',palette='viridis')
        plt.title('Avg Utilzation of Spaces by Meeting Room Capacities')
        plt.xlabel('Capacity')
        plt.ylabel('Avg Hours Used ')
        plt.show()

    
    def meeting_room_analysis_avg_occupancy(self):
        data=self.data[self.data['Space Type']=='Meeting Room']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Space Type','Capacity','Building'])

        sns.barplot(data=pivot,x='Capacity',y='Avg. Occupancy',hue='Building',palette='viridis')
        plt.title("Median Avg. Occupancy Across Capacities")

        plt.xlabel('Capacity')
        plt.ylabel('Avg Occupancy ')
        plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
        plt.show()
    
    def meeting_room_analysis_avg_occupancy_by_floor(self):

        data=self.data[self.data['Space Type']=='Meeting Room']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Space Type','Capacity','Building','Floor'])

        sns.barplot(data=pivot,x='Floor',y='Avg. Occupancy',hue='Building',palette='viridis')
        plt.title("Median Avg. Occupancy Across Floors")

        plt.xlabel('Floor')
        plt.ylabel('Avg Occupancy ')
        plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
        plt.show()


    def floor_analysis_avg_util(self):
        data=self.data[self.data['Space Type']=='Meeting Room']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Space Type','Floor','Building'])

        sns.barplot(data=pivot,x='Floor',y='Avg Utilization',hue='Building',palette='viridis')
        plt.xlabel('Floor')
        plt.ylabel('Avg Util ')
        plt.ylim(0,100)
        plt.show()

    
    def floor_analysis_avg_occupancy(self):
        data=self.data[self.data['Space Type']=='Meeting Room']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Capacity','Space Type','Floor','Building'])

        sns.barplot(data=pivot,x='Capacity',y='Avg. Occupancy',hue='Floor',palette='viridis')
        plt.xlabel('Floor')
        plt.ylabel('Avg. Occupancy')
        plt.ylim(0,100)
        plt.show()
    
    def scatter(self):
      data=self.data[self.data['Space Type']=='Meeting Room']
      sns.relplot(data=self.data,hue='Building', x='Avg Utilization', y='Capacity', col='Floor', kind='scatter', palette='viridis', col_wrap=3)
      plt.subplots_adjust(top=0.85)
      plt.suptitle("Scatter Plot of Avg Utilization vs. Capacity by Floor")
      plt.ylim(0,40)
      plt.show()





class Desks:
    def __init__(self,data:pd.DataFrame):
        self.data=data


    # def plot_avg_util_desk(self):
    #     desks=self.data[self.data['Space Type']=='Desks']
    #     sns.relplot(data=desks,hue='Building',x='Capacity',y='Avg Utilization',col='Floor',kind='scatter',palette='viridis',col_wrap=5)
    #     plt.subplots_adjust(top=0.85)
    #     plt.suptitle("Histogram of Avg Utilization vs. Capacity by Floor (Desks)")
    #     plt.xlim(0,5)
    #     plt.show()
    def plot_avg_util_desks_by_capacity(self):
        data=self.data[self.data['Space Type']=='Desks']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)',],index=['Space Type','Capacity','Building','Floor'],aggfunc='median')

        sns.barplot(data=pivot,x='Capacity',y='Avg Utilization',hue='Building',palette='viridis')
        plt.title("Median of Weekly Avg. Utilzation Across Capacities of Desks")
        plt.xlabel('Capacity')
        plt.ylabel('Avg Utilization (in %)')
        plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
        plt.show()
    
    def plot_avg_util_desks_by_floor(self):
        data=self.data[self.data['Space Type']=='Desks']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)',],index=['Space Type','Capacity','Building','Floor'],aggfunc='median')

        sns.barplot(data=pivot,x='Floor',y='Avg Utilization',hue='Capacity',palette='viridis')
        plt.title("Median of Weekly Avg. Utilzation Across Floors for Desks by capacity")
        plt.xlabel('Floors')
        plt.ylabel('Avg Utilization (in %)')
        plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
        plt.show()

    def plot_avg_occupancy(self):
        data=self.data[self.data['Space Type']=='Desks']
        pivot=data.pivot_table(values=['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)'],index=['Floor','Space Type','Capacity','Building'],aggfunc='median')

        sns.barplot(data=pivot,x='Floor',y='Avg. Occupancy',hue='Capacity',palette='viridis')
        plt.title("Median of Weekly Avg. Occupancy Across Capacities of Desks")
        plt.xlabel('Floor')
        plt.ylabel('Avg Occupancy (in %)')
        plt.legend(loc='upper left',fontsize=8, markerscale=0.7)
        plt.show()



class TimeSeries:
    def __init__(self,data: pd.DataFrame):
        self.data=data

    def time_series(self):
       
        #self.data.set_index('Week', inplace=True)
        #time=pd.self.data(index=pd.date_range("2000", freq="D", periods=3))
        #time=self.data.set_index('Week',inplace=True)

        #monthly=time #.resample('M').median()
        self.data.set_index('Week', inplace=True)
        monthly_data=self.data.resample('M')[['Avg Utilization','Avg. Occupancy','Avg Hours Used (Minutes)']].median()
        #return monthly
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_data.index, monthly_data['Avg Utilization'], marker='o', label='Avg Utilization (Median)')
        plt.plot(monthly_data.index, monthly_data['Avg. Occupancy'], marker='s', label='Avg Occupancy (Median)')
        plt.plot(monthly_data.index, monthly_data['Avg Hours Used (Minutes)'], marker='^', label='Avg Hours Used (Median)')
        plt.title("Monthly Median Metrics Over Time")
        plt.xlabel("Month")
        #plt.xlim(0,12)
        plt.ylabel("Median Value")
        plt.legend()
        plt.grid(False)
        plt.show()


    def meeting_room_utilization_trend(self):
        """Analyzes time-series trends in meeting room utilization."""
        
        """Tracks meeting room utilization over time based on capacity."""
        """Tracks meeting room utilization trends over time by binned capacity."""
        meeting_rooms = self.data[self.data["Space Type"] == "Desks"].copy()

        # Convert Week to datetime and set as index
        meeting_rooms["Week"] = pd.to_datetime(meeting_rooms["Week"])
        meeting_rooms.set_index("Week", inplace=True)

        # Bin capacities into categories
        bins = [0, 5, 10, 15, float("inf")]
        labels = ["0-5", "5-10", "10-15", "15+"]
       #meeting_rooms["Capacity Bin"] = pd.cut(meeting_rooms["Capacity"], bins=bins, labels=labels, right=False)

        # Resample by month and compute median utilization for each capacity bin
        capacity_trend = meeting_rooms.groupby("Capacity").resample("M")["Avg Utilization"].median().unstack(level=0)

        # Visualization
        plt.figure(figsize=(12, 6))
        markers = {"0-5": "o", "5-10": "s", "10-15": "D", "15+": "X"}
#sns.lineplot(data=capacity_trend[bin_label], marker=markers[bin_label], label=f"Capacity {bin_label}")

        for bin_label in capacity_trend.columns:
            sns.lineplot(data=capacity_trend[bin_label],alpha=0.7, marker="o", label=f"Capacity {bin_label}")

        plt.title("Utilization Trends by Desk Capacity ", fontsize=14, fontweight="bold")
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Median  Utilization (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title="Capacity", fontsize=10)
        plt.grid(False) #, linestyle="--", alpha=0.7)
        plt.show()

    
    

    def seasonality_analysis_by_building_floor(self):
        """Tracks seasonal trends in utilization for buildings and floors."""
        
      
        total = self.data.copy()

        total["Week"] = pd.to_datetime(total["Week"])
        total.set_index("Week", inplace=True)

        building_trend = total.groupby("Building").resample("M")["Avg Utilization"].median().unstack(level=0)
        floor_trend = total.groupby("Floor").resample("M")["Avg Utilization"].median().unstack(level=0)

        building_colors = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7"]  # Blue, Orange, Green, Yellow, Pink
        floor_colors = ["#56B4E9", "#E69F00", "#009E73", "#F0E442", "#D55E00", "#CC79A7","#CD5C5C","3b1c15","f622d4",'22f652',"070707"]  # More distinct colors

        plt.figure(figsize=(12, 6))
        for i, building in enumerate(building_trend.columns):
            sns.lineplot(x=building_trend.index, y=building_trend[building], marker="o", label=f"{building}",
                        color=building_colors[i % len(building_colors)], linewidth=2.5)

     

        plt.title("Seasonality Impact on Utilization by Building", fontsize=14, fontweight="bold")
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Median Utilization (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title="Building", fontsize=10, loc="upper left", frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

        plt.figure(figsize=(12, 6))
        for i, floor in enumerate(floor_trend.columns):
            sns.lineplot(x=floor_trend.index, y=floor_trend[floor],alpha=0.65, marker="o", label=f"Floor {floor}",
                        color=floor_colors[i % len(floor_colors)], linewidth=2.5)


        plt.title("Seasonality Impact on Utilization by Floor", fontsize=14, fontweight="bold")
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Median Utilization (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title="Floor", fontsize=10, loc="upper left", frameon=True)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

                
def main():
    '''setting up Environment class: fns--> load_data(self)
    clean_data(self):
    get_unique_vals_
    get_building_one_data_overall
    get_building_one_desk_only_data
    get_building_one_meeting_room_only_data
    get_building_two_data_overall
    get_building_two_desk_only_data
    get_building_2_meeting_room_only_data(self)
    get_pair_plot(self)
    get_correl_whole_datasest(self)
    get_correl_building_one(self):
    get_correl_building_1_
    get_full_occupancy
''' 
    env=Environment(CSV_PATH)
    #load & clean dataset:
    load_data=env.load_data()
    clean_data=env.clean_data()
    print()
    univariate=UnivariateAnalysis(clean_data)
    print(TimeSeries(clean_data).seasonality_analysis_by_building_floor())
    print(TimeSeries(clean_data).meeting_room_utilization_trend())
    print(BuildingAnalysis(clean_data).meeting_room_analysis_outliers())
    print(BuildingAnalysis(clean_data).outlier_heatmap())
    print(BuildingAnalysis(clean_data).plot_common_outliers())
    print("QUESTION 1: WHAT IS THE UTILIZATION AND OCCUPANCY OF MEETING ROOMS AND DESKS ACROSS BOTH BUILDINGS?")
    print('This is a visual analysis of the distribution of spaces by their metrics across both Calgary buildings:')
    hist_of_metrics=univariate.describe_avg_utilization_overall()
    print(hist_of_metrics)
    print("*******************************************************************************************************")
    print(univariate.hist_plot_avg_utilization_overall())
    print(f'''Here is my analysis of the above graph: 1. The number of desks in both buildings > Number of Meeting Rooms, 2. There are more counts of obs. for desks on the lower end of avg utilization. This implies, more desks are underutilized with a skewness  and a right tail.
    3. there are more desks in building 1 than in building 2.
    4. The avg utilization distribution across the dataset for meeting rooms is closer to normal than desks , but there are still alot of meeting rooms that have not been utilized properly
    NEXT STEPS: 
    1. Find the desks with util between 0-30 % and same with meeting rooms.
    ''')
    print("*******************************************************************************************************")
    print(univariate.box_plot_avg_utilization_overall())
    print('''insights:
    for desks in building 1, the data set is extremely skewed with outliers on the higher end of avg utilization

    for meeting spaces, there are outliers in both sides on the higher end of avg utilization)''')
    print("*******************************************************************************************************")
    print(univariate.decribe_avg_occupancy_overall())
    print("*******************************************************************************************************")
    print(univariate.hist_plot_avg_occupancy_overall())
    print("*******************************************************************************************************")
    print(univariate.box_plot_avg_occupancy_overall())
    print('''My analysis: there are not any outliers here and avg occupancy of meeting spaces> avg occupancy of desk spaces across both buildings''')
    print("*******************************************************************************************************")
    print(univariate.describe_avg_hours_overall())
    print("*******************************************************************************************************")
    print(univariate.describe_avg_hours_overall())
    print(univariate.box_plot_avg_hours_used_overall())
    print("*******************************************************************************************************")
    print('''There are outliers here as well on the higher end of avg hours used, in general avg hours used are low for desk spaces 
    ; next step is to find common outliers b/w avg hours used and avg utilization''')
    print('''One important insight i see here is that avg occupancy of meeting rooms> avg occupancy of desks
    but avg hours used of meeting space< avg hours used of desks. This means that meeting rooms are used more often but less intensively''')

    print('''Another crucial insight here is that even though there are more desks in building 1, the utilization of desks is higher in building 2,
    and we already know that Building 1: floors 1,2 3; Building 2: floors 4,5,6,7''')  

    print("*******************************************************************************************************")
    print('''Q2: WHAT IS THE UTILIZATION OF MEETING ROOM ACROSS VARIOUS CAPACITIES?''')
    meeting=MeetingRoom(clean_data)
    print(meeting.meeting_room_analysis_by_capacity())
    print('''
    Created a pivot table and aggregated the data based on the median as its a more robust measure due to outliers in the dataset.
    Meeting rooms with higher capacity are utilized more than meeting roo,s with lower capacity generally
    things to note: meeting room with capacity 7 is vastly underutilized, so let’s get rid of it and meeting rooms with size 8 can be used for 7 person meetings
    Another interesting thing to note that even though avg hours are varied for different capacities, the avg occupancy is pretty high so they are being used pretty often with the exception of meeting room 7 ''')
    print("*******************************************************************************************************")
    print(meeting.meeting_room_analysis_by_floor())
    print('''In the above fn i have plotted the median of avg weekly utilization of meeting room by the floor its on'''  )

    print("*******************************************************************************************************")
    print(meeting.meeting_room_analysis_avg_occupancy())

    print("*******************************************************************************************************")
    print(meeting.meeting_room_analysis_avg_occupancy_by_floor())
    print('''in the above code the median of avg weekly occupancy of meeting rooms has been plotted against floors''')
    print("*******************************************************************************************************")

    outlier_analysis=BuildingAnalysis(clean_data).outlier_analysis_plot()
    desks=Desks(clean_data)
    print("*******************************************************************************************************")
    print(desks.plot_avg_util_desks_by_capacity())
    print("*******************************************************************************************************")
    print(desks.plot_avg_util_desks_by_floor())
    print("*******************************************************************************************************")
    print(desks.plot_avg_occupancy())
    print("*******************************************************************************************************")
    time_series=TimeSeries(clean_data)
    print("*******************************************************************************************************")
    print(time_series.time_series())
    print("*******************************************************************************************************")


  






if __name__=="__main__":
    main()


