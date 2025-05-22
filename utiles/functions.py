import pandas as pd

import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_all_rows(df):
    with pd.option_context("display.max_rows", None, 
                       "display.max_columns", None,
                       "display.max_colwidth", None):
        display(df)

# ARS a USD
def transformacion_ars_usd(df):
    f=df["currency"]=="ARS"
    df.loc[f,"price"]=df.loc[f,"price"]/78.5
    df.loc[f,"currency"]="USD"
    return df

# 1EROS Filtros
def primeros_filtros(df_ent ,df_ap):
    df_ent = df_ent.loc[(df_ent["price"].notna()) & (df_ent["currency"] == "USD") & \
    (df_ent["l2"] == "Capital Federal") & (df_ent["operation_type"] == 'Venta') & \
    (df_ent["l3"].isin(df_ap["l3"].unique()))&\
    (df_ent["property_type"].isin(["Cochera", "Departamento", "Casa","PH"]))]

    df_ent=df_ent.drop_duplicates()
    return df_ent

def conversion_ph_to_casa(df):
    f=df["property_type"] =="PH"
    df.loc[f,"ph"] =1
    df.loc[~f,"ph"]=0
    df.loc[f,"property_type"] = "Casa"
    return df


# L3 & L4
def extraccion_l4_text(df):
    subgrupos_to_barrios = {
    # Recoleta
    'Barrio Norte': 'Recoleta',

    # Balvanera
    'Once': 'Balvanera',
    'Abasto': 'Balvanera',

    # Palermo
    'Palermo Hollywood': 'Palermo',
    'Palermo Soho': 'Palermo',
    'Palermo Viejo': 'Palermo',
    'Palermo Chico': 'Palermo',
    'Las Cañitas': 'Palermo',

    # Monserrat
    'Congreso': 'Monserrat',

    # Retiro
    'Catalinas': 'Retiro',

    # San Nicolás
    'Tribunales': 'San Nicolás',
    'Centro / Microcentro': 'San Nicolás',

    # Chacarita
    'Distrito Audiovisual': 'Colegiales',

    'Velez Sarsfield' : 'Versalles',

    'Parque Centenario':'Caballito',

}
    barrios_y_subbarrios_variantes = [
        "Agronomía|Agronomia", "Almagro", "Balvanera", "Barracas", "Belgrano", "Boca",
        "Boedo", "Caballito", "Chacarita", "Coghlan|Coglan", "Colegiales",
        "Constitución|Constitucion", "Flores", "Floresta", "Liniers", "Mataderos",
        "Monserrat", "Monte Castro", "Pompeya", "Nuñez|Nunez|Núñez", "Palermo",
        "Parque Avellaneda", "Parque Chacabuco", "Parque Chas",
        "Parque Patricios", "Paternal", "Puerto Madero", "Recoleta", "Retiro",
        "Saavedra", "San Cristobal", "San Nicolás|San Nicolas", "San Telmo",
        "Velez Sarsfield|Vélez Sarsfield|Velez", "Versalles", "Villa Crespo",
        "Villa del Parque", "Villa Devoto|Devoto", "Villa General Mitre|Villa Gral Mitre",
        "Villa Lugano|Lugano", "Villa Luro", "Villa Ortuzar|Villa Ortúzar",
        "Villa Pueyrredón|Villa Pueyrredon", "Villa Real", "Villa Riachuelo",
        "Villa Santa Rita", "Villa Soldati", "Villa Urquiza|Urquiza",

        # Subbarrios / L4 y zonas comunes
        "Palermo Hollywood", "Palermo Soho", "Palermo Viejo", "Palermo Chico",
        "Las Cañitas|Cañitas|Canitas", "Barrio Norte", "Once", "Abasto",
        "Congreso", "Catalinas", "Tribunales",
        "Centro / Microcentro|Microcentro|Centro",
        "Parque Centenario", "Distrito Audiovisual"]

    #Extraigo de title y describe
    for barrio in barrios_y_subbarrios_variantes:
        # f_desc=(df["description"].str.lower().str.contains(barrio.lower(),regex=True,na=False))
        f_title= (df["title"].str.lower().str.contains(barrio.lower(),regex=True,na=False))
        # df.loc[f_desc,"l4_descr"] = barrio.split(sep='|')[0]
        df.loc[f_title,"l4_title"] = barrio.split(sep='|')[0]

        # Lleno los nan con title
    f_l4_nan_1 = (df["l4"].isna()) & (df["l4_title"].notna())
    df.loc[f_l4_nan_1 , "l4"] = df.loc[f_l4_nan_1 , "l4_title"]

    
        # Lleno los nan de l3 con l4 y viceversa
    f_l3_nan = (df["l3"].isna())&(df["l4"].notna())
    df.loc[f_l3_nan,"l3"]= df.loc[f_l3_nan,"l4"]

    f_l4_nan = (df["l4"].isna())&(df["l3"].notna())
    df.loc[f_l4_nan,"l4"]= df.loc[f_l4_nan,"l3"]
    print(f"nan en l3: {df['l3'].isna().sum()}")
    print(f"nan en l4: {df['l4'].isna().sum()}")


        # Lleno los nan que faltan de l3 y l4 con describe (a esta altura ya tenemos la misma cantidad de nan en l3 y l4)
    for barrio in barrios_y_subbarrios_variantes:
        f_nan = df["l3"].isna()
        f_desc=(df.loc[f_nan,"description"].str.lower().str.contains(barrio.lower(),regex=True,na=False))
        df.loc[f_nan & f_desc,"l4_descr"] = barrio.split(sep='|')[0]
    
    f_l4_nan_2 = (df["l4"].isna()) & (df["l4_title"].isna()) & (df["l4_descr"].notna())
    df.loc[f_l4_nan_2 , "l4"] = df.loc[f_l4_nan_2 , "l4_descr"]
    df.loc[f_l4_nan_2 , "l3"] = df.loc[f_l4_nan_2 , "l4_descr"]

    print(f"nan en l3 final: {df['l3'].isna().sum()}")
    print(f"nan en l4 final: {df['l4'].isna().sum()}")
        # Corrijo barrios y subbarrios
    for subbarrio,barrio in subgrupos_to_barrios.items():
        f=df["l4"] == subbarrio
        df.loc[f,"l3"] = barrio

        f=df["l3"] == subbarrio
        df.loc[f,"l4"] = subbarrio
        df.loc[f,"l3"] = barrio

    return df

def eliminacion_nan_l3_l4(df):
    f_ap=df["price"].isna()
    f_ent=df["price"].notna() 
    f_nan=df["l3"].notna()
    df = df.loc[f_nan | f_ap]
    return df


# Lat & Lon
def tratamiento_outlayers_lat_lon(df):
    f_ap=df["price"].isna()
    f_ent=df["price"].notna()
    
    aux=df["lat"]
    df["lat"]=df["lon"]
    df["lon"]=aux

    f=df["lon"]>-60
    max_lat=df.loc[f&f_ap,"lat"].max()
    min_lat=df.loc[f&f_ap,"lat"].min()
    max_lon=df.loc[f&f_ap,"lon"].max()
    min_lon=df.loc[f&f_ap,"lon"].min()
    print(max_lat,min_lat,max_lon,min_lon)

    f_in=(df["lat"]<=max_lat) & (df["lat"]>=min_lat)&(df["lon"]<= max_lon)&(df["lon"]>= min_lon)
    f_out=~f_in
    df.loc[f_out,["lat","lon"]]=np.nan

    return df

def imputacion_nan_lat_lon(df):
    coordenadas_medias_l4=df.groupby("l3")[["lat","lon"]].mean()
    # print(coordenadas_medias_l4.loc["Palermo"])

    barrios=df["l3"].dropna().unique()
    for l in barrios :
        f_l=df["l3"] == l
        f_nan = df["lat"].isna()
        df.loc[f_l &f_nan ,"lat"] = coordenadas_medias_l4.loc[l]["lat"]
        df.loc[f_l &f_nan ,"lon"] = coordenadas_medias_l4.loc[l]["lon"]
    return df
    
def col_dataset(df):
    f_ap=df["price"].isna()
    f_ent=df["price"].notna()
    df["dataset"]=np.zeros(df.shape[0])
    df.loc[f_ent,"dataset"]=1
    return df

def eliminacion_nan_lat_lon(df):
    f_ap=df["price"].isna()
    f_ent=df["price"].notna()
    f=((df["lat"].notna()) & (df["lon"].notna()))
    df=df.loc[f | f_ap]
    return df

# rooms, bedrooms , bathrooms
def extraction_rooms_text(df):
    rooms = {k:f"{k} amb" for k in range(1,int(df["rooms"].max()+1))}
    for k,v in rooms.items():
        if k == 1 :
            f1=(df["title"].str.lower().str.contains(v,na=False)) | (df["title"].str.lower().str.contains("monoamb",na=False))
            df.loc[f1,"rooms_title"]=k
        else:
            fi=df["title"].str.lower().str.contains(v,na=False)
            df.loc[fi,"rooms_title"]=k

    rooms = {k:f"{k} amb" for k in range(1,int(df["rooms"].max()+1))}

    for k,v in rooms.items():
        if k == 1 :
            f1=(df["description"].str.lower().str.contains(v)) | (df["description"].str.lower().str.contains("monoamb"))
            df.loc[f1,"rooms_description"]=k
        else:
            fi=df["description"].str.lower().str.contains(v,na=False)
            df.loc[fi,"rooms_description"]=k
    
    return df

def categorize_rooms(row):
  vals=[row["rooms"] , row["rooms_title"],row["rooms_description"]]
  not_nulls = [v for v in vals if pd.notna(v)]

  if len(not_nulls)==0:
    return "Todas NaN"
  elif len(not_nulls)==1:
    return "Solo 1 valor"
  elif len(not_nulls)==2:
    if not_nulls[0]==not_nulls[1]:
      return "2 iguales, 1 NaN"
    else:
      return "2 distintos, 1 NaN"
  else:
    vals_unique=set(not_nulls)
    if len(vals_unique)==1:
      return "3 iguales"
    elif  len(vals_unique)==2:
      return "2 iguales, 1 distinto"
    else:
      return "3 distintos"

def rooms_def(row):
  vals=[row["rooms"] , row["rooms_title"],row["rooms_description"]]
  not_nulls = [v for v in vals if pd.notna(v)]
  if ((row["categorize_rooms"] == "3 iguales") | (row["categorize_rooms"] == "3 distintos")) :
    return row["rooms_title"]
  elif (row["categorize_rooms"] == "2 iguales, 1 distinto")|(row["categorize_rooms"] == "2 iguales, 1 NaN"):
    if row["rooms"]==row["rooms_description"]:
      return row["rooms"]
    else:
      return row["rooms_title"]
  elif (row["categorize_rooms"] == "2 distintos, 1 NaN"):
    if pd.isna(row["rooms_title"]):
      return row["rooms"]
    else:
      return row["rooms_title"]
  elif (row["categorize_rooms"] == "Solo 1 valor"):
    return not_nulls[0]
  else:
    return np.nan

def tratamiento_outlayers_ambientes(df):
   f_ap=df["price"].isna()
   f_ent=df["price"].notna()
   f_dpto=df["property_type"] =="Departamento"
   f_casa=df["property_type"] =="Casa"
   f_coch=df["property_type"] == "Cochera"
   
   # Rooms
   max_rooms_dpto = df.loc[f_ap & f_dpto,"rooms_def"].max()
   print(f"Max rooms dptos: {max_rooms_dpto}")
   f_max_dpto=df["rooms_def"]> max_rooms_dpto
   print(f"Pasando a nan {df.loc[f_dpto & f_max_dpto].shape[0]} rooms de deptos")
   df.loc[f_dpto & f_max_dpto,"rooms_def"] = np.nan

   max_rooms_casa = df.loc[f_ap & f_casa,"rooms_def"].max()
   print(f"Max rooms casas: {max_rooms_casa}")
   f_max_casa=df["rooms_def"]> max_rooms_casa
   print(f"Pasando a nan {df.loc[f_casa & f_max_casa].shape[0]} rooms de casas")
   df.loc[f_casa & f_max_casa,"rooms_def"] = np.nan

   # Bedrooms
   max_bedrooms_dpto = df.loc[f_ap & f_dpto,"bedrooms"].max()
   print(f"Max bedrooms dptos: {max_bedrooms_dpto}")
   f_max_dpto=df["bedrooms"]> max_bedrooms_dpto
   print(f"Pasando a nan {df.loc[f_dpto & f_max_dpto].shape[0]} bedrooms de deptos")
   df.loc[f_dpto & f_max_dpto,"bedrooms"] = np.nan

   max_bedrooms_casa = df.loc[f_ap & f_casa,"bedrooms"].max()
   print(f"Max bedrooms casas: {max_bedrooms_casa}")
   f_max_casa=df["bedrooms"]> max_bedrooms_casa
   print(f"Pasando a nan {df.loc[f_casa & f_max_casa].shape[0]} bedrooms de casas")
   df.loc[f_casa & f_max_casa,"bedrooms"] = np.nan

   # Bathrooms
   max_bathrooms_dpto = df.loc[f_ap & f_dpto,"bathrooms"].max()
   print(f"Max bathrooms dptos: {max_bathrooms_dpto}")
   f_max_dpto=df["bathrooms"]> max_bathrooms_dpto
   print(f"Pasando a nan {df.loc[f_dpto & f_max_dpto].shape[0]} bathrooms de deptos")
   df.loc[f_dpto & f_max_dpto,"bathrooms"] = np.nan

   max_bathrooms_casa = df.loc[f_ap & f_casa,"bathrooms"].max()
   print(f"Max bathrooms casas: {max_bathrooms_casa}")
   f_max_casa=df["bathrooms"]> max_bathrooms_casa
   print(f"Pasando a nan {df.loc[f_casa & f_max_casa].shape[0]} bathrooms de casas")
   df.loc[f_casa & f_max_casa,"bathrooms"] = np.nan
   return df

def imputacion_espacios(row):


    if row["property_type"] == "Cochera":
        rooms_def =0
        bedrooms = 0
        bathrooms = 0
        
    else:
      
        # Cuando rooms_def no es nan : Aca corrijo y recupero todos los que son nan en bathrooms y bedrooms pero no en rooms_def
        if not pd.isna(row["rooms_def"]):
            rooms_def=row["rooms_def"]
            
            # Caso de rooms ==1 o 2. Ponemos baños y bedrooms 1. Este es para todos
            if rooms_def <= 2:
                bedrooms = 1
                bathrooms = 1

            # Caso de rooms >2. bedrooms restamos 1 y bathrooms 2 
            elif rooms_def >2:
                bedrooms= rooms_def - 1

                if (row["bathrooms"] > rooms_def) | (pd.isna(row["bathrooms"])) |(row["bathrooms"]==0):
                    bathrooms = rooms_def - 2
                else:
                    bathrooms = row["bathrooms"]
        else:
            if (pd.notna(row["bedrooms"])) & ((row["bathrooms"]> row["bedrooms"]) |(pd.isna(row["bathrooms"]))) :
                bedrooms = row["bedrooms"]
                rooms_def = bedrooms + 1
                bathrooms = bedrooms
            elif (pd.notna(row["bedrooms"])) & (row["bathrooms"]<= row["bedrooms"]):
                bedrooms = row["bedrooms"]
                rooms_def = bedrooms + 1
                bathrooms = row["bathrooms"]

            elif (pd.isna(row["bedrooms"])) & (pd.notna(row["bathrooms"]) ):
                
                bathrooms=row["bathrooms"]
                bedrooms = bathrooms+1
                rooms_def = bathrooms +2
            
            elif (pd.isna(row["bedrooms"])) & (pd.isna(row["bathrooms"])) :
                bathrooms=np.nan
                bedrooms = np.nan
                rooms_def = np.nan

            else:
                print(f"ESTE CASO NO LO CONTEMPLE. Es el indice {row.name}")

                bathrooms=row["bathrooms"]
                bedrooms = row["bedrooms"]
                rooms_def = row["rooms_def"]

    return pd.Series({"rooms_def": rooms_def,"bedrooms": bedrooms,"bathrooms": bathrooms})

# Surface_covered & Surface_total
def rotacion_cov_tot(df):
   f_ap=df["price"].isna()
   f_ent=df["price"].notna()
   f=df["surface_covered"] > df["surface_total"]


   temp = df.loc[f, "surface_covered"].copy()
   df.loc[f, "surface_covered"] = df.loc[f, "surface_total"]
   df.loc[f, "surface_total"] = temp

   f=df["surface_covered"] > df["surface_total"]

   return df


def outliers_surface_ap(df,surface):
   f_ap=df["price"].isna()
   f_ent=df["price"].notna()
   f_dpto=df["property_type"] =="Departamento"
   f_casa=df["property_type"] =="Casa"
   f_coch=df["property_type"] == "Cochera"


   f_lsup_dpto=df[surface] > 1000 # Me quedo con el max de ap en covered :) --- corrijo en total
   print(f"valores max de deptos en ap de {surface} a pasar a nan {df.loc[f_ap & f_dpto & f_lsup_dpto,surface].shape[0]}")
   df.loc[f_ap & f_dpto & f_lsup_dpto,surface]= np.nan

   f_linf_dpto = df[surface] < 14 # Corrijo el minimo
   print(f"valores min de deptos en ap de {surface} a pasar a nan {df.loc[f_ap & f_dpto & f_linf_dpto,surface].shape[0]}")
   df.loc[f_ap & f_dpto & f_linf_dpto,surface] = np.nan

   print(f"Max y min de deptos en ap : {df.loc[f_ap&f_dpto,surface].agg(['max','min'])}")

        # Casa 
   f_lsup_casa=df[surface] > 10000 # Me quedo con el max de ap :)
   print(f"valores max de casas en ap de {surface} a pasar a nan {df.loc[f_ap & f_casa & f_lsup_casa,surface].shape[0]}")
   df.loc[f_ap & f_casa & f_lsup_casa,surface]= np.nan

   f_linf_casa=df[surface] < 20 # Corrijo el minimo
   print(f"valores min de casas en ap de {surface} a pasar a nan {df.loc[f_ap & f_casa & f_linf_casa,surface].shape[0]}")
   df.loc[f_ap & f_casa & f_linf_casa,surface]= np.nan
   print(f"Max y min de casas en ap : {df.loc[f_ap&f_casa,surface].agg(['max','min'])}")

        # Cochera
   f_lsup_coch=df[surface] > 10000 # Me quedo con el max de ap :)
   print(f"valores max de cocheras en ap de {surface} a pasar a nan {df.loc[f_ap & f_coch & f_lsup_coch,surface].shape[0]}")
   df.loc[f_ap & f_coch & f_lsup_coch,surface]= np.nan

   f_linf_coch=df[surface] < 8 # Corrijo el minimo
   print(f"valores min de cocheras en ap de {surface} a pasar a nan {df.loc[f_ap & f_coch & f_linf_coch,surface].shape[0]}")
   df.loc[f_ap & f_coch & f_linf_coch,surface]= np.nan
   print(f"Max y min de cocheras en ap : {df.loc[f_ap&f_coch,surface].agg(['max','min'])}")

   return df
def imputacion_covered_con_total_viceversa(df):
   f_cov_na = (df["surface_covered"].isna()) & (df["surface_total"].notna())
   f_tot_na= (df["surface_total"].isna()) & (df["surface_covered"].notna())
   print(f"nan en covered antes : {df['surface_covered'].isna().sum()}")
   print(f"nan en total antes : {df['surface_total'].isna().sum()}")

   df.loc[f_cov_na,'surface_covered'] = df.loc[f_cov_na,'surface_total']
   df.loc[f_tot_na,'surface_total'] = df.loc[f_tot_na,'surface_covered']
   print(f"nan en covered despues : {df['surface_covered'].isna().sum()}")
   print(f"nan en total despues : {df['surface_total'].isna().sum()}")
   return df

def outliers_surface_ent(df):
   
   f_ap=df["price"].isna()
   f_ent=df["price"].notna()
   
   for ppt in ["Departamento","Casa","Cochera"]:

    f_ppt=df["property_type"]==ppt
    min_cov = df.loc[f_ppt & f_ap ,"surface_covered"].min()
    max_cov = df.loc[f_ppt & f_ap ,"surface_covered"].max()
    min_tot = df.loc[f_ppt & f_ap ,"surface_total"].min()
    max_tot = df.loc[f_ppt & f_ap ,"surface_total"].max()



    f_total_out = (df["surface_total"]> max_tot) | (df["surface_total"]< min_tot)
    f_cov_out = (df["surface_covered"]> max_cov) | (df["surface_covered"]< min_cov)
    

    f_subset_total_out = f_ppt & f_total_out & f_ent 
    df.loc[f_subset_total_out , "surface_total"] = np.nan

    f_subset_cov_out = f_ppt & f_cov_out & f_ent 
    df.loc[f_subset_cov_out , "surface_covered"] = np.nan
   return df


def outliers_ratio_cov_total(df):
   f_ap=df["price"].isna()
   f_ent=df["price"].notna()
   df["cov_total"] = df["surface_covered"] / df["surface_total"] * 100
   for ppt in ["Departamento","Casa","Cochera"]:
      f_ppt=df["property_type"] == ppt
      if ppt =="Departamento":
        min_cov_total= df.loc[f_ap & f_ppt ,"cov_total"].min()
      elif ppt =="Casa":
         min_cov_total=30
      else:
         min_cov_total=70
         
      
      f_lim_inf = df["cov_total"] < min_cov_total

      f_out = f_ppt & f_lim_inf

      df.loc[f_out ,["surface_covered" , "surface_total"]] = np.nan
   
   df["cov_total"] = df["surface_covered"] / df["surface_total"] * 100

   return df


# Price_m2
def calculo_price_m2_real(df):
   df["price_m2_real"]=df["price"]/(df["surface_covered"]+0.001)
   return df
def outlaiers_price_m2_bruto(df):
   lim_inf=500
   lim_sup=10000
   f_lim_inf = df["price_m2_real"] <lim_inf
   f_lim_sup = df["price_m2_real"] > lim_sup
   f_out =  f_lim_inf | f_lim_sup
   f_in = ~f_out
   df=df.loc[f_in]
   return df

def calculo_price_m2_std(df):

    df["original_index"] = df.index
    df["price_m2"] = df["price"] / (df["surface_total"] + 0.001)
    gb = df.groupby(by=["property_type", "l3"])["price_m2"].mean().reset_index()
    df = df[df.columns.drop("price_m2")].merge(gb, on=["property_type", "l3"], how="left")
    df = df.set_index("original_index")
    df.index.name = "id"
    df["price_m2_x_surface_total"] = df["price_m2"] * df["surface_total"]
    return df

def outliers_price_m2_error(df):
    df["error"] = df["price_m2_real"] - df["price_m2"]
    f_out=np.abs(df["error"])> 1000
    df=df.loc[~f_out]
    return df

def outliers_prices_zonap(df):

   
    umbrales_por_l3 = {
    'Agronomía': {'min': 900, 'max': 3800},
        'Almagro': {'min': 900, 'max': 3800},
        'Balvanera': {'min': 700, 'max': 3600},
        'Barracas': {'min': 900, 'max': 3800},
        'Barrio Norte': {'min': 1700, 'max': 4500},
        'Belgrano': {'min': 1000, 'max': 5000},
        'Boedo': {'min': 800, 'max': 3700},
        'Boca': {'min': 500, 'max': 3100},
        'Caballito': {'min': 1000, 'max': 4100},
        'Chacarita': {'min': 1000, 'max': 3800},
        'Coghlan': {'min': 1000, 'max': 4100},
        'Colegiales': {'min': 1200, 'max': 4500},
        'Constitución': {'min': 400, 'max': 3400},
        'Flores': {'min': 700, 'max': 3800},
        'Floresta': {'min': 500, 'max': 3500},
        'Liniers': {'min': 700, 'max': 3800},
        'Mataderos': {'min': 600, 'max': 3500},
        'Monserrat': {'min': 700, 'max': 3700},
        'Monte Castro': {'min': 900, 'max': 3700},
        'Nuñez': {'min': 1000, 'max': 5000},
        'Palermo': {'min': 700, 'max': 8000},
        'Parque Avellaneda': {'min': 500, 'max': 3300},
        'Parque Chacabuco': {'min': 800, 'max': 3700},
        'Parque Chas': {'min': 1000, 'max': 3700},
        'Parque Patricios': {'min': 500, 'max': 3700},
        'Paternal': {'min': 700, 'max': 3700},
        'Pompeya': {'min': 500, 'max': 3000},
        'Puerto Madero': {'min': 1500, 'max': 8300}, 
        'Recoleta': {'min': 1000, 'max': 5000},
        'Retiro': {'min': 1000, 'max': 4500},
        'Saavedra': {'min': 1000, 'max': 3900},
        'San Cristobal': {'min': 600, 'max': 3400},
        'San Nicolás': {'min': 800, 'max': 3500},
        'San Telmo': {'min': 900, 'max': 4000},
        'Versalles': {'min': 700, 'max': 3600},
        'Velez Sarsfield': {'min': 700, 'max': 3600},
        'Villa Crespo': {'min': 1000, 'max': 4000},
        'Villa del Parque': {'min': 1000, 'max': 3700},
        'Villa Devoto': {'min': 1000, 'max': 3900},
        'Villa General Mitre': {'min': 700, 'max': 3600}, 
        'Villa Lugano': {'min': 400, 'max': 2800},
        'Villa Luro': {'min': 800, 'max': 3800},
        'Villa Ortuzar': {'min': 1000, 'max': 3800},
        'Villa Pueyrredón': {'min': 1000, 'max': 3700},
        'Villa Real': {'min': 700, 'max': 3600},
        'Villa Riachuelo': {'min': 400, 'max': 3100},
        'Villa Santa Rita': {'min': 700, 'max': 3800},
        'Villa Soldati': {'min': 400, 'max': 3000},
        'Villa Urquiza': {'min': 1000, 'max': 4200},
    }
    barrios = df["l3"].value_counts().index
    index_out=[]
    for l in barrios :
       f_l = df["l3"] == l
       f_out = (df["price_m2_real"] <umbrales_por_l3[l]["min"]) | (df["price_m2_real"] >umbrales_por_l3[l]["max"])
       index_out=index_out + list(df.loc[f_l & f_out].index)
    return index_out


    # ROOMS vs SURFACE_COVERED
def imputacion_rooms_surface(df,a_imputar=None):
    if a_imputar is None:
       print("Especifica variable a imputar")
    elif a_imputar=="surface_covered":
       
        gb_pt_rooms = df.groupby(["property_type","rooms_rec"])["surface_covered"].mean()
        for pt in ["Departamento","Casa"]:
            f_pt=df["property_type"]==pt
            
            rooms=df["rooms_rec"].dropna().unique()
            for r in rooms:
                f_r=df["rooms_rec"] == r
                f_cov_nan=df["surface_covered"].isna()
                if (pt,r) in gb_pt_rooms.index:
                    df.loc[f_pt & f_r & f_cov_nan , "surface_covered"] = gb_pt_rooms[(pt,int(r))] * r
    elif a_imputar =="rooms_rec":
       gb_pt_rooms = df.groupby(["property_type","rooms_rec"])["surface_covered"].mean()
       f_rooms_nan=(df["rooms_rec"].isna()) & (df["surface_covered"].notna())
       df.loc[f_rooms_nan,"rooms_rec"] = df.loc[f_rooms_nan].apply(lambda row : _imputacion_rooms_con_surface(row,gb_pt_rooms) , axis=1)
    return df

def _imputacion_rooms_con_surface(row,gb_pt_rooms):
   gb_df=pd.DataFrame(gb_pt_rooms)
   gb_df=gb_df.reset_index()
   if row["property_type"] == "Cochera":
      return 0
   else:
      pt = row["property_type"]
      surface=row["surface_covered"]
      f=gb_df["property_type"] == pt
      gb_df_pt=gb_df.loc[f,["rooms_rec","surface_covered"]]
      gb_df_pt["rooms_x_surface"] = gb_df_pt["rooms_rec"] * gb_df_pt["surface_covered"]
      gb_df_pt["diferencia"] = np.abs(gb_df_pt["rooms_x_surface"] - surface)
      min_dif = gb_df_pt["diferencia"].min()
      rooms = gb_df_pt.loc[gb_df_pt["diferencia"] == min_dif ,"rooms_rec"]
      return rooms.iloc[0]

def imputacion_espacios_2(row):


    if row["property_type"] == "Cochera":
        rooms_rec =0
        bedrooms_rec = 0
        bathrooms_rec = 0
        
    else:
      
        # Cuando rooms_rec no es nan : Aca corrijo y recupero todos los que son nan en bathrooms_rec y bedrooms pero no en rooms_def
        if not pd.isna(row["rooms_rec"]):
            rooms_rec=row["rooms_rec"]
            
            # Caso de rooms ==1 o 2. Ponemos baños y bedrooms_rec 1. Este es para todos
            if rooms_rec <= 2:
                bedrooms_rec = 1
                bathrooms_rec = 1

            # Caso de rooms >2. bedrooms_rec restamos 1 y bathrooms_rec 2 
            elif rooms_rec >2:
                bedrooms_rec= rooms_rec - 1

                if (row["bathrooms_rec"] > rooms_rec) | (pd.isna(row["bathrooms_rec"])) |(row["bathrooms_rec"]==0):
                    bathrooms_rec = rooms_rec - 2
                else:
                    bathrooms_rec = row["bathrooms_rec"]
        else:
            if (pd.notna(row["bedrooms_rec"])) & ((row["bathrooms_rec"]> row["bedrooms_rec"]) |(pd.isna(row["bathrooms_rec"]))) :
                bedrooms_rec = row["bedrooms_rec"]
                rooms_rec = bedrooms_rec + 1
                bathrooms_rec = bedrooms_rec
            elif (pd.notna(row["bedrooms_rec"])) & (row["bathrooms_rec"]<= row["bedrooms_rec"]):
                bedrooms_rec = row["bedrooms_rec"]
                rooms_rec = bedrooms_rec + 1
                bathrooms_rec = row["bathrooms_rec"]

            elif (pd.isna(row["bedrooms_rec"])) & (pd.notna(row["bathrooms"]) ):
                
                bathrooms_rec=row["bathrooms_rec"]
                bedrooms_rec = bathrooms_rec+1
                rooms_rec = bathrooms_rec +2
            
            elif (pd.isna(row["bedrooms_rec"])) & (pd.isna(row["bathrooms_rec"])) :
                bathrooms_rec=np.nan
                bedrooms_rec = np.nan
                rooms_rec = np.nan

            else:
                print(f"ESTE CASO NO LO CONTEMPLE. Es el indice {row.name}")

                bathrooms_rec=row["bathrooms_rec"]
                bedrooms_rec = row["bedrooms_rec"]
                rooms_rec = row["rooms_rec"]

    return pd.Series({"rooms_rec": rooms_rec,"bedrooms_rec": bedrooms_rec,"bathrooms_rec": bathrooms_rec})


      
         
      
