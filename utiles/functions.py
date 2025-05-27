import pandas as pd

import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PropertyType:
   property_types={"Departamento","Casa","Cochera","PH"}

   def __init__(self,prop,df_ent,df_ap):
      if prop not in self.property_types:
         raise ValueError(f"Tipo de propiedad {prop} no valido. Tiene que estar en {self.property_types}")
      # Carga de los datos
      self.prop = prop
      self.df_ent=df_ent
      self.df_ap=df_ap
      # Tratamiento iniciales
      self._transformacion_ars_usd()
      self._filtros_iniciales()
      self.df=self._concat_df()
      if self.prop =="Casa":
         self._conversion_ph_to_casa()

   def tratamiento_l3_l4(self):
      self._extraccion_l4_text()
      self._eliminacion_nan_l3_l4()

   def tratamiento_lat_lon(self):
      self._outliers_lat_lon()
      self._imputacion_nan_lat_lon()
      self._col_dataset()
      self._eliminacion_nan_lat_lon()


   def tratamiento_ambientes(self):
       # Tratamiento amb en cocheras
       if self.prop =="Cochera":
           self.df.drop(columns=["rooms","bedrooms","bathrooms"],inplace=True)
           return
       
       # Tratamiento rooms
       self._extraction_rooms_text()
       self.df["categorize_rooms"] = self.df.apply(self._categorize_rooms,axis=1)
       self.df["rooms_def"] = self.df.apply(self._rooms_def , axis=1)

       # Tratamiento rooms, bedrooms y bathrooms
       self._tratamiento_outlayers_ambientes()
       self.df[["rooms_rec","bedrooms_rec","bathrooms_rec"]]=self.df.apply(self._imputacion_espacios,axis=1)
   
   def tratamiento_surface_cov_tot(self):
       # Primero damos vuelta los cov>tot
       self._rotacion_cov_tot()
       # Segundo, pasamos outlayers de cada uno a nan
       self._outliers_surface_ap("surface_covered")
       print('')
       print("-------------")
       print('')
       self._outliers_surface_ap("surface_total")
       # Tercero, imputamos  covered con total
       self._imputacion_covered_con_total_viceversa()
       # Cuarto, volvemos nan los outlayers de ent con los limites de ap
       self._outliers_surface_ent()
       # Quinto, outliers de cov/total
       self._outliers_ratio_cov_total()
   
   def tratamiento_price_m2(self):
       print("1 -------------")
       self._calculo_price_m2_real()
       print("")
       print("2 -------------")
       self._outliers_price_m2_bruto()
       print("")
       print("3 -------------")
       self._calculo_price_m2_std()
       print("")
       print("4 -------------")
    #    self._outliers_price_m2_error()
       self._outliers_prices_zonap()

   def tratamiento_rooms_surfaceCovered(self):
       if self.prop=="Cochera":
           pass
       self._imputacion_rooms_surface(a_imputar="surface_covered")
       self.tratamiento_price_m2()
       self._imputacion_rooms_surface(a_imputar="rooms_rec")

       

    
#                       --------------------
# ********************* | METODOS INTERNOS | ***********************************
#                       --------------------

# Tratamientos iniciales -----------------------------------------------------------------------------------------------------------
   def _transformacion_ars_usd(self):
      f=self.df_ent["currency"]=="ARS"
      self.df_ent.loc[f,"price"]=self.df_ent.loc[f,"price"]/78.5
      self.df_ent.loc[f,"currency"]="USD"
      print("Transformacion ARS -> USD exitosa")
   
   def _filtros_iniciales(self):
      if self.prop == "Casa":
         self.df_ent = self.df_ent.loc[(self.df_ent["price"].notna()) & (self.df_ent["currency"] == "USD") & \
         (self.df_ent["l2"] == "Capital Federal") & (self.df_ent["operation_type"] == 'Venta') & \
         (self.df_ent["l3"].isin(self.df_ap["l3"].unique()))&\
         (self.df_ent["property_type"].isin(["Casa","PH"]))]
         self.df_ent=self.df_ent.drop_duplicates()
         self.df_ap=self.df_ap.loc[self.df_ap["property_type"].isin(["Casa","PH"])]
         print("Filtros iniciales exitosos")
      else:
         self.df_ent = self.df_ent.loc[(self.df_ent["price"].notna()) & (self.df_ent["currency"] == "USD") & \
         (self.df_ent["l2"] == "Capital Federal") & (self.df_ent["operation_type"] == 'Venta') & \
         (self.df_ent["l3"].isin(self.df_ap["l3"].unique()))&\
         (self.df_ent["property_type"]==self.prop)]
         self.df_ent=self.df_ent.drop_duplicates()
         self.df_ap=self.df_ap.loc[self.df_ap["property_type"] == self.prop]
         print("Filtros iniciales exitosos")

   def _concat_df(self):
      df = pd.concat([self.df_ent , self.df_ap] , axis=0)
      return df
   def _conversion_ph_to_casa(self):
      f = self.df["property_type"]=="PH"
      self.df["ph"] =0
      self.df.loc[f , "ph"] =1
      self.df.loc[f , "property_type"] = "Casa"
      print("Conversion PH -> CASA exitosa")
      
   def show_all_rows(self):
      with pd.option_context("display.max_rows", None, 
                       "display.max_columns", None,
                       "display.max_colwidth", None):
         display(self.df)
        

# Tratamiento l3 - l4 --------------------------------------------------------------------------------------------------------
   def _extraccion_l4_text(self):
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
            f_title= (self.df["title"].str.lower().str.contains(barrio.lower(),regex=True,na=False))
            # df.loc[f_desc,"l4_descr"] = barrio.split(sep='|')[0]
            self.df.loc[f_title,"l4_title"] = barrio.split(sep='|')[0]

            # Lleno los nan con title
        f_l4_nan_1 = (self.df["l4"].isna()) & (self.df["l4_title"].notna())
        self.df.loc[f_l4_nan_1 , "l4"] = self.df.loc[f_l4_nan_1 , "l4_title"]

        
            # Lleno los nan de l3 con l4 y viceversa
        f_l3_nan = (self.df["l3"].isna())&(self.df["l4"].notna())
        self.df.loc[f_l3_nan,"l3"]= self.df.loc[f_l3_nan,"l4"]

        f_l4_nan = (self.df["l4"].isna())&(self.df["l3"].notna())
        self.df.loc[f_l4_nan,"l4"]= self.df.loc[f_l4_nan,"l3"]
        print(f"nan en l3: {self.df['l3'].isna().sum()}")
        print(f"nan en l4: {self.df['l4'].isna().sum()}")


            # Lleno los nan que faltan de l3 y l4 con describe (a esta altura ya tenemos la misma cantidad de nan en l3 y l4)
        for barrio in barrios_y_subbarrios_variantes:
            f_nan = self.df["l3"].isna()
            f_desc=(self.df.loc[f_nan,"description"].str.lower().str.contains(barrio.lower(),regex=True,na=False))
            self.df.loc[f_nan & f_desc,"l4_descr"] = barrio.split(sep='|')[0]
        
        f_l4_nan_2 = (self.df["l4"].isna()) & (self.df["l4_title"].isna()) & (self.df["l4_descr"].notna())
        self.df.loc[f_l4_nan_2 , "l4"] = self.df.loc[f_l4_nan_2 , "l4_descr"]
        self.df.loc[f_l4_nan_2 , "l3"] = self.df.loc[f_l4_nan_2 , "l4_descr"]

        print(f"nan en l3 final: {self.df['l3'].isna().sum()}")
        print(f"nan en l4 final: {self.df['l4'].isna().sum()}")
            # Corrijo barrios y subbarrios
        for subbarrio,barrio in subgrupos_to_barrios.items():
            f=self.df["l4"] == subbarrio
            self.df.loc[f,"l3"] = barrio

            f=self.df["l3"] == subbarrio
            self.df.loc[f,"l4"] = subbarrio
            self.df.loc[f,"l3"] = barrio

   def _eliminacion_nan_l3_l4(self):
        print(f"shape antes: {self.df.shape}")
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna() 
        f_nan=self.df["l3"].notna()
        self.df = self.df.loc[f_nan | f_ap]
        print(f"shape después: {self.df.shape}")

# Tratamiento lat - lon  -----------------------------------------------------------------------------------------------------
   def _outliers_lat_lon(self):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        
        aux=self.df["lat"]
        self.df["lat"]=self.df["lon"]
        self.df["lon"]=aux

        f=self.df["lon"]>-60
        max_lat=self.df.loc[f&f_ap,"lat"].max()
        min_lat=self.df.loc[f&f_ap,"lat"].min()
        max_lon=self.df.loc[f&f_ap,"lon"].max()
        min_lon=self.df.loc[f&f_ap,"lon"].min()
        print(max_lat,min_lat,max_lon,min_lon)

        f_in=(self.df["lat"]<=max_lat) & (self.df["lat"]>=min_lat)&(self.df["lon"]<= max_lon)&(self.df["lon"]>= min_lon)
        f_out=~f_in
        self.df.loc[f_out,["lat","lon"]]=np.nan

   def _imputacion_nan_lat_lon(self):
        coordenadas_medias_l4=self.df.groupby("l3")[["lat","lon"]].mean()
        # print(coordenadas_medias_l4.loc["Palermo"])

        barrios=self.df["l3"].dropna().unique()
        for l in barrios :
            f_l=self.df["l3"] == l
            f_nan = self.df["lat"].isna()
            self.df.loc[f_l &f_nan ,"lat"] = coordenadas_medias_l4.loc[l]["lat"]
            self.df.loc[f_l &f_nan ,"lon"] = coordenadas_medias_l4.loc[l]["lon"]

   def _col_dataset(self):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        self.df["dataset"]=np.zeros(self.df.shape[0])
        self.df.loc[f_ent,"dataset"]=1

   def _eliminacion_nan_lat_lon(self):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        f=((self.df["lat"].notna()) & (self.df["lon"].notna()))
        self.df=self.df.loc[f | f_ap]
    
# Tratamiento ambientes ---------------------------------------------------------------------------------------------------

   def _extraction_rooms_text(self):
        rooms = {k:f"{k} amb" for k in range(1,int(self.df["rooms"].max()+1))}
        for k,v in rooms.items():
            if k == 1 :
                f1=(self.df["title"].str.lower().str.contains(v,na=False)) | (self.df["title"].str.lower().str.contains("monoamb",na=False))
                self.df.loc[f1,"rooms_title"]=k
            else:
                fi=self.df["title"].str.lower().str.contains(v,na=False)
                self.df.loc[fi,"rooms_title"]=k

        rooms = {k:f"{k} amb" for k in range(1,int(self.df["rooms"].max()+1))}

        for k,v in rooms.items():
            if k == 1 :
                f1=(self.df["description"].str.lower().str.contains(v)) | (self.df["description"].str.lower().str.contains("monoamb"))
                self.df.loc[f1,"rooms_description"]=k
            else:
                fi=self.df["description"].str.lower().str.contains(v,na=False)
                self.df.loc[fi,"rooms_description"]=k
   @staticmethod
   def _categorize_rooms(row):
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
   @staticmethod
   def _rooms_def(row):
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

   def _tratamiento_outlayers_ambientes(self):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        
        ambientes = ["rooms_def" , "bedrooms","bathrooms"]
        print(self.prop)
        for amb in ambientes:
            print(f"{amb}-----")
            max_amb = self.df.loc[f_ap,amb].max()
            print(f"Max {amb}_{self.prop} = {max_amb}")
            f_max=self.df[amb]> max_amb
            print(f"Pasando a nan {self.df.loc[ f_max].shape[0]} {amb} de {self.prop}")
            self.df.loc[f_max,amb] = np.nan

   @staticmethod   
   def _imputacion_espacios(row):

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

# Tratamiento surface_covered vs surface_total -----------------------------------------------------------
   def _rotacion_cov_tot(self):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        f=self.df["surface_covered"] > self.df["surface_total"]
        print(f"cantidad cov>tot en ent:{self.df.loc[f & f_ent].shape[0]}")
        print(f"cantidad cov>tot en ap:{self.df.loc[f & f_ap].shape[0]}")

        temp = self.df.loc[f, "surface_covered"].copy()
        self.df.loc[f, "surface_covered"] = self.df.loc[f, "surface_total"]
        self.df.loc[f, "surface_total"] = temp

        f=self.df["surface_covered"] > self.df["surface_total"]
        print(f"cantidad cov>tot en ent después de rotacion:{self.df.loc[f & f_ent].shape[0]}")
        print(f"cantidad cov>tot en ap después de rotacion:{self.df.loc[f & f_ap].shape[0]}")
   
   def _outliers_surface_ap(self,surface):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()

        if self.prop =="Departamento":
            f_lsup_dpto=self.df[surface] > 1000 # Me quedo con el max de ap en covered :) --- corrijo en total
            print(f"valores max de deptos en ap de {surface} a pasar a nan {self.df.loc[f_ap  & f_lsup_dpto,surface].shape[0]}")
            self.df.loc[f_ap  & f_lsup_dpto,surface]= np.nan

            f_linf_dpto = self.df[surface] < 14 # Corrijo el minimo
            print(f"valores min de deptos en ap de {surface} a pasar a nan {self.df.loc[f_ap  & f_linf_dpto,surface].shape[0]}")
            self.df.loc[f_ap  & f_linf_dpto,surface] = np.nan

            print(f"Max y min de deptos en ap : {self.df.loc[f_ap,surface].agg(['max','min'])}")

        elif self.prop =="Casa":
            f_lsup_casa=self.df[surface] > 10000 # Me quedo con el max de ap :)
            print(f"valores max de casas en ap de {surface} a pasar a nan {self.df.loc[f_ap & f_lsup_casa,surface].shape[0]}")
            self.df.loc[f_ap  & f_lsup_casa,surface]= np.nan

            f_linf_casa=self.df[surface] < 20 # Corrijo el minimo
            print(f"valores min de casas en ap de {surface} a pasar a nan {self.df.loc[f_ap  & f_linf_casa,surface].shape[0]}")
            self.df.loc[f_ap  & f_linf_casa,surface]= np.nan
            print(f"Max y min de casas en ap : {self.df.loc[f_ap,surface].agg(['max','min'])}")

        elif self.prop =="Cochera":
            f_lsup_coch=self.df[surface] > 10000 # Me quedo con el max de ap :)
            print(f"valores max de cocheras en ap de {surface} a pasar a nan {self.df.loc[f_ap  & f_lsup_coch,surface].shape[0]}")
            self.df.loc[f_ap  & f_lsup_coch,surface]= np.nan

            f_linf_coch=self.df[surface] < 8 # Corrijo el minimo
            print(f"valores min de cocheras en ap de {surface} a pasar a nan {self.df.loc[f_ap  & f_linf_coch,surface].shape[0]}")
            self.df.loc[f_ap  & f_linf_coch,surface]= np.nan
            print(f"Max y min de cocheras en ap : {self.df.loc[f_ap,surface].agg(['max','min'])}")

   def _imputacion_covered_con_total_viceversa(self):
        f_cov_na = (self.df["surface_covered"].isna()) & (self.df["surface_total"].notna())
        f_tot_na= (self.df["surface_total"].isna()) & (self.df["surface_covered"].notna())
        print(f"nan en covered antes : {self.df['surface_covered'].isna().sum()}")
        print(f"nan en total antes : {self.df['surface_total'].isna().sum()}")

        self.df.loc[f_cov_na,'surface_covered'] = self.df.loc[f_cov_na,'surface_total']
        self.df.loc[f_tot_na,'surface_total'] = self.df.loc[f_tot_na,'surface_covered']
        print(f"nan en covered despues : {self.df['surface_covered'].isna().sum()}")
        print(f"nan en total despues : {self.df['surface_total'].isna().sum()}")

   def _outliers_surface_ent(self):
   
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        
        min_cov = self.df.loc[ f_ap ,"surface_covered"].min()
        max_cov = self.df.loc[f_ap ,"surface_covered"].max()
        min_tot = self.df.loc[ f_ap ,"surface_total"].min()
        max_tot = self.df.loc[ f_ap ,"surface_total"].max()

        print(f"min cov = {min_cov}")
        print(f"max cov = {max_cov}")
        print(f"min tot = {min_tot}")
        print(f"max tot = {max_tot}")

        f_total_out = (self.df["surface_total"]> max_tot) | (self.df["surface_total"]< min_tot)
        f_cov_out = (self.df["surface_covered"]> max_cov) | (self.df["surface_covered"]< min_cov)
        

        f_subset_total_out = f_total_out & f_ent 
        print(f"datos de surface_total a nan: {self.df.loc[f_subset_total_out , 'surface_total'].shape[0]}")
        self.df.loc[f_subset_total_out , "surface_total"] = np.nan

        f_subset_cov_out =f_cov_out & f_ent 
        print(f"datos de surface_covered a nan: {self.df.loc[f_subset_cov_out , 'surface_covered'].shape[0]}")
        self.df.loc[f_subset_cov_out , "surface_covered"] = np.nan

   def _outliers_ratio_cov_total(self):
        f_ap=self.df["price"].isna()
        f_ent=self.df["price"].notna()
        self.df["cov_total"] = self.df["surface_covered"] / self.df["surface_total"] * 100
       
        if self.prop =="Departamento":
            min_cov_total= self.df.loc[f_ap ,"cov_total"].min()
        elif self.prop =="Casa":
            min_cov_total=30
        else:
            min_cov_total=70
            
        print(f"min : {min_cov_total}")
        
        f_lim_inf = self.df["cov_total"] < min_cov_total

        f_out =  f_lim_inf
        print(f"Cantidad de valores pasados a nan: {self.df.loc[f_out ,['surface_covered' , 'surface_total']].shape[0]}")

        self.df.loc[f_out ,["surface_covered" , "surface_total"]] = np.nan
        
        self.df["cov_total"] = self.df["surface_covered"] / self.df["surface_total"] * 100

# Tratamientos outliers price/m2 -----------------------------------------------------------------------------------------------------------
   def _calculo_price_m2_real(self):
        self.df["price_m2_real"]=self.df["price"]/(self.df["surface_covered"]+0.001)
   
   def _outliers_price_m2_bruto(self):
        lim_inf=500
        lim_sup=10000
        f_lim_inf = self.df["price_m2_real"] <lim_inf
        f_lim_sup = self.df["price_m2_real"] > lim_sup
        f_out =  f_lim_inf | f_lim_sup
        f_in = ~f_out
        print(f"shape antes de eliminar primeros outlaiers {self.df.shape}")
        self.df=self.df.loc[f_in]
        print(f"shape despues de eliminar primeros outlaiers {self.df.shape}")

   def _calculo_price_m2_std(self):
        self.df["original_index"] = self.df.index
        self.df["price_m2"] = self.df["price"] / (self.df["surface_total"] + 0.001)
        gb = self.df.groupby(by=[ "l3"])["price_m2"].mean().reset_index()
        self.df = self.df[self.df.columns.drop("price_m2")].merge(gb, on=["l3"], how="left")
        self.df = self.df.set_index("original_index")
        self.df.index.name = "id"
        self.df["price_m2_x_surface_total"] = self.df["price_m2"] * self.df["surface_total"]

   def _outliers_price_m2_error(self):
        self.df["error"] = self.df["price_m2_real"] - self.df["price_m2"]
        print(f"df shape antes:{self.df.shape}")
        f_out=np.abs(self.df["error"])> 1000
        print(f"Cantidad de registros a eliminar: {self.df.loc[f_out].shape}")
        self.df=self.df.loc[~f_out]
        print(f"df shape despues:{self.df.shape}")

   def _outliers_prices_zonap(self):
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
        barrios = self.df["l3"].value_counts().index
        index_out=[]
        for l in barrios :
            print(l)
            f_l = self.df["l3"] == l
            f_out = (self.df["price_m2_real"] <umbrales_por_l3[l]["min"]) | (self.df["price_m2_real"] >umbrales_por_l3[l]["max"])
            print(f"vamos a eliminar : {self.df.loc[f_l & f_out].shape[0]}")
            index_out=index_out + list(self.df.loc[f_l & f_out].index)
        
        print(f"len index_out: {len(index_out)}")
        print(f"shape antes : {self.df.shape}")
        self.df = self.df.loc[~ self.df.index.isin(index_out)]
        print(f"shape despues : {self.df.shape}")

# Tratamientos rooms - surface_covered  -----------------------------------------------------------------------------------------------------------

   def _imputacion_rooms_surface(self,a_imputar=None):
        if a_imputar is None:
            print("Especifica variable a imputar")
        elif a_imputar=="surface_covered":
            print(f"nan surface_covered antes : {self.df['surface_covered'].isna().sum()}")
            gb_rooms = self.df.groupby(["rooms_rec"])["surface_covered"].mean()
            
            rooms=self.df["rooms_rec"].dropna().unique()
            for r in rooms:
                print("***    ***    ***   ***   ***   ***   ***")
                print(r)
                f_r=self.df["rooms_rec"] == r
                f_cov_nan=self.df["surface_covered"].isna()
                if r in gb_rooms.index:
                    print(f"El valor medio de cada room es : {gb_rooms[int(r)]}")
                    self.df.loc[ f_r & f_cov_nan , "surface_covered"] = gb_rooms[int(r)] * r
            print(f"nan surface_covered despues : {self.df['surface_covered'].isna().sum()}")
        elif a_imputar =="rooms_rec":
            print(f"nan rooms_rec antes : {self.df['rooms_rec'].isna().sum()}")
            gb_rooms = self.df.groupby(["rooms_rec"])["surface_covered"].mean()
            f_rooms_nan=(self.df["rooms_rec"].isna()) & (self.df["surface_covered"].notna())
            self.df.loc[f_rooms_nan,"rooms_rec"] = self.df.loc[f_rooms_nan].apply(lambda row : self._imputacion_rooms_con_surface(row,gb_rooms) , axis=1)
            print(f"nan rooms_rec despues : {self.df['rooms_rec'].isna().sum()}")       

   @staticmethod
   def _imputacion_rooms_con_surface(row,gb_rooms):
        gb_df=pd.DataFrame(gb_rooms)
        gb_df=gb_df.reset_index()

        surface=row["surface_covered"]
        gb_df["rooms_x_surface"] = gb_df["rooms_rec"] * gb_df["surface_covered"]
        gb_df["diferencia"] = np.abs(gb_df["rooms_x_surface"] - surface)
        min_dif = gb_df["diferencia"].min()
        rooms = gb_df.loc[gb_df["diferencia"] == min_dif ,"rooms_rec"]
        return rooms.iloc[0]