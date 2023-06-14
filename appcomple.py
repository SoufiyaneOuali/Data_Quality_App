import pandas as pd
import numpy as np
from tkinter import ttk
from tkinter.filedialog import *
import tkinter.scrolledtext as st
from tkinter import * 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype    
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
    
from pandas.api.types import is_string_dtype
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def interacetow(df):
    global liste_of_selected_item
    global liste_percentage
    
    global liste 
    liste=(df.columns.values).tolist()
    # ==================================================================================================
    # ==================================================================================================    
    global list_selected
    list_selected=[]
    
    def second_window(df):
        
        def event_btn():
            if len(list_selected) != 0:
                parachoisir.delete('3.0','100.0')
                parachoisir.insert(END, "\n")
                parachoisir.insert(END,list_selected)
                
                window.destroy()
            else:
                parachoisir.delete('3.0','100.0')
                parachoisir.insert(END, "\n")
                parachoisir.insert(END, "you have to choose input first")
        
            
        def entryselected():
            global list_selected
            list_selected = [listNodes.get(i) for i in listNodes.curselection()]
            selectedoutput.delete('1.0','100.0')
            selectedoutput.insert(END, "\n")
            for i in list_selected :
                selectedoutput.insert(END,i)
                selectedoutput.insert(END,"\n")
                
                
        window = Toplevel()
        window.geometry("500x450")
        window.title("choose input")
        
        frame = Frame(window, width=250, height=250, borderwidth=2, relief=GROOVE, background="gray35" )
        frame.place(x=10,y=50)
        
        label2 = Label(window, text="Input", font=("Arial Black",15), foreground="gray35")
        label2.place(x=60, y=10)
        
        label2 = Label(window, text="Input Chosen", font=("Arial Black",15), foreground="gray35")
        label2.place(x=280, y=10)
        v=StringVar()
        listNodes = Listbox(frame,listvariable=v, width=20, height=15, font=("Helvetica", 12), selectmode=MULTIPLE)
    
        liste =(df.columns.values).tolist()
        j=1
        for i in liste:
            listNodes.insert(j, i)
            j=j+1
            
        listNodes.pack(side="left", fill="y")
        
        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=listNodes.yview)
        scrollbar.pack(side="right", fill="y")
        
        listNodes.config(yscrollcommand=scrollbar.set)
    
        canvas3 = Canvas(window, width=200, height=280, background="gray80", borderwidth=2, relief=GROOVE)
        selectedoutput = st.ScrolledText(canvas3, width=27, height=18, background="white" ,font=("arial ",10))
        
        selectedoutput.place(x=2, y=0) 
        canvas3.place(x=250,y=50)
            
        bouton3 = ttk.Button(window, text="submit", width=8,command=entryselected)
        bouton3.place(x=150, y=380)
        
        bouton3 = ttk.Button(window, text="OK", width=8,command=event_btn)
        bouton3.place(x=250, y=380)
        
        window.mainloop()
        
    
    # ==================================== SSVVMM ==============================================================
    
    global kernel
    global degree_liste
    degree_liste=[]
    global gamma_liste
    gamma_liste=[]
    global coef0_liste
    coef0_liste=[]
    global C_liste
    C_liste=[]
    
    
    global df_init
    
    
    #==============================================================================
    # =============================================================================
    class Model:
        def __init__(self):
            self.df_init = df.copy()
            self.df = df.dropna().reset_index().drop(['index'],axis=1)
            self.df_cod = self.df.copy()
            self.df_finale = df.copy()
            self.df_Result_liste = []
            self.model_liste = []
            self.sc_moyenne_liste = []
            self.std_liste = []
            self.col_dict={}
            self.date_liste=[]
            self.df_test=[]
            self.input=[]
        """    
        def Importer(self):
            global liste
            filename = askopenfilename(title="Ouvrir un fichier",filetypes=[('csv files','.csv'),
                                                                            ("Excel file","*.xlsx"),("Excel file 97-2003","*.xls")])
            df = pd.read_excel(filename)
            df["Index"] = [x for x in range(1, len(df.values)+1)]
            df =df[ ['Index'] + [ col for col in df.columns if col != 'Index' ] ]
            console.delete('4.0','100.0')
            console.insert(END, "\n")
            console.insert(END, "Le fichier est ouvert\n")
            self.df_init = df.copy()
            self.df = df.dropna().reset_index().drop(['index'],axis=1)
            self.df_cod = self.df.copy()
            self.df_finale = df.copy()
            self.update()
            
        """
    # ========================================================================================================
            
        def Afficher_Table(self,df):
                 try:
                     if((df.shape[0] ==0) and (df.shape[1]==0)):
                         console.insert(END, "Please open a file first \n")
                         
                     nb_colomns = df.shape[1]
                     aray = np.arange(1,nb_colomns+1)
                     tupl=tuple(aray)
                    
                     tree = ttk.Treeview(fenetre, columns = tupl, height = 5 , show ="headings")
                     tree.place(x=260, y=3, width=545 , height=420)
                         
                     # Add scrollbar
                     vsb1 = ttk.Scrollbar(fenetre , orient="vertical",command=tree.yview)
                     vsb1.place(x=805, y=3, height=420)
                     tree.configure(yscrollcommand=vsb1.set)
                             
                     vsb2 = ttk.Scrollbar(fenetre , orient="horizontal",command=tree.xview)
                     vsb2.place(x=260, y=410, width=560)
    
                     tree.configure(yscrollcommand=vsb1.set, xscrollcommand=vsb2.set)
                    
                    # Add headings
                     i=1
                     for name_attr in df.columns:
                         tree.heading(i, text = name_attr)
                         i=i+1
    
                        # Define column width
                     for i in range(1,nb_colomns+1):
                         tree.column(i, width = 80)
                
                     for index in range(0,df.shape[0]):
                         enrg = list(df.values[index,])
                         tree.insert('', END, value= enrg)
                
                     console.delete('3.0','100.0')
                     console.insert(END, "\n")
                     n1=df.shape[0]
                     n2=df.shape[1]
                     console.insert(END,"Number of lines  : "+str(n1)+"\n")
                     console.insert(END,"Nombre of colonnes : "+str(n2)+"\n")
                     
                 except:
                     console.delete('3.0','100.0')
                     console.insert(END, "\n")
                     console.insert(END,"Table is not exist yet   \n")
    
              
        def Codding(self):       
            valueDVEncode = str( DVEncode.get())
            valueEncoderT = str( EncoderT.get())
            col=valueDVEncode
            choix=valueEncoderT
            self.col_dict[col]= choix
        
            try:
                if(is_string_dtype(df[col])):
                    if(choix=="OneHotEncoder"):
                        OneHotEncoding = OneHotEncoder(handle_unknown="ignore")
                        a = OneHotEncoding.fit_transform(self.df[[col]]).astype(int) 
                        liste = []
                        liste = [col+'_'+str(i) for i in range(self.df[col].unique().shape[0])]
                        other = pd.DataFrame(data=a.toarray(),columns=liste)
                        self.df_cod = pd.concat([self.df_cod,other],axis=1)
                    if(choix=="OneLabelEncoder"):
                        OrdinalEncoding = OrdinalEncoder()  
                        other = OrdinalEncoding.fit_transform(self.df[[col]]).astype(int)
                        other = pd.DataFrame(other, columns=[col+'_'+'Num'])
                        self.df_cod = pd.concat([self.df_cod,other],axis=1)
                            
                    console.delete('3.0','100.0')
                    console.insert(END, "\n")
                    console.insert(END,"the variable is successfully converted  \n   ")
                
                            
                    parachoisir.delete('3.0','100.0')
                    parachoisir.insert(END, "\n")
                    parachoisir.insert(END,"variable :"+col+"\n")
                    parachoisir.insert(END,"encoding type :"+choix+"\n")
                   
                    
                else:
                    console.delete('3.0','100.0')
                    console.insert(END, "\n")
                    console.insert(END,"the variable is of type int \n")
                    
            except:
                    console.delete('3.0','100.0')
                    console.insert(END, "\n")
                    console.insert(END,"Error! variable not exists \n   ")
                    
            self.update()
            self.Afficher_Table(self.df_cod)
                
    
        def update(self):
            global liste
            liste =(self.df_cod.columns.values).tolist()
            
            DVEncode["value"] =liste
            DVEncode.current(0)
            
            DateVar["value"] =liste
            DateVar.current(0)
            
           
            Output["value"] =liste
            Output.current(0)
    
            corrterget["value"] =liste
            corrterget.current(0)
    
        def Transforme_DATE(self,df,df_cod,col):
            liste_1 = []
            liste_2 = []
            liste_3 = []
            
            from datetime import datetime
            
            for i in range(df.shape[0]):
                date = df[col].iloc[i]
                y = date.year
                m = date.month
                d = date.day
                liste_1.append(y)
                liste_2.append(m)
                liste_3.append(d)
    
            y = pd.Series(liste_1)
            m = pd.Series(liste_2)
            d = pd.Series(liste_3)
            df_date = pd.concat([y,m,d],axis=1) 
            df_date.columns = ['year','month','day']
            df_cod = pd.concat([df_cod,df_date],axis=1)
            return df_cod
        
        def Transforme_DATE_Train(self):
            try:
                valueDateVar = str( DateVar.get())
                col=valueDateVar
                self.date_liste.append(col)
                self.df_cod= self.Transforme_DATE(self.df,self.df_cod,col)
                self.Afficher_Table(self.df_cod)
                
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END,"Date variable is successfully converted \n   ")
                
                parachoisir.delete('3.0','100.0')
                parachoisir.insert(END, "\n")
                parachoisir.insert(END,"Date variable :"+valueDateVar+"\n")
                    
            except:
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END," Error: variable chosen not of type Date  \n   ")
                
        def Codding_Test(self,col,choix=0):
            if(choix==1):
                OneHotEncoding = OneHotEncoder(handle_unknown="ignore")
                OneHotEncoding.fit(self.df[[col]])
                a = OneHotEncoding.transform(self.df_test[[col]]).astype(int)
                liste = []
                liste = [col+'_'+str(i) for i in range(self.df[col].unique().shape[0])]
                other = pd.DataFrame(data=a.toarray(),columns=liste)
                self.df_test = pd.concat([self.df_test,other],axis=1)
            if(choix==0):
                OrdinalEncoding = OrdinalEncoder()  
                OrdinalEncoding.fit(self.df[[col]])
                other = OrdinalEncoding.transform(self.df_test[[col]]).astype(int)
                other = pd.DataFrame(other, columns=[col+'_'+'Num'])
                self.df_test = pd.concat([self.df_test,other],axis=1)
            
            console.insert(END, "\n")
            console.insert(END,"the test variable is successfully converted  \n   ")
                
            a = self.df_test[self.col_y]
            self.df_test = self.df_test.drop([self.col_y],axis=1)
            self.df_test[self.col_y] = a
        
        def Transforme_DATE_Test(self,col):
            self.df_test = self.Transforme_DATE(self.df_test,self.df_test,col)
            console.insert(END, "\n")
            console.insert(END,"the test date variable is converter with success  \n   ")
            a = self.df_test[self.col_y]
            self.df_test = self.df_test.drop([self.col_y],axis=1)
            self.df_test[self.col_y] = a
                
            
        def Correlation_Variable(self):
            try :
                Valuecorrterget=str( corrterget.get())
                corr = self.df_cod.corr()[Valuecorrterget].drop([Valuecorrterget],axis=0)
                
                x = corr.index
                
                fig=plt.figure(2,figsize=(9,7))
                fig=plt.barh(x, corr)
                fig=plt.show()
                canvas = FigureCanvasTkAgg(fig, master=fenetre)
                canvas.get_tk_widget().pack()
                canvas.draw()
                
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END,"Graphical visualization of the correlation of \n variables with the variable target :"+Valuecorrterget+"    \n   ")
                
            except :
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END,"Error !! choose Numerical Variable  \n   ")
                
        def Selection_fetcher(self):
            # global liste_of_selected_item
            
            #ValueInput=str(Input.get())
            global list_selected
            ValueOutput=str(Output.get())
            #cols_X=liste_of_selected_item
            cols_X=list_selected
            col_y =ValueOutput
            
            df_test = self.df_init[self.df_init.isnull()[col_y]].reset_index().drop(['index'],axis=1)
            self.df_test = df_test.drop([col_y],axis=1)
            self.df_test[col_y] = df_test[col_y]
            self.df_test_y = self.df_test.copy()
            
            self.col_y = col_y
            self.cols_X= list_selected
            
            for col in self.col_dict.keys():
                if(self.col_dict[col]=="OneLabelEncoder"):
                    self.Codding_Test(col,0)
                else :
                    self.Codding_Test(col,1)
                    
            
            for col in self.date_liste:
                self.Transforme_DATE_Test(col)
    
            self.y = self.df_cod[col_y].copy()
            self.X = self.df_cod[cols_X].copy()
            
            parachoisir.delete('3.0','100.0')
            parachoisir.insert(END, "\n")
            parachoisir.insert(END, "features chosen to build the model are :\n")
            parachoisir.insert(END, cols_X)
            parachoisir.insert(END, "\n target est :\n")
            parachoisir.insert(END, col_y)
           
           
            console.delete('3.0','100.0')
            console.insert(END, "\n")
            console.insert(END,"variables are successfully selected \n" )
            
            
        def Déscritiser(self):
            #try :
                n_clusters=int(entree_2.get())
                self.n_clusters = n_clusters
                target_array = np.array(self.y).reshape(-1, 1)
                k_means = KMeans(init='k-means++', n_clusters=n_clusters)
                y = k_means.fit_predict(target_array)
                
                self.target = self.y
                self.y = pd.DataFrame(y,columns=['y'])
                self.table = pd.concat([self.df_cod,self.y],axis=1)
                self.Afficher_Table(self.table)
               
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END,"discretization is successfully completed \n" )
    
       
    # Visualisation ************************************************************************************************************************
    
        def Choix_Parametre_listes(self):
            global liste_percentage
            
            name=str(ModelT.get())
            self.algorithme_att = name
            pourc_liste=liste_percentage
            pourc_liste = np.array(pourc_liste)
           
            parachoisir.delete('3.0','100.0')
            parachoisir.insert(END, "\n")
            parachoisir.insert(END,"choosen Model  :"+ name +"\n")
            parachoisir.insert(END,"pourcentage % : ")
            parachoisir.insert(END,pourc_liste)
            
            pourc_liste=[i/100  for i in pourc_liste]
            self.pc_liste_att =pourc_liste
    
            if(self.algorithme_att=='svm'):
                # kernel, degree_liste, gamma_liste, coef0_liste, C_liste = parametre_svm()
                kernel='rbf'
                self.kernel = kernel
                self.kernel = kernel
                degree_liste = [3]
                coef0_liste = [0]
                gamma_liste = [0.1,0.01,0.001,'scale']
                C_liste = [1]
      
                if(kernel=='rbf'):
                    param_dict = {'gamma': gamma_liste, 'C': C_liste}
                if(kernel=='poly'):
                    param_dict = {'degree': degree_liste, 'coef0': coef0_liste, 'C': C_liste}
                if(kernel=='sigmoid'):
                    param_dict = {'gamma': gamma_liste, 'coef0': coef0_liste, 'C': C_liste}
                    
                cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
                model = svm.SVC(kernel=kernel, gamma='scale')
                self.grid_model = GridSearchCV(estimator=model, param_grid=param_dict, cv=cv)
                
            if(self.algorithme_att=='DecisionTree'):
                
                #max_depth_liste, min_samples_leaf_liste, min_samples_split_liste = parametre_DecisionTree()
                
                max_depth_liste = [1,2,3]
                min_samples_leaf_liste = [1,2,3]
                min_samples_split_liste = [2,3]
                param_dict = {'max_depth': max_depth_liste, 'min_samples_leaf': min_samples_leaf_liste, 'min_samples_split': min_samples_split_liste}
                
                cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
                model = DecisionTreeClassifier()
                self.grid_model = GridSearchCV(estimator=model, param_grid=param_dict, cv=cv)
                
            if(self.algorithme_att=='KNeighbors'):
               
                #n_neighbors_liste, p_liste, weights_liste = parametre_KNeighbors()
                
                n_neighbors_liste = [4,5]
                p_liste = [1,2]
                weights_liste = ['uniform']
                param_dict = {'n_neighbors': n_neighbors_liste, 'p': p_liste, 'weights': weights_liste}
                param_dict = {'n_neighbors': n_neighbors_liste, 'p': p_liste, 'weights': weights_liste}
                
                cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
                model = KNeighborsClassifier()
                self.grid_model = GridSearchCV(estimator=model, param_grid=param_dict, cv=cv)
                
            if(self.algorithme_att=='RandomForest'):
                
                #n_estimators, max_depth_liste, min_samples_leaf_liste, min_samples_split_liste = parametre_RandomForest()
                
                n_estimators_liste = [100]
                max_depth_liste = [None,1,2,3]
                min_samples_split_liste = [2,3]
                min_samples_leaf_liste = [1,2,3]
                param_dict = {'n_estimators': n_estimators_liste, 'max_depth': max_depth_liste, 'min_samples_split': min_samples_split_liste, 'min_samples_leaf':min_samples_leaf_liste}
                
                cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
                model = RandomForestClassifier()
                self.grid_model = GridSearchCV(estimator=model, param_grid=param_dict, cv=cv)
                
            
            console.delete('3.0','100.0')
            console.insert(END, "\n")
            console.insert(END,"configuration of parameters is successfully completed\n" )
                
            
           
    
        def Entrainer_listes(self):
            
            if(self.algorithme_att=='svm'):
                self.grid_model.fit(self.X,self.y)
                
                if(self.kernel=='rbf'):
                    p1 = self.grid_model.best_params_['gamma']
                    p2 = self.grid_model.best_params_['C']
                    model = svm.SVC(kernel=self.kernel, gamma=p1, C=p2)
                if(self.kernel=='poly'):
                    p1 = self.grid_model.best_params_['degree']
                    p2 = self.grid_model.best_params_['C']
                    p3 = self.grid_model.best_params_['coef0']
                    model = svm.SVC(kernel=self.kernel, degree=p1, C=p2, coef0=p3, gamma='scale')
                if(self.kernel=='sigmoid'):
                    p1 = self.grid_model.best_params_['gamma']
                    p2 = self.grid_model.best_params_['C']
                    p3 = self.grid_model.best_params_['coef0']
                    model = svm.SVC(kernel=self.kernel, gamma=p1, C=p2, coef0=p3)
                    
            if(self.algorithme_att=='DecisionTree'):
                self.grid_model.fit(self.X,self.y)
                p1 = self.grid_model.best_params_['max_depth']
                p2 = self.grid_model.best_params_['min_samples_leaf']
                p3 = self.grid_model.best_params_['min_samples_split']
                model = DecisionTreeClassifier(max_depth = p1, min_samples_leaf = p2, min_samples_split = p3)
                
            if(self.algorithme_att=='KNeighbors'):
                self.grid_model.fit(self.X,self.y)
                p1 = self.grid_model.best_params_['n_neighbors']
                p2 = self.grid_model.best_params_['p']
                p3 = self.grid_model.best_params_['weights']
                model = KNeighborsClassifier(n_neighbors = p1, p = p2, weights = p3)
                
            if(self.algorithme_att=='RandomForest'):
                self.grid_model.fit(self.X,self.y)
                p1 = self.grid_model.best_params_['n_estimators']
                p2 = self.grid_model.best_params_['max_depth']
                p3 = self.grid_model.best_params_['min_samples_split']
                p4 = self.grid_model.best_params_['min_samples_leaf']
                model = RandomForestClassifier(n_estimators = p1, max_depth = p2, min_samples_split = p3, min_samples_leaf = p4)
                       
            if(self.algorithme_att=='Bayes'):
                model = GaussianNB()
                
            l = []
            sc_train_liste = []
            sc_test_liste = []
            for pc in self.pc_liste_att:
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=pc, random_state=0)
                #np.ravel(y_train,order='C')
                model.fit(X_train,y_train)
                sc_train = model.score(X_train,y_train)
                sc_test = model.score(X_test,y_test)
                sc_train_liste.append(sc_train)
                sc_test_liste.append(sc_test)
                l.append(str(int(pc*100))+" %")
                
            l1 = np.round(sc_train_liste,2)
            l2 = np.round(sc_test_liste,2)
            df_att = pd.DataFrame({'train':l1, 'test':l2,}, index = l)
            std = round(df_att['test'].std(),2)
            sc_moy = round(df_att['test'].mean(),2)
            
            self.sc_moyenne_liste.append(sc_moy)
            self.std_liste.append(std)
            self.model_liste.append(model)
            self.df_Result_liste.append(df_att)
            
            canvaResult = Canvas(fenetre,  width=560, height=420, background="white",
                                   borderwidth=2, relief=GROOVE)
            
            resul = st.ScrolledText(canvaResult, width=200, height=70, background="white" ,font=("arial ",11))  
            resul.insert('1.0', "\t\t\t\t  Result :\n")
            resul.insert(END, "\t\t ******************************************************\n\n")
            resul.insert(END,model)
            resul.insert(END, "\n\n")
            resul.insert(END,df_att)
            
            resul.focus()
            resul.configure(state ='disabled')
            resul.place(x=0, y=0)
            
            canvaResult.place(x=257, y=0)
    
            #print(model,"\n")
            #print(df_att)
            #print("\n La moyenne des scores de test :",sc_moy)
            #print("La variance entre les données :",std)
            
            console.delete('3.0','100.0')
            console.insert(END, "\n")
            console.insert(END,"the Model is being training")
            
            
            score.delete('4.0','100.0')
            score.insert(END, "\n")
            score.insert(END,"AVG test : ")
            score.insert(END,sc_moy)
            score.insert(END, "\n")
            score.insert(END,"STD test :")
            score.insert(END,std)
        
        
        def Comparer_Visualiser(self):
            try:
                canvaResult = Canvas(fenetre,  width=560, height=420, background=canva_color2,
                                       borderwidth=2, relief=GROOVE)
                resul = st.ScrolledText(canvaResult, background="white" ,font=("arial ",11))  
                resul.insert('1.0', "\t\t\t\t  Result :\n")
                resul.insert(END, "\t\t ******************************************************\n\n")
                
                for i in range(len(self.model_liste)):
                    #print("Le model",i+1,":")*
                    j=i+1
                    resul.insert(END, " Model : ")
                    resul.insert(END, j )
                    resul.insert(END, "\n")
                    #print("********************************************************************************************************** ")
                    resul.insert(END, " ************************************************************************")
                    resul.insert(END, "\n")
                    #print(self.model_liste[i],"\n")
                    resul.insert(END,self.model_liste[i])
                    resul.insert(END, "\n")
                    #print(self.df_Result_liste[i])
                    resul.insert(END,self.df_Result_liste[i])
                    resul.insert(END, "\n")
                    #print("\nLa moyenne des scores de test :",self.sc_moyenne_liste[i])
                    resul.insert(END,"tests AVG : ")
                    resul.insert(END, "\n")
                    
                    resul.insert(END,self.sc_moyenne_liste[i])
                    resul.insert(END, "\n")
                    #print("La variance entre les données :",self.std_liste[i],"\n")
                    resul.insert(END,"tests STD : ")
                    resul.insert(END, "\n")
            
                    resul.insert(END,self.std_liste[i])
                    resul.insert(END, "\n")
                        
                resul.focus()
                resul.configure(state ='disabled')
                resul.place(x=0, y=0)
                canvaResult.place(x=257, y=0)
            
                x = [i for i in range(1,len(self.std_liste)+1)]
                y1 = self.sc_moyenne_liste
                y2 = self.std_liste
                fig=plt.figure(1,figsize=(15,5))
                fig=plt.subplot(1,2,1)
                fig=plt.title("Avg Sccor")
                fig=plt.ylabel(" AVG Accurcy ")
                fig=plt.plot(x,y1, 'ro-', label="Model")
                fig=plt.legend()
                fig=plt.show()
                
                fig=plt.subplot(1,2,2)
                fig=plt.title("Std Sccor")
                fig=plt.ylabel("STD Accurcy")
                fig=plt.plot(x,y2, 'o-', label="Model")
                fig=plt.legend()
                fig=plt.show()
                canvas = FigureCanvasTkAgg(fig, master=fenetre)
                canvas.get_tk_widget().pack()
                canvas.draw()
            except:
               console.delete('3.0','100.0')
               console.insert(END, "\n")
               console.insert(END,"Error compare the model is failed!")
            
            
    
    
    # Le model *****************************************************************************************************************************
        
        def Choix_Parametre(self):
            
            try:
                name=str(Modelchoi.get())
                self.algorithme= name
                
                if(self.algorithme=='svm'):
                    #kernel, degree_liste, gamma_liste, coef0_liste, C_liste = parametre_svm()
                    C = 1
                    kernel = 'rbf'
                    degree = 3
                    coef0 = 0
                    gamma = 'scale'
                    self.model = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C)
                    
                if(self.algorithme=='DecisionTree'):
                    #max_depth_liste, min_samples_leaf_liste, min_samples_split_liste = parametre_DecisionTree()
                    max_depth = None
                    min_samples_leaf = 1
                    min_samples_split = 2
                    self.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
                
                if(self.algorithme=='KNeighbors'):
                    #n_neighbors_liste, p_liste, weights_liste = parametre_KNeighbors()
                    n_neighbors = 5
                    p = 2
                    weights = 'uniform'
                    self.model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights)
               
                if(self.algorithme=='Bayes'):
                    self.model = GaussianNB()
                
                if(self.algorithme=='RandomForest'):
                    #n_estimators, max_depth_liste, min_samples_leaf_liste, min_samples_split_liste = parametre_RandomForest()
                    n_estimators = 100
                    max_depth = None
                    min_samples_split = 2
                    min_samples_leaf = 1
                    self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    
                parachoisir.delete('3.0','100.0')
                parachoisir.insert(END, "\n")
                parachoisir.insert(END," The chosen model is : ")
                parachoisir.insert(END, "\n")
                parachoisir.insert(END,name)
    
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END,"the train model is successfully configured \n" )
    
            except:
                console.delete('3.0','100.0')
                console.insert(END, "\n")
                console.insert(END,"Error in the train model phase !! \n" )
            
    
        def Entrainer(self):
            self.model.fit(self.X,self.y)
                
        def Tester(self):
            self.Entrainer()
            sc = self.model.score(self.X,self.y)
            sc = round(sc,2)
            console.delete('3.0','100.0')
            console.insert(END, "\n")
            console.insert(END," The training is over \n")
            console.insert(END, "\n")
            console.insert(END," The test on all of the data: \n")
            
            score.delete('4.0','100.0')
            score.insert(END, "\n")
            score.insert(END," Accurcy ")
            score.insert(END, "\n")
            score.insert(END,sc)
            score.insert(END, " %")
            
    # Appliquer ****************************************************************************************************************************
    
    
            
        def Appliquer(self):
            console.delete('3.0','100.0')
            console.insert(END,"\n")
            console.insert(END," model application")
            console.insert(END,"\n")
    
            self.X_test_finale = self.df_test[self.cols_X].copy()
            
            y_predict = self.model.predict(self.X_test_finale)
            y_predict = pd.DataFrame(y_predict,columns=['y'])
            self.df_test_y = pd.concat([self.df_test,y_predict],axis=1)
            
            self.Afficher_Table(self.df_test_y)
            
            console.insert(END," the model is successfully applied")
    
    
    
        def Remplire(self):
            df = pd.concat([self.target,self.y],axis=1)
            m = []
            m = [round(df[df['y']==k][self.col_y].mean(),2) for k in range(self.n_clusters)]
            self.df_m = pd.DataFrame(m,columns=['mean'])
            for i in range(self.df_test.shape[0]):
                y = self.df_test_y['y'].iloc[i]
                self.df_test_y.loc[i,self.col_y] = self.df_m.loc[y,'mean']
            self.Afficher_Table(self.df_test_y)
            
            console.delete('3.0','100.0')
            console.insert(END,"\n")
            console.insert(END," the target is well Fill")
            
        def Sauvgarder(self):
            df_y = self.df_test_y.copy()
            df_y.index = self.df_finale[self.df_finale.isnull()[self.col_y]].index
            for i in df_y.index:
                self.df_finale.loc[i,self.col_y] = df_y.loc[i,self.col_y]
                
            self.Afficher_Table(self.df_finale)
            
            self.df_finale.to_csv('out.csv', index=False)  
            
            console.delete('3.0','100.0')
            console.insert(END,"\n")
            console.insert(END," the target is well Fill")
            
    
    
            
    
            
            
            
            
            
    Model = Model() 
    #liste =(df.columns.values).tolist()
    
    canva_color2 = "#FCFCFC"
    
    
    global colname
    global rowname
    global L_combobox
    # ****************************************************************************************************************************
    def about_us():
         mess1="\t \t \t   DATA QUALITY \n "
         mess="about us  : \n \n DataQuality App for Educatif porpuse "
        #messagebox.showinfo(title="Aout Us", message=mess)
         fenetrea = Tk()
         fenetrea.resizable(width=False, height=False)
         fenetrea.title("About Us")
         fenetrea.geometry("800x500")
        
         canvas2a = Canvas(fenetrea, width=750, height=450, background="white", borderwidth=2, relief=GROOVE)
         canvas2a.place(x=20,y=20)
         text=Text(canvas2a,width=68,height=21,font=("Arial",14),background="gray95",foreground="black")
         #text = st.ScrolledText(canvas2)  
         text.place(x=20,y=2000)
       
         text.pack()
         text.config(state="normal")
         text.insert(END, mess1)
         text.insert(END, mess)
    
         fenetrea.mainloop()
         
    #df=pd.read_excel("employe.xlsx")
    
    # fentre tow ---------------------------------------------------------------------------------
    fenetre = Tk()
    
    #class begiin 
    
    #-----------------------------------
    
    bouton_color = "#DEDEDE"
        
    fenetre.title("DATA QUALITY")
    fenetre.geometry("1080x650")
    # fenetre.iconbitmap("Logo.ico")
    fenetre.resizable(width=False, height=False)
    # Menu
    menu_G = Menu(fenetre)
    fenetre.config(menu=menu_G)
    menu1 = Menu(menu_G, tearoff=0)
    menub = Menu(menu_G, tearoff=0)
    menu_G.add_cascade(label="File", menu=menu1) 
    menu1.add_command(label="Import", command=lambda:Model.Importer()) 
    menu1.add_separator()
    menu1.add_command(label="Exit", command=fenetre.destroy)
    menu_G.add_cascade(label="About us ", menu=menub)
    menub.add_command(label="G.info", command=about_us)
    
    
    #-------------------------------------------------------------------------------------------------
    def userText(event):
        entree_1.delete(0,END)
        usercheck=True
        
    def userTextb(event):
        entree_2.delete(0,END)
        usercheck=True    
    
    liste_of_selected_item=[]
    
    def selceted(event):
        global liste_of_selected_item
        sele =Input.get()
        if(sele in liste_of_selected_item):
            liste_of_selected_item.remove(sele)
            console.delete('3.0','100.0')
            console.insert(END, "\n")
            console.insert(END," delted   :\n")
        else:
            liste_of_selected_item.append(sele)
               
        parachoisir.delete('3.0','100.0')
        parachoisir.insert(END, "\n")
        parachoisir.insert(END,liste_of_selected_item)
    
    liste_percentage=[]
    def pourcentage_selceted(event):
        global liste_percentage
        sele =float(testpercentage.get())
        if(sele in liste_percentage):
            liste_percentage.remove(sele)
            console.delete('3.0','100.0')
            console.insert(END, "\n")
            console.insert(END," delted   :\n")
        else:
            liste_percentage.append(sele)
               
        parachoisir.delete('3.0','100.0')
        parachoisir.insert(END, "\n")
        parachoisir.insert(END,liste_percentage) 
        
    def checkcmbo():
            value = str( L_combobox.get()) 
            fildir.insert(0 ,value)
        
    def config():
        pass
    
    def test():
        pass
    def fonc(event):
        pass
    
    bouton_text_size1 = 10
    
    

    # Variable encoding ----------------------------------------------------------------------------------------
    
    frame1 = Frame(fenetre, width=250, height=251, borderwidth=2, relief=GROOVE, background="gray35" )
    # Label
    label2 = Label(frame1, text="Variable Encoding", font=("Arial Black",15), background="gray35",foreground="White Smoke")
    label2.place(x=16, y=3)
    
    label2 = Label(frame1, text="Discreet Variable:", font=("Arial Black",13), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=35)
    DVEname = StringVar()
    
    label2 = Label(frame1, text="Variable  :", font=("Arial Black",10), background="gray35",foreground="White Smoke")
    label2.place(x=10,y=70)
    DVEncode = ttk.Combobox(frame1, textvariable=DVEname, width=20,state='readonly')
    DVEncode.place(x=95, y=70)
    DVEncode.bind("<<ComboboxSelected>>", fonc )
    DVEncode["value"] =liste
    DVEncode.current(0)
    
    
    encodtype = StringVar()
    label2 = Label(frame1, text="Encoding :", font=("Arial Black",10), background="gray35",foreground="White Smoke")
    label2.place(x=10,  y=100)
    colname = StringVar()
    EncoderT = ttk.Combobox(frame1, textvariable=encodtype, width=20,state='readonly')
    EncoderT.place(x=95, y=100)  
    a=["OneLabelEncoder","OneHotEncoder"]
    EncoderT["value"] =a
    EncoderT.current(0)           
    EncoderT.bind("<<ComboboxSelected>>")
    
    
    Encodone=Button(frame1, text="Encode", font=("arial ",10), width=8, height=1, background="gray30", fg='White'
                    ,command=lambda:Model.Codding())
    Encodone.place(x=80, y=130)
    
    label2 = Label(frame1, text="Date Varaiable:", font=("Arial Black",13), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=157)
    
    label2 = Label(frame1, text="Varaiable  :", font=("Arial Black",10), background="gray35",foreground="White Smoke")
    label2.place(x=10,y=188)
    
    dateva= StringVar()
    DateVar = ttk.Combobox(frame1, textvariable=dateva, width=20,state='readonly')
    DateVar.place(x=95, y=188)
    DateVar.bind("<<ComboboxSelected>>", fonc )
    DateVar["value"] =liste
    DateVar.current(2)
    
    
    Encodtwo=Button(frame1, text="Encode", font=("arial ",10), width=8, height=1, background="gray30", fg='White'
                    , command=lambda: Model.Transforme_DATE_Train())
    Encodtwo.place(x=80, y=216)
    
    frame1.place(x=0, y=0)
    
    # End encodage data ----------------------------------------------------------------------------------------
    
    # Visualizer data ----------------------------------------------------------------------------------------
    #liste =(df.columns.values).tolist()
    bouton_text_size3 = 10
    frame2 = Frame(fenetre, width=250, height=87, borderwidth=2, relief=GROOVE, background="gray35")
    
    label3 = Label(frame2, text="Correlation Table ",font=("Arial Black",14), background="gray35",foreground="White Smoke")
    label3.place(x=16, y=0)
    
    inputvall = StringVar()
    
    label2 = Label(frame2, text="Target  :", font=("Arial Black",10), background="gray35",foreground="White Smoke")
    label2.place(x=10,y=30)
    corrterget = ttk.Combobox(frame2, textvariable=inputvall, width=20,state='readonly')
    corrterget.place(x=95, y=30)
    corrterget.bind("<<ComboboxSelected>>", fonc )
    corrterget["value"] =liste
    corrterget.current(2)
    
    bouton3=Button(frame2, text="View", font=("arial ",10), width=8,  height=1, background="gray30", fg='White'
                   , command=lambda: Model.Correlation_Variable())
    bouton3.place(x=80, y=54)
    
    frame2.place(x=0,y=252)
    
    # END  Visualizer data -----------------------------------------------------------------------------
    # choose input and output ----------------------------------------------------------------------
    frame3 = Frame(fenetre, width=250, height=146, borderwidth=2, relief=GROOVE, background="gray35" )
    
    label2 = Label(frame3, text="Variable module", font=("Arial Black",15), background="gray35",foreground="White Smoke")
    label2.place(x=25, y=3)
    label2 = Label(frame3, text="Input :", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=10, y=40)
    
    """
    inputval = StringVar()
    Input = ttk.Combobox(frame3, textvariable=inputval, width=20,state='readonly')
    Input.place(x=95, y=45)
    Input.bind("<<ComboboxSelected>>", selceted )
    Input["value"] =liste
    Input.current(0)
    """
    
    bouton3=Button(frame3, text="choose", font=("arial Black ",9), width=15, height=1, background="White", fg='gray30'
                   , command=lambda: second_window( Model.df_cod))
    bouton3.place(x=110, y=45)
    
    
    label2 = Label(frame3, text="Output :", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=10, y=75)
    
    outputval = StringVar()
    Output = ttk.Combobox(frame3, textvariable=outputval, width=20,state='readonly')
    Output.place(x=95, y=80)
    Output.bind("<<ComboboxSelected>>", fonc )
    Output["value"] =liste
    Output.current(0)
    
    bouton3=Button(frame3, text="Submit", font=("arial ",12), width=10, height=1, background="gray30", fg='White'
                   , command=lambda: Model.Selection_fetcher())
    bouton3.place(x=80, y=110)
    
    frame3.place(x=0, y=340)
    
    frame3a = Frame(fenetre, width=250, height=117, borderwidth=2, relief=GROOVE, background="gray35" )
    label2 = Label(frame3a, text="Discretize the target:", font=("Arial Black",14), background="gray35",foreground="White Smoke")
    label2.place(x=16, y=0)
    
    label2 = Label(frame3a, text="Number of class:", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=10, y=27)
    
    classnomber = StringVar()
    entree_2 = Entry(frame3a, textvariable=classnomber, width=24,background="white",foreground="black", 
                            font=("Arial Rounded MT Bold ",9))
    entree_2.place(x=34, y=57)
    entree_2.insert(0,"5")
    entree_2.bind("<Button>",userTextb)
    
    bouton3=Button(frame3a, text="Discretize", font=("arial ",12), width=10, height=1, background="gray30", fg='White'
                   , command=lambda:Model.Déscritiser() )
    bouton3.place(x=80, y=80)
    frame3a.place(x=0, y=487)
    # END choose input and output ----------------------------------------------------------------------
    
    bouton_text_size2 = 10
    
    # Test Model ----------------------------------------------------------------------
    bouton_text_size2 = 10
    frame2 = Frame(fenetre, width=250, height=210, borderwidth=2, relief=GROOVE, background="gray35")
    # Label
    label2 = Label(frame2, text="Test the Modules:", font=("Arial Black",14), background="gray35",foreground="White Smoke")
    label2.place(x=25, y=3)
    
    label2 = Label(frame2, text="Modules:", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=35)
    
    modeltype = StringVar()
    ModelT = ttk.Combobox(frame2, textvariable=modeltype, width=25,state='readonly')
    ModelT.place(x=34, y=65)
    a=["svm","DecisionTree","KNeighbors","Bayes","RandomForest"]
    ModelT["value"] =a
    ModelT.current(0) 
    ModelT.bind("<<ComboboxSelected>>", fonc )
    
    label2 = Label(frame2, text="test percentage:", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=85)
    
    testpercen = StringVar()
    testpercentage = ttk.Combobox(frame2, textvariable=testpercen, width=25,state='readonly')
    testpercentage.place(x=34, y=115)
    a=["5","10","15","20","25"]
    testpercentage["value"] =a
    testpercentage.current(0) 
    testpercentage.bind("<<ComboboxSelected>>", pourcentage_selceted )
    
    
    label2 = Label(frame2, text="module parameter:", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=135)
    bouton3=Button(frame2, text="config", font=("arial ",12), width=9, height=1, background="gray30", fg='White'
                   , command=lambda: Model.Choix_Parametre_listes())
    
    bouton3.place(x=20, y=165)
    
    bouton3=Button(frame2, text="Trainer", font=("arial ",12), width=9, height=1, background="gray30", fg='White'
                   , command=lambda: Model.Entrainer_listes())
    bouton3.place(x=120, y=165)
    
    
    frame2.place(x=830,y=0)
    # END Test Model ----------------------------------------------------------------------
    #Compare MODELE ----------------------------------------------------------------------------------------
    frame3B = Frame(fenetre, width=250, height=80, borderwidth=2, relief=GROOVE,background="gray35")
    
    label3 = Label(frame3B, text="Compare Modules ",font=("Arial Black",13), background="gray35",foreground="White Smoke")
    label3.place(x=40, y=0)
    
    bouton3=Button(frame3B, text="Compare", font=("arial ",12), width=9, height=1, background="gray30", fg='White'
                   , command=lambda: Model.Comparer_Visualiser())
    bouton3.place(x=80, y=35)
    
    frame3B.place(x=830,y=211)
    
    #end compare model ----------------------------------------------------------------------------------------
    
    # choose Model ----------------------------------------------------------------------
    bouton_text_size2 = 10
    frame2 = Frame(fenetre, width=250, height=174, borderwidth=2, relief=GROOVE, background="gray35")
    # Label
    label2 = Label(frame2, text="choose a Module :", font=("Arial Black",14), background="gray35",foreground="White Smoke")
    label2.place(x=25, y=3)
    
    label2 = Label(frame2, text="Modules:", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=35)
    
    madelchoix = StringVar()
    Modelchoi = ttk.Combobox(frame2, textvariable=madelchoix, width=25,state='readonly')
    Modelchoi.place(x=34, y=65)
    a=["svm","DecisionTree","KNeighbors","Bayes","RandomForest"]
    Modelchoi["value"] =a
    Modelchoi.current(0) 
    Modelchoi.bind("<<ComboboxSelected>>", fonc )
    
    
    label2 = Label(frame2, text="module parameter:", font=("Arial Black",12), background="gray35",foreground="White Smoke")
    label2.place(x=5, y=90)
    bouton3=Button(frame2, text="config", font=("arial ",12), width=9, height=1, background="gray30", fg='White'
                   , command=lambda: Model.Choix_Parametre())
    bouton3.place(x=20, y=125)
    
    bouton3=Button(frame2, text="Trainer", font=("arial ",12), width=9, height=1, background="gray30", fg='White'
                   , command=lambda: Model.Tester())
    bouton3.place(x=120, y=125)
    
    frame2.place(x=830,y=292)
    
    # # END chossen Model ----------------------------------------------------------------------
    
    #CHOOSE FEATURE TO APPLIQUE MODELE ----------------------------------------------------------------------------------------
    frame3B = Frame(fenetre, width=250, height=134, borderwidth=2, relief=GROOVE,background="gray35")
    
    label3 = Label(frame3B, text="Fill Target ",font=("Arial Black",15), background="gray35",foreground="White Smoke")
    label3.place(x=60, y=0)
    
    
    bouton3=Button(frame3B, text="Apply", font=("arial ",13), width=12, height=1, background="gray30",
                   fg='White', command=lambda: Model.Appliquer())
    bouton3.place(x=60, y=40)
    
    bouton3=Button(frame3B, text="Replenish", font=("arial ",13), width=12, height=1, background="gray30",
                   fg='White', command=lambda:Model.Remplire())
    bouton3.place(x=60, y=80)
    
    frame3B.place(x=830,y=468)
    
    #CHOOSE FEATURE TO APPLIQUE MODELE ----------------------------------------------------------------------------------------
    frame4 = Frame(fenetre, width=1100, height=50, borderwidth=2, relief=GROOVE,background="gray35")
    
    bouton4=Button(frame4, text="Initial table",font=("arial ",13), width=10, height=1, background="gray30", fg='White',
                   command=lambda:Model.Afficher_Table(Model.df_init))
    bouton4.place(x=80, y=5)
    
    bouton4=Button(frame4, text="Empty Table ",font=("arial ",13), width=10, height=1, background="gray30", fg='White',
                   command=lambda:Model.Afficher_Table(Model.df_test))
    bouton4.place(x=830, y=5)
    
    bouton4=Button(frame4, text="Save",font=("arial ",13), width=10, height=1, background="gray30", fg='White',
                   command=lambda:Model.Sauvgarder() )
    bouton4.place(x=953, y=5)
    label1 = Label(frame4, 
                   text="Created , Developed and Designed by : Anass Houdou , Ouali Soufiyane , Jai Otman \n Directed By: Prof .El far Mohamed   BDSAS  2019/2020",
                   font=("Arial ",9), background="gray35",foreground="White Smoke")
    label1.place(x=300, y=5)
    frame4.place(x=0, y=605)
    
    #----------------------------------------------------------------------------------------
    # Canva zone de tableua
    canva_color1 = "#FCFCFC"
    canvas1 = Canvas(fenetre, width=560, height=420, background=canva_color1, borderwidth=2, relief=GROOVE)
    canvas1.place(x=257, y=0)
    # Canva zone sortie
    
    #para ,console ---------------------------------------------------------
    canvas3 = Canvas(fenetre, width=360, height=160, background="gray80", borderwidth=2, relief=GROOVE)
    
    parachoisir = st.ScrolledText(canvas3, width=50, height=5, background=canva_color2 ,font=("arial ",10))  
    parachoisir.insert('1.0', "chosen parameter:\n")
    parachoisir.insert(END, "***********************\n\n")
    parachoisir.place(x=2, y=0)
    
    console = st.ScrolledText(canvas3, width=50, height=5, background=canva_color2 ,font=("arial ",10))  
    console.insert('1.0', "console :\n")
    console.insert(END, "************\n\n")
    console.place(x=2, y=86)
    
    canvas3.place(x=257,y=434)
    #END :para ,console ---------------------------------------------------------
    #SCORE ---------------------------------------------------------
    canvas2 = Canvas(fenetre, width=186, height=161, background=canva_color2, borderwidth=2, relief=GROOVE)
    
    score = st.ScrolledText(canvas2, width=21, height=9, background=canva_color2 ,font=("arial ",11))  
    score.insert('1.0', "\t score :\n")
    score.insert(END, "           ***************\n\n")
    score.place(x=2, y=2)
    
    canvas2.place(x=628,y=433)
    
    #end SCORE ---------------------------------------------------------
    fenetre.mainloop()
