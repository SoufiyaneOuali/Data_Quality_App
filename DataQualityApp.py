import pandas as pd
import numpy as np
from tkinter import ttk
from tkinter.filedialog import *
import tkinter.scrolledtext as st
from tkinter import * 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

canva_color2 = "#FCFCFC"


global colname
global rowname
global L_combobox

global df
        
from appcomple import *   

# ****************************************************************************************************************************
def Importer():
    global df
    filename = askopenfilename(title="Ouvrir un fichier",filetypes=[('csv files','.csv'),
                                                                    ("Excel file","*.xlsx"),("Excel file 97-2003","*.xls")])
    df = pd.read_excel(filename)
    df["Index"] = [x for x in range(1, len(df.values)+1)]
    df =df[ ['Index'] + [ col for col in df.columns if col != 'Index' ] ]
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    text.insert(END, "Le fichier est ouvert\n")
    
    liste =(df.columns.values).tolist()
    a=["None"]
    L_combobox["value"] =a+liste
    L_combobox.current(0)

    
# ****************************************************************************************************************************
def Table():
    global df
    if((df.shape[0] ==0) and (df.shape[1]==0)):
        text.insert(END, "SVP ouvrir un fichier tout d'abord \n")
        
    nb_colomns = df.shape[1]
    aray = np.arange(1,nb_colomns+1)
    tupl=tuple(aray)
    
    tree = ttk.Treeview(fenetre, columns = tupl, height = 5 , show ="headings")
    tree.place(x=260, y=3, width=800, height=425)
    
    # Add scrollbar
    vsb1 = ttk.Scrollbar(fenetre , orient="vertical",command=tree.yview)
    vsb1.place(x=1050, y=3, height=430)
    tree.configure(yscrollcommand=vsb1.set)

    vsb2 = ttk.Scrollbar(fenetre , orient="horizontal",command=tree.xview)
    vsb2.place(x=260, y=420, width=800)

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
    
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    
    n1=df.shape[0]
    n2=df.shape[1]
    text.insert(END,"Nombre de lignes   : "+str(n1)+"\n")
    text.insert(END,"Nombre de colonnes : "+str(n2)+"\n")
    
    
# ****************************************************************************************************************************
"""
def duplicate():
    global df
    da =df.drop(columns ="Index")
    da = da[da.duplicated(keep=False)]
    da["Index"] = df["Index"]
    da =da[ ['Index'] + [ col for col in da.columns if col != 'Index' ] ]
    df_double=pd.DataFrame(da)
    
    dab =df_double.drop(columns ="Index")
    duprecord = dab.groupby(dab.columns.tolist()).size().reset_index().rename(columns={0:'records'})
    df_reco=pd.DataFrame(duprecord)
    nb=len(df_double)
    return df_double, df_reco, nb
"""

def duplicate():
    global df
    df.drop(columns ="Index", inplace=True)
    duplicateRowsDF = df[df.duplicated(keep=False)]
    df_double=pd.DataFrame(duplicateRowsDF)
    duprecord = df_double.groupby(df_double.columns.tolist()).size().reset_index().rename(columns={0:'records'})
    df_reco=pd.DataFrame(duprecord)

    df["Index"] = [x for x in range(1, len(df.values)+1)]
    df =df[ ['Index'] + [ col for col in df.columns if col != 'Index' ] ]
    #number of dublicate of each rows 
    nb = len(df_double)
    return df_double, df_reco, nb  
    
def Lignes_doublouns():
    
    df_double, df_reco ,nb = duplicate()
    nb_colomns = df_double.shape[1]
    aray = np.arange(1,nb_colomns+1)
    tupl=tuple(aray)
    
    tree = ttk.Treeview(fenetre, columns = tupl, height = 5 , show ="headings")
    tree.place(x=260, y=3, width=800, height=425)
    
    # Add scrollbar
    vsb1 = ttk.Scrollbar(fenetre , orient="vertical",command=tree.yview)
    vsb1.place(x=1050, y=3, height=430)
    tree.configure(yscrollcommand=vsb1.set)

    vsb2 = ttk.Scrollbar(fenetre , orient="horizontal",command=tree.xview)
    vsb2.place(x=260, y=420, width=800)

    tree.configure(yscrollcommand=vsb1.set, xscrollcommand=vsb2.set)
    
    # Add headings
    i=1
    for name_attr in df_double.columns:
        tree.heading(i, text = name_attr)
        i=i+1

    # Define column width
    for i in range(1,nb_colomns+1):
        tree.column(i, width = 80)

    for index in range(0,df_double.shape[0]):   
        enrg = list(df_double.values[index,])
        tree.insert('', END, value= enrg)
    
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    
    if(df_reco.shape[0]==0):
        text.insert(END, "Il ñ y a pas des lignes doublons dans la table\n")
        
    if(df_reco.shape[0]!=0):
        text.insert(END, "Les lignes doublouns : \n")
        text.insert(END, str(df_reco)+"\n")
        
        
    
# ****************************************************************************************************************************
def rowmissing():   
    df_incompl = df[df.isnull().values.any(axis=1)]
    nb = len(df_incompl)
    return df_incompl, nb
       
def Lignes_incomplétudes():
    

    df_incompl,nb = rowmissing() 
    
    nb_colomns = df_incompl.shape[1]
    aray = np.arange(1,nb_colomns+1)
    tupl=tuple(aray)
    
    tree = ttk.Treeview(fenetre, columns = tupl, height = 5 , show ="headings")
    tree.place(x=260, y=3, width=800, height=425)
    
    # Add scrollbar
    vsb1 = ttk.Scrollbar(fenetre , orient="vertical",command=tree.yview)
    vsb1.place(x=1050, y=3, height=430)
    tree.configure(yscrollcommand=vsb1.set)

    vsb2 = ttk.Scrollbar(fenetre , orient="horizontal",command=tree.xview)
    vsb2.place(x=260, y=420, width=800)

    tree.configure(yscrollcommand=vsb1.set, xscrollcommand=vsb2.set)
    
    # Add headings
    i=1
    for name_attr in df_incompl.columns:
        tree.heading(i, text = name_attr)
        i=i+1

    # Define column width
    for i in range(1,nb_colomns+1):
        tree.column(i, width = 80)

    for index in range(0,df_incompl.shape[0]):  
        enrg = list(df_incompl.values[index,])
        tree.insert('', END, value= enrg)
    
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    
    if(df_incompl.shape[0]==0):
        text.insert(END, "Il ñ y a pas des lignes incomplet dans la table\n")
        
    if(df_incompl.shape[0]!=0):
        text.insert(END, "Les lignes incomplétudes :\n")
        text.insert(END, str(df_incompl)+"\n")
    
    
    
     
# ****************************************************************************************************************************
def détaille():
    
    a, b, n3 = duplicate()
    c, n4 = rowmissing() 
    n5=0
    n6 = n3 + n4 + n5
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    text.insert(END,"Nombre des doublouns     : "+str(n3)+"\n")
    text.insert(END,"Nombre des incomplétudes : "+str(n4)+"\n")
    text.insert(END,"Nombre des incohérences  : "+str(n5)+"\n")
    text.insert(END,"-----------------------------\n")
    text.insert(END,"Nombre des invalidités   : "+str(n6)+"\n")
    
    
# ****************************************************************************************************************************   
def Afficher():    
    v = valuea.get()
    if(v==1):
        Table()
    elif(v==2):
        Lignes_doublouns()
    elif(v==3):
        Lignes_incomplétudes()
    elif(v==4):
        détaille()
    else:
        text.delete('4.0','100.0')
        text.insert(END, "\n")
        text.insert(END, "choisir une option  ..\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# ****************************************************************************************************************************
    

    
    
def Supprimer():
    global df 
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    namecol=str(colname.get())
    namerow=str(rowname.get())
    valuecol = str( L_combobox.get()) 
    text.insert(END, "les données que vous avez entré est : ("+ valuecol + ", "+namerow+") . \n\n")
    text.insert(END, "l'état : \n")
    
    if( ( (len(namerow)==0) and (valuecol=="None") ) or ( ((namerow)=="row value") and (valuecol=="None")) or (valuecol=="None") or (len(valuecol)==0) ):
        text.insert(END, "SVP choisir un column ou une ligne\n")
    elif((len(namerow)==0) and (valuecol!="Index")):
        if (valuecol in df.columns):
            df.drop(columns=valuecol, inplace=True)
            text.insert(END, "La suppression est terminée avec succès\n")
        else:
            text.insert(END, "la colonne ne se trouve pas dans la table\n")
    elif((len(namerow)==0) and (valuecol=="Index")):
            text.insert(END, "error: Tu peut pas supprimer l'index\n")
    else :
        if((type(namerow)==str) and (is_string_dtype(df[valuecol]))):
            if (df[valuecol].str.contains(namerow).any()):
                indexNames = df[ ( df[valuecol]==namerow )].index
                df.drop(indexNames, inplace=True)
                text.insert(END,"La suppression est terminée avec succès\n")
            else:
               text.insert(END,"La ligne que vous avez entré ne se trouve pas dans la table\n")  
        elif((type(namerow)==str) and (is_numeric_dtype(df[valuecol]))):
            try : 
                if (int(namerow) in df[valuecol].values ):
                    indexNames = df[ ( df[valuecol]==int(namerow) )].index
                    df.drop(indexNames, inplace=True)
                    text.insert(END,"La suppression est terminée avec succès\n") 
                else:
                    text.insert(END,"La ligne que vous avez entré ne se trouve pas dans la table \n")
            except: 
                text.insert(END,"error: col de type int , valeur de ligne string !!")
    
   # else :
    #    print("La suppression est échouee \n")
                
            
def Supprimer_doubl():
    global df
    text.delete('4.0','100.0')
    text.insert(END, "\n")
    df.drop(columns ="Index",inplace=True)
    df.drop_duplicates(keep="first", inplace=True)
    df["Index"] = [x for x in range(1, len(df.values)+1)]
    df =df[ ['Index'] + [ col for col in df.columns if col != 'Index' ] ]    
    text.insert(END, "Les lignes doublouns sont supprimer\n")


# ****************************************************************************************************************************


def Rempfentere():
    global df
    fenetre.destroy()
    interacetow(df)
    
def Exécuter():
    
    v = value2.get()
    
    if(v==5):
        Supprimer()
    elif(v==6):
        Supprimer_doubl()
    elif(v==7):
        Rempfentere()
    else:
        text.delete('4.0','100.0')
        text.insert(END, "\n")
        text.insert(END, "choisir une option  ..\n")
    
    
   
    
    
    
    
    
    
# ****************************************************************************************************************************
def Vérifier():
    
    a, b, n3 = duplicate()
    c, n4 = rowmissing() 
    n5=0
    n6 = n3 + n4 + n5 
    text.delete('4.0','100.0')
    text.insert(END,"\n")
    if(n6==0):
        text.insert(END,"Le fichier est valide\n")
    else:
        text.insert(END,"Le fichier n'est pas valide\n")

        
# ****************************************************************************************************************************
def Enregistrer():
    global df
    df.to_csv('out.csv', index=False)
    text.delete('4.0','100.0')
    text.insert(END,"\n")
    text.insert(END,"Le fichier est bien enregistrer\n")
    

#((((((((((((((((((((( fenetre ))))))))))))))))))))) 
#((((((((((((((((((((( fenetre ))))))))))))))))))))) 

    
    
    
from tkinter import messagebox   
    
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
    
bouton_color = "#DEDEDE"

fenetre = Tk()

    
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
menu1.add_command(label="Import", command=Importer) 
menu1.add_separator()
menu1.add_command(label="Exit", command=fenetre.destroy)
menu_G.add_cascade(label="About us ", menu=menub)
menub.add_command(label="G.info", command=about_us)

# Frame 1
bouton_text_size1 = 10

frame1 = Frame(fenetre, width=250, height=250, borderwidth=2, relief=GROOVE, background="gray35" )
# Label
label2 = Label(frame1, text="Show Table ", font=("Arial Black",16), background="gray35",foreground="White Smoke")
label2.place(x=35, y=0)

# Boutons radio 
valuea = IntVar() 

valuea.set(1)
R_bouton1 = Radiobutton(frame1, text="Table" ,width=20,background="gray35",foreground="gray70", 
                        font=("Arial Rounded MT Bold ",14),indicator=0,variable=valuea, value=1 ).place(x=4, y=40)
R_bouton2 = Radiobutton(frame1, text="duplicate data",width=20,background="gray35",foreground="gray70",
                        font=("Arial Rounded MT Bold ",14),indicator=0, variable=valuea, value=2).place(x=4, y=80)
R_bouton3 = Radiobutton(frame1, text="missing values", width=20,background="gray35",foreground="gray70",
                        font=("Arial Rounded MT Bold ",14),indicator=0,variable=valuea, value=3).place(x=4, y=120)
R_bouton4 = Radiobutton(frame1, text="more info", width=20,background="gray35",foreground="gray70",
                        font=("Arial Rounded MT Bold ",14),indicator=0,variable=valuea, value=4).place(x=4, y=165) 
# Bouton
bouton1=Button(frame1, text="Show", font=("arial ",13), width=11, height=1, background="gray30", fg='White', command=Afficher)
bouton1.place(x=60, y=210)

frame1.place(x=0, y=0)


# Frame 2
bouton_text_size2 = 10
frame2 = Frame(fenetre, width=250, height=268, borderwidth=2, relief=GROOVE, background="gray35")
# Label
label1 = Label(frame2, text="Update Table", font=("Arial Black",16), background="gray35",foreground="White Smoke")
label1.place(x=35, y=0)
# Boutons radio 
value2 = IntVar() 
value2.set(1)


R2_bouton2 = Radiobutton(frame2, text="Delete ligne",width=20,background="gray35",foreground="gray70", 
                        font=("Arial Rounded MT Bold ",14),indicator=0,variable=value2, value=5 ).place(x=4, y=55)

def userText(event):
    entree_1.delete(0,END)
    usercheck=True
    
def userTextb(event):
    entree_2.delete(0,END)
    usercheck=True    
    
def fonc(event):
    global value
    value = L_combobox.get()
    
def checkcmbo():
        value = str( L_combobox.get()) 
        fildir.insert(0 ,value)

# colname = StringVar()
# entree_1 = Entry(frame2, textvariable=colname, width=28,background="white",foreground="black", 
#                         font=("Arial Rounded MT Bold ",9))
# entree_1.place(x=15, y=95)
# entree_1.insert(0,"col name")
# entree_1.bind("<Button>",userText)



colname = StringVar()
# liste =(df.columns.values).tolist()
L_combobox = ttk.Combobox(frame2, textvariable=colname, width=30,state='readonly')
# a=["None"]
# L_combobox["value"] =a+liste
L_combobox.place(x=15, y=95)
L_combobox.bind("<<ComboboxSelected>>", fonc )
# L_combobox.current(0)


rowname = StringVar()
entree_2 = Entry(frame2, textvariable=rowname, width=28,background="white",foreground="black", 
                        font=("Arial Rounded MT Bold ",9))
entree_2.place(x=15, y=120)
entree_2.insert(0,"row value")
entree_2.bind("<Button>",userTextb)


# e2= Entry(frame2,textvariable=rowname, width=15,background="gray70",foreground="black",font=("Arial Rounded MT Bold ",9))
# e2.place(x=120, y=95)
# e2.insert(0,"Enter password")
# e2.bind("<Button>",passText)

R2_bouton3 = Radiobutton(frame2, text="delete Duplicate",width=20,background="gray35",foreground="gray70", 
                        font=("Arial Rounded MT Bold ",14),indicator=0, variable=value2, value=6).place(x=4, y=147)

R2_bouton3 = Radiobutton(frame2, text="Fill empty",width=20,background="gray35",foreground="gray70", 
                        font=("Arial Rounded MT Bold ",14),indicator=0, variable=value2, value=7).place(x=4, y=190)
# Bouton
bouton2=Button(frame2, text="Run", font=("arial ",13), width=11, height=1, background="gray30", fg='White', command=Exécuter)
bouton2.place(x=60, y=228)
frame2.place(x=0, y=255)


# Frame 3
bouton_text_size3 = 10
frame3 = Frame(fenetre, width=250, height=80, borderwidth=2, relief=GROOVE, background="gray35")

label3 = Label(frame3, text="File Validation ",
               font=("Arial Black",16), background="gray35",foreground="White Smoke")
label3.place(x=30, y=0)
bouton3=Button(frame3, text="Verify", font=("arial ",13), width=11, height=1, background="gray30", fg='White', command=Vérifier)
bouton3.place(x=60, y=38)

frame3.place(x=0, y=524)



# Canva zone de tableua
canva_color1 = "#FCFCFC"
canvas1 = Canvas(fenetre, width=800, height=430, background=canva_color1, borderwidth=2, relief=GROOVE)
canvas1.place(x=257, y=0)
# Canva zone sortie
canvas2 = Canvas(fenetre, width=800, height=145, background=canva_color2, borderwidth=2, relief=GROOVE)
canvas2.place(x=257,y=450)
# text
text = st.ScrolledText(canvas2, width=98, height=8, background=canva_color2 ,)  
text.insert('1.0', "Zone de sortie :\n")
text.insert(END, "***************\n\n")
text.place(x=4, y=0)
# Frame 4
bouton_text_size4 = 11
frame4 = Frame(fenetre, width=1100, height=50, borderwidth=2, relief=GROOVE,background="gray35")

bouton4=Button(frame4, text="Save",font=("arial ",13), width=10, height=1, background="gray30", fg='White', command=Enregistrer)
bouton4.place(x=953, y=5)

frame4.place(x=0, y=605)

label1 = Label(frame4, 
               text="Creatd , Developped and Designed  by : Anass Houdou , Ouali Soufiyane , Jai Otman \n Directed By: Prof .El far Mohamed   BDSAS  2019/2020",
               font=("Arial ",9), background="gray35",foreground="White Smoke")
label1.place(x=300, y=5)

fenetre.mainloop()
    