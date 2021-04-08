#!/usr/bin/env python
# coding: utf-8

# In[1]:

import bnlearn as bn
import json as js
import sys

def probabilidadConjunta(model, test):
    #se define un arreglo unidimensional para registrar el resultado de la inferencia
    arreglo = []
    indice  = 0
    
    for fila in test.index: #recorriendo las filas
        valor = "{"
        for columna in test: #recorriendo las columnas
            if columna in ('estado'): 
               continue
            else:
               try:
                  valor = valor + "\""+columna + "\":" +test[columna][fila] + ", "
               except:
                  valor = valor + "\""+columna + "\":" +str(test[columna][fila]) + ", "
                
        valor = valor[:-2] + '}'
        print("FILA N°: " + str(fila) + " -> P(\"Estado\" | " + "[" + valor + "]")
        res = js.loads(valor)
        
        arreglo.append([])
        #try:
        q1 = bn.inference.fit(model, variables=['estado'], evidence=res)
           
        if (q1.get_value(estado=0) > q1.get_value(estado=1)):
             arreglo[indice].append(0)
             arreglo[indice].append(q1.get_value(estado=0))
        else:
             arreglo[indice].append(1)
             arreglo[indice].append(q1.get_value(estado=1))
           
           #arreglo.append(max([q1.get_value(estado=0), q1.get_value(estado=1)]))
        #except:
        #   arreglo[indice].append(-1)
        #   arreglo[indice].append(-1)
        #   e = sys.exc_info()[1]
        #   print(e.args[0]) 
        
        indice += 1
        valor = "{"
    
    return arreglo


# In[ ]:




