import numpy as np
from math import sqrt as sq
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style


#------------------------------------------------------------------------------


N = 100.    #-> quantidade de fios de macarrão por barra
g = 9.81    #-> ac. gravidade (mm/s²)

#>> Propriedades de material:
"""
Referência: 
    
<https://www.ucam-campos.br/wp-content/uploads/2016/04/dados_para_projeto_pontes_de_espaguete.pdf>
"""    
E = 3530.394  #-> Módulo de Elasticidade (N/mm²)
A = 2.545*N   #-> Área de seção (mm²)
         

#------------------------------------------------------------------------------
         

def rigidez(A, E, x1, x2):                               #-> x1 = (x, y) ; x2 = (x, y): coordenadas dos nós x1 e x2, respectivamente
    L = sq((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)          #-> Tamanho do elemento                 
    theta = np.arctan2((x2[1]-x1[1]), (x2[0]-x1[0]))     #-> Inclinação do elemento
    c = np.cos(theta)                                      
    s = np.sin(theta)   


    c2 = c**2
    s2 = s**2
    cs = c*s              
     

    k = (A*E/L)*np.array((                               #-> Matriz de rigidez local
        (c2, cs, -c2, -cs),
        (cs, s2, -cs, -s2),
        (-c2, -cs, c2, cs),
        (-cs, -s2, cs, s2),
        ))
    return k                
         

#------------------------------------------------------------------------------

#>> Definição dos nós:
nodes = (     
        (0,  0.,   0.),
        (1,  75.,  150.),
        (2,  150., 0.),
        (3,  225., 150.),
        (4,  300., 0.),
        (5,  375., 150.),
        (6,  450., 0.),
        (7,  525,  150.),
        (8,  600., 0.),
        (11, 675., 150.),
        (12, 750., 0.),
        (13, 825., 150.), 
        (14, 900., 0.)
        ) #-> node=(numero_do_no,dist_x,dist_y)


#>> Definição dos elementos:
elementos =(
           (0,  0,  1),
           (1,  1,  3),
           (2,  3,  5),
           (3,  5,  7),
           (4,  7,  9),
           (5,  9,  11),
           (6,  11, 12),
           (7,  10, 12),
           (8,  8,  10),
           (9,  6,  8),
           (10, 4,  6),
           (11, 2,  4),
           (12, 0,  2),
           (13, 1,  2),
           (14, 2,  3),
           (15, 3,  4),
           (16, 4,  5),
           (17, 5,  6),
           (18, 6,  7),
           (19, 7,  8),
           (20, 8,  9),
           (21, 9,  10),
           (22, 10, 11)
           ) #-> elements=(numero_do_elemento, numero_primeiro_no, numero_segundo_no)


#>> Definição do carregamento:
forcas = (
        (2,  1, -3.*g),  #-> massa de 3.00kg
        (4,  1, -4.5*g), #-> massa de 4.50kg
        (6,  1, -6.*g),  #-> massa de 6.00kg
        (8,  1, -4.5*g), #-> massa de 4.50kg
        (10, 1, -3.*g),  #-> massa de 3.00kg
        ) #-> força(numero_do_no, GL_do_no, valor_da_forca)


#>> Definição das condições de contorno:
contorno = (
           (0,  0, 0.),
           (0,  1, 0.),
           (12, 0, 0.),
           (12, 1, 0.),
           ) #-> contorno(numero_do_no, numero_de_GL_do_no, valor_da_condicao_de_contorno)


#------------------------------------------------------------------------------


#Listas para armazenar os GL
u_gamma = []  #-> Vetor com as posições dos nós 'travados'
u_val   = []  #-> Respectivos valores
for cc in contorno:
    u_gamma.append(cc[0]*2+cc[1])
    u_val.append(cc[2])

forcas_global = np.zeros(2*len(nodes))
for f in forcas:
    forcas_global[2*f[0] + f[1]] += f[2]



rigidez_global = np.zeros((2*len(nodes),2*len(nodes)))
for e in elementos:
    #Nós elementares:
    node_a = e[1]   #-> Start node
    node_b = e[2]   #-> End node
 
    
    #Posições elementares:
    x1 = nodes[node_a][1:3]
    x2 = nodes[node_b][1:3]
    
    
    #Rigidez local através de 'nested for'
    rigidez_local = rigidez(A, E, x1, x2)
    for i in range(2):   #-> Range do número de nós por elemento
        for j in range(2):  #-> Range do número de GL por nó
            for m in range(2):  #-> Range do número de nós por elemento
                for n in range(2): #-> Range do número de GL por nó
                    rigidez_global[2*e[i+1]+j][2*e[m+1]+n] += rigidez_local[2*i+j][2*m+n]

rigidez_global_2 = np.copy(rigidez_global)
forcas_global_2 = forcas_global

rigidez_global = np.delete(rigidez_global, u_gamma, 0)
rigidez_global = np.delete(rigidez_global, u_gamma, 1)
forcas_global = np.delete(forcas_global, u_gamma, 0)

u = np.linalg.solve(rigidez_global, forcas_global)


#>> Condições de contorno: aplicando método dos deslocamentos livres:
u_livres = [i for i in range(2*len(nodes)) if i not in u_gamma] #-> Generator

k11 = rigidez_global_2[np.ix_(u_livres,u_livres)]
k21 = rigidez_global_2[np.ix_(u_gamma,u_livres)]
k12 = rigidez_global_2[np.ix_(u_livres,u_gamma)]
k22 = rigidez_global_2[np.ix_(u_gamma,u_gamma)]

f1 = forcas_global_2[u_livres]

u_1 = np.linalg.solve(k11, f1 - np.matmul(k12,u_val))   #-> Deslocamentos
f_react_1 = np.matmul(k21, u_1) + np.matmul(k22, u_val) #-> Forças de reação


#------------------------------------------------------------------------------


#>> Armazenando os resultados obtidos em dicts:
Reactions_1 = {
    'Node_00:Fx': f_react_1[0], 'Node_00:Fy': f_react_1[1],
    'Node_12:Fx': f_react_1[2], 'Node_12:Fy': f_react_1[3],
    } #-> Forças de reação nos engastes

Nodal_Displacements_1 = {
    'Node_01:(X)':u_1[0],'Node_01:(Y)':u_1[1],    
    'Node_02:(X)':u_1[2],'Node_02:(Y)':u_1[3],
    'Node_03:(X)':u_1[4],'Node_03:(Y)':u_1[5],
    'Node_04:(X)':u_1[6],'Node_04:(Y)':u_1[7],
    'Node_05:(X)':u_1[8],'Node_05:(Y)':u_1[9],
    'Node_06:(X)':u_1[10],'Node_06:(Y)':u_1[11],
    'Node_07:(X)':u_1[12],'Node_07:(Y)':u_1[13],
    'Node_08:(X)':u_1[14],'Node_08:(Y)':u_1[15],
    'Node_09:(X)':u_1[16],'Node_09:(Y)':u_1[17],
    'Node_10:(X)':u_1[18],'Node_10:(Y)':u_1[19],
    'Node_11:(X)':u_1[20],'Node_11:(Y)':u_1[21],
    } #-> Deslocamentos nodais encontrados


Elementary_Load = {
    'S_00':((u_1[0]-0.00)*(E/(150.*sq(2)))),
    'S_01':((u_1[4]-u_1[0])*(E/(150.))),
    'S_02':((u_1[4]-u_1[0])*(E/(150.))),
    'S_03':((u_1[8]-u_1[4])*(E/(150.))),
    'S_04':((u_1[16]-u_1[12])*(E/(150.))),
    'S_05':((u_1[20]-u_1[14])*(E/(150.))),
    'S_06':((0.00-u_1[20])*(E/(150.*sq(2)))),
    'S_07':((0.00-u_1[18])*(E/(150.))),
    'S_08':((u_1[20]-u_1[14])*(E/(150.))),
    'S_09':((u_1[14]-u_1[10])*(E/(150.))),
    'S_10':((u_1[10]-u_1[6])*(E/(150.))),
    'S_11':((u_1[6]-u_1[2])*(E/(150.))),
    'S_12':((u_1[2]-0.00)*(E/(150.))),
    'S_13':((u_1[2]-u_1[0])*(E/(150.*sq(2)))),
    'S_14':((u_1[4]-u_1[2])*(E/(150.*sq(2)))),
    'S_15':((u_1[6]-u_1[4])*(E/(150.*sq(2)))),
    'S_16':((u_1[8]-u_1[6])*(E/(150.*sq(2)))),
    'S_17':((u_1[10]-u_1[8])*(E/(150.*sq(2)))),
    'S_18':((u_1[12]-u_1[10])*(E/(150.*sq(2)))),
    'S_19':((u_1[14]-u_1[12])*(E/(150.*sq(2)))),
    'S_20':((u_1[16]-u_1[14])*(E/(150.*sq(2)))),
    'S_21':((u_1[18]-u_1[16])*(E/(150.*sq(2)))),
    'S_22':((u_1[20]-u_1[18])*(E/(150.*sq(2))))
    } #-> Tensões em cada elemento: Sigma[n] = (ux[i]-ux[j])*(E/L[n])


#>> Resultados gerados (visualização através do terminal):
print('-='*30)
print('\n\t\t\t\t',Fore.RED + Back.GREEN + Style.BRIGHT,'RESULTADOS', Style.RESET_ALL)


print(f'{Fore.RED + Style.BRIGHT}\n\n>> Deslocamento máximo (em módulo):{Style.RESET_ALL}\n{max(abs(u_1))}mm')


print(f'\n{Fore.RED + Style.BRIGHT}>> Deslocamentos nodais:{Style.RESET_ALL}')
for key, value in Nodal_Displacements_1.items():
    print(f'{key} = {value}mm')


print(f'\n{Fore.RED + Style.BRIGHT}>> Forças de reação:{Style.RESET_ALL}')
for key, value in Reactions_1.items():
    print(f'{key} = {value}N')


print(f'\n{Fore.RED + Style.BRIGHT}>> Esforços internos em cada elemento:{Style.RESET_ALL}')
for key, value in Elementary_Load.items():
    if Elementary_Load[key] > 0:
        print(f'{key} = {value}N/mm²  -> this is an element under tractive strength!')
    elif Elementary_Load[key] < 0:
        print(f'{key} = {value}N/mm² -> this is an element under compressive strength!')
    else:
        print(f'{key} = {value}N/mm² -> this is a static element!')


#------------------------------------------------------------------------------


scale = 150.

for i in u_gamma:
    u = np.insert(u, i, 0.)

fig, ax = plt.subplots()

#>> Forma indeformada:
for elem in elementos:
    ax.plot((nodes[elem[1]][1], nodes[elem[2]][1]), (nodes[elem[1]][2], nodes[elem[2]][2]), '.r')
    ax.plot((nodes[elem[1]][1], nodes[elem[2]][1]), (nodes[elem[1]][2], nodes[elem[2]][2]), 'c')

#>> Forma deformada:    
for elem in elementos:
    ax.plot((nodes[elem[1]][1]+scale*u[elem[1]*2], nodes[elem[2]][1]) + scale*u[elem[2]*2],(nodes[elem[1]][2]+scale*u[elem[1]*2+1], nodes[elem[2]][2]+scale*u[elem[2]*2+1]), '--k')

plt.axis('equal')
plt.grid(b=bool, which='both', axis='both', color='k', linestyle=':', linewidth=0.5)
plt.title('.::. Forma Deformada e Indeformada da Estrutura .::.')
plt.xlabel('X\n[mm]')
plt.ylabel('Y\n[mm]')
    
    
    
#--------------------------------------------------------------------------End.
