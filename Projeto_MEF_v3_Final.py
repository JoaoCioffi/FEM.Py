#%%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import eig 

def gauss():
    g = np.array((
        (-(3./5.)**0.5,5./9.),
        (0.,8./9.),
        (+(3./5.)**0.5,5./9.),
        ))
    return g

def shape_functions(x):
    phi = np.array((
        1./2.+x/2.,
        1./2.-x/2.,
        ))
    dphi = np.array((
        +1./2.,
        -1./2,
        )) 
    
    psi = np.array((
        -(x+1.)**2.*(x-2.)/4.,
        +(x-1.)**2.*(x+2.)/4.,
        +(x-1.)*(x+1.)**2./4.,
        +(x-1.)**2.*(x+1.)/4.,
        ))
    
    dpsi = np.array((
        +3./4. - 3.*x**2./4.,
        -3./4. + 3.*x**2./4.,
        +3.*x**2./4. + x/2. - 1./4.,
        +3.*x**2./4. - x/2. - 1./4.,
        ))
    
    ddpsi = np.array((
        -3.*x/2.,
        +3.*x/2.,
        +3.*x/2. + 1./2.,
        +3.*x/2. - 1./2.,
        ))
    return phi,dphi,psi,dpsi,ddpsi

def s_derivates(x,x1,x2):
    __,dphi,__,dpsi,ddpsi = shape_functions(x)
    
    alpha = np.arctan2(x2[2]-x1[2],x2[1]-x1[1])
    
    dxdxi = dphi[0]*x1[1] + dphi[1]*x2[1]
    dydxi = dphi[0]*x1[2] + dphi[1]*x2[2]
    
    if dxdxi ==0.:
        dxidx = 0.
        
    else:
        dxidx = dxdxi**-1
        
    if dydxi ==0.:
        dxidy = 0.
    else:
        dxidy = dydxi**-1
        
    dxidt = np.cos(alpha)*dxidx + np.sin(alpha)*dxidy
    
    dphidt = np.multiply(dphi,dxidt)
    dpsidt = np.multiply(dpsi,dxidt)
    dpsiddt = np.multiply(ddpsi,dxidt**2.)

    return dphidt,dpsidt,dpsiddt

def b_matrixes(x,x1,x2):
    dphi,__,ddpsi = s_derivates(x,x1,x2)
    bn = np.array((
        (dphi[0],   0.,   0., dphi[1],  0., 0.)
        ))
    
    bb = np.array((
        (0., ddpsi[0], ddpsi[2],   0., ddpsi[1], ddpsi[3])
        ))

    return bn, bb

def local_stiffness(x1,x2,area,intertia,young_modulus):
    L = ((x2[1]-x1[1])**2. + (x2[2]-x1[2])**2.)**0.5
    
    alpha = np.arctan2(x2[2]-x1[2],x2[1]-x1[1])
    
    k = np.zeros((6,6))
    
    g = gauss()
    for g_point,g_weight in g:
        xg1 = 0.
        xg2 = L
        
        dsdxi = (xg2-xg1)/2.
        bn, bb = b_matrixes(g_point,x1,x2)
        
        k += g_weight*dsdxi*(young_modulus*area*np.outer(bn,bn) + young_modulus*intertia*np.outer(bb,bb))
        
    c = np.cos(alpha)
    s = np.sin(alpha)

    t = np.array((
    (c, -s, 0., 0., 0., 0.),
    (s, c, 0., 0., 0., 0.),
    (0., 0., 1., 0., 0., 0.),
    (0., 0., 0., c, -s, 0.),
    (0., 0., 0., s, c, 0.),
    (0., 0., 0., 0., 0., 1.),
    ))     

    k = np.matmul(np.matmul(np.transpose(t),k),t)  
    return k

def n_matrix(x,x1,x2):
    phi,__,psi,__,__ = shape_functions(x)
    __,dpsi,__ = s_derivates(x, x1, x2)
    n = np.array((
        (phi[0],  0.,  0.,  phi[1],  0.,  0.),
        (  0., psi[0], psi[2],  0., psi[1], psi[3]),
        (0., dpsi[0], dpsi[2],  0., dpsi[1], dpsi[3]),
        ))
    
    return n

def local_mass(x1,x2,area,area_moment,rho):
    L = ((x2[1]-x1[1])**2. + (x2[2]-x1[2])**2.)**0.5
    
    alpha = np.arctan2(x2[2]-x1[2],x2[1]-x1[1])
    
    m = np.zeros((6,6))
    
    
    g = gauss()
    for g_point,g_weight in g:
        xg1 = 0.
        xg2 = L
        
        dsdxi = (xg2-xg1)/2.
        n = n_matrix(g_point,x1,x2)
        
        m += dsdxi*g_weight*rho*area*np.matmul(np.transpose(n),n)
        
    c = np.cos(alpha)
    s = np.sin(alpha)
    t = np.array((
    (c, -s, 0., 0., 0., 0.),
    (s, c, 0., 0., 0., 0.),
    (0., 0., 1., 0., 0., 0.),
    (0., 0., 0., c, -s, 0.),
    (0., 0., 0., s, c, 0.),
    (0., 0., 0., 0., 0., 1.),
    ))     

    m = np.matmul(np.matmul(np.transpose(t),k),t)  
    return m

a, b = 18., 18.             #-> medidas da seção (18mm x 18mm) - 100 spaghetis por viga (10 x 10)
young_modulus = 360
rho = 1.5483*10**(-6)
area = a*b
area_moment = (a*b**3)/12.
g = 9.81

#%%
# Nº de divisões (cortes) por viga
# Para plotar com 13 nós descomente e insire num = 0
# Para plotar com 36 nós descomente e insire num = 1
# Para plotar com 82 nós descomente e insire num = 3
num = 3

# Discretização dos nós
#nodes =  np.array(np.loadtxt("13nos.txt")) # Para plotar com 13 nós descomente e insire num = 0
#nodes =  np.array(np.loadtxt("36nos.txt")) # Para plotar com 36 nós descomente e insire num = 1
nodes =  np.array(np.loadtxt("82nos.txt")) # Para plotar com 82 nós descomente e insire num = 3

# Discretização dos elementos
#elements = np.array(np.loadtxt("23elementos.txt")) # Para plotar com 23 elementos descomente e insire num = 0
#elements = np.array(np.loadtxt("46elementos.txt")) # Para plotar com 46 elementos descomente e insire num = 1
elements = np.array(np.loadtxt("92elementos.txt")) # Para plotar com 92 elementos descomente e insire num = 3

# Nº de nós
number_of_nodes = 13 + num*23
# Nº de elementos
number_of_elements = (num+1)*23

# Condições de contorno: (nó, GL (0|1|2), condição de contorno (0-Engaste))
boundary = (
    (0, 0, 0.),
    (0, 1, 0.),
    (0, 2, 0.),
    ((num+1)*12, 0, 0.),
    ((num+1)*12, 1, 0.),
    ((num+1)*12, 2, 0.)
    ) #-> contorno(numero_do_no, numero_de_GL_do_no, valor_da_condicao_de_contorno)


#%%

# Forças aplicadas: (0 - GDL para lado, 1 - GDL para cima, 2 - GDL para rotação)
forcas = (
        ((num+1)*2,  1, -3*g), #-> massa de 30kg
        ((num+1)*4,  1, -4.5*g), #-> massa de 45kg
        ((num+1)*6,  1, -6.0*g), #-> massa de 60kg
        ((num+1)*8,  1, -4.5*g), #-> massa de 45kg
        ((num+1)*10, 1, -3.0*g), #-> massa de 30kg
        ) #-> força(numero_do_no, GL_do_no, valor_da_forca)

#%%

# Carregando o vetor de forças globais
forcas_global = np.zeros(3*len(nodes))

for f in forcas:
    #print(f)
    forcas_global[3*f[0] + f[1]] += f[2]

#%%

u_gamma = []
u_val = []
for i in boundary:
    u_gamma.append(3*i[0]+i[1])
    u_val.append(i[2])

u_not_gamma = [i for i in range(3*number_of_nodes) if i not in u_gamma]

K = np.zeros((3*number_of_nodes,3*number_of_nodes,))
M = np.zeros((3*number_of_nodes,3*number_of_nodes,))

# Calculando as matrizes K e M
for elem in elements:
    x1 = nodes[int(elem[1])]
    x2 = nodes[int(elem[2])]
    k = local_stiffness(x1,x2,area,area_moment,young_modulus)
    m = local_mass(x1,x2,area,area_moment,rho)
    
    index = (3*int(elem[1]),3*int(elem[1])+1,3*int(elem[1])+2,3*int(elem[2]),3*int(elem[2])+1,3*int(elem[2])+2)
    
    K[np.ix_(index,index)] += k
    M[np.ix_(index,index)] += m

#%%

# Condição de substituição direta:
for (i,j) in zip(u_gamma,u_val):
    K[i][i] = 1.

    forcas_global[i] = j

# Solver sistema linear
u = np.linalg.solve(K, forcas_global)


# Resolução alternativa (MATLAB)
np.savetxt("K.txt", K, delimiter="   ")
np.savetxt("forca.txt", forcas_global, delimiter="   ")

#%%

#DESLOCAMENTO MÁXIMO
Deslocamento_maximo = np.max(u)
print(Deslocamento_maximo)
#DESLOCAMENTO MÉDIO
Deslocamento_medio = np.mean(u)
print(Deslocamento_medio)

# Salva os dados para uma analise de convergência
#resp = [num,Deslocamento_maximo,Deslocamento_medio]
#np.savetxt("Convergencia_1.txt", resp, delimiter=" ")

#%%

# Carrega os dados da análise de convergência
conver = np.array(np.loadtxt("Dados_Convergencia_1.txt"))

# Análise da convergência - Gráfico
plt.figure(4)
plt.plot((conver[0],conver[3], conver[6] ),(conver[1],conver[4],conver[7]),'-b',(conver[0],conver[3], conver[6] ),(conver[2],conver[5],conver[8]),'-k')
plt.axis([0,100,-1,1])
plt.grid(b=bool, which='both', axis='both', color='k', linestyle=':', linewidth=0.5)
plt.title('.::. Análise de Convergência .::.')
plt.xlabel('nº Nós')
plt.ylabel('Deslocamento\n[mm]')

#%%

# Plots
scale = 15

for n in range(len(elements)):
    # Figura da Estrutura 1
    x1_1_plot = nodes[int(elements[n][1])][1]
    y1_1_plot = nodes[int(elements[n][1])][2]
    x2_1_plot = nodes[int(elements[n][2])][1]
    y2_1_plot = nodes[int(elements[n][2])][2]
    plt.figure(1)
    plt.plot([x1_1_plot,x2_1_plot],[y1_1_plot,y2_1_plot],'.m')
    plt.plot([x1_1_plot,x2_1_plot],[y1_1_plot,y2_1_plot],'c')
    plt.axis('equal')
    plt.grid(b=bool, which='both', axis='both', color='k', linestyle=':', linewidth=0.5)
    plt.title('.::. Estrutura 1 .::.')
    plt.xlabel('X\n[mm]')
    plt.ylabel('Y\n[mm]')

    # Figura da Estrutura 1 Deformada 
    plt.figure(2)
    u1_1_plot = u[int(elements[n][1])]
    v1_1_plot = u[int(elements[n][1])+1]
    u2_1_plot = u[int(elements[n][2])]
    v2_1_plot = u[int(elements[n][2])+1]
    plt.plot([(x1_1_plot+scale*u1_1_plot),(x2_1_plot+scale*u2_1_plot)],
             [(y1_1_plot+scale*v1_1_plot),(y2_1_plot+scale*v2_1_plot)],'r',alpha=1)
    plt.axis('equal')
    plt.grid(b=bool, which='both', axis='both', color='k', linestyle=':', linewidth=0.5)
    plt.title('.::. Forma Deformada da Estrutura 1 .::.')
    plt.xlabel('X\n[mm]')
    plt.ylabel('Y\n[mm]')

    
    # Figura da Estrutura 1 Indeformada + Deformada
    plt.figure(3)
    plt.plot([x1_1_plot,x2_1_plot],[y1_1_plot,y2_1_plot],'c')
    plt.plot([(x1_1_plot+scale*u1_1_plot),(x2_1_plot+scale*u2_1_plot)],
             [(y1_1_plot+scale*v1_1_plot),(y2_1_plot+scale*v2_1_plot)],'r',alpha=1)
    plt.axis('equal')
    plt.grid(b=bool, which='both', axis='both', color='k', linestyle=':', linewidth=0.5)
    plt.title('.::. Forma Indeformada e Deformada da Estrutura 1 .::.')
    plt.xlabel('X\n[mm]')
    plt.ylabel('Y\n[mm]')


#%% Cálculo das frequências de vibração

K = np.delete(K,u_gamma,0)
K = np.delete(K,u_gamma,1)
M = np.delete(M,u_gamma,0)
M = np.delete(M,u_gamma,1)

omega2, Phi = eig(K,M)

for i in u_gamma:
    Phi = np.insert(Phi, i, np.zeros(3*number_of_nodes-len(u_gamma)),axis=0)
    
values = np.argsort(omega2)
Phi = np.transpose(Phi)
Phi = Phi[values]

if num == 1:
    print ('Primeiro modo: ', omega2[0], '\nSegundo modo: ', omega2[1], '\nTerceiro modo: ', omega2[2])
else:
    print ('Primeiro modo: ', omega2[0], '\nSegundo modo: ', omega2[1], '\nTerceiro modo: ', omega2[2], '\nQuarto modo: ', omega2[3], '\nQuinto modo: ', omega2[4], '\nSexto modo: ', omega2[5])

#%% MODOS DE VIBRAR

fig, axis = plt.subplots()
scale = 150.

#>> Modo de vibrar: para o n-ésimo modo de vibrar -> mode = (n - 1)
mode = 5   #  Altere o modo


for i in elements:
    
    axis.plot((nodes[int(i[1])][1], nodes[int(i[2])][1]),(nodes[int(i[1])][2], nodes[int(i[2])][2]),'k--')
    plt.grid(b=bool, which='both', axis='both', color='k', linestyle=':', linewidth=0.5)
    plt.title('.::. Modo de flambagem - 06 .::.')
    plt.xlabel('X\n(mm)')
    plt.ylabel('Y\n(mm)')
    
    axis.plot((nodes[int(i[1])][1] + scale*Phi[mode][3*int(i[1])], nodes[int(i[2])][1] + scale*Phi[mode][3*int(i[2])]),(nodes[int(i[1])][2] + scale*Phi[mode][3*int(i[1])+1], nodes[int(i[2])][2] + scale*Phi[mode][3*int(i[2])+1]),'r-')


    
plt.axis('equal')

#%%