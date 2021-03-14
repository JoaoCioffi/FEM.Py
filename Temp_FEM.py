import numpy as np

#-----------------------------------------------------------------------------
print('-=' * 30)


#>> Inputs:
q= 1.e+05
h = 5.e+03
k = 190.
dx = 2.5e-03
dy = 4.e-03
T_inf = 15 + 273


"""
_______________________________________________________________________________
Considerando:
    

A =    (1,1)  (1,2)  (1,3)  ...  (1,n)
       (2,1)  (2,2)  (2,3)  ...  (2,n)
       (3,1)  (3,2)  (3,3)  ...  (3,n)
         .      .      .    ...    .
         .      .      .    ...    . 
         .      .      .    ...    .
       (n,1)  (n,2)  (n,3)  ...  (n,n)


R = (1,1)
    (2,1)
    (3,1)
      .
      .
      .
    (n,1)
    


T = (T1)
    (T2)
    (T3)
     .
     .
     .
    (Tn)

_______________________________________________________________________________
                    Para a resolução do sistema linear
    
                                A * T = R
                
    sendo np.shape(A) = (n,n) ; np.shape(T) = (n,1) ; np.shape(R) = (n,1)
_______________________________________________________________________________
"""

#>> Equações do sistema:
A = np.zeros((28,28)) #-> matriz dos coeficientes
R = np.zeros((28,1))  #-> matriz resposta

#>> Eq. (i):
A[0][0] = -((dy/dx) + (dx/dy))
A[0][1] = (dy/dx)
A[0][5] = (dx/dy)

#>> Eq. (ii):
A[1][3] = (dy/dx)    
A[1][4] = -((dy/dx) + (dx/dy))
A[1][9] = (dx/dy)

#>> Eq. (iii):
A[2][0] = (dy/dx)
A[2][1] = -2*((dy/dx) + (dx/dy))
A[2][2] = (dy/dx)
A[2][6] = 2*(dx/dy)

#>> Eq. (iv):
A[3][1] = (dy/dx)
A[3][2] = -2*((dy/dx) + (dx/dy))
A[3][3] = (dy/dx)
A[3][7] = 2*(dx/dy)

#>> Eq. (v):
A[4][2] = (dy/dx)
A[4][3] = -2*((dy/dx) + (dx/dy))
A[4][4] = (dy/dx)
A[4][8] = 2*(dx/dy)

#>> Eq. (vi):
A[5][0] = (dx/dy)
A[5][5] = -((dx/dy) + (dy/dx) + (h*dx/k))
A[5][6] = (dy/dx)
R[5][0] = -h*(dx/k)*T_inf

#>> Eq. (vii):
A[6][13] = -((dx/dy) + (dy/dx) + (h*dx/k))
A[6][14] = (dy/dx)
A[6][18] = (dx/dy)
R[6][0] = -h*(dx/k)*T_inf

#>> Eq. (viii):
A[7][5] = (dy/dx) 
A[7][7] = (dy/dx)
A[7][1] = 2*(dx/dy)
A[7][6] = -2*((dx/dy) + (dy/dx) + (h*dx/k))
R[7][0] = -2*h*(dx/k)*T_inf

#>> Eq. (ix):
A[8][13] = (dy/dx) 
A[8][15] = (dy/dx)
A[8][19] = 2*(dx/dy)
A[8][14] = -2*((dx/dy) + (dy/dx) + (h*dx/k))
R[8][0] = -2*h*(dx/k)*T_inf

#>> Eq. (x):
A[9][6] = (dy/dx)
A[9][8] = 2*(dy/dx)
A[9][10] = (dx/dy)
A[9][2] = 2*(dx/dy)
A[9][7] = -((3*(dy/dx)) + (3*(dx/dy)) + ((h/k)*(dx+dy)))
R[9][0] = ((-h/k)*(dx+dy))*T_inf

#>> Eq. (xi):
A[10][14] = (dy/dx)
A[10][16] = 2*(dy/dx)
A[10][10] = (dx/dy)
A[10][20] = 2*(dx/dy)
A[10][15] = -((3*(dy/dx)) + (3*(dx/dy)) + ((h/k)*(dx+dy)))
R[10][0] = ((-h/k)*(dx+dy))*T_inf

#>> Eq. (xii):
A[11][7] = (dx/dy)
A[11][15] = (dx/dy)
A[11][11] = 2*(dy/dx)
A[11][10] = -2*((dx/dy) + (dy/dx) + (h*(dy/k)))
R[11][0] = -2*h*(dy/k)*T_inf

#>> Eq. (xiii):
A[12][7] = (dy/dx)
A[12][9] = (dy/dx)
A[12][3] = (dx/dy)
A[12][11] = (dx/dy)
A[12][8] = -2*((dx/dy) + (dy/dx))

#>> Eq. (xiv):
A[13][10] = (dy/dx)
A[13][12] = (dy/dx)
A[13][8] = (dx/dy)
A[13][16] = (dx/dy)
A[13][11] = -2*((dx/dy) + (dy/dx))

#>> Eq. (xv):
A[14][15] = (dy/dx)
A[14][17] = (dy/dx)
A[14][11] = (dx/dy)
A[14][21] = (dx/dy)
A[14][16] = -2*((dx/dy) + (dy/dx))

#>> Eq. (xvi):
A[15][19] = (dy/dx)
A[15][21] = (dy/dx)
A[15][25] = (dx/dy)
A[15][15] = (dx/dy)
A[15][20] = -2*((dx/dy) + (dy/dx))

#>> Eq. (xvii):
A[16][20] = (dy/dx)
A[16][22] = (dy/dx)
A[16][26] = (dx/dy)
A[16][16] = (dx/dy)
A[16][21] = -2*((dx/dy) + (dy/dx))

#>> Eq. (xviii):
A[17][4] = (dx/dy)
A[17][12] = (dx/dy)
A[17][8] = 2*(dy/dx)
A[17][9] = -2*((dx/dy) + (dy/dx))   

#>> Eq. (xix):
A[18][9] = (dx/dy)
A[18][17] = (dx/dy)
A[18][11] = 2*(dy/dx)
A[18][12] = -2*((dx/dy) + (dy/dx))  
    
#>> Eq. (xx):
A[19][12] = (dx/dy)
A[19][22] = (dx/dy)
A[19][16] = 2*(dy/dx)
A[19][17] = -2*((dx/dy) + (dy/dx))  
    
#>> Eq. (xxi):
A[20][17] = (dx/dy)
A[20][27] = (dx/dy)
A[20][21] = 2*(dy/dx)
A[20][22] = -2*((dx/dy) + (dy/dx))  

#>> Eq. (xxii):
A[21][13] = (dx/dy)
A[21][23] = (dy/dx)
A[21][19] = 2*(dy/dx)
A[21][18] = -2*((dx/dy) + (dy/dx))

#>> Eq. (xxiii):
A[22][18] = (dx/dy)
A[22][24] = (dy/dx)
A[22][23] = -((dx/dy) + (dy/dx))
R[22][0] = -q*(dx/k)

#>> Eq. (xxiv):
A[23][22] = (dx/dy)
A[23][26] = (dy/dx)
A[23][27] = -((dx/dy) + (dy/dx))
R[23][0] = -q*(dx/k) 

#>> Eq. (xxv):
A[24][23] = (dy/dx) 
A[24][25] = (dy/dx)
A[24][19] = 2*(dy/dx)
A[24][24] = -2*((dx/dy) + (dy/dx))
R[24][0] = -2*q*(dx/k) 

#>> Eq. (xxvi):
A[25][24] = (dy/dx) 
A[25][26] = (dy/dx)
A[25][20] = 2*(dy/dx)
A[25][25] = -2*((dx/dy) + (dy/dx))
R[25][0] = -2*q*(dx/k) 

#>> Eq. (xxvii):
A[26][25] = (dy/dx) 
A[26][27] = (dy/dx)
A[26][21] = 2*(dy/dx)
A[26][26] = -2*((dx/dy) + (dy/dx))
R[26][0] = -2*q*(dx/k) 

#>> Eq. (xviii):
A[27][18] = (dy/dx) 
A[27][20] = (dy/dx)
A[27][14] = (dx/dy)
A[27][24] = (dx/dy)
A[27][19] = -2*((dx/dy) + (dy/dx))

T = np.linalg.solve(A,R)

print('\n>> Valores obtidos:\n')
for i in range(0,len(T)):
    print(f'Node{i} Temp = {(T[i][0])-273}°C')

print('-=' * 30)