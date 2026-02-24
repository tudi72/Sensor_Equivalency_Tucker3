from colorama import Fore 
from sklearn.base import TransformerMixin
from pathlib import Path 
from scipy import linalg
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import warnings 
import sys 
import os 

FILE = os.path.basename(__file__)
TAG = FILE[:-3]


class Tucker3(TransformerMixin):
    def __init__(self,LowRankApprox=np.array([2,2,2]),maxIter=200,tolCriteria=1e-5,ErrorScaled=True):
        self.LowRankApprox = LowRankApprox 
        self.maxIter = maxIter 
        self.numIter = 0 
        self.tolCriteria = tolCriteria
    
        self.fitScores = None # Nbat x R1 
        self.varLoadings = None # Nvar x R2 
        self.timeLoadings = None # timepoints x R3  
        self.coreTensor = None  # Nvar x 1 

        self.DModXTraining = None # Nbat x 1 
        self.ning = None # Nbat x 1 

        self.scaling = None
        self.ErrorScaled = ErrorScaled 
        self.TSQRTraining = None

    
    def train(self,input,verbose=True): 
        Nbat,Nvar,timePoints = input.shape 
        Rank = self.LowRankApprox 

        X_1         = np.reshape(input, (Nbat, Nvar * timePoints),order="F")
        U, _ , _    = linalg.svd(X_1,full_matrices=False)

        if Rank[0] <= U.shape[1]:
            end         = Rank[0]
            A           = U[:, 0:end]
        else: 
            A           = U 
            Arand       = np.random.rand(U.shape[0],Rank[0] - U.shape[1])
            A           = np.concatenate((A,Arand),axis=0)
            
        X_2         = np.reshape(np.transpose(input, (1,0,2)), ((Nvar, Nbat * timePoints)),order='F')
        U,_,_       = linalg.svd(X_2,full_matrices=False)
        
        if Rank[1] <= U.shape[1]:
            end = Rank[1]
            B   = U[:,0:end]
        else: 
            B   = U 
            Brand = np.random.rand(U.shape[0],Rank[1] - U.shape[1])
            B       = np.concatenate((B,Brand), axis=0)
        
        X_3         = np.reshape(np.transpose(input, (2,0,1)), (timePoints, Nbat * Nvar), order='F')
        U, _, _     = linalg.svd(X_3,full_matrices=False) 

        if Rank[2] <= U.shape[1]: 
            end = Rank[2]
            C   = U[:, 0:end] 
        else: 
            C   = U 
            Crand = np.random.rand(U.shape[0],Rank[2] - U.shape[1])
            C = np.concatenate((C,Crand),axis=0)

        if verbose: 
            print(TAG,"train","Multilinear SVD completed for 1st guess of A,B,C matrices...")
        
        kronCB = np.kron(C,B) 
        G_1     = A.T @ X_1 @ kronCB 
        GT      = np.reshape(G_1,Rank,order='F')
        G_2     = np.reshape(np.transpose(GT, (1,0,2)),(Rank[1], Rank[0] * Rank[2]),order='F')
        G_3     = np.reshape(np.transpose(GT, (2,0,1)),(Rank[2], Rank[0] * Rank[1]),order='F')
        
        convTest = 1 
        rel_error = []
        numIter = 0 

        x_vec = np.reshape(X_1,(Nbat * Nvar * timePoints,1),order='F')

        # Train loop 
        while(convTest > self.tolCriteria) and (numIter < self.maxIter):
            
            regress_factor      = G_1 @ kronCB.T 
            A                   = X_1 @ regress_factor.T @ linalg.pinv(regress_factor @ regress_factor.T)
            norm                = np.linalg.norm(A,ord=2,axis=0,keepdims=True)
            A                   = A / norm 

            kronCA              = np.kron(C,A)
            regress_factor      = G_2 @ kronCA.T 
            B                   = X_2 @ regress_factor.T @ linalg.pinv(regress_factor @ regress_factor.T)
            norm                = np.linalg.norm(B,ord=2,axis=0,keepdims=True)
            B                   = B / norm 

            kronBA              = np.kron(B,A)
            regress_factor      = G_3 @ kronBA.T 
            C                   = X_3 @ regress_factor.T @ linalg.pinv(regress_factor @ regress_factor.T)
            norm                = np.linalg.norm(C,ord=2,axis=0,keepdims=True)
            C                   = C / norm 

            kronCB              = np.kron(C,B)
            F                   = np.kron(kronCB,A)  
            g_vec               = (linalg.pinv(F.T @ F) @ F.T) @ x_vec

            GT                  = np.reshape(g_vec,Rank,order='F') 
            G_1                 = np.reshape(GT,(Rank[0],Rank[1] * Rank[2]),order='F') 
            G_2                 = np.reshape(np.transpose(GT, (1,0,2)),(Rank[1], Rank[0] * Rank[2]),order='F')
            G_3                 = np.reshape(np.transpose(GT, (2,0,1)),(Rank[2], Rank[0] * Rank[1]),order='F')

            X1_est              = A @ G_1 @ kronCB.T  
            residual            = X_1 - X1_est 
            error               = np.linalg.norm(residual,ord='fro') / np.linalg.norm(X_1,ord='fro')
            rel_error.append(error)

            if numIter > 10: 
                convTest = abs(rel_error[numIter] - rel_error[numIter - 1]) / rel_error[numIter - 1]
            
            if verbose: 
                if numIter <= 10:  
                    print("ALS iteration",numIter," relative error", error) 
                # else: 
                #     print("ALS iteration",numIter," relative error", error," convergence test", convTest) 

            self.fitScores      = A # Nbat x R1 
            self.varLoadings    = B # Nvar x R2 
            self.timeLoadings   = C # timePoints x R3 
            self.coreTensor     = GT # R1 X R2 X R3   
            self.numIter        = numIter 
            numIter             +=1 
            
    def fit(self,input,verbose=True): 

        Nbat, Nvar, timePoints = input.shape 

        self.train(input,verbose=verbose) 

        if verbose: 
            print(TAG,"fit",f"Successfully trained Tucker3 for {self.numIter} iterations...")
        if self.ErrorScaled: 
            TRess       = self.calcOOMD(input,metric="Res") 
            Ress_2      = np.reshape(np.transpose(TRess,(1,0,2)), (Nvar, Nbat * timePoints), order='F') 
            XT_scaled   = input.copy() 
            convTest    = 1 
            numIter     = 0 
            cum_factor  = np.ones((Nvar)) 

            while(convTest > self.tolCriteria) and (numIter < self.maxIter): 
                std_Ress    = np.std(Ress_2,axis=1)
                cum_factor  = cum_factor * std_Ress 
                X_2         = np.reshape(np.transpose(XT_scaled, (1,0,2)), (Nvar,Nbat * timePoints),order='F') 
                X_2         /= std_Ress[:, np.newaxis]
                XT_scaled   = np.transpose(np.reshape(np.transpose(X_2, (1,0)), (Nbat,timePoints,Nvar),order='F'), (0,2,1))

                self.train(XT_scaled,verbose=verbose) 
                TRess       = self.calcOOMD(XT_scaled,metric='Res')
                Ress_2      = np.reshape(np.transpose(TRess,(1,0,2)), (Nvar, Nbat * timePoints),order='F')

                if numIter > 2: 
                    convTest    = np.linalg.norm(std_Ress - np.ones((Nvar, 1))) 
                
                if verbose: 
                    print("Error",numIter, "Convergence Test", convTest," error std",std_Ress) 
                
                numIter +=1 

            self.scaling = cum_factor 


        self.DModXTraining = self.calcOOMD(input,metric="DModX")
        self.TSQRTraining  = self.calcIMD(input=input,metric="HotellingT2") 

        return self 

    def transform(self,input,verbose=True): 

        if self.varLoadings is None or self.timeLoadings is None: 
            raise ValueError("Model has not yet been fit...") 
        
        Nbat, Nvar, timePoints  = input.shape 
        transformDat            = input.copy() 
        XT_scaled               = transformDat
        Rank                    = self.LowRankApprox 
        B                       = self.varLoadings 
        C                       = self.timeLoadings
        GT                      = self.coreTensor 

        G_1                     = np.reshape(GT, (Rank[0], Rank[1] * Rank[2]),order='F')
        
        if self.scaling is not None: 
            X_2         = np.reshape(np.transpose(transformDat, (1,0,2)), (Nvar,Nbat * timePoints),order='F') 
            X_2         /= self.scaling[:, np.newaxis]
            XT_scaled   = np.transpose(np.reshape(np.transpose(X_2, (1,0)), (Nbat,timePoints,Nvar),order='F'), (0,2,1))

            X_1         = np.reshape(XT_scaled, (Nbat, Nvar * timePoints), order='F') 
        else: 
            X_1         = np.reshape(transformDat, (Nbat, Nvar * timePoints), order='F') 

        kronCB                  = np.kron(C,B) 
        regress_coef            = G_1 @ kronCB.T 
        inv_FFt                 = linalg.pinv(regress_coef @ regress_coef.T)
        scores                  = X_1 @ regress_coef.T @ inv_FFt 
        XT_est                  = self.inverse_transform(scores,scale="model")

        return scores, XT_scaled, XT_est 

    def fit_transform(self,input,verbose=True): 
        if self.varLoadings is None: 
            self.fit(input,verbose=verbose) 
            scores = self.fitScores.copy() 
            return {
                'scores'        : scores,
                'varLoadings'   : self.varLoadings,
                'timeLoadings'  : self.timeLoadings,
                'coreTensor'    : self.coreTensor
            }         
        else: 
            raise ValueError("Model has already been fitted ...") 
    
    def inverse_transform(self,scores,scale='model'):

        if self.varLoadings is None or self.timeLoadings is None: 
            raise ValueError("Model has not been fitted...") 
        
        Nbat,Rank_1     = scores.shape 
        Rank            = self.LowRankApprox
        B               = self.varLoadings 
        C               = self.timeLoadings 
        GT              = self.coreTensor 

        Nvar, _         = B.shape 
        timePoints, _   = C.shape 

        if Rank_1 != Rank[0]: 
            raise ValueError("Input scores has different number of columns vs model components") 
        
        G_1             = np.reshape(GT, (Rank[0], Rank[1] * Rank[2]),order='F')
        kronCB          = np.kron(C,B) 

        outData         = scores @ G_1 @ kronCB.T 
        outData         = np.reshape(outData, (Nbat, Nvar, timePoints), order='F')
        
        # TODO assumes that self.scaling is defined
        if self.ErrorScaled and scale == 'Data':# and self.scaling is not None: 
            outDataT    = np.reshape(outData,(Nbat,Nvar,timePoints),order='F') 
            X_2         = np.reshape(np.transpose(outDataT, (1,0,2)), (Nvar,Nbat * timePoints),order='F') 
            X_2         /= self.scaling[:, np.newaxis]

            XT_scaled   = np.transpose(np.reshape(np.transpose(X_2, (1,0)), (Nbat,timePoints,Nvar),order='F'), (0,2,1))

            outData     = XT_scaled 
        
        return outData 

    def calcIMD(self,scores=None,input=None,metric='HotellingT2'):

        if self.varLoadings is None: 
            raise ValueError("Model has not yet been fitted...") 
        
        if metric == "HotellingT2":
            if (input is None) and (scores is None): 
                raise ValueError("No values provided...") 
            elif (input is not None) and (scores is not None): 
                warnings.warn("Both scores and data re give, but operates only with data...") 
                outT2               = self.calcIMD(input=input)
            elif (input is not None) and (scores is None): 
                scores, _, _        = self.transform(input)
                outT2               = self.calcIMD(scores=scores) 
            
            elif (input is None) and (scores is not None): 
                Nbat, Rank_1        = scores.shape 
                Rank                = self.LowRankApprox  
                if Rank_1 != Rank[0]: 
                    raise ValueError("Input scores have more columns latent variable than model was fitted...")
                else: 
                    cov_fitScores   = np.cov(self.fitScores.T) 
                    TSQR_cov        = scores @ linalg.pinv(cov_fitScores) @ scores.T 
                    outT2           = np.diagonal(TSQR_cov)
            else: 
                raise ValueError("Unknwon issue in IMD method for metric = Hotelling...")

        return outT2

    def calcContribution(self,XT_scaled): 
        GT      = self.coreTensor 
        Rank    = self.LowRankApprox 
        A       = self.fitScores 
        B       = self.varLoadings 
        C       = self.timeLoadings 
        Nbat    = A.shape[0] 
        Nvar    = B.shape[0]
        timePoints = C.shape[0] 

        X_1     = np.reshape(XT_scaled, (Nbat, Nvar * timePoints),order='F')
        G_1     = np.reshape(GT, (Rank[0], Rank[1] * Rank[2]), order='F')
        kronCB  = np.kron(C,B) 
        regress_coef = G_1 @ kronCB.T 
        A_est   = X_1 @ regress_coef.T @ linalg.pinv(regress_coef @ regress_coef.T)

        cov_A   = np.cov(A.T) 
        TSQR_cov    = A_est @ np.linalg.pinv(cov_A) @ A_est.T 
        TSQR    = np.diagonal(TSQR_cov) 

        time = int(X_1.shape[1] // Nvar) 
        Nlaten  = int(A_est.shape[1])

        T_X  = np.reshape(X_1, (Nbat, Nvar, time), order='F') 
        T_uf    = np.reshape(regress_coef, (Nlaten,Nvar,time), order='F')
        con_TSQR = np.zeros((TSQR.size,Nvar))
        for j in range(Nvar):
            X_1_var = T_X[:, j, :]
            uf_var  = T_uf[:, j, :]
            con_T_est = X_1_var @ uf_var.T @ linalg.pinv(regress_coef @ regress_coef.T)
            con_TSQR_cov = A_est @ np.linalg.pinv(cov_A) @ con_T_est.T
            con_TSQR[:, j] = np.diag(con_TSQR_cov) 
        
        Xnew_est = A_est @ G_1 @ kronCB.T 
        E_1 = Xnew_est - X_1 
        T_SPE_mat = E_1 @ E_1.T 

        r,c = T_SPE_mat.shape 
        if r> 1 and c > 1 :
            T_SPE = np.diag(T_SPE_mat).reshape(1,-1)
            T_SPE = T_SPE.flatten() 
        else: 
            T_SPE = T_SPE_mat.flatten() 
        
        # ===========
        iSPE = E_1**2 
        iSPE_batch = np.zeros((Nbat, Nvar))
        ET = np.reshape(E_1,[Nbat, Nvar, timePoints],order='F') 
        E_2 = np.reshape(np.transpose(ET,(1,0,2)), (Nvar, Nbat * timePoints),order='F')

        for i in range(Nvar):
            E_2[i,:] = E_2[i,:] * self.scaling[i]
        ET_rescaled = np.transpose(np.reshape(np.transpose(E_2,(1,0)), (Nbat,timePoints,Nvar),order='F'),(0,2,1))
        squared_sum     = np.sum(ET_rescaled**2,axis=2) 
        relative_contr  = squared_sum / np.sum(squared_sum, axis=1, keepdims=True) 
        iSPE_batch = pd.DataFrame(relative_contr) 

        return {
            'T_SPE'     : T_SPE,
            'TSQR'      : TSQR,
            'con_TSQR'  : con_TSQR, 
            'iSPE'      : iSPE,
            'iSPE_batch': relative_contr
        }

    def calcOOMD(self,input,metric='Res'):

        if metric == 'Ress_Data': 
            transformDat            = input.copy() 
            Nbat,Nvar,timePoints    = input.shape 
            scores,_, _             = self.transform(transformDat) 
            modeledData             = self.inverse_transform(scores,scale='Data') 
            outOOMD                 = transformDat - modeledData 
        elif metric == 'Res': 

            transformDat            = input.copy() 
            Nbat,Nvar,timePoints    = input.shape
            scores,scaledData, _    = self.transform(transformDat) 
            modeledData             = self.inverse_transform(scores,scale='model') 
            outOOMD                 = scaledData - modeledData 

        elif metric == 'QRes': 
            transformDat            = input.copy() 
            Nbat,Nvar,timePoints    = input.shape 
            outOOMD                 = np.zeros((Nbat,1))
            scores,scaledData, _    = self.transform(transformDat) 
            modeledData             = self.inverse_transform(scores,scale='model') 

            scaledData_1            = np.reshape(scaledData, (Nbat, Nvar * timePoints),order='F')
            modeledData_1           = np.reshape(modeledData, (Nbat, Nvar * timePoints), order='F')
            
            resids                  = scaledData_1 - modeledData_1 
            nanMask                 = np.isnan(resids) 
            notNull                 = ~nanMask 

            for row in range(Nbat): 
                outOOMD[row,0]  = resids[row, notNull[row,:]] @ resids[row, notNull[row, :]].T

        elif metric == 'DModX': 
            Nbat_fit, R_1           = self.fitScores.shape 
            Nbat, Nvar, timePoints  = input.shape 
            outOOMD                 = self.calcOOMD(input,metric='QRes') 
            A0                      = 1 
            X_1                     = np.reshape(input, (Nbat, Nvar * timePoints), order='F') 
            nanMask                 = np.isnan(X_1) 
            notNull                 = ~nanMask 

            K                       = notNull.sum(axis=1) 
            factor                  = np.sqrt(Nbat_fit / ((Nbat_fit - R_1 - A0) * (K - R_1))) 
            outOOMD                 = factor.reshape(-1,1) * np.sqrt(outOOMD) 
        else: 
            raise ValueError("Input metric not recognized...") 

        return outOOMD 
















