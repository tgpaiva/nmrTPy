import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from Dosyfuncs import conv_grad_b

def save_results(savepath, nameoffile, resultsarray, fitmodel = 'Exp',  Nexp = '1', fiterror = 'n', sterror = 'n', prettyexcel = 'n', onlyD = None):

    if fitmodel == 'Exp':   

        if prettyexcel == 'n':

            if Nexp == 1:

                if fiterror == 'n' and sterror == 'n':

                    try:
                        df = pd.DataFrame(resultsarray,columns= ['P', 'D1'])

                    except:
                        pass

                    df = pd.DataFrame([resultsarray], columns= ['P', 'D1'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

                elif fiterror == 'y' and sterror == 'n':

                    df = pd.DataFrame(resultsarray,columns= ['P', 'D1', 'chisqr'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

                elif fiterror == 'n' and  sterror == 'y':

                    df = pd.DataFrame(resultsarray,columns= ['P', 'D1', 'Perr', 'D1err'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

                elif fiterror == 'y' and sterror == 'y':

                    df = pd.DataFrame(resultsarray,columns= ['P', 'D1', 'Perr', 'D1err', 'chisqr'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

            elif Nexp == 2:

                if fiterror == 'n' and sterror == 'n':

                    df = pd.DataFrame(resultsarray,columns= ['P1', 'P2', 'D1', 'D2'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

                elif fiterror == 'y' and sterror == 'n':

                    df = pd.DataFrame(resultsarray,columns= ['P1','P2','D1','D2','chisqr'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

                elif fiterror == 'n' and  sterror == 'y':

                    df = pd.DataFrame(resultsarray,columns= ['P1','P2','D1','D2','P1err','P2err','D1err','D2err'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

                elif fiterror == 'y' and sterror == 'y':

                    df = pd.DataFrame(resultsarray,columns= ['P1','P2','D1','D2','P1err','P2err','D1err','D2err', 'chisqr'])
                    df.to_csv(savepath + nameoffile + 'out.csv', index=False)

        elif prettyexcel == 'y':

            if onlyD == 'y':

                if Nexp == 1:

                    z = np.empty((0,2))

                    for D, Derr in np.nditer  ([resultsarray[0], resultsarray[1]]):

                        z = np.vstack( (z,  (np.format_float_scientific(D,5),  np.format_float_scientific(Derr,5))))

                    f = np.empty(0)

                    for D, Derr in np.nditer  ([z[:,0], z[:,1]]):

                        q = ( np.array_str(D) + ' ' u"\u00B1" ' ' + np.array_str(Derr))
                        f = np.hstack((q,f))

                    df = pd.DataFrame(f ,columns= ['D'])
                    df.to_excel(savepath + nameoffile + 'out.xlsx')

            else:

                if Nexp == 1:

                    if fiterror == 'n' and sterror == 'n':

                        z = np.empty((0,2))

                        for P , D in np.nditer  ([resultsarray[: ,0], resultsarray[: ,1]]):
                            z  = np.vstack( (  (np.format_float_scientific(P,5),  np.format_float_scientific(D,5))   ,z) ) 


                        df = pd.DataFrame(z ,columns= ['P','D'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')
                    
                    elif fiterror == 'y' and sterror == 'n':

                        z = np.empty((0,3))

                        for P , D, Fiterr in np.nditer  ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2]]):
                            z  = np.vstack(((np.format_float_scientific(P,5),  np.format_float_scientific(D,5), np.format_float_scientific(Fiterr,5)), z)) 


                        df = pd.DataFrame(z ,columns= ['P','D','chisqr'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')
                    

                    elif fiterror == 'n' and  sterror == 'y':

                        z = np.empty((0,4))

                        for P , D, Perr, Derr in np.nditer  ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2],resultsarray[ :,3] ]):
                            z  = np.vstack(((np.format_float_scientific(P,5),  np.format_float_scientific(D,5),np.format_float_scientific(Perr,5),np.format_float_scientific(Derr,5))   ,z) ) 

                        f = np.empty((0,2))

                        for P , D, Perr, Derr in np.nditer  ([z[: ,0], z[: ,1], z[ :,2],z[ :,3] ]):
                            q , w = [( np.array_str(P) + ' ' u"\u00B1" ' ' + np.array_str(Perr)), ( np.array_str(D) + ' ' u"\u00B1" ' ' + np.array_str(Derr))]
                            f = np.vstack((f,(q,w)))

                        df = pd.DataFrame(f ,columns= ['P','D'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')

                    elif fiterror == 'y' and sterror == 'y':

                        z = np.empty((0,5))

                        for P , D, Perr, Derr, Fiterr in np.nditer  ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2], resultsarray[ :,3], resultsarray[ :,4] ]):
                            z  = np.vstack( (  (np.format_float_scientific(P,5),  np.format_float_scientific(D,5),np.format_float_scientific(Perr,5), np.format_float_scientific(Derr,5), np.format_float_scientific(Fiterr,5) )   ,z) ) 

                        f = np.empty((0,3))

                        for P , D, Perr, Derr, Fiterr in np.nditer  ([z[: ,0], z[: ,1], z[ :,2], z[ :,3], z[ :,4]]):
                            q , w, s = [( np.array_str(P) + ' ' u"\u00B1" ' ' + np.array_str(Perr)), ( np.array_str(D) + ' ' u"\u00B1" ' ' + np.array_str(Derr)), Fiterr]
                            f = np.vstack((f, (q,w,s)))

                        df = pd.DataFrame(f ,columns= ['P','D', 'chisqr'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')
                    
                elif Nexp == 2:

                    if fiterror == 'n' and sterror == 'n':

                        z = np.empty((0,4))

                        for P1 ,P2, D1, D2 in np.nditer ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2], resultsarray[ :,3], ]):
                            z  = np.vstack( (  (np.format_float_scientific(P1,5), np.format_float_scientific(P2,5) , np.format_float_scientific(D1,5), np.format_float_scientific(D2,5)) ,z) ) 

                        df = pd.DataFrame(z ,columns= ['P1','P2','D1','D2'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')    

                    elif fiterror == 'y' and sterror == 'n':

                        z = np.empty((0,5))

                        for P1 ,P2, D1, D2, Fiterr in np.nditer ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2], resultsarray[ :,3], resultsarray[ :,4]]):
                            z  = np.vstack( (  (np.format_float_scientific(P1,5), np.format_float_scientific(P2,5) , np.format_float_scientific(D1,5), np.format_float_scientific(D2,5), np.format_float_scientific(Fiterr,5)  ) ,z) ) 

                        df = pd.DataFrame(z ,columns= ['P1','P2','D1','D2','chisqr'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')                            
                    
                    elif fiterror == 'n' and  sterror == 'y':

                        z = np.empty((0,8))

                        for P1 ,P2, D1, D2, P1err, P2err, D1err, D2err in np.nditer ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2], resultsarray[ :,3], resultsarray[ :,4], resultsarray[ :,5], resultsarray[ :,6], resultsarray[ :,7]]):
                            z  = np.vstack(((np.format_float_scientific(P1,5), np.format_float_scientific(P2,5) , np.format_float_scientific(D1,5), np.format_float_scientific(D2,5) , np.format_float_scientific(P1err,5), np.format_float_scientific(P2err,5), np.format_float_scientific(D1err,5),np.format_float_scientific(D2err,5))   ,z) ) 

                        f = np.empty((0,4))

                        for P1 ,P2, D1, D2, P1err, P2err, D1err, D2err in np.nditer  ([z[: ,0], z[: ,1], z[ :,2],z[ :,3],z[ :,4],z[ :,5],z[ :,6],z[ :,7]]):
                            q1 ,w1, q2, w2 = [( np.array_str(P1) + ' ' u"\u00B1" ' ' + np.array_str(P1err)), 
                            ( np.array_str(P2) + ' ' u"\u00B1" ' ' + np.array_str(P2err)), 
                            ( np.array_str(D1) + ' ' u"\u00B1" ' ' + np.array_str(D1err)), 
                            ( np.array_str(D2) + ' ' u"\u00B1" ' ' + np.array_str(D2err))]
                            f = np.vstack((f, (q1, w1, q2, w2)))

                        df = pd.DataFrame(f ,columns= ['P1','P2','D1','D2'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')

                    elif fiterror == 'y' and sterror == 'y':

                        z = np.empty((0,9))

                        for P1 ,P2, D1, D2, P1err, P2err, D1err, D2err, Fiterr in np.nditer ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2], resultsarray[ :,3], resultsarray[ :,4], resultsarray[ :,5], resultsarray[ :,6], resultsarray[ :,7], resultsarray[ :,8]]):
                            z  = np.vstack(((np.format_float_scientific(P1,5), np.format_float_scientific(P2,5) , np.format_float_scientific(D1,5), np.format_float_scientific(D2,5) , np.format_float_scientific(P1err,5), np.format_float_scientific(P2err,5), np.format_float_scientific(D1err,5),np.format_float_scientific(D2err,5),np.format_float_scientific(Fiterr,5))   ,z) ) 

                        f = np.empty((0,5))

                        for P1 ,P2, D1, D2, P1err, P2err, D1err, D2err, Fiterr in np.nditer  ([z[: ,0], z[: ,1], z[ :,2],z[ :,3],z[ :,4],z[ :,5],z[ :,6],z[ :,7], z[ :,8]]):
                            q1 ,w1, q2, w2, chisqr = [( np.array_str(P1) + ' ' u"\u00B1" ' ' + np.array_str(P1err)), 
                            ( np.array_str(P2) + ' ' u"\u00B1" ' ' + np.array_str(P2err)), 
                            ( np.array_str(D1) + ' ' u"\u00B1" ' ' + np.array_str(D1err)), 
                            ( np.array_str(D2) + ' ' u"\u00B1" ' ' + np.array_str(D2err)), Fiterr]
                            f = np.vstack((f, (q1, w1, q2, w2, chisqr)))

                        df = pd.DataFrame(f ,columns= ['P1', 'P2', 'D1', 'D2', 'chisqr'])
                        df.to_excel(savepath + nameoffile + 'out.xlsx')

    elif fitmodel == 'StretchedEXP':

        z = np.empty((0,6))

        for P ,D, B, Perr, Derr, Berr in np.nditer ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2], resultsarray[ :,3], resultsarray[ :,4], resultsarray[ :,5]]):
            z  = np.vstack( (  (np.format_float_scientific(P,2), np.format_float_scientific(D,2) , np.format_float_scientific(B,2), np.format_float_scientific(Perr,2) , np.format_float_scientific(Derr,2), np.format_float_scientific(Berr,2)),z ) )

        f = np.empty((0,3))

        for P ,D, B, Perr, Derr, Berr in np.nditer  ([z[: ,0], z[: ,1], z[ :,2],z[: ,3],z[: ,4],z[: ,5] ]):
            Pop , Diffcoeff, Betavalue = [( np.array_str(P) + ' ' u"\u00B1" ' ' + np.array_str(Perr)), ( np.array_str(D) + ' ' u"\u00B1" ' ' + np.array_str(Derr)), ( np.array_str(B) + ' ' u"\u00B1" ' ' + np.array_str(Berr))]
            f = np.vstack(((Pop, Diffcoeff, Betavalue),f))

        df = pd.DataFrame(f ,columns= ['P','D','B'])
        df.to_excel(savepath + nameoffile + 'out.xlsx')

    elif fitmodel == 'ARR':

        if prettyexcel == 'y':

            if sterror == 'y':
              
                z = np.empty((0,4))

                for P , D, Perr, Derr in np.nditer  ([resultsarray[: ,0], resultsarray[: ,1], resultsarray[ :,2],resultsarray[ :,3] ]):

                    z  = np.vstack( (z,  (np.format_float_scientific(P,2),  np.format_float_positional(D/1E3, 2 ), np.format_float_scientific(Perr,2), np.format_float_positional(Derr/1E3, 2 ))   ) ) 

                f = np.empty((0,2))

                for P , D, Perr, Derr in np.nditer  ([z[: ,0], z[: ,1], z[ :,2],z[ :,3] ]):
                    q , w = [( np.array_str(P) + ' ' u"\u00B1" ' ' + np.array_str(Perr)), ( np.array_str(D) + ' ' u"\u00B1" ' ' + np.array_str(Derr))]
                    f = np.vstack((f,(q,w)))

                df = pd.DataFrame(f ,columns= ['tau','Ea'])
                df.to_excel(savepath + nameoffile + 'out.xlsx')
   
    return print(df)



def residual(params, x, y_data, Nexp, eps = None):

    if Nexp == 1:
        
        A1 = params['A1']
        D1 = params['D1']

        model =  A1 * np.exp(-x * D1)

        if eps is None:
        
            return (model - y_data)
        
        else:
        
            return (model-y_data) / eps

    elif Nexp == 2:
        
        A1 = params['A1']
        A2 = params['A2']
        D1 = params['D1']
        D2 = params['D2']

        model = A1 * np.exp(-x * D1) + A2 * np.exp(-x * D2)

        if eps is None:
        
            return (model - y_data)
        
        else:
        
            return (model-y_data) / eps

    return 

def T1fit(params, x, y_data, fitmodel = None, Nexp = None, eps = None, fitchisqr = None, stderror = None):

    if fitmodel == 'Exp':

        out = minimize(residual, params, args=(x, y_data , Nexp, eps), method='leastsq')
        fit = np.array(out.params)
        
        if Nexp == 1: 

            if stderror == 'y':
        
                paramserror = np.array((out.params['A1'].stderr, out.params['D1'].stderr))
                fit = np.hstack((fit, paramserror))
        
            else:
                pass
            
        elif Nexp == 2:

            if stderror == 'y':
         
                paramserror = np.array((out.params['A1'].stderr, out.params['A2'].stderr, out.params['D1'].stderr, out.params['D2'].stderr))
                fit = np.hstack((fit, paramserror))
         
            else:
                pass

        if fitchisqr == 'y':
            
            fiterror = np.array(out.chisqr)
            fit = np.hstack((fit, fiterror))

        else:
            pass

        plt.figure()
        plt.plot(x,out.residual,'ko')
        plt.plot(x,np.repeat(0,len(x)),'r--')

        if Nexp == 1:
        
            def func(x,A1,D1):
                return A1 * np.exp(-x * D1)
            
            yEXP=func(x,fit[0],fit[1])
            plt.figure()
            plt.plot(x,y_data,'ko')
            plt.plot(x,yEXP,'r--')
            plt.yscale('log')

        elif Nexp == 2:
        
            def func(x,A1,A2,D1,D2):
                return A1 * np.exp(-x * D1) + A2 * np.exp(-x * D2)

            yEXP=func(x,fit[0],fit[1],fit[2],fit[3])
            plt.figure()
            plt.plot(x,y_data,'ko')
            plt.plot(x,yEXP,'r--')
            plt.yscale('log')
            # plt.xscale('log')

    elif fitmodel == 'StretchedEXP':

        out = minimize(others_residual, params, args = (x, y_data,  fitmodel ,eps), method = 'leastsq')
        fit=np.array(out.params)
        
        if stderror == 'y':
            paramserror = np.array((out.params['A'].stderr, out.params['D'].stderr, out.params['Beta'].stderr))
            fit = np.hstack((fit, paramserror))
     
        else:
            pass
            
        if fitchisqr == 'y':
            fiterror = np.array (out.chisqr)
            fit = np.hstack((fit, fiterror))
       
        else:
            pass

        # PLOT fitting

        plt.figure()
        plt.plot(x,out.residual,'ko')
        plt.plot(x,np.repeat(0,len(x)),'r--')

        def func(x,A,D, Beta):
            return A * np.exp(-(x * D) ** Beta)
       
        yEXP = func(x,fit[0],fit[1],fit[2])
        plt.figure()
        plt.plot(x,y_data,'ko')
        plt.plot(x,yEXP,'r--')
        plt.yscale('log')

        # plt.xscale('log')

    elif fitmodel == 'Gamma':

        out = minimize(others_residual, params, args = (x, y_data,  fitmodel ,eps), method = 'leastsq')
        fit=np.array(out.params)

        if stderror == 'y':

            paramserror = np.array((out.params['A'].stderr, out.params['D'].stderr, out.params['Sigma'].stderr))
            fit = np.hstack((fit, paramserror))
       
        else:
            pass
            
        if fitchisqr == 'y':
  
            fiterror = np.array (out.chisqr)
            fit = np.hstack((fit, fiterror))
       
        else:
            pass

        # PLOT fitting

        plt.figure()
        plt.plot(x,out.residual,'ko')
        plt.plot(x,np.repeat(0,len(x)),'r--')

        def func(x,A, D, Sigma):
            return A * (1 + x * Sigma ** 2 / D) ** - (D ** 2 / Sigma ** 2) 

        yEXP=func(x,fit[0],fit[1],fit[2])
        plt.figure()
        plt.plot(x,y_data,'ko')
        plt.plot(x,yEXP,'r--')
        plt.yscale('log')

    return fit
    
def get_Parameters(Nexp):

    params = Parameters()
    
    if Nexp == 1:
    
        params.add('A1', value=0.5,min = 0 , max = 2)
        params.add('D1', value=1E-10,min=1E-15, max = 1E-7)
    
    elif Nexp == 2:
    
        params.add('A1', value=0.5,min = 0, max =1)
        params.add('A2', expr = '1 - A1')
        params.add('D1', value= 1E-12,min=1E-15,max=1E-9)
        params.add('D2', value= 1E-14,min=1E-15,max=1E-9)
    
    return params


def fit_routine(grad_matrix, data_array, PeakArray, fitmodel = None, Nexp = None, eps = None, norm = None, fiterror = None, sterror = None, printonlyD = None, DOSY = False, save = 'n', datasetname = None, prettyexcel_ = 'n'):

    if fitmodel == 'Exp':
    
        if eps != None:

            nrcol = 2

            if Nexp == 1:

                if fiterror == 'y' and sterror == 'n':
                    nrcol = 3

                elif fiterror == 'n' and  sterror == 'y':
                    nrcol = 4

                elif fiterror == 'y' and sterror == 'y':
                    nrcol = 5

            elif Nexp == 2:

                nrcol = 4
                
                if fiterror == 'y' and sterror == 'n':            
                    nrcol = 5
                
                elif fiterror == 'n' and sterror == 'y':                
                    nrcol = 8
                
                elif fiterror == 'y' and sterror == 'y':
                    nrcol = 9

            Results = np.empty((PeakArray.shape[2], nrcol , 0))
        
        if len(PeakArray.shape) == 4:

            for spect in range(len(PeakArray)):

                Results_spect= np.empty((0, nrcol))

                for peak in range (PeakArray.shape[2]):
                    bvalues = grad_matrix[spect]

                    if DOSY == 'y':                   
                        bvalues = conv_grad_b(bvalues, data_array[spect])

                    if eps != None:                   
                        epsilon = PeakArray[spect,:,peak,1]
                    
                    else:
                        epsilon = None

                    onepeak = PeakArray[spect,:,peak,0]

                    if norm == 'y':

                        onepeak = onepeak/onepeak[0]
                        onepeak = [i for i in onepeak if i >= 0.05]
                        bvalues = bvalues[:len(onepeak)]
                        
                        if eps != None:                        
                            epsilon = epsilon[:len(onepeak)]

                    else:
                        pass

                    par=get_Parameters(Nexp)
                    fitresults=T1fit(par, bvalues, onepeak,fitmodel, Nexp, eps = epsilon, fitchisqr = fiterror, stderror = sterror)
                    Results_spect = np.vstack((fitresults,Results_spect))   
                Results = np.dstack((Results_spect, Results))

            if printonlyD == 'y':
                
                if Nexp == 1:
                    
                    if len(Results) > 1:

                        erro = np.sqrt(np.sum(np.power(Results[:,3,:],2),axis = 0)) / len(Results)
                        Results = np.average(Results[:,1,:], axis = 0)
                        Results = np.vstack([Results,erro])
                    
                    else:
                        
                        Results = Results[0,1,:]
                # elif Nexp == 2:
            else:
                pass

        elif len(PeakArray.shape) == 3:
            
            Results_spect = np.empty((0, nrcol))

            for peak in range (PeakArray.shape[1]):
                bvalues = grad_matrix

                if DOSY == 'y':

                    bvalues = conv_grad_b(grad_matrix, data_array)

                onepeak = PeakArray[:,peak,0]

                if eps != None:

                    epsilon = PeakArray[:,peak,1]
                
                else:
                
                    epsilon = None

                if norm == 'y':

                    onepeak = onepeak/onepeak[0]
                    onepeak = [i for i in onepeak if i >= 0.05]
                    bvalues = bvalues[:len(onepeak)]

                    if eps != None:

                        epsilon = epsilon[:len(onepeak)]
                else:
                    pass

                par=get_Parameters(Nexp)
                fitresults=T1fit(par, bvalues, onepeak, fitmodel , Nexp, eps = epsilon, fitchisqr = fiterror, stderror = sterror)
                Results_spect = np.vstack((fitresults,Results_spect))   
                Results = Results_spect
                
            # Results = np.dstack((Results_spect, Results))

            if printonlyD == 'y':

                if Nexp == 1:

                    if len(Results) > 1:

                        erro = np.sqrt(np.sum(np.power(Results[:,3],2),axis = 0)) / len(Results)
                        Results = np.average(Results[:,1],axis = 0)
                        Results = np.vstack([Results,erro])

                    else:
                        
                        Results = Results[0,1,:]
                    # elif Nexp == 2:
            else:
                pass

########################################################################################

        elif PeakArray.shape[1] == 1:

            onepeak = PeakArray[:,0]

            if DOSY == 'y':                   
                bvalues = conv_grad_b(bvalues, data_array)

            bvalues = grad_matrix

            if norm == 'y':
                onepeak = onepeak/onepeak[0]
                onepeak = [i for i in onepeak if i >= 0.05]
                bvalues = bvalues[:len(onepeak)]

            par=get_Parameters(Nexp)
            fitresults = T1fit(par, bvalues, onepeak, fitmodel , Nexp, eps = eps, fitchisqr = fiterror, stderror = sterror)

            Results = fitresults
            # Results_spect = np.vstack((fitresults, Results_spect))   
            # Results = Results_spect            

####################################################################################################################################
    else:

        nrcol = 3
        
        if fiterror == 'y' and sterror == 'n':

            nrcol = 4
        
        elif fiterror == 'n' and  sterror == 'y':

            nrcol = 6
        
        elif fiterror == 'y' and sterror == 'y':

            nrcol = 7

        Results_spect= np.empty((0, nrcol))

        for peak in range (PeakArray.shape[1]):

            bvalues = grad_matrix

            if DOSY == 'y':
                    
                    bvalues = conv_grad_b(grad_matrix, data_array)

            onepeak = PeakArray[:,peak,0]

            if eps != None:

                epsilon = PeakArray[:,peak,1]

            else:
                
                epsilon = None

            if norm == 'y':

                onepeak = onepeak/onepeak[0]
                onepeak = [i for i in onepeak if i >= 0.05]
                bvalues = bvalues[:len(onepeak)]

                if eps != None:

                    epsilon = epsilon[:len(onepeak)]

            else:
                pass

            par = other_parameters(fitmodel)

            fitresults = T1fit(par, bvalues, onepeak, fitmodel, eps = epsilon, fitchisqr = fiterror, stderror = sterror)
            Results_spect = np.vstack((fitresults, Results_spect))   
            Results = Results_spect

    if save == 'y':

        save_results('/Users/tiagopaiva/', datasetname, Results, fitmodel = fitmodel , Nexp = Nexp, fiterror = fiterror, sterror = sterror, prettyexcel = prettyexcel_)

    return Results

#%% Other models

def other_parameters(fitmodel):

    if fitmodel == 'StretchedEXP':

        params = Parameters()
        params.add('A', value = 1)
        params.add('D', value=1E-12, min=1E-15, max = 1E-10)
        params.add('Beta', value = 0.5, min = 0, max = 0.9)

    elif fitmodel == 'Gamma':

        params = Parameters()
        params.add('A', value = 1, min = 0.7, max = 2)
        params.add('D', value=1E-12, min=1E-15, max = 1E-10)
        params.add('Sigma', value = 1E-12, min = 1E-15, max = 1E-10)

    return params


def others_residual(params, x, y_data, fitmodel = None, eps=None):
    
    if fitmodel == 'StretchedEXP':
    
        A = params['A']
        D = params['D']
        Beta = params['Beta']

        model =  A * np.exp(-(x * D) ** Beta)

        if eps is None:

            return (model - y_data)
    
        else:

            return (model - y_data) / eps

    elif fitmodel == 'Gamma':

        A = params['A']
        D = params['D']
        Sigma = params['Sigma']

        model =  A * (1 + x * Sigma ** 2 / D) ** - (D ** 2 / Sigma ** 2) 

        if eps is None:

            return (model - y_data)
    
        else:
            return (model - y_data) / eps
    
    return         

def fit_others_routine(grad_matrix, data_array, PeakArray, Fitmodel = None, eps = None, norm = None, fiterror = None, sterror = None, DOSY = None):

    nrcol = 3

    if fiterror == 'y' and sterror == 'n':
        nrcol = 4
    
    elif fiterror == 'n' and  sterror == 'y':
        nrcol = 6
    
    elif fiterror == 'y' and sterror == 'y':
        nrcol = 7

    Results_spect= np.empty((0, nrcol))

    for peak in range (PeakArray.shape[1]):

        bvalues = grad_matrix

        if DOSY == 'y':
                
                bvalues = conv_grad_b(grad_matrix, data_array)

        onepeak = PeakArray[:,peak,0]

        if eps == 'y':

            epsilon = PeakArray[:,peak,1]

        else:
            epsilon = None

        if norm == 'y':

            onepeak = onepeak/onepeak[0]
            onepeak = [i for i in onepeak if i >= 0.05]
            bvalues = bvalues[:len(onepeak)]

            if eps == 'y':

                epsilon = epsilon[:len(onepeak)]
        else:
            pass

        par = other_parameters(Fitmodel)

        fitresults = T1fit(par, bvalues, onepeak, Fitmodel, eps = epsilon, fitchisqr = fiterror, stderror = sterror)
        Results_spect = np.vstack((fitresults,Results_spect))   

    return Results_spect 
