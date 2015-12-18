import sys
from numpy import *
from pylab import *
from scipy import integrate, interpolate
from lmfit import minimize, Parameters, Parameter, report_errors 

#----------------------------------------------------------------------
# constants (gas constant R, parameters from the spectroscopy)
#----------------------------------------------------------------------

R = 8.31451       # gas constant

V1, Te1 = 2, 444           # it would be better to create a list,
V2, Te2 = 2, 453           # or even dictionary for these temperatures,
V3, Te3 = 4, 479           # but I prefer this form 
V4, Te4 = 3, 499           # just for convenient visualization
V5, Te5 = 3, 507           # of F_spectroscopy
V6, Te6 = 6, 536
V7, Te7 = 1, 570
V8, Te8 = 1, 589
V9, Te9 = 1, 639
V10, Te10 = 1, 712
V11, Te11 = 2, 1009
V12, Te12 = 1, 1182
V13, Te13 = 1, 1136         # V - weights of Einstein functions
V14, Te14 = 2, 1161         # Te - Einstein temperatures


#--------Integration lambda-function and Debye function-------

xt = lambda x:(x**4*exp(x))/(exp(x) - 1)**2

def F_debye(T,Q):
    y,err = integrate.quad(xt,0,(Q/T))
    return 3*(T/Q)**3*y

#--------Einstein function------------------------------------

def F_einstein(T,E):
    return (E/T)**2*exp(-E/T)/(1 - exp(-E/T))**2


#-----------------------------------------------------------
# sum of F_E and F_D weighted correctly
#-----------------------------------------------------------

def F_spectroscopy(x):

    return R*(V1*F_einstein(x,Te1)
              + V2*F_einstein(x,Te2)
              + V3*F_einstein(x,Te3)
              + V4*F_einstein(x,Te4)
              + V5*F_einstein(x,Te5)
              + V6*F_einstein(x,Te6)
              + V7*F_einstein(x,Te7)
              + V8*F_einstein(x,Te8)
              + V9*F_einstein(x,Te9)
              + V10*F_einstein(x,Te10)
              + V11*F_einstein(x,Te11)
              + V12*F_einstein(x,Te12)
              + V13*F_einstein(x,Te13)
              + V14*F_einstein(x,Te14))



def F_vibrational(x,qe1,qD1,qD2):

    return R*(8*F_einstein(x,qe1)      # proposed
              + 12*F_debye(x,qD1)      # model of temperatures
              + 7*F_debye(x,qD2))      # and their weights (8, 12, 7)
                                       # 2 Debye / 1 Einstein


#----------------------------------------------------------------------
# lmfit
#----------------------------------------------------------------------

def rsd_lmfit(params,x,y):              # residuals function
                                        # calculating y - y_model
    t_E1 = params['T_E1'].value
    t_D1 = params['T_D1'].value
    t_D2 = params['T_D2'].value
    A = params['A'].value

    int_C = []          # modelled y_model(x)

    for j in range(len(x)):

        ## there could be used an alternative eq. for fitting in this form:
        ## int_C.append(F_spectroscopy(x[j])
        ##           + F_vibrational(x[j],t_E1,t_D1,t_D2) + A*x[j]*y[j]**2)

        int_C.append(F_spectroscopy(x[j])
                     + F_vibrational(x[j],t_E1,t_D1,t_D2)
                     + A*x[j]*(F_spectroscopy(x[j])
                                + F_vibrational(x[j],t_E1,t_D1,t_D2))**2)

    return y - int_C    # residuals


def main():

    #---------------------MAIN CODE-------------------------------------
    # import data
    #----------------------------------------------------------------------

    data=genfromtxt(sys.argv[1])

    T_exp = data[:,0]       # temperature
    C_p = data[:,1]         # heat capacity subset (without phase transition)
       
    len_T = len(T_exp)      # number of experimental points

    #---------------------------------------------------------------------

    params = Parameters()

    params.add('T_E1',value = 50.0)    # proposed set    
    params.add('T_D1',value = 250.0)        
    params.add('T_D2',value = 150.0)         
    params.add('A',value = 1e-7,min = 0, max = 1e-7)        

    #---------------------------------------------------------------------
    # save temperatures for different number of exp_points (OPTIONAL)
    #---------------------------------------------------------------------

    f0 = open(sys.argv[2],'w')
    f0.write('N_T'+'\t'+'T_E1,K'+'\t'+'T_D1,K'
             +'\t'+'\t'+'T_D2,K'+'\n')

    T1_einstein = []        # results of regression
    T1_debye = []
    T2_debye = []
    N = []

    for i in range(15, len_T, 10):        # perform fitting for number
                                          # of points from 26 to maximum

        params['T_E1'].value = 50.0       # proposed set    
        params['T_D1'].value = 250.0        
        params['T_D2'].value = 150.0         
        params['A'].value = 1e-7  

        rezult = minimize(rsd_lmfit,params,                   # lmfit
                 args=(T_exp[:i],C_p[:i]),method = 'leastsq') # minimize
        
        params = rezult.params

        T1_einstein.append(rezult.params['T_E1'].value)     # lists of
        T1_debye.append(rezult.params['T_D1'].value)        # results
        T2_debye.append(rezult.params['T_D2'].value)
        N.append(i)
        
        f0.write(str(i)+'\t'+str(params['T_E1'].value)      # write
                 +'\t'+str(params['T_D1'].value)            # results
                 +'\t'+str(params['T_D2'].value)+'\n')       # to file
        
    f0.close()

    #--------------------Plot section---------------------------------------

    x_spline = linspace(array(N).min(),array(N).max(),300)

                # splines are used for better visualization

    T_E1_smooth = interpolate.spline(N,T1_einstein,x_spline)
    T_D1_smooth = interpolate.spline(N,T1_debye,x_spline)
    T_D2_smooth = interpolate.spline(N,T2_debye,x_spline)

                # scatter-spline pairs for each temperature

    title(r'Debye and Einstein temperatures of Pb$_5$Ge$_3$O$_{11}$',fontsize=18)
    plot(N,T1_einstein,'ro',label=r'$\theta_{E_1}$')
    hold('on')
    plot(x_spline,T_E1_smooth,'--')

    plot(N,T1_debye,'co',label=r'$\theta_{D_1}$')
    plot(x_spline,T_D1_smooth,'--')

    plot(N,T2_debye,'bo',label=r'$\theta_{D_2}$')
    plot(x_spline,T_D2_smooth,'--')

    xlabel(r'$N_T$ - number of fitting points',fontsize = 15)
    ylabel('Temperature (K)',fontsize = 15)

    legend(loc = 7)

    show()

#-------------------------------------------------------------------

if __name__ == '__main__':
    main()
