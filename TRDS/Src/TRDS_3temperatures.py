import sys
from numpy import *
from pylab import *
from scipy import integrate, interpolate
from lmfit import minimize, Parameters, Parameter, report_errors 

#----------------------------------------------------------------------
# constants (gas constant R, parameters from the spectroscopy)
#----------------------------------------------------------------------

R = 8.31451       # gas constant

Te1 = 1451
Te2 = 635
Te3 = 1683
Te4 = 879         # Einstein temperatures

V1 = 2
V2 = 4
V3 = 6
V4 = 6            # weights of Einstein functions


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
              +V2*F_einstein(x,Te2)
              +V3*F_einstein(x,Te3)
              +V4*F_einstein(x,Te4))



def F_vibrational(x,qe1,qe2,qD):

    return R*(11*F_einstein(x,qe1)      # proposed
              +4*F_einstein(x,qe2)      # model of temperatures
              +9*F_debye(x,qD))         # and their weights


#----------------------------------------------------------------------
# lmfit
#----------------------------------------------------------------------

def rsd_lmfit(params,x,y):              # residuals function
                                        # calculating y - y_model
    t_e1 = params['T_E1'].value
    t_e2 = params['T_E2'].value
    t_D = params['T_D'].value
    A = params['A'].value

    int_C = []          # modelled y_model(x)

    for j in range(len(x)):

        ## there could be used an alternative eq. for fitting in this form:
        ## int_C.append(F_spectroscopy(x[j])
        ##           + F_vibrational(x[j],t_e1,t_e2,t_D) + A*x[j]*y[j]**2)

        int_C.append(F_spectroscopy(x[j])
                     + F_vibrational(x[j],t_e1,t_e2,t_D)
                     + A*x[j]*(F_spectroscopy(x[j])
                                + F_vibrational(x[j],t_e1,t_e2,t_D))**2)

    return y - int_C    # residuals



def main():

    #---------------------MAIN CODE-------------------------------------
    # import data
    #----------------------------------------------------------------------

    data=genfromtxt(sys.argv[1])

    T_exp = data[:,0]       # temperature
    C_p = data[:,1]*R       # heat capacity subset (without phase transition)

        # !!!NOTE!!! TRDS_bg.dat file comprises heat capacity,
        # normalized by R: C_p/R(T)
        
    len_T = len(T_exp)      # number of experimental points

    #---------------------------------------------------------------------

    params = Parameters()

    params.add('T_E1',value = 150.0,min = 0)    # proposed set    
    params.add('T_E2',value = 200.0,min = 0)        
    params.add('T_D',value = 100.0,min = 0)         
    params.add('A',value = 1e-7,min = 0)        

    #---------------------------------------------------------------------
    # save temperatures for different number of exp_points (OPTIONAL)
    #---------------------------------------------------------------------

    f0 = open(sys.argv[2],'w')
    f0.write('N_T'+'\t'+'T_E1,K'+'\t'+'T_E2,K'
             +'\t'+'\t'+'T_D,K'+'\n')

    T1_einstein = []        # results of regression
    T2_einstein = []
    T1_debye = []
    N = []

    for i in range(26, len_T, 10):        # perform fitting for number
                                          # of points from 26 to maximum

        params.add('T_E1',value = 150.0,min = 0)    # proposed set    
        params.add('T_E2',value = 200.0,min = 0)        
        params.add('T_D',value = 100.0,min = 0)         
        params.add('A',value = 1e-7,min = 0) 

        rezult = minimize(rsd_lmfit,params,                   # lmfit
                 args=(T_exp[:i],C_p[:i]),method = 'leastsq') # minimize
        
        params = rezult.params

        T1_einstein.append(rezult.params['T_E1'].value)     # lists of
        T2_einstein.append(rezult.params['T_E2'].value)     # results
        T1_debye.append(rezult.params['T_D'].value)
        N.append(i)
        
        f0.write(str(i)+'\t'+str(params['T_E1'].value)      # write
                 +'\t'+str(params['T_E2'].value)            # results
                 +'\t'+str(params['T_D'].value)+'\n')       # to file
        
    f0.close()

    #--------------------Plot section---------------------------------------

    x_spline = linspace(array(N).min(),array(N).max(),300)

                # splines are used for better visualization

    T_E1_smooth = interpolate.spline(N,T1_einstein,x_spline)
    T_E2_smooth = interpolate.spline(N,T2_einstein,x_spline)
    T_D1_smooth = interpolate.spline(N,T1_debye,x_spline)

                # scatter-spline pairs for each temperature

    title(r'Debye and Einstein temperatures of Rb$_3$D(SO$_4$)$_2$',fontsize=18)
    plot(N,T1_einstein,'ro',label=r'$\theta_{E_1}$')
    hold('on')
    plot(x_spline,T_E1_smooth,'--')

    plot(N,T2_einstein,'yo',label=r'$\theta_{E_2}$')
    plot(x_spline,T_E2_smooth,'--')

    plot(N,T1_debye,'bo',label=r'$\theta_D$')
    plot(x_spline,T_D1_smooth,'--')

    xlabel(r'$N_T$ - number of fitting points',fontsize = 15)
    ylabel('Temperature (K)',fontsize = 15)

    yscale('log')
    legend(loc = 2)

    show()

#---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
