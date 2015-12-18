import sys
from numpy import *        # yes, I know I had better specified 'em!
from pylab import *
from scipy import integrate, interpolate, stats
from matplotlib import pyplot
from lmfit import minimize, Parameters, Parameter, report_errors

from numpy import random   # must override numpy (for random.choice)

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

#----------------bootstrapping--------------------------------------

def bootstrap(y,r):         # returns array with randomly sampled
                            # residuals added to y(t)
        rnd=[]
        for z in random.choice(list(r),len(r),replace = True):
            rnd.append(z)
        return y+rnd

#-------------------------------------------------------------------


def main():

    #---------------------MAIN CODE-------------------------------------
    #      -- import data --
    #
    # a special subset of C_p(T) (formed in 'TRDS_bg.dat' file)
    # is used for the regression (no low temperature region, no phase transition)
    #
    # full data, stored in 'TRDS_fullPT.dat' is used later
    # for excess heat capacity calculation
    #
    #----------------------------------------------------------------------

    data = genfromtxt(sys.argv[1])

    T_exp = data[:,0]       # temperature subset (without phase transition)
    C_p = data[:,1]*R       # heat capacity subset (without phase transition)

        # !!!NOTE!!! TRDS_bg.dat file comprises heat capacity,
        # normalized by R: C_p/R(T)

    len_T = len(T_exp)      # number of experimental points

    data_full = genfromtxt(sys.argv[2])

    T_full = data_full[:,0] # temperature (full span)
    C_full = data_full[:,1] # heat capacity (complete data including phase transition)

    #---------------------------------------------------------------------

    params = Parameters()

    params.add('T_E1',value = 150.0,min = 0)    # proposed set    
    params.add('T_E2',value = 200.0,min = 0)        
    params.add('T_D',value = 100.0,min = 0)         
    params.add('A',value = 1e-7,min = 0)        

    #---------------------------------------------------------------------
    # signal fitting procedure
    #---------------------------------------------------------------------


    rezult = minimize(rsd_lmfit,params,                   # lmfit
            args = (T_exp,C_p),method = 'leastsq')        # minimize

    # leastsq, nelder, lbfgsb, anneal, powell, cg, newton, cobyla, slsqp
        
    params = rezult.params
    signal_parameters = params      # saved for report

    residuals = rsd_lmfit(params,T_exp,C_p)   # residuals
                                              # will be used for Runs test

    #----------------------------------------------------------------------
    #report results of calculations
    #----------------------------------------------------------------------

    print('\nLMFIT report:\n')
    report_errors(signal_parameters)    # results from the 'signal' fit

    C_sp = []       # term from spectroscopy

    for i in range(len(T_exp)):
        C_sp.append(F_spectroscopy(T_exp[i]))


    C_vibr = []     # vibrational term with parameters from regression!

    for i in range(len(T_exp)):
        C_vibr.append(F_vibrational(T_exp[i],params['T_E1'].value,
                                    params['T_E2'].value,params['T_D'].value))

    C_calc = []     # full composition of heat capacity

    for i in range(len(T_exp)):
        C_calc.append(C_sp[i] + C_vibr[i]
                      + params['A'].value*T_exp[i]*(C_sp[i]+C_vibr[i])**2)

    #------------------------------------------------------------------------
    # saving results
    #------------------------------------------------------------------------


    C_bg = []       # background (if there's no phase transition)
    DC_p = []       # the excess heat capacity DC_p = (C_full - C_bg)

    for k in range(len(C_full)):
        
        spe_vib = (F_spectroscopy(T_full[k])
                   + F_vibrational(T_full[k],
                                   params['T_E1'].value,
                                   params['T_E2'].value,
                                   params['T_D'].value))
                  #-------------------------------------------------------
                  # a term that happens twice in general C_p(T) formula

        C_bg.append(spe_vib + params['A'].value*T_full[k]*(spe_vib)**2)
        DC_p.append(C_full[k] - C_bg[k])

    f1 = open(sys.argv[3],'w')
    f1.write('T,K'+'\t'+'C_bg'+'\t'+'DC_p'+'\n')

    for i in range(len(T_full)):
        f1.write(str(T_full[i])+'\t'+str(C_bg[i])+'\t'+str(DC_p[i])+'\n')
        
    f1.close()


    #----------------------------------------------------------------------
    # bootstrapping section
    #----------------------------------------------------------------------


    T_E1_bootstr = []       # arrays of computed parameters
    T_E2_bootstr = []       # during bootstrap procedure
    T_D_bootstr = []
    A_bootstr = []

    iteration = int(input('\nHow many bootstrapping iterations? '))
                            # number of boostrap procedures

    for i in range(iteration):
        
        params.add('T_E1',value = 150.0,min = 0)
        params.add('T_E2',value = 200.0,min = 0)
        params.add('T_D',value = 100.0,min = 0)
        params.add('A',value = 1e-7,min = 0)

        rezult = minimize(rsd_lmfit,params,
                         args=(T_exp,bootstrap(C_p,residuals)),
                         method='leastsq')
        params = rezult.params
        
        T_E1_bootstr.append(params['T_E1'].value)
        T_E2_bootstr.append(params['T_E2'].value)
        T_D_bootstr.append(params['T_D'].value)
        A_bootstr.append(params['A'].value)


    #----------------------------------------------------------------------
    # quantiles/errors and means
    #----------------------------------------------------------------------


    err_T_E1 = stats.mstats.mquantiles(T_E1_bootstr,[0.0, 1.0])
    err_T_E2 = stats.mstats.mquantiles(T_E2_bootstr,[0.0, 1.0])
    err_T_D = stats.mstats.mquantiles(T_D_bootstr,[0.0, 1.0])
    err_A = stats.mstats.mquantiles(A_bootstr,[0.0, 1.0])

            # 0.25, 0.75 -quantiles may be used instead of the full span


    #----------------------------------------------------------------------
    # Runs test
    #----------------------------------------------------------------------


    np = nm = 0     # number of positive and negative residuals, respectively
    nR = 1          # observed number of runs (changes of sign)


    if residuals[0] < 0:
        nm += 1

    for i in range(1,len(residuals)):       # loop for calculating
                                            # nm and nR
        if residuals[i] < 0:
            nm += 1

            if residuals[i-1] > 0:
                nR += 1

        elif residuals[i-1] < 0:
            nR += 1

    np = len(residuals) - nm                # np - number of positive residuals

    Runs = 1 + (2*np*nm)/(np + nm)          #expected number of runs

    sigma_R = sqrt(2*nm*np*(2*nm*np - np - nm)/((np + nm - 1)*(np + nm)**2))
                                    #variance of the expected number of runs

    if nR <= Runs:
        Z = (nR - Runs + 0.5)/sigma_R
    else:                                  # estimated standard normal
        Z = (nR - Runs - 0.5)/sigma_R      # distribution (Z-score)


    #----------------------------------------------------------------------
    #report results of calculations (Bootstrap)
    #----------------------------------------------------------------------


    print('\nBootstrap report:\n\nT_E1 =',
          "%.4g" % median(T_E1_bootstr),'\t( -',
          "%.4f" % (100*((median(T_E1_bootstr)-err_T_E1[0])/median(T_E1_bootstr))),'% / +',
          "%.4f" % (100*((err_T_E1[1]-median(T_E1_bootstr))/median(T_E1_bootstr))),'% )')

    print('T_E2 =',"%.4g" % median(T_E2_bootstr),'\t( -',
          "%.4f" % (100*((median(T_E2_bootstr)-err_T_E2[0])/median(T_E2_bootstr))),'% / +',
          "%.4f" % (100*((err_T_E2[1]-median(T_E2_bootstr))/median(T_E2_bootstr))),'% )')

    print('T_D =',"%.4g" % median(T_D_bootstr),'\t( -',
          "%.4f" % (100*((median(T_D_bootstr)-err_T_D[0])/median(T_D_bootstr))),'% / +',
          "%.4f" % (100*((err_T_D[1]-median(T_D_bootstr))/median(T_D_bootstr))),'% )')

    print('A =',"%.4g" % median(A_bootstr),'\t( -',
          "%.4f" % (100*((median(A_bootstr)-err_A[0])/median(A_bootstr))),'% / +',
          "%.4f" % (100*((err_A[1]-median(A_bootstr))/median(A_bootstr))),'% )')

                # NOTE! since the statistical approach
                # has been used, medians are more relevant
                # instead of means

    print('\nRuns test:\n\n Numbers of points:\n n_m =',nm,'\n n_p =',np,'\n\n'
          'Observed number of runs n_R =',nR,'\n'
          'Expected number of runs R =',"%.3f" % Runs,'+/-',"%.3f" % sigma_R,'\n'
          'The standard normal distribution score Z =',"%.3f" % Z,)



    #--------------------Plot section---------------------------------------


    #--------------------------
    # experiment and fitting

    title(r'Decomposition of the heat capacity of Rb$_3$D(SO$_4$)$_2$',fontsize=18)
    plot(T_full,C_full,'bo',label='Experiment')
    hold('on')
    plot(T_exp,C_sp,'r-',label='Internal (Einstein)')
    plot(T_exp,C_vibr,'m-', label='External (Debye & Einstein)')
    plot(T_full,C_bg,'g-',label=r'Including $C_p-C_V$')

    legend(loc=7)
    xlabel(r'$T,K$',fontsize=15)
    ylabel(r'$C_p$, JK$^{-1}$mol$^{-1}$',fontsize=15)
    subplots_adjust(top=0.85)

    show()

    #--------------------------
    # full data and DCp

    subplot(211)    # heat capacity itself

    title(r'Heat capacity of Rb$_3$D(SO$_4$)$_2$',fontsize=18)
    plot(T_full,C_full,'bo')
    plot(T_full,C_bg,'r-')
    xlim([35,165])
    ylim([75,185])

    xlabel(r'$T,K$',fontsize=15)
    ylabel(r'$C_p$, JK$^{-1}$mol$^{-1}$',fontsize=15)

    subplot(212)    # the excess heat capacity

    title(r'The excess heat capacity of Rb$_3$D(SO$_4$)$_2$',fontsize=18)
    plot(T_full,DC_p,'ko')
    pyplot.axhline(linewidth=1, color='r')
    xlim([35,165])
    ylim([-3,13])

    text(100,6,'Phase transition \n $T_C\\approx 78.5 K$',fontsize = 11)
    xlabel(r'$T,K$',fontsize=15)
    ylabel(r'$\Delta C_p$, JK$^{-1}$mol$^{-1}$',fontsize=15)
    subplots_adjust(hspace=0.43)

    show()

    #--------------------------
    # residuals

    title(r'Residuals $C_p - C_{p_{model}}$',fontsize=18)
    stem(T_exp,residuals,linefmt='g--',markerfmt='bs',basefmt='r-')

    xlabel(r'$T,K$',fontsize=15)
    ylabel(r'Unweighted residuals, JK$^{-1}$mol$^{-1}$',fontsize=15)
    show()

    #--------------------------
    # Histograms

    suptitle('Histograms of fitting parameters', fontsize=18)

    subplot(221)
    hist(T_E1_bootstr, color = 'green')
    xlabel(r'$\theta_{E_1}, K$',fontsize=15)
    ylabel('Frequency',fontsize=15)

    subplot(222)
    hist(T_E2_bootstr, color = 'green')
    xlabel(r'$\theta_{E_2}, K$',fontsize=15)
    ylabel('Frequency',fontsize=15)

    subplot(223)
    hist(T_D_bootstr, color = 'green')
    xlabel(r'$\theta_D, K$',fontsize=15)
    ylabel('Frequency',fontsize=15)

    subplot(224)
    hist(A_bootstr, color = 'green')
    xlabel(r'$A$, $J^{-1}mol$',fontsize=15)
    ylabel('Frequency',fontsize=15)
    tight_layout()

    subplots_adjust(top=0.9)
    show()

#---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
