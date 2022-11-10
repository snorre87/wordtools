from scipy.optimize import curve_fit
import numpy as np
def line(x, a, b):
    return a * x + b
def log_line(x,a,b):
    return a*np.log(x)+b
def exp_line(x,a,b,c):
    return a*(np.e**(x*b))+c
def exp_line2(x,a,b,c):
    return a*(np.e**((1/x)*b))+c
def square_func(x,a,b,c):
    return a*x+b*x**2+c

def log_func(x,a,b,c):
  return c*x + np.log(x+1)*a+b

def rsquare(x,y,func,parameters):
    x = np.array(x)
    y = np.array(y)
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    mean = np.mean(y)
    ypred = np.array(func(x,*parameters))
    sum_square = sum((y-mean)**2)
    res = sum((y-ypred)**2)
    return 1-(float(res)/sum_square)

def plot_fit(x,y,add_linear=False,ax=False,parameters=False,linear_only=False):
    func2fit = {}
    if linear_only:
        funcs = {'linear':line}
    else:
        funcs = {'linear':line,'exp':exp_line,'sqrt':square_func,'log':log_func}
    for name,func in funcs.items():

        try:
            popt, pcov = curve_fit(func, x,y,maxfev=2000)
        except:
            continue
        rq = rsquare(x,y,func,popt)
        xfine = np.linspace(min(x),max(x),50)
        func2fit[name] = {'r2':rq,'xfine':xfine,'popt':popt,'func':func}

    if ax==False:
      fig,ax = plt.subplots()
    # pick the best
    else:
      fig = None
    best_name = max(func2fit,key=lambda x: func2fit[x]['r2'])

    best = func2fit[best_name]
    xfine,popt,func,rq = best['xfine'],best['popt'],best['func'],best['r2']
    ax.plot(xfine,func(xfine,*popt),lw=3,label='%s fit, R2: %r'%(best_name,round(rq,2)),alpha=0.5)
    print(best_name)
    if add_linear and best_name!='linear':
        best_name = 'linear'
        best = func2fit[best_name]
        xfine,popt,func,rq = best['xfine'],best['popt'],best['func'],best['r2']
        ax.plot(xfine,func(xfine,*popt),lw=3,label='%s R2: %r'%(best_name,round(rq,)),alpha=0.5)
    ax.legend()
    pred = func(np.array(x),*popt)
    if parameters:
      return fig,ax,rq,pred,func,popt
    else:
      return fig,ax,rq,pred
