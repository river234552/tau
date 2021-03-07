import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import torch
import pdb
import math
from vast.DistributionModels.weibull import weibull



# train  = np.genfromtxt('tauijbb-32-initial-pairs-gaps.csv',delimiter=',')
# ijbbdata = torch.tensor(train,dtype=torch.float32)
# train  = np.genfromtxt('mnist-initial-pairs-gaps.csv',delimiter=',')
# mnist = torch.tensor(train,dtype=torch.float32)
# train  = np.genfromtxt('LFW-initial-pairs-gaps.csv',delimiter=',')
# lfwdata = torch.tensor(train,dtype=torch.float32)
# train  = np.genfromtxt('IJBB_512-initial-pairs-gaps.csv',delimiter=',')
# ijbb512 = torch.tensor(train,dtype=torch.float32)


tw =  weibull(translateAmountTensor=.001)

#tw =  weibull()


def nan_to_num(t,mynan=0.):
    if torch.all(torch.isfinite(t)):
        return t
    if len(t.size()) == 0:
        return torch.tensor(mynan)
    return torch.cat([nan_to_num(l).unsqueeze(0) for l in t],0)


def get_tau(data,maxval,name,tailfrac=.999,pcent=.999,usehigh=True,maxmodeerror=.05):
    tau = -1
    while(tau < 0):
      nbin=100
      nscale = 10
      fullrange = torch.linspace(0,maxval,nbin)
      fsize = max(3,int(tailfrac*len(data)))    
      if(usehigh):
          tw.FitHighTrimmed(data.view(1,-1),fsize)
      else:
          tw.FitLowReversed(data.view(1,-1),fsize)
      parms = tw.return_all_parameters()
      if(usehigh):
          tau=  parms['Scale']*np.power(-np.log((1-pcent)),(1/parms['Shape'])) - parms['translateAmountTensor'] + parms['smallScoreTensor']
      else:
          tau = parms['translateAmountTensor']- parms['smallScoreTensor']-(parms['Scale']*np.power(-np.log((pcent)),(1/parms['Shape'])))
      if(math.isnan(tau)):
          print( name , "Parms", parms)        
          tau = torch.mean(data)
      wmode = float(parms['translateAmountTensor']- parms['smallScoreTensor']+ (parms['Scale']*np.power((parms['Shape']-1)/(parms['Shape']),1./parms['Shape']          )))

      wscoresj = tw.wscore(fullrange)
      probj = nan_to_num(tw.prob(fullrange))
      if(torch.sum(probj) > .001):
          probj = probj/torch.sum(probj)
  #    print( name , "Prob mean, max", torch.mean(probj),torch.max(probj))
      datavect=data.numpy()
      histc,hbins = np.histogram(datavect,bins=nbin,range=[0,1])
      imode = hbins[np.argmax(histc[0:int(tau*nbin+1)])]
  #    print( name , "Data mode min median mean std, max", round(imode,3),round(imin,3),round(imedian,3),round(imean,3),round(istd,3),round(imax,3))
      merror = abs(imode-wmode)
      if(merror > maxmodeerror):
          #print( "   Outliers detected imode wmode modeerror ", round(imode,3),round(wmode,3),merror)
          #outlier detected, reduce tail fraion and force loop
          tailfrac = tailfrac - .05
          tau = -1
    print(name," EVT Tau with datafraction ", round(tailfrac*100,2)," Percentile ",pcent*100,"   is ", float(tau.numpy()))
    return tau
    

def get_tau_and_plot(data,maxval,name,tailfrac=.999,pcent=.999,usehigh=True,maxmodeerror=.05):
    tau = -1
    while(tau < 0):
      nbin=100
      nscale = 10
      fullrange = torch.linspace(0,maxval,nbin)
      torch.Tensor.ndim = property(lambda self: len(self.shape))
      shape = 10;
      fsize = int(len(data))
      imin = float(torch.min(data[:fsize]))
      imedian = float(torch.median(data[:fsize]))            
      imean = float(torch.mean(data[:fsize]))
      istd = float(torch.std(data[:fsize]))
      imax = float(torch.max(data[:fsize]))
      fsize = max(3,int(tailfrac*len(data)))    
      if(usehigh):
          tw.FitHighTrimmed(data.view(1,-1),fsize)
          #            tw.FitHigh(data.view(1,-1),int(tailfrac*len(data)))        
      else:
          tw.FitLowReversed(data.view(1,-1),fsize)
          #        tw.FitLow(data.view(1,-1),int(tailfrac*len(data)))        
      parms = tw.return_all_parameters()
          #        print( name , "Parms", parms)
      if(usehigh):
          tau=  parms['Scale']*np.power(-np.log((1-pcent)),(1/parms['Shape'])) - parms['translateAmountTensor'] + parms['smallScoreTensor']
      else:
          tau = parms['translateAmountTensor']- parms['smallScoreTensor']-(parms['Scale']*np.power(-np.log((pcent)),(1/parms['Shape'])))
      if(math.isnan(tau)):
          print( name , "Parms", parms)        
          tau = torch.mean(data)
      wmode = float(parms['translateAmountTensor']- parms['smallScoreTensor']+ (parms['Scale']*np.power((parms['Shape']-1)/(parms['Shape']),1./parms['Shape']          )))
      wscoresj = tw.wscore(fullrange)
      probj = nan_to_num(tw.prob(fullrange))
      if(torch.sum(probj) > .001):
          probj = probj/torch.sum(probj)
  #    print( name , "Prob mean, max", torch.mean(probj),torch.max(probj))
      datavect=data.numpy()
      histc,hbins = np.histogram(datavect,bins=nbin,range=[0,1])
      imode = hbins[np.argmax(histc[0:int(tau*nbin+1)])]
  #    print( name , "Data mode min median mean std, max", round(imode,3),round(imin,3),round(imedian,3),round(imean,3),round(istd,3),round(imax,3))
      merror = abs(imode-wmode)
      if(merror > maxmodeerror):
          #print( "   Outliers detected imode wmode modeerror ", round(imode,3),round(wmode,3),merror)
          #outlier detected, reduce tail fraion and force loop
          tailfrac = tailfrac - .05
          tau = -1
    print(name," EVT Tau with datafraction ", round(tailfrac*100,2)," Percentile ",pcent*100,"   is ", float(tau.numpy()))
    
    probj = nscale*probj
    plt.plot(fullrange,wscoresj.view(-1,1).numpy(),label="Weibull CDF");
    plt.plot(fullrange,(probj.view(-1,1).numpy()),label="Scaled Weibull PDF");
    histc,hbins = np.histogram(datavect,bins=nbin,range=[0,1])
    plt.plot(fullrange,nscale*histc/(np.sum(histc)),label="Scaled Data frequency");
    plt.axvline(x=tau,ymin=0,ymax=1,color='r',ls=":",label="Tau")
    legend = plt.legend(loc='center right', shadow=True, fontsize='x-large')
    tstring  = "EVT-modeling " +name + "   Tau = "+str(np.round(float(tau),4))
    title = plt.title(loc='center',  fontsize='x-large',label=tstring)
    tmeanstring= 'data mean='+str(round(float(imean),3))+'mode='+str(round(float(imode),3))+'  std='+str(round(float(istd),3))+'  max=' +str(round(float(imax),3))
    plt.text(.35,.9,tmeanstring)

    plt.savefig(name+"-fit.png", format="png")
    plt.clf()
    return tau

        
dfrac = 1.0
probgoal=.99
usehigh=True
#usehigh=False
maxmodeerror=.05

csvfiles=[ "IJBB_32-nearest-neighbor-points.csv", "IJBB_64-nearest-neighbor-points.csv", "IJBB_128-nearest-neighbor-points.csv","IJBB_256-nearest-neighbor-points.csv", "IJBB_512-nearest-neighbor-points.csv", "IJBB_1024-nearest-neighbor-points.csv", "IJBB_1845-nearest-neighbor-points.csv",  "mnist-nearest-neighbor-points.csv", "imagenet_2012-nearest-neighbor-points.csv","LFW-nearest-neighbor-points.csv"]	     

taudict=dict()
for x in csvfiles:
    title=x.split('-')[0]
    if(not title in taudict):
        train  = np.genfromtxt(x,delimiter=',')
        data = torch.tensor(train,dtype=torch.float32)
        tau = get_tau_and_plot(data,1,title,dfrac,probgoal,usehigh,maxmodeerror)
        if tau>0:
            taudict[title]=float(tau)
            
print("LFw with just tau no plot should match" ,float(get_tau(data,1,title,dfrac,probgoal,usehigh,maxmodeerror)))

print (" Results:" ,taudict)
