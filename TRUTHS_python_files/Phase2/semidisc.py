import subprocess as subproc
import io
from dataclasses import dataclass

import numpy as np

@dataclass
class geom:
    vza: float
    vaa: float
    sza: float
    saa: float

class semiD:

    def __init__(self):

        #self.semiD_exe="/home/tristan/src/nadim/front_end/nadim"
        self.semiD_exe="/home/users/ndouglas/TRUTHS/semiDiscrete/nadim"

        self.wvl=[640.,850.,950.]
        
        self.params={}
        self.params['LAI']=None


    def gen_input(self,geom_list):
        """Generate and angles and wavelength 
        file for semi discrete
        """
        
        f=io.StringIO()
        #nangs=len(self.vza)
        nangs=len(geom_list)
        nwvls=len(self.wvl)
        print(nangs,nwvls,end=" ",file=f)
        for w in self.wvl:
            print("%0.1f "%w,end="",file=f)
        print("",file=f)
        #for i in range(nangs):
            #print(f"{self.vza[i]:.2f}",end=" ",file=f) 
            #print(f"{self.vaa[i]:.2f}",end=" ",file=f) 
            #print(f"{self.sza[i]:.2f}",end=" ",file=f) 
            #print(f"{self.saa[i]:.2f}",file=f)
        for g in geom_list:
            print(f"{g.vza:.2f}",end=" ",file=f) 
            print(f"{g.vaa:.2f}",end=" ",file=f) 
            print(f"{g.sza:.2f}",end=" ",file=f) 
            print(f"{g.saa:.2f}",file=f)
   
        return(f)

    def gen_args(self):

        args=[self.semiD_exe,]
        for arg in self.params:
            if self.params[arg] is not None:
                args.append("-%s"%arg)
                args.append(str(self.params[arg]))
        
        return(args)

    def run(self,geom_list):

        #nangs=len(self.vza)
        #nwvls=len(self.wvl)
        #refl=np.zeros([nangs,nwvls])  
        refl=np.zeros([len(geom_list), len(self.wvl)])
        
        angs=self.gen_input(geom_list)
        args=self.gen_args()
        p=subproc.run(args,input=angs.getvalue(),encoding='ascii',stdout=subproc.PIPE)
        for (i,line) in enumerate(p.stdout.split("\n")):
            if i==0:
                continue
            fields=line.split()
            for j in range(4,len(fields)):
                refl[i-1,j-4]=float(fields[j])
         
        return(refl)


if __name__=="__main__":

    geom_list = []
    geom_list.append(geom(vza=20., vaa=30., sza=0., saa=0.))
    geom_list.append(geom(vza=30., vaa=40., sza=10., saa=0.))
    geom_list.append(geom(vza=45., vaa=50., sza=10., saa=0.))
    geom_list.append(geom(vza=40., vaa=0., sza=30., saa=0.))
    
    s=semiD()
    s.params['LAI']=1.0
    s.params['cab']=100.3
    print(s.run(geom_list))
    
    
