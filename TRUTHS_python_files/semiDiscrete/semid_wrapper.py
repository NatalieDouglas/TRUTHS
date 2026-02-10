import subprocess as subproc
import io

import numpy as np

class semiD:

    def __init__(self):

        self.semiD_exe="/home/tristan/src/nadim/front_end/nadim"

        self.vza=[20.,30.,45.,40,]
        self.sza=[ 0.,10.,10.,20]
        self.vaa=[30.,40.,50.,20]
        self.saa=[ 0.,0.,0.,0]

        self.wvl=[640.,850.,950.]
        
        self.params={}
        self.params['LAI']=None


    def gen_input(self):
        """Generate and angles and wavelength 
        file for semi discrete
        """
        
        f=io.StringIO()
        nangs=len(self.vza)
        nwvls=len(self.wvl)
        print(nangs,nwvls,end=" ",file=f)
        for w in self.wvl:
            print("%0.1f "%w,end="",file=f)
        print("",file=f)
        for i in range(nangs):
            print(f"{self.vza[i]:.2f}",end=" ",file=f) 
            print(f"{self.vaa[i]:.2f}",end=" ",file=f) 
            print(f"{self.sza[i]:.2f}",end=" ",file=f) 
            print(f"{self.saa[i]:.2f}",file=f)
   
        return(f)

    def gen_args(self):

        args=[self.semiD_exe,]
        for arg in self.params:
            if self.params[arg] is not None:
                args.append("-%s"%arg)
                args.append(str(self.params[arg]))
        
        return(args)

    def run(self):

        nangs=len(self.vza)
        nwvls=len(self.wvl)
        refl=np.zeros([nangs,nwvls])  
        
        angs=self.gen_input()
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

    s=semiD()
    s.params['LAI']=1.0
    s.params['cab']=100.3
    print(s.run())
    
    
