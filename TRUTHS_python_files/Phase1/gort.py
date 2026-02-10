from dataclasses import dataclass
import subprocess as subproc

import numpy as np


@dataclass
class geom:
    vza: float
    vaa: float
    sza: float
    saa: float


class gort:

    def __init__(self):

        self.exe = "/home/users/ndouglas/TRUTHS/GORT/gortt"
        self.wavl = [640., 850.]

        self.params = {}
        # scene physical properties
        self.params['LAI'] = None       # leaf area index
        self.params['HB'] = None        # ratio of the centroid height range to the vertical crown radius
        self.params['BR'] = None        # ratio of the vertical to horizontal crown radius
        self.params['PCC'] = None       # set the projected crown cover (at nadir)

        # prospect leaf
        self.params['N'] = None       # leaf structure variable
        self.params['Cab'] = None     # leaf chlorophyll content (µg.cm-2)
        self.params['Cw'] = None      # equivalent leaf water thickness (cm)
        self.params['Car'] = None     # carotenoid content (µg.cm-2)
        self.params['Anth'] = None    # anthocyanin content (µg.cm-2)
        self.params['Cbrown'] = None  # brown pigment content (arbitrary units)
        self.params['Cm'] = None      # leaf mass per unit area (g.cm-2) S

        self.flags = {}

        self.flags["prnprop"] = True
        self.flags["energy"] = True
        

    def gen_args(self):
        """generate the GORT command line
        arguments from the params dict"""

        args = [self.exe, ]
        for arg in self.params:
            if self.params[arg] is not None:
                args.append("-%s" % arg)
                args.append(str(self.params[arg]))
        for arg in self.flags:
            if self.flags[arg] is True:
                args.append("-%s" % arg)

        return(args)

    def gen_input(self, geom_list):
        """generate the wavlength and
        angles input file for GORT
        """
        inpt = ""
        inpt += str(len(geom_list))+" "
        inpt += str(len(self.wavl))+" "
        for w in self.wavl:
            inpt += str(w)+" "
        inpt += "\n"

        for g in geom_list:
            inpt += "%f %f %f %f\n" % (g.vza, g.vaa, g.sza, g.saa)

        return(inpt)

    def run(self, geom_list, verbose=False):

        brdf = np.zeros([len(geom_list), len(self.wavl)])
        enrg = np.zeros([len(geom_list), len(self.wavl)*3])
        comp = np.zeros([len(geom_list), 4])
        args = self.gen_args()
        p = subproc.Popen(args, encoding='ascii', stdin=subproc.PIPE,
                          stdout=subproc.PIPE)
        rawout = p.communicate(input=self.gen_input(geom_list))
        for (i, line) in enumerate(rawout[0].rstrip("\n").split("\n")):
            if verbose:
                print(line)
            if i == 0:
                continue
            for n in range(len(self.wavl)):
                brdf[i-1, n] = float(line.split()[4+n])
            if self.flags["prnprop"]:
                for n in range(4):
                    comp[i-1, n] = float(line.split()[5+len(self.wavl)+n])
            if self.flags["energy"]:
                for n in range(len(self.wavl)):
                    for m in range(3):
                        enrg[i-1,-3*(n+1)+m]=float(line.split()[-3*(n+1)+m])


        return(brdf, comp, enrg)


if __name__ == "__main__":

    geom_list = []
    geom_list.append(geom(vza=10., vaa=0., sza=20, saa=45.))
    geom_list.append(geom(vza=15., vaa=10., sza=30, saa=45.))

    g = gort()
    
    #note: the following turn Prospect and Price models off
    g.params["alb_leaf"]=0.2
    g.params["alb_soil"]=0.1
 
    brdf, comp, enrg = g.run(geom_list, verbose=True)
    print(brdf)
    print(comp)
    print(enrg)

