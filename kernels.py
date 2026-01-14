#!/usr/bin/python

import numpy as np
from brdfFile import brdfFile

def sec( x ):
  return 1./np.cos( x )

np.sec = sec

class kernelBRDF( brdfFile ):
  '''
  A class for the AMBRALS style semi-empirical BRDF kernel models.
  
  Tristan Quaife 19/09/09
  '''

  def __init__( self, br=1.0, hb=2.0 ):                 
    self.br = br
    self.hb = hb
    
    #default kernels:
    self.nKernels=3    
    self.kernels=[]
    self.kernels.append( self.isotropic )
    self.kernels.append( self.rossThick )
    self.kernels.append( self.liSparse )
        
    #init all the stuff from brdfFile:
    brdfFile.__init__( self )
    

  def dtor( self, d ):  
    return d*np.pi/180.0 


  def correctAngles( self, vza, sza, raa ):
    
    vaa=0
    saa=raa
    
    if vza < 0.0:
      vaa+=np.pi 
      vza*=-1.0
    if sza < 0.0:
      saa+=np.pi 
      sza*=-1.0
    while saa > 2 * np.pi:
      saa-=2*np.pi
    while vaa > 2*np.pi:
      vaa-=2*np.pi
    while saa < 0: 
      saa+=2*np.pi
    while vaa < 0:
      vaa+=2*np.pi
    raa=saa-vaa 
    raa = abs((raa - 2.0 * np.pi *  round(0.5 + raa * 0.5 / np.pi )))

    return vza, sza, raa


  def phaseAngle( self, vza, sza, raa ):
    return np.arccos( ( np.cos( vza )*np.cos( sza ) + 
                 np.sin( vza )*np.sin( sza )*np.cos( raa ) ) )


  def thetaPrime( self, theta, br=None ):
    if br==None: br=self.br
    return np.arctan( br*np.tan( theta ) );
  

  def liDistance( self, vzp, szp, raa ):
      return np.sqrt( np.tan( szp )**2 + np.tan( vzp )**2 - 
                     2.0*np.tan( szp )*np.tan( vzp )*np.cos( raa ) )


  def liT( self, vzp, szp, raa, hb=None ):
    if hb==None: hb=self.hb

    d = self.liDistance( vzp, szp, raa )  
    cosT = hb*( np.sqrt( d**2+( np.tan( szp )*np.tan( vzp )*np.sin( raa ) )**2) )/( np.sec( szp )+np.sec( vzp ))
          
    #correct for cosT>1
    if np.isscalar( cosT ):
      if cosT>1:cosT=1.0
    else:
      i=0
      for x in cosT:
        if cosT[i]>1:cosT[i]=1.0
        i+=1
      
    return np.arccos(cosT)   


  def liOverlap( self, vza, sza, raa, hb=None, br=None ):
    if br==None: br=self.br
    if hb==None: hb=self.hb
    
    szp=self.thetaPrime( sza, br=br )
    vzp=self.thetaPrime( vza, br=br )
    t=self.liT( vzp, szp, raa, hb=hb )

    return ( 1.0/np.pi )*( t-np.sin( t )*np.cos( t ) )*( np.sec( szp )+np.sec( vzp ) );


  def isotropic( self, vza, sza, raa ):
    return 1.0


  def rossThick(  self, vza, sza, raa ):
    p = self.phaseAngle( vza, sza, raa )
    return ((((( np.pi/2 )-p )*np.cos( p ))+np.sin( p ))/
               ( np.cos( vza )+np.cos( sza )))-( np.pi/4.0 )


  def rossThin(  self, vza, sza, raa ):
    p = self.phaseAngle( vza, sza, raa )
    return ((((( np.pi/2 )-p )*np.cos( p ))+np.sin( p ))/
               ( np.cos( vza )*np.cos( sza )))-( np.pi/2.0 )


  def liSparse( self, vza, sza, raa, hb=None, br=None, reciprocal=True ):
    if br==None: br=self.br
    if hb==None: hb=self.hb
    
    szp=self.thetaPrime( sza, br=br )
    vzp=self.thetaPrime( vza, br=br )
    pap=self.phaseAngle( vzp, szp, raa )
    o=self.liOverlap( vza, sza, raa, hb=hb, br=br )
    
    if reciprocal==False:
      return o-np.sec( szp )-np.sec( vzp )+0.5*( 1.0 + np.cos( pap ) )*np.sec( vzp )

    return o-np.sec( szp )-np.sec( vzp )+0.5*( 1.0 + np.cos( pap ) )*np.sec( vzp )*np.sec( szp )


  def liDense( self, vza, sza, raa, hb=None, br=None ):
    if br==None: br=self.br
    if hb==None: hb=self.hb
    
    szp=self.thetaPrime( sza, br=br )
    vzp=self.thetaPrime( vza, br=br )
    pap=self.phaseAngle( vzp, szp, raa )
    o=self.liOverlap( vza, sza, raa, hb=hb, br=br )

    return (((1.0 + np.cos( pap ))*np.sec( vzp ) )/(np.sec( szp )+np.sec( vzp ) - o ) )-2.0


  #Walthall kernels
  def walthallLinMultCos( self, vza, sza, raa ):
    return vza * sza * np.cos( raa )

  def walthallSqMult( self, vza, sza, raa ):
    return vza * vza * sza * sza

  def walthallSqSum( self, vza, sza, raa ):
    return vza * vza + sza * sza


  def kernelMatrixRTkLSp( self, filename=None ):
    '''This function is superseded by kernelMatrix'''
  
    if filename != None:
      self.readBRDF( filename )
  
    elif self.gotData==False:
      self.errMsgNoData(  )
      sys.exit( )

    K=[]
    for i in range( self.nAngles ):    
      vza=self.dtor( self.vza_arr[i] )
      vaa=self.dtor( self.vaa_arr[i] )
      sza=self.dtor( self.sza_arr[i] )
      saa=self.dtor( self.saa_arr[i] )
      [vza, sza, raa]=self.correctAngles( vza, sza, vaa-saa )
    
      iso = self.isotropic( vza, sza, raa )
      vol = self.rossThick( vza, sza, raa )
      geo = self.liSparse( vza, sza, raa )
      K.append( [iso,vol,geo] )
        
    return np.array( K )
    

  def kernelMatrix( self, filename=None ):
  
    if filename != None:    
      self.readBRDF( filename )
  
    elif self.gotData==False:    
      self.errMsgNoData(  )
      sys.exit( )

    K=[]
    for i in range( self.nAngles ):
      vza=self.dtor( self.vza_arr[i] )
      vaa=self.dtor( self.vaa_arr[i] )
      sza=self.dtor( self.sza_arr[i] )
      saa=self.dtor( self.saa_arr[i] )
      [vza, sza, raa]=self.correctAngles( vza, sza, vaa-saa )
    
      kVals=[]
      for i in range( self.nKernels ):
        kVals.append( self.kernels[i]( vza, sza, raa ) )
      K.append( kVals )
        
    return np.array( K )


  def solveKernelBRDF( self, R_dict, filename=None ):
    K = self.kernelMatrix( filename )
    KW=[]
    #for i in range( self.nWavelengths ):
    #  [ kw, resd, rnk, sng ]=np.linalg.lstsq(  K, np.array(self.getObs(i)), rcond=None )
    #  KW.append( kw )

    for i in range( self.nWavelengths ):
      R=R_dict[i]
      work1=np.dot(K.T,np.linalg.inv(R))
      work2=np.linalg.inv(np.dot(work1,K))
      work3=np.dot(work2,work1)
      KW.append(np.dot(work3,self.getObs(i)))
                  
    return np.array( KW )

#  def WeightsOfDetermination( self, filename=None, obserr ):
#    K = self.kernelMatrix( filename )
#    W = self.solveKernelBRDF( filename )
#    Kinv = np.linalg.inv(K)
#    work1=np.dot(Kinv,W)
#    work2=np.dot(W.T,work1)
#    exp_uncert=obserr*np.sqrt(work2)
#      
#    return exp_uncert

  def predictWSAlbedoRTkLSp( self, kw ):
    '''
    Predict the WHITE SKY albedo for the provided Kw
    Assumes that the kernels in use are Iso, RTk and LSp. 
    Does not check that this is the case!
    '''
    
    #kernel integrals for WS:
    aW=[1, 0.189184, -1.377622]
   
    alb=[]         
    for i in range( self.nWavelengths ):
        a=0
        for j in range( self.nKernels ):
            a+=kw[i,j]*aW[j]    
    
        alb.append(a)  
      
    return alb


  def predictBSAlbedoRTkLSp( self, kw, sza ):
    '''
    Predict the BLACK SKY albedo for the provided Kw and sza
    Assumes that the kernels in use are Iso, RTk and LSp. 
    Does not check that this is the case!
    
    See Table 1 in:
    https://modis.gsfc.nasa.gov/data/atbd/atbd_mod09.pdf
    '''
    sza=np.deg2rad(sza)
      
    #kernel integrals for BS:
    g0k=[1.0, -0.007574, -1.284909]
    g1k=[0.0, -0.070987, -0.166314]
    g2k=[0.0,  0.307588,  0.041840]
    
    alb=[]         
    for i in range( self.nWavelengths ):
        a=0
        for j in range( self.nKernels ):
            a+=kw[i,j]*(g0k[j]+g1k[j]*sza*sza+g2k[j]*sza*sza*sza)    
    
        alb.append(a)  
      
    return alb

  def predict_brfs( self, kw ):
    """forward model BRFs based on provided
    kernel weights (kw).
    """
  
    for i in range( self.nAngles ):      
      vza=self.dtor( self.vza_arr[i] )
      vaa=self.dtor( self.vaa_arr[i] )
      sza=self.dtor( self.sza_arr[i] )
      saa=self.dtor( self.saa_arr[i] )
    
      [vza, sza, raa]=self.correctAngles( vza, sza, vaa-saa )

      for w in range( self.nWavelengths ):
          rho=0.
          for j in range( self.nKernels ):
              rho+=kw[w,j] * self.kernels[j]( vza, sza, raa )
    
          self.brf[i][w]=rho
    


  def printMatrix( self, M ):  
    s=np.shape( M )
    for i in range( s[0] ):
      for j in range( s[1] ):
        print("%g"%M[i,j], end=' ')
      print("")



if __name__=="__main__":

  k=kernelBRDF( )    

  if True: 
    '''Test kernel values over range of angles'''
    
    vza_list=list(range(-80,90,2)) #in degrees
    sza=k.dtor( 30 )
    raa=k.dtor( 0 )
      
    for (i,vza) in enumerate(vza_list):
        vza=k.dtor( vza )
        iso = k.isotropic( vza, sza, raa )
        vol = k.rossThick( vza, sza, raa )
        geo = k.liSparse( vza, sza, raa )

        print(np.rad2deg(vza), iso, vol, geo) 


