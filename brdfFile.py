#!/usr/bin/python

import os
import sys

class brdfFile( object ):

	def __init__ ( self ):

		self.__version__ = 0.1

		self.nWavelengths = 0
		self.nAngles = 0
		self.wavelengths=[]
		self.vza_arr=[]
		self.vaa_arr=[]
		self.sza_arr=[]
		self.saa_arr=[]
		self.brf=[]
		self.brfObs=[]

		self.hasTimeData=False	
		self.time_arr=[]
		
		#use colOffset to account for
		#any additional columns of data
		#e.g. time and/or QA
		self.colOffset=0

		self.verbose=True
		self.gotData=False
		self.gotObs=False
		self.reporErrors=True
		self.quitOnErr=True

	

	def errMsgNWavelengths( self, n ):
		self.errMsg( "Number of wavelengths found (%d) does not match number specified (%d)"%(n,self.nWavelengths) )

	def errMsgWavebandOutOfRange( self, n ):
		self.errMsg( "Waveband requested (%d) is out of range"%n )

	def errMsgNAngles( self, n ):
		self.errMsg( "Number of angles found (%d) does not match number specified (%d)"%(n,self.nAngles) )

	def errMsgNoData( self ):
		self.errMsg( "No BRDF data found in class. " )

	def errMsgNObs( self ):
		self.errMsg( "Number of observed BRFs is >0 but less than number of wavelengths specified" )

	def errMsg( self, errString ):
		if self.reporErrors:
			print(self.__class__.__name__+": "+errString, file=sys.stderr)
		if self.quitOnErr: sys.exit()


	def readBRDF ( self, filename ):

		f=open( filename )

		na=0
		for line in f.readlines(  ):
			na+=1
			
			#read header line
			if na==1:
				
				offset=0
				#skip magic number if present				
				if line.split()[0]=="BRDF" : 
					offset=1
				elif line.split()[0]=="BRDFTIME" :
					self.hasTimeData=True
					self.colOffset=1
					offset=1
				
				#read number of angles and wavelengths
				self.nAngles=int( line.split()[0+offset] )
				self.nWavelengths=int( line.split()[1+offset] )
				
				nw=0
				#read in wavelengths
				for nw in range( len( line.split() ) -offset -2 ):
					self.wavelengths.append(  float( line.split()[nw+2+offset] ) )
				
					
				#check number of wavelengths found	
				if (nw+1)!=self.nWavelengths:
					if self.verbose: 
						print(nw, len( line.split() ) -offset -2, end=' ', file=sys.stderr) 
						self.errMsgNWavelengths( nw+1 )
					self.nWavelengths=nw+1

			#read angle lines
			else:
				
				#ignore blank lines
				if len( line.split() ) == 0:
					na-=1
					continue
				
				if self.hasTimeData:
					self.time_arr.append( float( line.split()[0] ) )

				self.vza_arr.append( float( line.split()[0+self.colOffset] ) )
				self.vaa_arr.append( float( line.split()[1+self.colOffset] ) )
				self.sza_arr.append( float( line.split()[2+self.colOffset] ) )
				self.saa_arr.append( float( line.split()[3+self.colOffset] ) )
			
				#brfs (if they exist)
				
				self.brf.append([])
				self.brfObs.append([])

				for i in range( self.nWavelengths ):
					self.brf[na-2].append( None )
					self.brfObs[na-2].append( None )
				
				#read in observed brfs if they exist
				#and check that the number is == to
				#the number of specified wavelengths
				
				l=(len(line.split())-4)
				if l>0 and l<self.nWavelengths:
					self.errMsgNObs( )
				elif l==self.nWavelengths:
					self.gotObs=True
					
				for i in range( 4+self.colOffset, len(line.split()) ):
					self.brfObs[na-2][i-4-self.colOffset] = float( line.split()[i] )
					

		#check number of angles found
		if (na-1)!=self.nAngles:
			if self.verbose: 
				self.errMsgNAngles( na-1 )
			self.nAngles=na-1

		f.close()

		self.gotData=True


	def getObs( self, waveband ):
	
	
		obs_arr=[]
	
		if waveband >= self.nWavelengths:
			self.errMsgWavebandOutOfRange( waveband )


		for i in range( self.nAngles ):
			obs_arr.append( self.brfObs[ i ][ waveband ] )

		return( obs_arr )




	def printBRDF( self, brf=None ):

		if self.gotData==False:
			self.errMsgNoData(  )
			sys.exit( )
	
	
		if brf==None:
			brf=self.brf
	
		print(self.nAngles, self.nWavelengths, end=' ')
		
		for i in range( self.nWavelengths ):
			print("%g"%self.wavelengths[ i ], end=' ')
			
		print("")

		
		for i in range( self.nAngles ):
			if self.hasTimeData:
				print("%g"%self.time_arr[ i ], end=' ') 
			print("%g %g %g %g"%(self.vza_arr[i],self.vaa_arr[i],self.sza_arr[i],self.saa_arr[i]), end=' ')
			for j in range( len( brf[ i ] ) ):
				if brf[ i ][ j ]!=None: 
					print("%g"%brf[ i ][ j ], end=' ')
			print("")
		



if __name__=="__main__":

	b=brdfFile( )
	b.readBRDF( "./test.brdf" )
	b.printBRDF( )
	print("-------")
	b.printBRDF( brf=b.brfObs )
