#!/usr/bin/env python
"""
This is a Python interface to the Dry Sensor Interface (DSI) headset by Wearable Sensing
LLC.  It uses the DSI API, loaded from the libDSI dynamic library via ctypes.  The dynamic
library must be in the same directory as this Python file.  Function prototypes are parsed
automatically at import time from DSI.h, so DSI.h must also be in the same directory.

Most of the C functions are reinterpreted as object methods:  this module defines
classes Headset, Source and Channel to wrap them, and adds two helper methods:
Headset.Sources() and Headset.Channels().  It also defines various global functions,
and the decorators SampleCallback and MessageCallback.  Examples of how to use the
decorators, and a minimal Test() function, are provided at the bottom of this file.

Normal usage would be to import this file and use the classes and functions the module
provides. As a quick test, the Test() function can be run by executing this file directly,
with the serial port address as the first command-line argument, and (optionally) the
reference Source name or the word 'impedances' as the second.

The Python source file also contains copyright and disclaimer information.
"""

# This file is part of the Application Programmer's Interface (API) for Dry Sensor Interface
# (DSI) EEG systems by Wearable Sensing. The API consists of code, headers, dynamic libraries
# and documentation.  The API allows software developers to interface directly with DSI
# systems to control and to acquire data from them.
# 
# The API is not certified to any specific standard. It is not intended for clinical use.
# The API, and software that makes use of it, should not be used for diagnostic or other
# clinical purposes.  The API is intended for research use and is provided on an "AS IS"
# basis.  WEARABLE SENSING, INCLUDING ITS SUBSIDIARIES, DISCLAIMS ANY AND ALL WARRANTIES
# EXPRESSED, STATUTORY OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OR THIRD PARTY RIGHTS.
# 
# (c) @YEARS@ Wearable Sensing LLC


# TODO:  enum

__all__ = [
	'Headset', 'Source', 'Channel',
	'SampleCallback', 'MessageCallback',
	'DSIException',
	'IfStringThenRawString', 'IfStringThenNormalString',
]
# global DSI_* functions from the dylib will be appended to this
import numpy as np
import os, sys, ctypes
if sys.version >= '3': unicode = str; basestring = ( bytes, str )  # bytes is already defined, unicode is not
else: bytes = str # unicode is already defined, bytes is not
def IfStringThenRawString( x ):
	"""
	A string is likely to be either raw bytes already, or utf-8-encoded unicode. A simple
	quoted string literal may or may not be raw bytes, depending on Python version. This
	is a problem.
	
	If x is a string then, regardless of Python version and starting format, return the
	"raw bytes" version of it so that we can send it over a serial port, pass it via
	ctypes to a C function, etc.
	
	If x is not a string, return it unchanged (so you can use this function to filter a
	whole list of arguments agnostically).
	
	See also IfStringThenNormalString()
	"""
	if isinstance( x, str ): x = x.encode( 'utf-8' )
	return x
def IfStringThenNormalString( x ):
	"""
	A string is likely to be either raw bytes or utf-8-encoded unicode. Depending on
	Python version, either the raw bytes or the unicode might be treated as a "normal"
	string (i.e. the type you get from an ordinary quoted string literal, and the type
	can be print()ed without adornment). This is a problem.
	
	If x is a string then, regardless of Python version and starting format, return the 
	"normal string" version of it so that we can print it, use it for formatting, make an
	Exception out of it, get on with our lives, etc.
	
	If x is not a string, return it unchanged (so you can feel free to use this function
	to filter a whole list of arguments agnostically).
	
	See also IfStringThenRawString()
	"""
	if str is not bytes and isinstance( x, bytes ): x = x.decode( 'utf-8' )
	return x


class Headset:
	def __init__( self, arg=None ):
		self.ptr = arg
		self.__needs_cleanup = False
		if arg is None or isinstance( arg, basestring ): # treat arg as port specifier string rather than a pointer
			self.ptr = dll.DSI_Headset_New(IfStringThenRawString( arg ))
			self.__needs_cleanup = True
	def __del__( self ):
		if self.__needs_cleanup:
			try: dll.DSI_Headset_Delete( self.ptr )
			except: pass
			else:
				self.__needs_cleanup = False
	def Sources( self ):  return [ self.GetSourceByIndex( i )  for i in range( self.GetNumberOfSources()  ) ]
	def Channels( self ): return [ self.GetChannelByIndex( i ) for i in range( self.GetNumberOfChannels() ) ]
class Source:
	def __init__( self, ptr ): self.ptr = ptr

class Channel:
	def __init__( self, ptr ): self.ptr = ptr

class ProcessingStage:
	def __init__( self, ptr ): self.ptr = ptr

class DSIException( Exception ): pass


SampleCallback  = ctypes.CFUNCTYPE( None,            ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p )
MessageCallback = ctypes.CFUNCTYPE( ctypes.c_int,    ctypes.c_char_p, ctypes.c_int )

@MessageCallback
def NullMessageCallback( msg, lvl=0 ): return 1

@SampleCallback
def NullSampleCallback( headsetPtr, packetTime, userData ): pass

__allprototypes__ = []
def LoadAPI( dllname = None ):
	import platform, re, inspect, ctypes.util

	if dllname is None:
		uname = platform.system()
		machine = platform.machine().lower()
		if machine.startswith( 'armv' ): machine = machine.rstrip( 'l' )
		try: maxsize = sys.maxsize    # not available in Python 2.5
		except: maxsize = sys.maxint  # not available in Python 3
		executable_architecture= '64bit' if maxsize > 2 ** 32 else '32bit'
		# we must catch the case of 32-bit Python running on 64-bit machine, so we're interested
		# in this more than the underlying platform.machine(). And for some reason the official
		# python.org docs recommend using sys.maxsize in this way rather than using
		# platform.architecture()[0]

		if not machine or machine in [ 'i386', 'x86_64', 'amd64' ]:
			#arch = executable_architecture
			arch = 'i386' if executable_architecture.startswith( '32' ) else 'x86_64'
		else:
			arch = machine

		if   uname.lower().startswith( 'win' ): dllxtn = '.dll'
		elif uname.lower().startswith( 'darwin' ): dllxtn = '.dylib'
		else: dllxtn = '.so'
		dllname = 'libDSI-' + uname + '-' + arch + dllxtn

	headername = 'DSI.h'
	
	global dllpath, headerpath	
	whereami = os.path.dirname( os.path.abspath( inspect.getfile( inspect.currentframe() ) ) )
	dllpath = ctypes.util.find_library( dllname )  # try the usual places first: current working dir, then $DYLD_LIBRARY_PATH and friends (posix) or %PATH% (Windows)
	if dllpath is None: dllpath = os.path.join( whereami, dllname ) # if failed, try right next to this .py file
	if not os.path.isfile( dllpath ): dllpath = None
	if dllpath is None: raise OSError( "failed to find dynamic library " + dllname )
	dllpath = os.path.abspath( dllpath )
	whereisdll = os.path.dirname( dllpath )
	dll = ctypes.CDLL( dllpath )
	headerpath = os.path.join( whereisdll, headername )  # expect to find header next to dynamic library, wherever it was
	if not os.path.isfile( headerpath ): raise OSError( "failed to find header %s in directory %s" % ( headername, whereisdll ) )
	
	prototypes = [ line.split( ' , ' ) for line in open( headerpath ).readlines() if line.strip().startswith( 'DSI_API_FUNC\x28' ) ]

	ctypetypes = {
		'DSI_Headset'         : ctypes.c_void_p,
		'DSI_Source'          : ctypes.c_void_p,
		'DSI_Channel'         : ctypes.c_void_p,
		'DSI_ProcessingStage' : ctypes.c_void_p,
		'void *'              : ctypes.c_void_p,
		'const void *'        : ctypes.c_void_p,
		'const char *'        : ctypes.c_char_p,
		'size_t'              : ctypes.c_size_t,
		'bool_t'              : getattr( ctypes, 'c_bool', ctypes.c_int ),
		'int'                 : ctypes.c_int,
		'unsigned int'        : ctypes.c_uint,
		'double'              : ctypes.c_double,
		'void'                : None,
		'DSI_MessageCallback' : MessageCallback,
		'DSI_SampleCallback'  : SampleCallback,
		'DSI_SourceSelection' : ctypes.c_int,
	}

	classes = { 'DSI_Headset' : Headset, 'DSI_Source': Source, 'DSI_Channel': Channel, 'DSI_ProcessingStage': ProcessingStage }

	def wrapfunction( funcptr, outputClass, doc ):
		def function( *args ):
			args = [ IfStringThenRawString( arg ) for arg in args ]
			output = funcptr( *args )
			err = dll.DSI_ClearError()
			if err: raise( DSIException( IfStringThenNormalString( err ) ) )
			if outputClass: output = outputClass( output )
			return IfStringThenNormalString( output )
		function.__doc__ = doc
		return function
	
	def wrapmethod( funcptr, outputClass, doc ):
		def method( self, *args ):
			args = [ IfStringThenRawString( arg ) for arg in args ]
			output = funcptr( self.ptr, *args )
			err = dll.DSI_ClearError()
			if err: raise( DSIException( IfStringThenNormalString( err ) ) )
			if outputClass: output = outputClass( output )
			return IfStringThenNormalString( output )
		method.__doc__ = doc
		return method

	globalFuncs = {}
	
	def clean( s ): return re.sub( r'\/\*.*\*\/', '', s ).strip()
	
	for prototype in prototypes:
		
		restype = clean( prototype[ 0 ].split( ' ', 1 )[ 1 ] )
		funcname = clean( prototype[ 1 ] )
		args = clean( prototype[ 2 ] )
		doc = restype + ' ' + funcname + args + ';'
		__allprototypes__.append( doc )
		args = args.strip( '()' ).split( ',' )
		funcptr = getattr( dll, funcname )
		funcptr.restype = ctypetypes[ restype ]
		outputClass = classes.get( restype, None )
		for prefix, cls in classes.items():
			if funcname.startswith( prefix + '_' ):
				methodname = funcname[ len( prefix ) + 1 : ]
				setattr( cls, methodname, wrapmethod( funcptr, outputClass, doc ) )
				break
		else:
			if funcname.startswith( 'DSI_' ): funcname = funcname[ 4 : ]
			globalFuncs[ funcname ] = wrapfunction( funcptr, outputClass, doc )
		args = [ arg.strip().rsplit( ' ', 1 ) for arg in args ]
		if args != [ [ 'void' ] ]:  funcptr.argtypes = tuple( [ ctypetypes[ arg[ 0 ].strip() ] for arg in args ] )	
	return dll, globalFuncs

dll, globalFuncs = LoadAPI()	
locals().update( globalFuncs )
__all__ += globalFuncs.keys()
del globalFuncs
del LoadAPI
del Headset.New    # only Headset.__init__() should be calling DSI_Headset_New()
del Headset.Delete # only Headset.__del__() should be calling DSI_Headset_Delete()
del os, sys, ctypes


#########################################################################################
##### Example code ######################################################################
#########################################################################################

import sys

@MessageCallback
def ExampleMessageCallback( msg, lvl=0 ):
	if lvl <= 3:  # ignore messages at debugging levels higher than 3
		print( "DSI Message (level %d): %s" % ( lvl, IfStringThenNormalString( msg ) ) )
	return 1

@SampleCallback
def ExampleSampleCallback_Signals( headsetPtr, packetTime, userData ):
	h = Headset( headsetPtr )
	strings = [ '%s=%+08.2f \n' % ( IfStringThenNormalString( ch.GetName() ), ch.ReadBuffered() ) for ch in h.Channels() ]
	print( ( '%8.3f:   \n' % packetTime ) + ', '.join( strings ) )
	sys.stdout.flush()

@SampleCallback
def ExampleSampleCallback_Impedances( headsetPtr, packetTime, userData ):
	h = Headset( headsetPtr )
	#fmt = '%s = %5.3f'
	fmt = '%s'
	#strings = [ fmt % ( IfStringThenNormalString( src.GetName() ), src.GetImpedanceEEG() ) for src in h.Sources() if src.IsReferentialEEG() and not src.IsFactoryReference() ]
	strings = np.array([ fmt % (IfStringThenNormalString( src.GetName())) for src in h.Sources() if src.IsReferentialEEG() and not src.IsFactoryReference() ])
	strings = strings.reshape(20,1)
	new_strings = strings[[10, 11, 5, 4, 6, 16, 17, 9, 3, 8, 12, 19, 13, 18, 14, 0, 9, 15, 16, 1],:]
	#strings.append( fmt % ( 'CMF @ ' + h.GetFactoryReferenceString(), h.GetImpedanceCMF() ) )
	#print( ( '%8.3f:   ' % packetTime ) + ', '.join( strings ) )
	allstrings = np.append(strings, np.array(range(len(strings))).reshape(len(strings),1), axis=1)
	print(allstrings)
	sys.stdout.flush()

def Test( port, arg='' ):
	"""
	`arg` is either a specification of the desired reference, or the
	string "impedances"
	"""
	h = Headset() # if we did not want to change the callbacks first, we could simply say h = Headset( port )
	h.SetMessageCallback( ExampleMessageCallback )  # could set this to NullMessageCallback instead if we wanted to shut it up
	h.Connect( port )
	if arg.lower().startswith( 'imp' ):
		print('impedance mode')
		h.SetSampleCallback( ExampleSampleCallback_Impedances, 0 )
		h.StartImpedanceDriver()
	else:
		print('normal mode')
		h.SetSampleCallback( ExampleSampleCallback_Signals, 0 )
		if len( arg.strip() ): h.SetDefaultReference( arg, True )

	h.StartDataAcquisition()  # calls StartDataAcquisition(), then Idle() for 2 seconds, then StopDataAcquisition(), then Idle() for 2 seconds
	# NB: your application will probably want to use Idle( seconds ) in its main loop instead of Receive()
	h.GetInfoString()
	h.Idle(1)
	h.StopDataAcquisition()

if __name__ == '__main__':
	args = getattr( sys, 'argv', [ '' ] )
	if sys.platform.lower().startswith( 'win' ): default_port = 'COM5'
	else:                                        default_port = '/dev/cu.DSI7-0009.BluetoothSeries'

	# first command-line argument: serial port address
	if len( args ) > 1: port = args[ 1 ]
	else: port = default_port

	# second command-line argument:  name of the Source to be used as reference, or the word 'impedances'
	if len( args ) > 2: ref = args[ 2 ]
	else: ref = ''

	Test( port, '' )
