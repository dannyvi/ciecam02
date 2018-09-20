"""a set of function convert colorspace among rgb ciexyz ciecam02."""
# Author: dannyv
import numpy as np
cdef extern from "math.h":
     double fabs(double)
     double sin(double)
     double round(double)
cimport numpy as np

cdef float convert2rgb(float C):
    if ( C > 0.0031308):
        return 1.055 * (C**(1/2.4)) - 0.055
    else:
        return 12.92*C


def xyz2rgb(list xyz):
    """convert xyz 2 rgb mode
    Args:
        xyz: a list like [x, y, z]  0<x<95.047, 0<y<100.0 0<z<108.883
    Returns:
        a list contains rgb value like [r,g,b]
        the rgb value type is integer range from  0  to 255
    """
    cdef float X = xyz[0]/100.0           # 0 --- 95.047
    cdef float Y = xyz[1]/100.0           # 0 --- 100.00
    cdef float Z = xyz[2]/100.0           # 0 --- 108.883

    cdef float varR = X * 3.2406 - Y * 1.5372 - Z * 0.4986
    cdef float varG = X * (-0.9689) + Y * 1.8758 + Z * 0.0415
    cdef float varB = X * 0.0557 - Y * 0.2040 + Z * 1.0570

    cdef int R = <int>(round(convert2rgb(varR)*255))
    cdef int G = <int>(round(convert2rgb(varG)*255))
    cdef int B = <int>(round(convert2rgb(varB)*255))

    return [R,G,B]

cdef float convert2xyz(float C):
    if ( C > 0.04045):
        C = pow(((C+0.055)/1.055),2.4)
    else:
        C = C/12.92
    C *= 100
    return C

def rgb2xyz(list rgb):
    """convert rgb 2 xyz mode
    Args:
        rgb: a list like [r,g,b]  r,g,b is integer value range from 0 to 255
    Returns:
        a list contains x,y,z value like [x,y,z]
        the xyz value is float  0<x<95.047, 0<y<100.0 0<z<108.883
    """
    cdef float R = rgb[0]/255.0
    cdef float G = rgb[1]/255.0
    cdef float B = rgb[2]/255.0

    R = convert2xyz(R)
    G = convert2xyz(G)
    B = convert2xyz(B)

    cdef double X = R*0.4124 + G*0.3576 + B*0.1805
    cdef double Y = R*0.2126 + G*0.7152 + B*0.0722
    cdef double Z = R*0.0193 + G*0.1192 + B*0.9505
    return [X,Y,Z]

whitepoint = {'white':[95.05,100.00,108.88],
              'c':[109.85,100.0,35.58]}

env = {'dim':[0.9,0.59,0.9],\
       'average':[1.0,0.69,1.0],\
       'dark':[0.8,0.525,0.8]}
lightindensity = {'default':80.0,'high':318.31,'low':31.83}
bgindensity = {'default':16.0,'high':20.0,'low':10.0}

currentwhite = whitepoint['white']
currentenv = env['average']
currentlight = lightindensity['default']
currentbg = bgindensity['default']
def setconfig(a='white',b='average',c='default',d='default'):
    currentwhite = whitepoint[a]
    currentenv = env[b]
    currentlight = lightindensity[c]
    currentbg = bgindensity[d]


Mcat02 = np.array([[0.7328,0.4296,-0.1624],\
                   [-0.7036,1.6975,0.0061],\
                   [0.0030,0.0136,0.9834]])
#Xw,Yw,Zw=currentwhite  #[95.05,100.0,108.88]
cdef double Xw = currentwhite[0]
cdef double Yw = currentwhite[1]
cdef double Zw = currentwhite[2]
#Nc,c,F = currentenv    #[1.0,0.69,1.0]           #print
cdef double Nc = currentenv[0]
cdef double c = currentenv[1]
cdef double F = currentenv[2]
cdef double LA = currentlight   #80.0
cdef double Yb = currentbg      #16.0
RwGwBw = Mcat02.dot(np.array([Xw,Yw,Zw]))
cdef double Rw = RwGwBw[0]
cdef double Gw = RwGwBw[1]
cdef double Bw = RwGwBw[2]
cdef double D = F*(1 - (1/3.6) * (np.e**((-LA-42)/92)))
if D>1:D=1.0
if D<0:D=0.0
#Dr,Dg,Db = [Yw*D/Rw+1-D, Yw*D/Gw+1-D, Yw*D/Bw+1-D]
cdef double Dr = Yw*D/Rw+1-D
cdef double Dg = Yw*D/Gw+1-D
cdef double Db = Yw*D/Bw+1-D
#Rwc,Gwc,Bwc=[Dr*Rw,Dg*Gw,Db*Bw]
cdef double Rwc = Dr*Rw
cdef double Gwc = Dg*Gw
cdef double Bwc = Db*Bw
cdef double k = 1/(5*LA+1)
cdef double FL = 0.2*(k**4)*(5*LA) + 0.1*((1-(k**4))**2)*((5*LA)**(1/3.0))
cdef double n = Yb/Yw
if n>1: n=1.0
if n<0: n=0.0

#Nbb = Ncb = 0.725*((1.0/n)**0.2)
cdef double Ncb = 0.725*((1.0/n)**0.2)
cdef double Nbb = 0.725*((1.0/n)**0.2)
cdef double z = 1.48 + n**0.5
M_1cat02 = np.array([[1.096241,-0.278869,0.182745],\
                     [0.454369, 0.473533,0.072098],\
                     [-0.009628,-0.005698,1.015326]])
MH = np.array([[ 0.38971, 0.68898,-0.07868],\
               [-0.22981, 1.18340, 0.04641],\
               [ 0.00000, 0.00000, 1.00000]])
M_1hpe = np.array([[1.910197,-1.112124,0.201908],\
                   [0.370950, 0.629054,-0.000008],\
                   [0.000000, 0.000000, 1.000000]])
Rw_Gw_Bw_ = MH.dot(M_1cat02.dot([Rwc,Gwc,Bwc]))
cdef double Rw_ = Rw_Gw_Bw_[0]
cdef double Gw_ = Rw_Gw_Bw_[0]
cdef double Bw_ = Rw_Gw_Bw_[0]
colordata = [[20.14,0.8,0],[90,0.7,100],[164.25,1.0,200],\
             [237.53,1.2,300],\
             [380.14,0.8,400]]
cdef double Rwa_ = (400 * ((FL*Rw_/100)**0.42))/(27.13+((FL*Rw_/100)**0.42))+0.1
cdef double Gwa_ = (400 * ((FL*Gw_/100)**0.42))/(27.13+((FL*Gw_/100)**0.42))+0.1
cdef double Bwa_ = (400 * ((FL*Bw_/100)**0.42))/(27.13+((FL*Bw_/100)**0.42))+0.1
cdef double Aw =Nbb * (2*Rwa_+Gwa_+(Bwa_/20) - 0.305)

def xyz2cam02(list XYZ,XwYwZw=[95.05,100.00,108.88],Nc=0.9,c=0.59,F=0.9,LA=80.0,Yb=16.0):
    """convert xyz 2 cam02 data

    Args:
        XYZ:    contains [X,Y,Z] value
        XwYwZw: target white point xyz, provide a default value
        Nc,c,F: environment parameter
                CIE defined 3 environments which named
                    dim:     0.9, 0.59, 0.9  for display device
                    average: 1.0, 0.69, 1.0  for normal print
                    dark:    0.8, 0.525,0.8  for project device
                which are not imutable
        LA:     light indensity by unit lcd/cm2
        Yb:     the background light indensity in the scene
    Returns:
        retvalue: [h,H,J,Q,C,M,s]

            h,H:  Hue value
                  h: the same procedure as CIELAB ranging 0 to 360
                  H: hue quadrature value ranging from 0  to 400
            J,Q:  Lightness and Brightness
            C,M,s: C:chroma, M:Colorfulness, s:saturation
    """
    cdef double R,G,B,Rc,Gc,Bc,R_,G_,B_,Ra_,Ga_,Ba_,a,b,h,H
    cdef double etemp,A,J,Q,t,C,M,s
    cdef double Xw,Yw,Zw,Rw,Gw,Bw,Dr,Dg,Db,Rwc,Gwc,Bwc,D
    cdef double k,FL,n,Ncb,Nbb,z,Rw_,Gw_,Bw_,Aw
    #Xw,Yw,Zw = XwYwZw
    Rw,Gw,Bw = Mcat02.dot(np.array(XwYwZw))
    D = F*(1 - (1/3.6) * (np.e**((-LA-42)/92)))
    if D>1:D=1.0
    if D<0:D=0.0
    Xw,Yw,Zw = XwYwZw
    Dr = Yw*D/Rw+1-D
    Dg = Yw*D/Gw+1-D
    Db = Yw*D/Bw+1-D

    Rwc = Dr*Rw
    Gwc = Dg*Gw
    Bwc = Db*Bw

    k = 1/(5*LA+1)
    FL = 0.2*(k**4)*(5*LA) + 0.1*((1-(k**4))**2)*((5*LA)**(1/3.0))

    n = Yb/Yw
    if n>1: n=1.0
    if n<0: n=0.0

    Ncb = 0.725*((1.0/n)**0.2)
    Nbb = 0.725*((1.0/n)**0.2)
    z = 1.48 + n**0.5

    Rw_,Gw_,Bw_ = MH.dot(M_1cat02.dot([Rwc,Gwc,Bwc]))


    Rwa_ = (400 * ((FL*Rw_/100)**0.42))/(27.13+((FL*Rw_/100)**0.42))+0.1
    Gwa_ = (400 * ((FL*Gw_/100)**0.42))/(27.13+((FL*Gw_/100)**0.42))+0.1
    Bwa_ = (400 * ((FL*Bw_/100)**0.42))/(27.13+((FL*Bw_/100)**0.42))+0.1

    Aw =Nbb * (2*Rwa_+Gwa_+(Bwa_/20) - 0.305)

    R,G,B    = Mcat02.dot(XYZ)
    Rc = Dr*R
    Gc = Dg*G
    Bc = Db*B
    #Rc,Gc,Bc = [Dr*R,Dg*G,Db*B]
    #step 5
    R_,G_,B_ = MH.dot(M_1cat02.dot([Rc,Gc,Bc]))
    #step 6
    Ra_ = (400 * ((FL*R_/100)**0.42))/(27.13+((FL*R_/100)**0.42))+0.1
    Ga_ = (400 * ((FL*G_/100)**0.42))/(27.13+((FL*G_/100)**0.42))+0.1
    Ba_ = (400 * ((FL*B_/100)**0.42))/(27.13+((FL*B_/100)**0.42))+0.1
    #step 7
    a = Ra_ - 12* Ga_/11 + Ba_/11
    b = (1/9.0) * (Ra_+Ga_-2*Ba_)
    h = np.arctan2(b,a)
    if h<0:
        h+=np.pi*2
    h = h *180/np.pi
    try:
        assert(h>=0 and h<360)
    except AssertionError:
        print 'bad hue value'
    #step8
    H = 0
    #Hi = 0
    datai = []
    datai1= []
    if h<colordata[0][0]:
        h_ = h+360
    else:
        h_ = h
    etemp = (np.cos(h_*np.pi/180+2)+3.8) * 0.25
    for i in range(5):
        if h_>=colordata[i][0] and h_<colordata[i+1][0]:
            datai = colordata[i]
            datai1 = colordata[i+1]
            break
    H = datai[2] + ((100*(h_-datai[0])/datai[1])/\
                    (((h_-datai[0])/datai[1])+(datai1[0]-h_)/datai1[1]))
    #step 9
    A = Nbb * (2*Ra_+Ga_+(Ba_/20.0) - 0.305)
    #step10
    J = 100*((A/Aw)**(c*z))
    #step 11
    Q = (4/c) * ((J/100.0)**0.5) * (Aw + 4) * (FL**0.25)
    #step 12
    t = ((50000/13.0)*Nc*Ncb*etemp*((a**2+b**2)**0.5))/(Ra_+Ga_+(21/20.0)*Ba_)
    C = t**0.9*((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)
    M = C*(FL**0.25)
    s = 100*((M/Q)**0.5)
    return [h,H,J,Q,C,M,s]

def rgb2jch(list color):
    """provide a quick convertion from rgb to cam02 JCH value

    XwYwZw default to [95.05,100.00,108.88]
    Nc,c,F default to dim mode 0.9,0.59,0.9
    LA,Yb  default to 80.0 16.0

    Args:
        color: need to provide [r,g,b] list ranging from 0 to 255 integer

    Returns:
        [J,C,H]
         J: Lightness ranging from 0 to 100 float
         C: Chroma
         H: Hue quadrature ranging from 0 to 400
    """
    cdef double R = color[0]/255.0
    cdef double G = color[1]/255.0
    cdef double B = color[2]/255.0

    #    def convert(C):

    R = convert2xyz(R)
    G = convert2xyz(G)
    B = convert2xyz(B)

    cdef double X = R*0.4124 + G*0.3576 + B*0.1805
    cdef double Y = R*0.2126 + G*0.7152 + B*0.0722
    cdef double Z = R*0.0193 + G*0.1192 + B*0.9505
    #XYZ = rgb2xyz(color)
    cdef double Rc,Gc,Bc,R_,G_,B_,Ra_,Ga_,Ba_,a,b,h,H
    cdef double etemp,A,J,Q,t,C,M,s
    R,G,B    = Mcat02.dot([X,Y,Z])
    Rc = Dr*R
    Gc = Dg*G
    Bc = Db*B
    #Rc,Gc,Bc = [Dr*R,Dg*G,Db*B]
    #step 5
    R_,G_,B_ = MH.dot(M_1cat02.dot([Rc,Gc,Bc]))
    #step 6
    Ra_ = (400 * ((FL*R_/100)**0.42))/(27.13+((FL*R_/100)**0.42))+0.1
    Ga_ = (400 * ((FL*G_/100)**0.42))/(27.13+((FL*G_/100)**0.42))+0.1
    Ba_ = (400 * ((FL*B_/100)**0.42))/(27.13+((FL*B_/100)**0.42))+0.1
    #step 7
    a = Ra_ - 12* Ga_/11 + Ba_/11
    b = (1/9.0) * (Ra_+Ga_-2*Ba_)
    h = np.arctan2(b,a)
    if h<0:
        h+=np.pi*2
    h = h *180/np.pi
    try:
        assert(h>=0 and h<360)
    except AssertionError:
        print 'bad hue value'
    #step8
    H = 0
    #Hi = 0
    datai = []
    datai1= []
    if h<colordata[0][0]:
        h_ = h+360
    else:
        h_ = h
    etemp = (np.cos(h_*np.pi/180+2)+3.8) * 0.25
    for i in range(5):
        if h_>=colordata[i][0] and h_<colordata[i+1][0]:
            datai = colordata[i]
            datai1 = colordata[i+1]
            break
    H = datai[2] + ((100*(h_-datai[0])/datai[1])/\
                    (((h_-datai[0])/datai[1])+(datai1[0]-h_)/datai1[1]))
    #step 9
    A = Nbb * (2*Ra_+Ga_+(Ba_/20.0) - 0.305)
    #step10
    J = 100*((A/Aw)**(c*z))
    #step 11
    Q = (4/c) * ((J/100.0)**0.5) * (Aw + 4) * (FL**0.25)
    #step 12
    t = ((50000/13.0)*Nc*Ncb*etemp*((a**2+b**2)**0.5))/(Ra_+Ga_+(21/20.0)*Ba_)
    C = t**0.9*((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)
    M = C*(FL**0.25)
    s = 100*((M/Q)**0.5)
    #value = xyz2cam02(XYZ)
    #return [h,H,J,Q,C,M,s]
    return [J,C,H]#value[1]*360/400]

def jch2xyz(list jch):
    """provide a quick convertion from cam02 JCH value to xyz

    XwYwZw default to [95.05,100.00,108.88]
    Nc,c,F default to dim mode 0.9,0.59,0.9
    LA,Yb  default to 80.0 16.0

    Args:
         jch: need to provide [J,C,H] list
            J: Lightness ranging from 0 to 100 float
            C: Chroma
            H: Hue quadrature ranging from 0 to 400
    Returns:
         XYZ list like [X,Y,Z]
    """
    cdef double H,J,C,h_,t,etemp,e,A,a,b,p1,p2,p3,p4,p5,h
    cdef double Ra_,Ga_,Ba_,R_,G_,B_,Rc,Gc,Bc,R,G,B,X,Y,Z
    H = jch[2]*400/360.0
    J = jch[0]*1.0
    C = jch[1]*1.0
    h_ = 0
    for i in range(4):
        if H>=colordata[i][2] and H<colordata[i+1][2]:
            h_ = ((H-colordata[i][2])*(colordata[i+1][1]*colordata[i][0]-colordata[i][1]*colordata[i+1][0])\
                  - 100*colordata[i][0]*colordata[i+1][1])/\
                ((H-colordata[i][2])*(colordata[i+1][1]-colordata[i][1]) - 100*colordata[i+1][1])
            if h_>360:
                h_ -= 360
            break
    t = 0
    if J!=0:
        t = (C/(((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)))**(1/0.9)
    #etemp =
    etemp = (np.cos(h_*np.pi/180+2)+3.8) * 0.25
    e = (50000/13.0)*Nc*Ncb*etemp
    A = Aw*((J/100)**(1/(c*z)))
    a = b =0
    p2 = A/Nbb +0.305
    p3 = 21/20.0
    h = h_*np.pi/180
    if t!=0:
        p1 = e/t #t!=0
        if abs(np.sin(h))>abs(np.cos(h)):
            p4 = p1/np.sin(h)
            b = (p2*(2+p3)*(460.0/1403))/\
                (p4+(2+p3)*(220.0/1403)*(np.cos(h)/np.sin(h))-27.0/1403+\
                 p3*(6300.0/1403))
            a = b*(np.cos(h)/np.sin(h))
        elif abs(np.cos(h))>abs(np.sin(h)):
            p5 = p1/np.cos(h)
            a = (p2*(2+p3)*(460.0/1403))/\
                (p5+(2+p3)*(220.0/1403)-\
                (27.0/1403 - p3*(6300.0/1403))*(np.sin(h)/np.cos(h)))
            b = a*(np.sin(h)/np.cos(h))
    # step 6
    Ra_ = (460*p2 + 451*a + 288*b)/1403.0
    Ga_ = (460*p2 - 891*a - 261*b)/1403.0
    Ba_ = (460*p2 - 220*a - 6300*b)/1403.0
    R_ = np.sign(Ra_-0.1)*(100.0/FL)*(((27.13*abs(Ra_-0.1))/(400-abs(Ra_-0.1)))**(1/0.42))
    G_ = np.sign(Ga_-0.1)*(100.0/FL)*(((27.13*abs(Ga_-0.1))/(400-abs(Ga_-0.1)))**(1/0.42))
    B_ = np.sign(Ba_-0.1)*(100.0/FL)*(((27.13*abs(Ba_-0.1))/(400-abs(Ba_-0.1)))**(1/0.42))
    #step8
    Rc,Gc,Bc = Mcat02.dot(M_1hpe.dot([R_,G_,B_]))
    R,G,B    = [Rc/Dr,Gc/Dg,Bc/Db]
    X,Y,Z    = M_1cat02.dot([R,G,B])
    return [X,Y,Z]

cdef int * cjch2rgb(double JJ,double CC,double HH):
    cdef double H,J,C,h_,t,etemp,e,A,a,b,p1,p2,p3,p4,p5,h
    cdef double Ra_,Ga_,Ba_,R_,G_,B_,Rc,Gc,Bc,R,G,B,X,Y,Z
    cdef int RRGGBB[3]
    #RRGGBB = <int *>mem.PyMem_Malloc(sizeof(int)*3)
    H = HH*400/360.0
    J = JJ*1.0
    C = CC*1.0
    h_ = 0
    for i in range(4):
        if H>=colordata[i][2] and H<colordata[i+1][2]:
            h_ = ((H-colordata[i][2])*(colordata[i+1][1]*colordata[i][0]-colordata[i][1]*colordata[i+1][0])\
                  - 100*colordata[i][0]*colordata[i+1][1])/\
                ((H-colordata[i][2])*(colordata[i+1][1]-colordata[i][1]) - 100*colordata[i+1][1])
            if h_>360:
                h_ -= 360
            break
    t = 0
    if J!=0:
        t = (C/(((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)))**(1/0.9)
    #etemp =
    etemp = (np.cos(h_*np.pi/180+2)+3.8) * 0.25
    e = (50000/13.0)*Nc*Ncb*etemp
    A = Aw*((J/100)**(1/(c*z)))
    a = b =0
    p2 = A/Nbb +0.305
    p3 = 21/20.0
    h = h_*np.pi/180
    if t!=0:
        p1 = e/t #t!=0
        if fabs(sin(h))>fabs(np.cos(h)):
            p4 = p1/sin(h)
            b = (p2*(2+p3)*(460.0/1403))/\
                (p4+(2+p3)*(220.0/1403)*(np.cos(h)/sin(h))-27.0/1403+\
                 p3*(6300.0/1403))
            a = b*(np.cos(h)/sin(h))
        elif fabs(np.cos(h))>fabs(sin(h)):
            p5 = p1/np.cos(h)
            a = (p2*(2+p3)*(460.0/1403))/\
                (p5+(2+p3)*(220.0/1403)-\
                (27.0/1403 - p3*(6300.0/1403))*(sin(h)/np.cos(h)))
            b = a*(sin(h)/np.cos(h))
    # step 6
    Ra_ = (460*p2 + 451*a + 288*b)/1403.0
    Ga_ = (460*p2 - 891*a - 261*b)/1403.0
    Ba_ = (460*p2 - 220*a - 6300*b)/1403.0
    R_ = np.sign(Ra_-0.1)*(100.0/FL)*(((27.13*fabs(Ra_-0.1))/(400-fabs(Ra_-0.1)))**(1/0.42))
    G_ = np.sign(Ga_-0.1)*(100.0/FL)*(((27.13*fabs(Ga_-0.1))/(400-fabs(Ga_-0.1)))**(1/0.42))
    B_ = np.sign(Ba_-0.1)*(100.0/FL)*(((27.13*fabs(Ba_-0.1))/(400-fabs(Ba_-0.1)))**(1/0.42))
    #step8
    Rc,Gc,Bc = Mcat02.dot(M_1hpe.dot([R_,G_,B_]))
    R,G,B    = [Rc/Dr,Gc/Dg,Bc/Db]
    X,Y,Z    = M_1cat02.dot([R,G,B])
    X /= 100.0
    Y /= 100.0
    Z /= 100.0
    #cdef float X = color[0]/100.0           # 0 --- 95.047
    #cdef float Y = color[1]/100.0           # 0 --- 100.00
    #cdef float Z = color[2]/100.0           # 0 --- 108.883

    cdef double varR = X * 3.2406 - Y * 1.5372 - Z * 0.4986
    cdef double varG = X * (-0.9689) + Y * 1.8758 + Z * 0.0415
    cdef double varB = X * 0.0557 - Y * 0.2040 + Z * 1.0570

    #def convert(C):
    cdef int RR = <int>(round(convert2rgb(varR)*255))
    cdef int GG = <int>(round(convert2rgb(varG)*255))
    cdef int BB = <int>(round(convert2rgb(varB)*255))
    #print RR,GG,BB
    RRGGBB[0] = RR
    RRGGBB[1] = GG
    RRGGBB[2] = BB
    return RRGGBB
    #return [X,Y,Z]
#    xyz = jch2xyz(jch)
    #return xyz2rgb(xyz)
def jch2rgb(list jch):
    """provide a quick convertion from cam02 JCH value to rgb

    XwYwZw default to [95.05,100.00,108.88]
    Nc,c,F default to dim mode 0.9,0.59,0.9
    LA,Yb  default to 80.0 16.0

    Args:
         jch: need to provide [J,C,H] list
            J: Lightness ranging from 0 to 100 float
            C: Chroma
            H: Hue quadrature ranging from 0 to 400
    Returns:
         RGB list like [R,G,B]
    """
    cdef double j,c,h,r,g,b
    j = jch[0]
    c = jch[1]
    h = jch[2]
    cdef int * rgb
    rgb = cjch2rgb(j,c,h)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return [r,g,b]

#"""
