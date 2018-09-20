import numpy as np


def xyz2infinitergb(xyz):
    xyz = xyz/100.0

    M_1 = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758,  0.0415],
                    [0.0557, -0.2040,  1.0570]]).T
    RGB = xyz.dot(M_1)
    RGB = np.where(RGB <= 0, 0.00000001, RGB)
    RGB = np.where(RGB > 0.0031308,
                   1.055*(RGB**0.4166666)-0.055,
                   12.92*RGB)

    RGB = np.around(RGB*255)

    return RGB


def xyz2rgb(xyz):
    xyz = xyz/100.0

    M_1 = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758,  0.0415],
                    [0.0557, -0.2040,  1.0570]]).T
    RGB = xyz.dot(M_1)
    RGB = np.where(RGB <= 0, 0.00000001, RGB)
    RGB = np.where(RGB > 0.0031308,
                   1.055*(RGB**0.4166666)-0.055,
                   12.92*RGB)

    RGB = np.around(RGB*255)
    RGB = np.where(RGB <= 0, 0, RGB)
    RGB = np.where(RGB > 255, 255, RGB)
    RGB = RGB.astype('uint8')

    return RGB


def rgb2xyz(color):
    color = color/255.0
    color = np.where(color > 0.04045, np.power(((color+0.055)/1.055), 2.4),
                     color/12.92)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]]).T

    return color.dot(M)*100


whitepoint = {'white': [95.05, 100.00, 108.88],
              'c': [109.85, 100.0, 35.58]}

env = {'dim': [0.9, 0.59, 0.9],
       'average': [1.0, 0.69, 1.0],
       'dark': [0.8, 0.525, 0.8]}
lightindensity = {'default': 80.0, 'high': 318.31, 'low': 31.83}
bgindensity = {'default': 16.0, 'high': 20.0, 'low': 10.0}

currentwhite = whitepoint['white']
currentenv = env['average']
currentlight = lightindensity['default']
currentbg = bgindensity['default']


def setconfig(a='white', b='average', c='default', d='default'):
    currentwhite = whitepoint[a]
    currentenv = env[b]
    currentlight = lightindensity[c]
    currentbg = bgindensity[d]


Mcat02 = np.array([[0.7328, 0.4296, -0.1624],
                   [-0.7036, 1.6975, 0.0061],
                   [0.0030, 0.0136, 0.9834]])
Xw, Yw, Zw = currentwhite
Nc, c, F = currentenv
LA = currentlight
Yb = currentbg
Rw, Gw, Bw = Mcat02.dot(np.array([Xw, Yw, Zw]))
D = F*(1 - (1/3.6) * (np.e**((-LA-42)/92)))
if D > 1:
    D = 1
if D < 0:
    D = 0
Dr, Dg, Db = [Yw*D/Rw+1-D, Yw*D/Gw+1-D, Yw*D/Bw+1-D]
Rwc, Gwc, Bwc = [Dr*Rw, Dg*Gw, Db*Bw]
k = 1/(5*LA+1)
FL = 0.2*(k**4)*(5*LA) + 0.1*((1-(k**4))**2)*((5*LA)**(1/3.0))
n = Yb/Yw
if n > 1:
    n = 1
if n < 0:
    n = 0.000001
Nbb = Ncb = 0.725*((1.0/n)**0.2)
z = 1.48 + n**0.5
M_1cat02 = np.array([[1.096241, -0.278869, 0.182745],
                     [0.454369, 0.473533, 0.072098],
                     [-0.009628, -0.005698, 1.015326]])
MH = np.array([[0.38971, 0.68898, -0.07868],
               [-0.22981, 1.18340, 0.04641],
               [0.00000, 0.00000, 1.00000]])
M_1hpe = np.array([[1.910197, -1.112124, 0.201908],
                   [0.370950, 0.629054, -0.000008],
                   [0.000000, 0.000000, 1.000000]])
Rw_, Gw_, Bw_ = MH.dot(M_1cat02.dot([Rwc, Gwc, Bwc]))
colordata = [[20.14, 0.8, 0], [90, 0.7, 100], [164.25, 1.0, 200],
             [237.53, 1.2, 300],
             [380.14, 0.8, 400]]
Rwa_ = (400 * ((FL*Rw_/100)**0.42))/(27.13+((FL*Rw_/100)**0.42))+0.1
Gwa_ = (400 * ((FL*Gw_/100)**0.42))/(27.13+((FL*Gw_/100)**0.42))+0.1
Bwa_ = (400 * ((FL*Bw_/100)**0.42))/(27.13+((FL*Bw_/100)**0.42))+0.1
Aw = Nbb * (2*Rwa_+Gwa_+(Bwa_/20) - 0.305)


def xyz2cam02(XYZ):
    RGB = XYZ.dot(Mcat02.T)
    RcGcBc = RGB*np.array([Dr, Dg, Db])
    # step 5
    R_G_B_ = RcGcBc.dot(M_1cat02.T).dot(MH.T)
    # step 6
    R_G_B_in = np.power(FL*R_G_B_/100, 0.42)
    Ra_Ga_Ba_ = (400 * R_G_B_in)/(27.13 + R_G_B_in) + 0.1
    # step 7
    a = Ra_Ga_Ba_[:, 0] - 12 * Ra_Ga_Ba_[:, 1]/11 + Ra_Ga_Ba_[:, 2]/11
    b = (1/9.0) * (Ra_Ga_Ba_[:, 0]+Ra_Ga_Ba_[:, 1]-2*Ra_Ga_Ba_[:, 2])
    h = np.arctan2(b, a)

    h = np.where(h < 0, (h+np.pi*2)*180/np.pi, h*180/np.pi)
    huue = np.where(h < colordata[0][0], h+360, h)
    etemp = (np.cos(huue*np.pi/180+2)+3.8) * 0.25
    coarray = np.array([20.14, 90, 164.25, 237.53, 380.14])
    position_ = coarray.searchsorted(huue)

    def TransferHue(h_, i):
        datai = colordata[i-1]
        datai1 = colordata[i]
        Hue = datai[2] + ((100*(h_-datai[0])/datai[1]) /
                          (((h_-datai[0])/datai[1])+(datai1[0]-h_)/datai1[1]))
        return Hue
    ufunc_TransferHue = np.frompyfunc(TransferHue, 2, 1)
    H = ufunc_TransferHue(huue, position_).astype('float')
    # step 9
    A = Nbb * (2*Ra_Ga_Ba_[:, 0] +
               Ra_Ga_Ba_[:, 1]+(Ra_Ga_Ba_[:, 2]/20.0) - 0.305)
    # step10
    J = 100*((A/Aw)**(c*z))
    # step 11
    Q = (4/c) * ((J/100.0)**0.5) * (Aw + 4) * (FL**0.25)
    # step 12
    t = ((50000/13.0)*Nc*Ncb*etemp*((a**2+b**2)**0.5)) /\
        (Ra_Ga_Ba_[:, 0]+Ra_Ga_Ba_[:, 1]+(21/20.0)*Ra_Ga_Ba_[:, 2])
    C = t**0.9*((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)
    M = C*(FL**0.25)
    s = 100*((M/Q)**0.5)
    return np.array([h, H, J, Q, C, M, s]).T


def rgb2jch(color):
    XYZ = rgb2xyz(color)
    value = xyz2cam02(XYZ)
    return value[:, [2, 4, 1]]*np.array([1.0, 1.0, 0.9])


def jch2xyz(jch):
    JCH = jch*np.array([1.0, 1.0, 10/9.0])
    J = JCH[:, 0]
    C = JCH[:, 1]
    H = JCH[:, 2]
    coarray = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    position_ = coarray.searchsorted(H)

    def TransferHue(H_, i):
        C1 = colordata[i-1]
        C2 = colordata[i]
        h = ((H_-C1[2])*(C2[1]*C1[0]-C1[1]*C2[0])-100*C1[0]*C2[1]) /\
            ((H_-C1[2])*(C2[1]-C1[1]) - 100*colordata[i][1])
        if h > 360:
            h -= 360
        return h
    ufunc_TransferHue = np.frompyfunc(TransferHue, 2, 1)
    h_ = ufunc_TransferHue(JCH[:, 2], position_).astype('float')
    J = np.where(J <= 0, 0.00001, J)
    C = np.where(C <= 0, 0.00001, C)
    t = (C/(((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)))**(1/0.9)
    t = np.where(t - 0 < 0.00001, 0.00001, t)
    etemp = (np.cos(h_*np.pi/180+2)+3.8) * 0.25
    e = (50000/13.0)*Nc*Ncb*etemp
    A = Aw*((J/100)**(1/(c*z)))

    pp2 = A/Nbb + 0.305
    p3 = 21/20.0
    hue = h_*np.pi/180
    pp1 = e/t

    def evalAB(h, p1, p2):
        if abs(np.sin(h)) >= abs(np.cos(h)):
            p4 = p1/np.sin(h)
            b = (p2*(2+p3)*(460.0/1403)) /\
                (p4+(2+p3)*(220.0/1403)*(np.cos(h)/np.sin(h))-27.0/1403 +
                 p3*(6300.0/1403))
            a = b*(np.cos(h)/np.sin(h))
        else:  # abs(np.cos(h))>abs(np.sin(h)):
            p5 = p1/np.cos(h)
            a = (p2*(2+p3)*(460.0/1403)) /\
                (p5+(2+p3)*(220.0/1403) -
                 (27.0/1403 - p3*(6300.0/1403))*(np.sin(h)/np.cos(h)))
            b = a*(np.sin(h)/np.cos(h))
        return np.array([a, b])
    ufunc_evalAB = np.frompyfunc(evalAB, 3, 1)
    abinter = np.row_stack(ufunc_evalAB(hue, pp1, pp2))
    a = abinter[:, 0]
    b = abinter[:, 1]

    Ra_ = (460*pp2 + 451*a + 288*b)/1403.0
    Ga_ = (460*pp2 - 891*a - 261*b)/1403.0
    Ba_ = (460*pp2 - 220*a - 6300*b)/1403.0
    R_ = np.sign(Ra_-0.1)*(100.0/FL) *\
        (((27.13*np.abs(Ra_-0.1))/(400-np.abs(Ra_-0.1)))**(1/0.42))
    G_ = np.sign(Ga_-0.1)*(100.0/FL) *\
        (((27.13*np.abs(Ga_-0.1))/(400-np.abs(Ga_-0.1)))**(1/0.42))
    B_ = np.sign(Ba_-0.1)*(100.0/FL) *\
        (((27.13*np.abs(Ba_-0.1))/(400-np.abs(Ba_-0.1)))**(1/0.42))

    RcGcBc = (np.array([R_, G_, B_]).T).dot(M_1hpe.T).dot(Mcat02.T)
    RGB = RcGcBc/np.array([Dr, Dg, Db])
    XYZ = RGB.dot(M_1cat02.T)
    return XYZ


def jch2rgb(jch):
    xyz = jch2xyz(jch)
    return xyz2rgb(xyz)


if __name__ == "__main__":
    a = np.array([[20, 20, 20],
                  [56, 34, 199],
                  [255, 255, 255]
                  ])
    print(rgb2jch(a))
