ciecam02    An easier and fast transformer between rgb and ciecam02 color space
===============================================================================

Converts color between rgb and a simplified form of the CIECAM02 model named
JCH, wrapped in the numpy arrays in a quick algorithm, for a practical use for
doing algorithms of images, displaying, etc.

introduce
---------

[Munsell color system](img/Moncell-system.png)

![img](https://en.wikipedia.org/wiki/File:Munsell-system.svg)

[CIECAM02](https://en.wikipedia.org/wiki/CIECAM02) approximately linearize 
Moncell color system.

Color type rgb could be use widely in display devices and image formats. The
data form is an integer list [r, g, b], where r g b is among 0 - 255.

CIECAM02 produces multiple correlates, like H, J, z, Q, t, C, M, s. Some of
them represent similar concepts, such as C means chroma and M colorfulness
s saturation correlate the same thing in different density. We need only 3 
major property of these arguments to completely represent a color, and we 
can get other properties or reverse algorithms.

Color type jch is a float list like [j, c, h], where 0.0 < j < 100.0,
0.0 < h < 360.0, and 0.0 < c. the max value of c does not limit, and may 
produce exceeds when transform to rgb. The effective value of max c varies.
Probablly for red color h 0.0, and brightness j 50.0, c reach the valid 
maximum, values about 160.0.

And jch comes from the CIECAM02 model outputs as an float list like
[j, c, h], and some distortion was made to obtain a proper proportion.

j values the same as J, the brightness.

c values the same as C, the chroma.

h compress the original H from 0-400 to 0-360 by simply * 0.9 for 
represents in a polar coordinates.

install
--------

    pip install ciecam02

Usage
-----

Basic functions:

    import numpy as np
    from ciecam02 import rgb2jch, jch2rgb
    color = np.array([[20, 20, 20],
                  [56, 34, 199],
                  [255, 255, 255]
                  ])
                  
    output: [[  6.942311     0.38424796 242.41395946]
             [ 21.4432157   74.80048318 284.3167947 ]
             [ 99.99968129   1.49090566 242.41103965]]




a set of function convert colorspace among rgb ciexyz ciecam02.
functions:
        xyz2rgb
        rgb2xyz
        xyz2cam02
        rgb2jch
        jch2xyz
        jch2rgb
