import math
pi=math.pi

# def volume1(r,s):
#     return 4./3.*pi*r^3 + pi*rrs
#
# def volume2(r,s,l):
#     return 4/3pir^3+pirrl+(pirr+2rl)s
#
# def volume3(r,s,l,h):
#     return 4/3pir3+pir2(l+h)+sqrt(3)/2*hlr+(pir2+2lr+3hr+sqrt(3))/2*hl)s
#


def volume1(r, s):
    return 4.0 / 3.0 * pi * r ** 3 + pi * r * r * s

def volume2(r, s, l):
    return 4.0 / 3.0 * pi * r ** 3 + pi * r * r * l + (pi * r * r + 2 * r * l) * s

def volume3(r, s, l, h):
    return 4.0 / 3.0 * pi * r ** 3 + pi * r ** 2 * (l + h) + math.sqrt(3) / 2 * h * l * r + (
                pi * r ** 2 + 2 * l * r + 2 * h * r + math.sqrt(3) / 2 * h * l) * s


r=0.65
s=0.427
l=0.613
h1=0.465
h2=0.44
v0=volume1(r,s)
v1=volume2(r,s,0.9*l)
v2=volume2(r,s,h2)
v3=volume3(r,s,l,h1)
print(v0/v0,v1/v0,v2/v0,v3/v0)