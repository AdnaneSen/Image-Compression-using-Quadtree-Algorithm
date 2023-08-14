import matplotlib.pyplot as plt
from timeit import default_timer
from PIL import Image
from math import sqrt,log10
import gc
import sys
from PIL import ImageFilter


im = Image.open("voiture.jpg")

px = im.load()
W,H = im.size
print(W,H)

def peindre(x,y,w,h,r,g,b):
    
    for i in range(w):
        for j in range(h):
            px[x+i,y+j] = (r,g,b)
            

def moyenne(x,y,w,h):
    R,G,B=0,0,0
    for i in range(w):
        for j in range(h):
            r,g,b=px[x+i, y+j]
            R+=r
            G+=g
            B+=b
    n=w*h
    return(R/n,G/n,B/n)
    
def ecart_type(x,y,w,h):
    Rm,Gm,Bm=moyenne(x,y,w,h)
    R,G,B=0,0,0
    n=w*h
    for i in range(w):
        for j in range(h):
            r,g,b=px[x+i, y+j]
            R+=r**2
            G+=g**2
            B+=b**2
    return(sqrt(R/n-Rm**2),sqrt(G/n-Gm**2),sqrt(B/n-Bm**2))

def est_homogene(x,y,w,h,seuil):
    return sum(ecart_type(x,y,w,h))/3 <= seuil

def quadripartition(x,y,w,h):
    assert w>0 and h>0 and not w==h==1
    i = (w+1)//2
    j = (h+1)//2
    return (
		(x, y, i, j),
		(x+i, y, w-i, j) if w>1 else None,
		(x, y+j, i, h-j) if h>1 else None,
		(x+i, y+j, w-i, h-j) if w>1 and h>1 else None)
    
class Noeud:
    def __init__(self,x,y,w,h,r,g,b,hg,hd,bg,bd):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.r = r
        self.g = g
        self.b = b
        self.hg = hg
        self.hd = hd
        self.bg = bg
        self.bd = bd
        
    def __str__(self,prefix=""):
        # if len(prefix) > 4:
        #     return ""
        return "\n".join((f"{prefix}({self.x},{self.y}) enfants :",
                    self.hg.__str__(prefix+"  ") if self.hg !=None else prefix+ "  None",
                    self.hd.__str__(prefix+"  ") if self.hd !=None else prefix+ "  None",
                    self.bg.__str__(prefix+"  ") if self.bg !=None else prefix+ "  None",
                    self.bd.__str__(prefix+"  ") if self.bd !=None else prefix+ "  None",
                ))
        
        
def arbre(x,y,w,h,seuil):
    r,g,b = moyenne(x,y,w,h)
    if est_homogene(x,y,w,h,seuil):
        return Noeud(x,y,w,h,r,g,b,None,None,None,None)
    else:
        hg,hd,bg,bd = quadripartition(x,y,w,h)
        return Noeud(x,y,w,h,r,g,b,
                     arbre(*hg,seuil) if hg != None else None,
                     arbre(*hd,seuil) if hd != None else None,
                     arbre(*bg,seuil) if bg != None else None,
                     arbre(*bd,seuil) if bd != None else None)

def compter(N):
    if N == None:
        return 0
    
    else:
        return 1 + compter(N.hg) + compter(N.hd)+ compter(N.bg) + compter(N.bd)

def peindre_arbre(n):
    if n == None:
        return
    
    if n.hg == n.hd == n.bg == n.bd == None:
        peindre(n.x,n.y,n.w,n.h,round(n.r),round(n.g),round(n.b))
    else:
        peindre_arbre(n.hg)
        peindre_arbre(n.hd)
        peindre_arbre(n.bg)
        peindre_arbre(n.bd)

def peindre_profondeur(n, m, p=0):
	if n == None:
		return
	if n.hg==n.hd==n.bg==n.bd==None:
		peindre(n.x, n.y, n.h, n.w, (255*p//m, 255*p//m, 255*p//m))
	else:
		peindre_profondeur(n.hg, m, p+1)
		peindre_profondeur(n.hd, m, p+1)
		peindre_profondeur(n.bg, m, p+1)
		peindre_profondeur(n.bd, m, p+1)
        
def EQ(n):
    if n == None:
        return 0
    
    if n.hg == n.hd == n.bg == n.bd == None:
        eq = 0
        for i in range(n.w):
            for j in range(n.h):
                r,g,b = px[n.x+i,n.y+j]
                eq += (r-n.r)**2 + (g-n.g)**2 + (b-n.b)**2
                
        return eq
    
    else:
        return EQ(n.hg) + EQ(n.hd) + EQ(n.bg) + EQ(n.bd)
                
def PSNR(n):
    return 20*log10(255) - 10*log10(EQ(n)/3 / n.w / n.h)


#Optimisation---------------------------------------------------------------------------------------------
#Formules permettant de calculer l'indice de l'enfant: haut gauche : 4*i+1 ; haut droit : 4*i+2 ; bas gauche : 4*i+3 ; bas droit : 4*i+4
#Formules permettant de calculer l'indice du parent : parent : E((i-1)/4)

class Noeud2:
    def __init__(self,x,y,w,h,r,g,b):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.r = r
        self.g = g
        self.b = b
        
    def __str__(self,prefix=""):
        return "\n".join((f"{prefix}({self.x},{self.y}) enfants :",
                    self.hg.__str__(prefix+"  ") if self.hg !=None else prefix+ "  None",
                    self.hd.__str__(prefix+"  ") if self.hd !=None else prefix+ "  None",
                    self.bg.__str__(prefix+"  ") if self.bg !=None else prefix+ "  None",
                    self.bd.__str__(prefix+"  ") if self.bd !=None else prefix+ "  None",
                ))


def creation_liste(x,y,w,h,seuil, liste, i): 
    r,g,b = moyenne(x,y,w,h) # le triplet par la moyenne de la région réctangulaire d'image
    liste[i] = Noeud2(x,y,w,h,round(r),round(g),round(b))  #Création du noeud d'indice i
    if not est_homogene(x,y,w,h,seuil): # Si la région n'est pas homogène
        liste2 = quadripartition(x,y,w,h) # On la répartie en 4 sous régions
        if len(liste) <= (i + 1)*4:  # Si la longueur de la liste n'est pas suffisante
             for k in range((i + 1)*4 - len(liste) + 1):
                liste.append(None)   # On remplit la liste jusqu'au dernier fils
        k = 1
        for fils in liste2:
            if fils == None:
                k+=1
            else:
                liste = creation_liste(*fils, seuil, liste, 4*i + k) 
                k += 1
    return liste

def arbre_implicite(x,y,w,h,seuil): 
    return creation_liste(x,y,w,h,seuil,[None],0)

def peindre_arbre_implicite (x,y,w,h,seuil):
    Arbre = arbre_implicite(x,y,w,h,seuil)
    l = len (Arbre)
    for i in range (l):
        if Arbre[i] != None:
            if 4*i + 4 < l:
                if Arbre[4*i + 1] == None and Arbre[4*i + 2] == None and Arbre[4*i + 3] == None and Arbre[4*i + 4] == None:
                    (r,g,b) = (Arbre[i].r,Arbre[i].g,Arbre[i].b)
                    peindre(Arbre[i].x, Arbre[i].y, Arbre[i].w, Arbre[i].h,r,g,b)
            else:
                (r,g,b) = (Arbre[i].r,Arbre[i].g,Arbre[i].b)
                peindre(Arbre[i].x,Arbre[i].y, Arbre[i].w, Arbre[i].h,r,g,b)

def temps_creation_arbre():
    X=[5*i+10 for i in range(12)]
    Y1=[]
    Y2=[]
    Ecart_relatif=[]
    for j in range (12):
        i1 = default_timer()
        arbre(0,0,W,H,10+j*5)
        delta1 = default_timer() - i1
        Y1.append(delta1)
        
        i2 = default_timer()
        arbre_implicite(0,0,W,H,10+j*5)
        delta2 = default_timer() - i2
        Y2.append(delta2) 

    plt.figure()
    plt.plot(X, Y1,'x:' , label='arbre explicite')
    plt.plot(X, Y2, 'x:', label='arbre implicite')
    plt.title("Comparaison du temps d'éxécution des deux méthode en fonction du seuil")
    plt.legend()
    #plt.show() 
    
def comparer_memoire():
    X=[5*i+10 for i in range(12)]
    Y1=[]
    Y2=[]
    
    for i in range(12):
        gc.collect()
        i1 = sys.getallocatedblocks()
        arbre(0,0,W,H,5+i*5)
        delta1 = sys.getallocatedblocks() - i1
        Y1.append(delta1)
        
        gc.collect()
        i2 = sys.getallocatedblocks()
        arbre_implicite(0,0,W,H,5+i*5)
        delta2 = sys.getallocatedblocks() - i2
        Y2.append(delta2)
        
    plt.figure()
    plt.plot(X, Y1,'x:' , label='arbre explicite')
    plt.plot(X, Y2, 'x:', label='arbre implicite')
    plt.title("Comparaison de l'espace de créaction de l'arbre pour les deux méthodes")
    plt.legend()
    plt.show()
 

def EQ2(noeud):
      if noeud == None:
          return(0)
      elif noeud.hg==noeud.hd==noeud.bg==noeud.bd==None:
          eq2 = 0
          for i in range(noeud.x,noeud.x+noeud.l):
              for j in range(noeud.y,noeud.y+noeud.h):
                  r, g, b = px[i, j]
                  eq2 += (r-noeud.r)**2 + 1.5*(g-noeud.v)**2 + (b-noeud.b)**2 #double importance du vert 
                  eq2 += 0.5*((noeud.r+noeud.v+noeud.b)/3-(r+g+b)/3)**(2) #contrastes
                  
          return(eq2)
      else:
          return(EQ(noeud.hg) + EQ(noeud.hd) + EQ(noeud.bg) + EQ(noeud.bd))
      
def PSNR2(Noeud):
    return(20*log10(255)-log10(EQ2(Noeud)/(3*W*H)))

def homogeniete(x,y,w,h,seuil,noeud):
    return( (sum(ecart_type(x,y,w,h))/3 <= seuil) and (EQ2(noeud)*10**(-5)<=seuil) )

def flouGaussien(w = W, h = H, x = 0, y = 0) :
    
    if x + w > W :
        w = W - x
    if y + h > H :
        h = H - y
    
    for i in range(1, w - 1) :
        for j in range(1, h - 1) :
            r = 0.05*(px[i-1, j-1][0] + px[i-1, j+1][0] + px[i+1, j-1][0] + px[i+1, j+1][0]) + 0.15*(px[i, j-1][0] + px[i, j+1][0] + px[i-1, j][0] + px[i+1, j][0]) + 0.2*px[i, j][0]
            g = 0.05*(px[i-1, j-1][1] + px[i-1, j+1][1] + px[i+1, j-1][1] + px[i+1, j+1][1]) + 0.15*(px[i, j-1][1] + px[i, j+1][1] + px[i-1, j][1] + px[i+1, j][1]) + 0.2*px[i, j][1]
            b = 0.05*(px[i-1, j-1][2] + px[i-1, j+1][2] + px[i+1, j-1][2] + px[i+1, j+1][2]) + 0.15*(px[i, j-1][2] + px[i, j+1][2] + px[i-1, j][2] + px[i+1, j][2]) + 0.2*px[i, j][2]
            px[i, j] = (int(r), int(g), int(b))


if __name__ == '__main__':     
    seuil=15
    A = arbre(0,0,W,H,seuil)
    peindre_arbre(A)
    # B= arbre_implicite(0,0,W,H,seuil)
    # peindre_arbre_implicite (0,0,W,H,seuil)
    print('PSNR 1 : ' + str(PSNR(A)))
    print('PSNR 2 : ' + str(PSNR2(A)))
    im.show()
    im.save('image_compresséebis.png')