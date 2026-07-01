import numpy as np
import math
import scipy.ndimage as nd

def near_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n):
    """Near field displacement for a moment tensor eource in isotropic media"""
    
    nt=X0.shape[0]
    X=np.zeros(nt)
    r=float(np.linalg.norm(x))
    gi=x[i]/r
    gj=x[j]/r
    gn=x[n]/r
    dij=(i==j)
    dni=(n==i)
    dnj=(n==j)
    a=r/alpha
    b=r/beta
    ia=round(a/dt)
    ib=round(b/dt)
    factor=1.0/(4*math.pi*rho)*(15*gn*gi*gj-3*gn*dij-3*gi*dnj-3*gj*dni)/r**4
    for it in range(nt):
        val=0
        for itau in range(ia,ib):
            if (it-itau>=0) and (it-itau<nt):
                val += itau*dt*X0[it-itau]*dt
        X[it] = factor*val
    return X

def mid_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n):
    """Mid field displacement for a moment tensor source in isotropic media"""
    
    nt=X0.shape[0]
    X=np.zeros(nt)
    r=float(np.linalg.norm(x))
    gi=x[i]/r
    gj=x[j]/r
    gn=x[n]/r
    dij=(i==j)
    dni=(n==i)
    dnj=(n==j)
    a=r/alpha
    b=r/beta
    factor1=1.0/(4*math.pi*rho*alpha**2)*(6*gn*gi*gj-gn*dij-gi*dnj-gj*dni)/r**2
    factor2=1.0/(4*math.pi*rho*beta**2)*(6*gn*gi*gj-gn*dij-gi*dnj-2*gj*dni)/r**2
    temp1=np.copy(X0)
    temp2=np.copy(X0)
    temp1=nd.shift(temp1,a/dt,mode='nearest')
    temp2=nd.shift(temp2,b/dt,mode='nearest')
    X = factor1*temp1 - factor2*temp2
    return X

def far_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n):
    """Far field displacement for a moment tensor source in isotropic media"""
    
    nt=X0.shape[0]
    X=np.zeros(nt)
    r=float(np.linalg.norm(x))
    gi=x[i]/r
    gj=x[j]/r
    gn=x[n]/r
    dij=(i==j)
    dni=(n==i)
    dnj=(n==j)
    a=r/alpha
    b=r/beta
    factor1=1.0/(4*math.pi*rho*alpha**3)*gn*gi*gj/r
    factor2=1.0/(4*math.pi*rho*beta**3)*gj*(gn*gi-dni)/r
    temp1=np.copy(X0)
    temp2=np.copy(X0)
    temp1=nd.shift(temp1,a/dt,mode='nearest')
    temp2=nd.shift(temp2,b/dt,mode='nearest')
    for it in range(1,nt-1):
        X[it] = factor1*0.5*(temp1[it+1]-temp1[it-1]) - factor2*0.5*(temp2[it+1]-temp2[it-1])
    X /= dt
    return X

def total_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n):
    return near_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n) + mid_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n) + far_field_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n)

def near_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k):
    """Near field strain s_nk for a moment tensor source in isotropic media"""
    
    nt=X0.shape[0]
    X=np.zeros(nt)
    r=float(np.linalg.norm(x))
    gi=x[i]/r
    gj=x[j]/r
    gn=x[n]/r
    gik=((i==k)*r**2 - x[i]*x[k])/r**3
    gjk=((j==k)*r**2 - x[j]*x[k])/r**3
    gnk=((n==k)*r**2 - x[n]*x[k])/r**3
    rk=x[k]/r
    dij=(i==j)
    dni=(n==i)
    dnj=(n==j)
    a=r/alpha
    b=r/beta
    ak=rk/alpha
    bk=rk/beta
    ia=round(a/dt)
    ib=round(b/dt)
    factor=1.0/(4*math.pi*rho)*(15*gn*gi*gj-3*gn*dij-3*gi*dnj-3*gj*dni)/r**4
    factork=1.0/(4*math.pi*rho)*(r**2*(15*(gnk*gi*gj+gn*gik*gj+gn*gi*gjk) - 3*gnk*dij - 3*gik*dnj - 3*gjk*dni) - 4*x[k]*(15*gn*gi*gj-3*gn*dij-3*gi*dnj-3*gj*dni))/r**6
    for it in range(nt):
        val=0
        for itau in range(ia,ib):
            if (it-itau>=0) and (it-itau<nt):
                val += itau*dt*X0[it-itau]*dt
        X[it] += factork*val
        if (it-ib>=0) and (it-ib<nt):
            X[it] += factor*b*X0[it-ib]*bk
        if (it-ia>=0) and (it-ia<nt):
            X[it] -= factor*a*X0[it-ia]*ak
    return X

def mid_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k): 
    """Mid field strain s_nk for a moment tensor source in isotropic media"""
    
    nt=X0.shape[0]
    X=np.zeros(nt)
    r=float(np.linalg.norm(x))
    gi=x[i]/r
    gj=x[j]/r
    gn=x[n]/r
    gik=((i==k)*r**2 - x[i]*x[k])/r**3
    gjk=((j==k)*r**2 - x[j]*x[k])/r**3
    gnk=((n==k)*r**2 - x[n]*x[k])/r**3
    rk=x[k]/r
    dij=(i==j)
    dni=(n==i)
    dnj=(n==j)
    a=r/alpha
    b=r/beta
    ak=rk/alpha
    bk=rk/beta
    factor1=1.0/(4*math.pi*rho*alpha**2)*(6*gn*gi*gj-gn*dij-gi*dnj-gj*dni)/r**2
    factor2=1.0/(4*math.pi*rho*beta**2)*(6*gn*gi*gj-gn*dij-gi*dnj-2*gj*dni)/r**2
    factor1k=1.0/(4*math.pi*rho*alpha**2)*(r*(6*(gnk*gi*gj+gn*gik*gj+gn*gi*gjk) - gnk*dij - gik*dnj - gjk*dni) - 2*rk*(6*gn*gi*gj-gn*dij-gi*dnj-gj*dni))/r**3
    factor2k=1.0/(4*math.pi*rho*beta**2)*(r*(6*(gnk*gi*gj+gn*gik*gj+gn*gi*gjk) - gnk*dij - gik*dnj - 2*gjk*dni) - 2*rk*(6*gn*gi*gj-gn*dij-gi*dnj-2*gj*dni))/r**3
    temp1=np.copy(X0)
    temp2=np.copy(X0)
    temp1=nd.shift(temp1,a/dt,mode='nearest')
    temp2=nd.shift(temp2,b/dt,mode='nearest')
    for it in range(1,nt-1):
        X[it] = factor1k*temp1[it] - factor1*ak*0.5*(temp1[it+1]-temp1[it-1])/dt
        X[it] = X[it] - factor2k*temp2[it] + factor2*bk*0.5*(temp2[it+1]-temp2[it-1])/dt
    return X

def far_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k):
    """Far field strain s_nk for a moment tensor source in isotropic media"""
    
    nt=X0.shape[0]
    X=np.zeros(nt)
    r=float(np.linalg.norm(x))
    gi=x[i]/r
    gj=x[j]/r
    gn=x[n]/r
    gik=((i==k)*r**2 - x[i]*x[k])/r**3
    gjk=((j==k)*r**2 - x[j]*x[k])/r**3
    gnk=((n==k)*r**2 - x[n]*x[k])/r**3
    rk=x[k]/r
    dij=(i==j)
    dni=(n==i)
    dnj=(n==j)
    a=r/alpha
    b=r/beta
    ak=rk/alpha
    bk=rk/beta
    factor1=1.0/(4*math.pi*rho*alpha**3)*gn*gi*gj/r
    factor2=1.0/(4*math.pi*rho*beta**3)*gj*(gn*gi-dni)/r
    factor1k=1.0/(4*math.pi*rho*alpha**3)*(r*(gnk*gi*gj + gn*gik*gj + gn*gi*gjk) - rk*gn*gi*gj)/r**2
    factor2k=1.0/(4*math.pi*rho*beta**3)*(r*(gjk*(gn*gi-dni) + gj*(gnk*gi+gn*gik)) - rk*gj*(gn*gi-dni))/r**2
    temp1=np.copy(X0)
    temp2=np.copy(X0)
    temp1=nd.shift(temp1,a/dt,mode='nearest')
    temp2=nd.shift(temp2,b/dt,mode='nearest')
    for it in range(1,nt-1):
        X[it] = factor1k*0.5*(temp1[it+1]-temp1[it-1])/dt - factor1*ak*(temp1[it+1]-2*temp1[it]+temp1[it-1])/dt**2
        X[it] = X[it] - factor2k*0.5*(temp2[it+1]-temp2[it-1])/dt + factor2*bk*(temp2[it+1]-2*temp2[it]+temp2[it-1])/dt**2
    return X

def total_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k):
    return near_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k) + mid_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k) + far_field_strain_iso_mt(X0,dt,alpha,beta,rho,x,i,j,n,k)

def strain_tensor(s11,s22,s33,s12,s13,s23):
    """Build the 3x3 strain tensor from the 6 independent components"""
    
    nt=s11.shape[0]
    e=np.zeros((nt,3,3))
    e[:,0,0] = s11
    e[:,1,1] = s22
    e[:,2,2] = s33
    e[:,0,1] = s12
    e[:,0,2] = s13
    e[:,1,2] = s23
    e[:,1,0] = e[:,0,1]
    e[:,2,0] = e[:,0,2]
    e[:,2,1] = e[:,1,2]
    
    return e

def directional_strain(e,l):
    """Compute the scalar strain in a given direction"""
    
    nt=e.shape[0]
    lel=np.zeros(nt)
    for it in range(nt):
        lel[it] = np.dot(l,np.matmul(e[it,:,:],l))
    
    return lel


# load modeled geophone and DAS traces (the geophone file must contain 3 traces, 
# the DAS file must contain a single trace)
#geo=np.loadtxt('../tests/das_verification1_geo.txt') # SPECIFIED BY USER
#das=np.loadtxt('../tests/das_verification1_das.txt') # SPECIFIED BY USER
#geo=geo[:,1] # SPECIFIED BY USER depending on which column contains the actual seismic samples
#das=das[:,1] # SPECIFIED BY USER

ncomp=3
#geo=np.reshape(geo,(ncomp,int(geo.shape[0]/ncomp)))
#geo=np.transpose(geo)

# source wavelet as parameterized in GEOSX
dt=0.001 # SPECIFIED BY USER
f0=10. # SPECIFIED BY USER

#nt=geo.shape[0]
nt=1000
w=np.zeros(nt)
tpeak=1.0/f0
pi = math.pi
lam=(f0*pi)**2
for it in range(0,min(nt,int(2.9*tpeak/dt)+1)):
    t=it*dt
    w[it]=2.0*lam*(2.0*lam*(t-tpeak)*(t-tpeak)-1.0)*math.exp( -lam*(t-tpeak)*(t-tpeak))

# analytical geophone
alpha=4000. # SPECIFIED BY USER P-wave speed
beta=2000. # SPECIFIED BY USER S-wave speed
rho=2000. # SPECIFIED BY USER density
x=np.array([100.,100.,100.]) # SPECIFIED BY USER receiver x,y,z location

M=np.identity(3)

geosyn=np.zeros((nt,ncomp))
for n in range(ncomp):
    for p in range(3):
      for q in range(3):
        geosyn[:,n]+=total_field_iso_mt(M[p,q]*w,dt,alpha,beta,rho,x,p,q,n)


# analytical DAS
dip=0.523599 # SPECIFIED BY USER ; DAS channel dip with respect to the horizontal
az=0.785398 # SPECIFIED BY USER ; DAS channel azimuth with respect to the x-axis counter-clockwise when looking down
gl=8. # SPECIFIED BY USER ; gauge length

l=np.array([math.cos(dip)*math.cos(az),math.cos(dip)*math.sin(az),math.sin(dip)]) # DAS fiber direction 
xstart=x-0.5*gl*l
xend=x+0.5*gl*l
Nquad=7 # number of quadrature points to estimate the integral
dassyn=np.zeros((nt))
for i in range(Nquad):
    xavg = xstart+gl*(i+1-0.5)/(Nquad)*l
    strain=np.zeros((nt,3,3))
    for n in range(3):
        for k in range(3):
            for p in range(3):
              for q in range(3):
                strain[:,n,k] += total_field_strain_iso_mt(M[p,q]*w,dt,alpha,beta,rho,xavg,p,q,n,k)
    dassyn += directional_strain(strain,l)
print(f"DAS fiber direction: {l}")
print(f"DAS fiber endpoints: {xstart}, {xend}")
dassyn = dassyn / Nquad
np.savetxt("dassyn",dassyn)
np.savetxt("ux.txt",geosyn[:,0])
np.savetxt("uy.txt",geosyn[:,1])
np.savetxt("uz.txt",geosyn[:,2])
for i in range(3):
  for j in range(3):
    np.savetxt("strain_"+str(i)+str(j),strain[:,i,j])
#strain at endopoints
geosyndiff=np.zeros((nt,ncomp))
for n in range(ncomp):
    for p in range(3):
      for q in range(3):
        geosyndiff[:,n]+=total_field_iso_mt(M[p,q]*w,dt,alpha,beta,rho,xend,p,q,n)-total_field_iso_mt(M[p,q]*w,dt,alpha,beta,rho,xstart,p,q,n) 
np.savetxt("uxdiff.txt",geosyndiff[:,0])
np.savetxt("uydiff.txt",geosyndiff[:,1])
np.savetxt("uzdiff.txt",geosyndiff[:,2])

