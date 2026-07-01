import os
import sys
import argparse
import numpy as np

def geos_params_model_convert(x_in,model,tomodel):
    scalar_input=np.isscalar(x_in)
    x_in=np.atleast_1d(x_in)
    x_out=np.zeros(len(x_in))
    if not "multi:" in model:
        if model=="c":
            x_out[:]=x_in[:]
        elif model=="1/c":
            x_out[:]=1/x_in[:]
        elif model=="1/c2":
            if tomodel:
                x_out[:]=1/(x_in[:]*x_in[:])
            else:
                x_out[:]=np.sqrt(1/x_in[:])
        else:
            raise argparse.ArgumentTypeError("String '%s' does not match 'c', '1/c', '1/c2' or 'multi:*' required format"%(model))
    else:
        #multivars only available model is c,rho
        x_out[:]=x_in[:]

    if scalar_input:
        return x_out[0]
        #return np.squeeze(x_out)
    return x_out
    
def geos_params_to_model(x_in,model):
    return geos_params_model_convert(x_in,model,True)
            
def geos_params_from_model(x_in,model):
    return geos_params_model_convert(x_in,model,False)
            
def grad_model_convert(grad_in,geos_params,model,tomodel):
    if not len(grad_in)==len(geos_params):
        raise argparse.ArgumentTypeError("grad and geos_params must have the same size: len(grad)=%d, len(geos_params)=%d"%(len(grad_in),len(geos_params)))
    grad_out=np.zeros(len(grad_in))
    if not "multi:" in model:
        #initially gradient was with respect to c
        if model=="c":
            grad_out[:]=grad_in[:]
        elif model=="1/c":
            if tomodel:
                grad_out[:]= - geos_params[:] * geos_params[:] * grad_in[:]
            else:
                grad_out[:]= - grad_in[:] / (geos_params[:] * geos_params[:])
        elif model=="1/c2":
            if tomodel:
                grad_out[:]= - (geos_params[:] * geos_params[:] * geos_params[:] / 2) * grad_in[:]
            else:
                grad_out[:]= - (2 / (geos_params[:] * geos_params[:] * geos_params[:])) * grad_in[:]
        else:
            raise argparse.ArgumentTypeError("String '%s' does not match 'c', '1/c', '1/c2' or 'multi:*' required format"%(model))
    else:
        if tomodel:
            #multivars only available model is c,rho
            #while grad1 from geos with respect to 1/(rho*c**2)
            #and grad2 with respect to 1/rho
            #
            #grad_rho=-1/(rho^2*c^2)grad1-1/rho^2 grad2
            #grad_c=-2/(rho*c^3)grad1
            n1=len(grad_in)//2
            x1=geos_params[:n1]
            x2=geos_params[n1:]
            grad_out[n1:] = -1/(x2[:]*x2[:]*x1[:]*x1[:]) * grad_in[:n1] - 1/(x2[:]*x2[:]) * grad_in[n1:]
            grad_out[:n1] = -2/(x2[:]*x1[:]*x1[:]*x1[:]) * grad_in[:n1]
            #nullify grad relative to density to see if FWI is working
            #grad_out[n1:] = 0.0
        else:
            raise argparse.ArgumentTypeError("grad from model to Kappa/b not yet implemented")
            #grad_out[:]=grad_in[:]

    return grad_out

def grad_to_model(grad_in,x,model):
    return grad_model_convert(grad_in,x,model,True)

def grad_from_model(grad_in,x,model):
    return grad_model_convert(grad_in,x,model,False)

def model_min_max(min_in,max_in,model,tomodel):
    scalar_input=np.isscalar(min_in)
    min_in=np.atleast_1d(min_in)
    max_in=np.atleast_1d(max_in)
    if not len(min_in)==len(max_in):
        raise argparse.ArgumentTypeError("min and max are not the same size len(min)=%d, len(max)=%d"%(len(min_in),len(max_in)))
        
    if tomodel:
        min_out=geos_params_to_model(min_in,model)
        max_out=geos_params_to_model(max_in,model)
    else:
        min_out=geos_params_from_model(min_in,model)
        max_out=geos_params_from_model(max_in,model)

    for i in range(len(min_out)):
        if min_out[i] > max_out[i]:
            min_out[i]=min_out[i]+max_out[i]
            max_out[i]=min_out[i]-max_out[i]
            min_out[i]=min_out[i]-max_out[i]

    if scalar_input:
        return min_out[0],max_out[0]
        #return np.squeeze(min_out),np.squeeze(max_out)

    return min_out,max_out

def min_max_to_model(min_in,max_in,model):
    return model_min_max(min_in,max_in,model,True)

def min_max_from_model(min_in,max_in,model):
    return model_min_max(min_in,max_in,model,False)
