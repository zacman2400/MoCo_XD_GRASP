from torch.utils import checkpoint
import torch
import numpy as np
import random
import torch_optimizer as optim
import cupy
import sigpy
import torch
import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
from interpol import grid_pull
import random
import os
#from multi_scale_low_rank_image import MultiScaleLowRankImage
print('a')
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

import argparse
import numpy as np
import sigpy as sp
import logging
import h5py
import torch
import cupy as xp


#deforms complex data (splits into two channel)
def warp1(img,flow,mps,complex=True):
    img=img.cuda()
    #img=torch.reshape(img,[1,1,304, 176, 368])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])

    spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
    size=shape
    vectors=[]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors)
    grid  = torch.stack(grids) # y, x, z
    grid  = torch.unsqueeze(grid, 0)  #add batch
    grid = grid.type(torch.FloatTensor)
    new_locs=grid.cuda()+flow
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])

    new_locs = new_locs.permute(0, 2, 3, 4, 1) 
   # new_locs = new_locs[..., [2,1,0]]
    new_locsa = new_locs[..., [0,1,2]]
    if complex==True:
        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        im_out=torch.complex(ima_real,ima_imag)
    else:
        im_out=grid_pull(torch.squeeze((img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
  
    return im_out

#solves for template-image to a given respiratory dynamic
def for_field_solver(deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,im_template,ksp,coord,dcf,mps,iter_adj,RO,block_torch,ishape,T1,T2,interp,res,spokes_per_bin,weight_dc,weight_smoother):
  
    #readout images and deformation fields during training
    deform_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_still=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    im_tester=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
   
    mps=torch.from_numpy(mps).cpu()
    deform=[deformL_param_adj[0],deformR_param_adj[0],deformL_param_adj[1],deformR_param_adj[1],deformL_param_adj[2],deformR_param_adj[2]] 
  
    optimizer0=torch.optim.Adam([deform[i] for i in range(6)],.001)

    
    for io in range(iter_adj):
          
            flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            count=0
            lossz=np.zeros([20])

         
            K=random.sample(range(T1,T2), T2-T1)
            print(K)
            for j in K:
                jo=j
                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
                flowa=deforma
                flowa=torch.nn.functional.interpolate(flowa, size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
                im_test=torch.from_numpy(im_template).cuda().unsqueeze(0)
                scale0=torch.abs(im_test).max()
                con0=im_test #con
                im_out=warp1(con0/scale0,flowa,mps,complex=True)
                
                #select phase
                ksp_ta=torch.from_numpy(ksp[:,j,:RO]).cuda()/all0
                coord_t=torch.from_numpy(coord[j,:RO]).cuda()
                dcf_t=torch.from_numpy(dcf[j,:RO]).cuda()
                deform_look[j-T1]=(flowa[:,0,:,30,:].detach().cpu().numpy())
                image_look[j-T1]=np.abs(im_out[:,30,:].detach().cpu().numpy())
                loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa)
                
                loss_for=torch.utils.checkpoint.checkpoint(_updateb,im_out.unsqueeze(0),ksp_ta,dcf_t,coord_t,mps) 
                loss_L=0
                loss_R=0
                loss_L0=0
                loss_R0=0
                for i in range(3):
                  
                   
                    loss_L0=loss_L0+torch.norm(deformL_param_adj[i],'fro')**2
                    loss_R0=loss_R0+torch.norm(deformR_param_adj[i][:,:,:,1:]-deformR_param_adj[i][:,:,:,:-1],'fro')**2
                loss=loss_for*weight_dc/1+loss_grad0*weight_smoother+loss_L0*1e-7+loss_R0*1e-7 
                (loss).backward()
              
                (optimizer0).step()
             
                optimizer0.zero_grad()
            
               
            
            import imageio
            imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=5)
            imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=5)
            imageio.mimsave('image_tester.gif', [np.abs(im_tester[i,:,:])*1e15 for i in range(50)], fps=5)
           
    return deformL_param_adj,deformR_param_adj,image_look



#initializes motion fields
def gen_MSLR(T,rank,block_size_adj,block_size_for,scale,mps):

    import cupy
    import numpy as np
    
    import math
    import torch
    from math import ceil

    block_torch0=[]
    block_torch1=[]
    blockH_torch=[]
    #deformL_adj=[]
    #deformR_adj=[]
    deformL_for=[]
    deformR_for=[]
    ishape0a=[]
    ishape1a=[]
    j=0
    deformL_param_adj=[]
    deformR_param_adj=[]
    deformL_param_for=[]
    deformR_param_for=[]
    #gen


    block_size0=block_size_adj
    block_size1=block_size_for

    #Ltorch=[]
    #Rtorch=[]
    import torch_optimizer as optim


    for jo in block_size0:
        print(jo)

        b_j = [min(i, jo) for i in [mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale]]
        print(b_j)
        s_j = [(b+1)//2  for b in b_j]
        i_j = [ceil((i - b + s) / s) * s + b - s 
        for i, b, s in zip([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], b_j, s_j)]
        import sigpy as sp
        block=sp.linop.BlocksToArray(i_j, b_j, s_j)
       # print(block.shape)
        C_j = sp.linop.Resize([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], i_j,
                                      ishift=[0] * 3, oshift=[0] * 3)
       # b_j = [min(i, j) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
        w_j = sp.hanning(b_j, dtype=cupy.float32, device=0)**0.5
        W_j = sp.linop.Multiply(block.ishape, w_j)
        block_final=C_j*block*W_j
        ishape1a.append(block_final.ishape)
        block_torch1.append(sp.to_pytorch_function(block_final,input_iscomplex=False,output_iscomplex=False))
       
        temp0=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*1
  
        temp0=1e3*temp0/torch.sum(torch.square(torch.abs(temp0)))**0.5
        print(temp0.max())
        deformL_param_adj.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
       # tempa=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],dtype=torch.float16,device='cuda')
        deformR_param_adj.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
        deformL_param_for.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
        deformR_param_for.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
    return deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,block_torch1,ishape1a

def gen(block_torcha,deformL_param,deformR_param,ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo):
    jb=int(jo[0])
   # print(jb)
    deform_patch_adj=torch.matmul(deformL_param,deformR_param[:,:,:,jb:jb+1])
    deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(ishape0[0]),int(ishape1[0]),int(ishape2[0]),int(ishape3[0]),int(ishape4[0]),int(ishape5[0])])
    deformx_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[0])).unsqueeze(0)
    deformy_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[1])).unsqueeze(0)
    deformz_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[2])).unsqueeze(0)
   # deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
    return deformx_adj,deformy_adj,deformz_adj

def flows(deformL_param_adj,deformR_param_adj,j,block_torch1,ishape1a):
        jo=torch.ones([1])*j
        deform_adj=[]
        deform_for=[]
        #count=int(counta[0])
        for count in range(3):
           # print(count)
            ishape0=ishape1a[count][0]*torch.ones([1])
            ishape1=ishape1a[count][1]*torch.ones([1])
            ishape2=ishape1a[count][2]*torch.ones([1])
            ishape3=ishape1a[count][3]*torch.ones([1])
            ishape4=ishape1a[count][4]*torch.ones([1])
            ishape5=ishape1a[count][5]*torch.ones([1])
       # deformx0,deformy1,deformz0=torch.utils.checkpoint.checkpoint(gen,block_torch0,deformL_param_adj0,deformR_param_adj0,ishape00,ishape10,ishape20,ishape30,ishape40,ishape50,jo)
            deformx,deformy,deformz=gen(block_torch1[count],deformL_param_adj[count],deformR_param_adj[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo)
           # deform_for.append(torch.cat([deformx,deformy,deformz],axis=0))
           # deformx,deformy,deformz=torch.utils.checkpoint.checkpoint(gen,block_torch[count],deformL_param_for[count],deformR_param_for[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo,preserve_rng_state=False)
            deform_adj.append(torch.cat([deformx,deformy,deformz],axis=0))
        flow=deform_adj[0]+deform_adj[1]+deform_adj[2] #+deform_adj[2] #+deform_adj[3] #+deform_adj[4] #+deform_adj[3]+deform_adj[4]+deform_adj[5] #+deform_adj[6]+deform_adj[7]
        flow=flow.unsqueeze(0)
        return flow
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) #*w0
       
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) #*w1
      
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) #*w2
       # dt = torch.abs(y_pred[1:, :, :, :, :] - y_pred[:-1, :, :, :, :])

      
        dy = dy
        dx = dx
        dz = dz
            #dt=dt*dt

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

     
        return grad
    
f=Grad()

def calculate_sense0(img_t,ksp,mps_c,coord_t,dcf):
        torch.cuda.empty_cache()
       # r = torch.cuda.memory_reserved(0) /1e9
        ksp=torch.reshape(ksp,[-1])
        coord_t=torch.reshape(coord_t,[-1,3])
        dcf=torch.reshape(dcf,[-1])
        mps_c=mps_c.cuda()
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], torch.reshape(coord_t,[-1,3]), oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        M_t=torch.cat([torch.reshape(torch.real(img_t*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),torch.reshape(torch.imag(img_t*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])],axis=3)
       
        
       
        torch.cuda.empty_cache()
       
        torch.cuda.empty_cache()
        e_tc=F_torch.apply(M_t)
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
       # e_tca=(e_tca/torch.abs(e_tca).max())*torch.abs(ksp).max()
       # e_tc_update=e_tca
      #  print(ksp.shape)
      #  print(e_tca.shape)
       # print(torch.abs(e_tca).max())
       # print(torch.abs(ksp).max())
        #loss_self1=torch.nn.MSELoss()
        torch.cuda.empty_cache()
        resk=((ksp-e_tca)*(dcf)**0.5)
        torch.cuda.empty_cache()
        #loss=(torch.norm(resk)) #/torch.norm(ksp,2)+torch.norm(resk,1)/torch.norm(ksp,1) #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
       
        loss=torch.norm(resk,2)**2 #*index_all/ksp.shape[0]
       # r = torch.cuda.memory_reserved(0) /1e9
       # print(r)
      #  print(r)
        ksp=ksp.detach().cpu().numpy()
        coord_t=coord_t.detach().cpu().numpy()
        dcf=dcf.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return (loss)

def _updateb(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0
  #print(mpsa.shape)
  #print(img_t.shape)
  
  for c in range(mpsa.shape[0]):
   # torch.cuda.empty_cache()
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,img_t,ksp_t[c],mpsa[c],coord_t,dcf_t)
  
  print(loss_t)
  return loss_t



class MotionResolvedRecon(object):
    def __init__(self, deform_for,deform_rev,kspg, coordg, dcfg,index_all, mps,  B,
                 lamda=1e-6, alpha=1, beta=0.5,
                 max_power_iter=10, max_iter=120,
                 device=0, margin=10,
                 coil_batch_size=None, comm=None, show_pbar=True, **kwargs):
        self.B = B
        self.index_all=index_all
        self.C = mps.shape[0]
        self.mps = mps
        #print(self.mps.shape)
        self.device = sp.Device(device)
        self.xp = xp
        self.deform_for=deform_for #from template image to respiratory phase dynamic
        self.deform_adj=deform_adj #from respiratory phase dynamic to template image
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.max_iter = max_iter
        self.max_power_iter = max_power_iter
        self.comm = None
        self.show_pbar=True
       # if comm is not None:
        #    self.show_pbar = show_pbar and comm.rank == 0

        self.img_shape = mps.shape[1:]
       # print(self.img_shape)

       # bins = np.percentile(resp, np.linspace(0 + margin, 100 - margin, B + 1))
        self.bksp=[]
        self.bcoord=[]
        self.bdcf=[]
        print('bin')
        for i in range(self.B):
            self.bksp.append(cupy.array(kspg[i]))
            self.bcoord.append(cupy.array(coordg[i]))
            self.bdcf.append(cupy.array(dcfg[i]))
       
      #  self._normalize()
    def get_adj_field(self,t):
         flow=0
         mps=self.mps
         deform_adj=[]
         for count in range(3):
            #self.scale=3
            deform_patch_adj=torch.matmul(self.deformL_param_adj[count].cuda(),self.deformR_param_adj[count][:,:,:,t:t+1].cuda())
            deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[0])).unsqueeze(0)
            deformy_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[1])).unsqueeze(0)
            deformz_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[2])).unsqueeze(0)
            deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
            flow=flow+deform_adj[count] 
         #print(flow.shape)
         #flow=flow.unsqueeze(0)
         #flow=Bspline.compute_flow(flow)
         self.scale=3
         flow=torch.reshape(flow,[1,3,self.mps.shape[1]//self.scale,self.mps.shape[2]//self.scale,self.mps.shape[3]//self.scale])
        
         flow=torch.nn.functional.interpolate(flow, size=[self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*3
         flow_adj=torch.reshape(flow,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
         
          
         return flow_adj
    def get_for_field(self,t):
        flow=0
        mps=self.mps
        deform_for=[]
        for count in range(3):
            deform_patch_for=torch.matmul(self.deformL_param_for[count].cuda(),self.deformR_param_for[count][:,:,:,t:t+1].cuda())
            deform_patch_for=torch.reshape(deform_patch_for,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[0])).unsqueeze(0)
            deformy_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[1])).unsqueeze(0)
            deformz_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[2])).unsqueeze(0)
            deform_for.append(torch.cat([deformx_for,deformy_for,deformz_for],axis=0))
            flow=flow+deform_for[count] 
         #flow=flow.unsqueeze(0)
         #flow=Bspline.compute_flow(flow)
        self.scale=3
        flow=torch.reshape(flow,[1,3,self.mps.shape[1]//self.scale,self.mps.shape[2]//self.scale,self.mps.shape[3]//self.scale])
        
        #flow=torch.reshape(flow,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        #scale=1
        flow_for=torch.nn.functional.interpolate(flow, size=[self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*3
        flow_for=torch.reshape(flow_for,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        
        return flow_for
        
      def warp(self,flow,img):
        img=img.cuda()
        mps=self.mps
        img=torch.reshape(img,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
         
        spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
        shape=(mps.shape[1],mps.shape[2],mps.shape[3])
        size=shape
        vectors=[]
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        new_locs=grid.cuda()+flow
        shape=(mps.shape[1],mps.shape[2],mps.shape[3])
      #  for i in range(len(shape)):
      #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1) 
       # new_locs = new_locs[..., [2,1,0]]
        new_locsa = new_locs[..., [0,1,2]]

        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
         #ima_real=torch.nn.functional.grid_sample(torch.real(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
         #ima_imag=torch.nn.functional.grid_sample(torch.imag(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
        im_out=torch.complex(ima_real,ima_imag)
        im_out=torch.squeeze(im_out)
    


        
        return im_out
        

    def _normalize(self):
        # Normalize using first phase.
        #with device:
            mrimg_adj = 0
            for c in range(self.C):
                device=0
                mrimg_c = sp.nufft_adjoint(
                    self.bksp[0][c] * self.bdcf[0], self.bcoord[0],
                    self.img_shape)
                #print(mrimg_c.shape)
               # T=self.xp.conj(sp.to_device(self.mps[c], device))
               # print(T.shape)
                mrimg_c *= self.xp.conj(sp.to_device(self.mps[c], device))
                mrimg_adj += mrimg_c

           # if comm is not None:
           #     comm.allreduce(mrimg_adj)

            # Get maximum eigenvalue.
            F = sp.linop.NUFFT(self.img_shape, self.bcoord[0])
            W = sp.linop.Multiply(F.oshape, self.bdcf[0])
            max_eig = sp.app.MaxEig(F.H * W * F,
                                    max_iter=self.max_power_iter,
                                    dtype=cupy.complex64, device=device,
                                    show_pbar=self.show_pbar).run()

            # Normalize
            self._normalize
            self.alpha /= max_eig
            self.lamda *= max_eig * self.xp.abs(mrimg_adj).max().item()
            
            
            
        

    def gradf(self, mrimg):
        out = self.xp.zeros_like(mrimg)
        for b in range(self.B):
           # print(b)
            for c in range(self.mps.shape[0]):
              
                mps_c = sp.to_device(self.mps[c], self.device)
                out[b] += sp.nufft_adjoint(
                    self.bdcf[b] * (sp.nufft(mrimg[b] * mps_c, self.bcoord[b])
                                    - self.bksp[b][c]),
                    self.bcoord[b],
                    oshape=mrimg.shape[1:]) * self.xp.conj(mps_c)

        if self.comm is not None:
            self.comm.allreduce(out)

        eps = 1e-31
        
        #align to template space
        aligned=torch.zeros_like(mrimg)
        for b in range(self.B):
            interm=torch.as_tensor(mrimg[b],device='cuda') 
            flow_adj=self.get_adj_field(b)
            aligned[b]=self.warp(flow_adj,interm)
        for b in range(self.B):
            if b > 0:
                #take gradient steps in respiratory phase space
                diff = aligned[b] - aligned[b - 1]
                flow_for=self.get_for_field(b)
                diff_for=self.warp(flow_for,diff)
                sp.axpy(out[b], self.lamda, diff_for / (self.xp.abs(diff_for) + eps))

            if b < self.B - 1:
                diff = aligned[b] - aligned[b + 1]
                flow_for=self.get_for_field(b)
                diff_for=self.warp(flow_for,diff)
                sp.axpy(out[b], self.lamda, diff_for / (self.xp.abs(diff_for) + eps))
        import imageio
       
        return out

    def run(self):
        done = False
        while not done:
            try:
                with tqdm(total=self.max_iter, desc='MotionResolvedRecon',
                          disable=not self.show_pbar) as pbar:
                    with self.device:
                        mrimg = self.xp.zeros([self.B,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]],dtype=self.mps.dtype)
                      
                        for it in range(self.max_iter):
                            g = self.gradf(mrimg)
                            sp.axpy(mrimg, -self.alpha, g)
                            import imageio
                            imageio.mimsave('./rec_image.gif', [np.abs(np.array(mrimg[i][:,35,:].get()))*1e15 for i in range(6)], fps=4)

                            gnorm = self.xp.linalg.norm(g.ravel()).item()
                            print(gnorm)
                            if np.isnan(gnorm) or np.isinf(gnorm):
                                raise OverflowError('LowRankRecon diverges.')

                            pbar.set_postfix(gnorm=gnorm)
                            pbar.update()

                        done = True
            except OverflowError:
                self.alpha *= self.beta

        return mrimg,self.bksp,self.bcoord,self.bdcf
