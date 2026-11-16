"""Conservative FV and reusable local responses on the archived polygon mesh."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .nofvm_2d import graph_matrix
from .nofvm_polygon_mesh import PolygonMesh


def outer_faces(mesh):
    ids=[]
    for side in range(1,5):
        selected=np.flatnonzero(mesh.boundary_tag==side)
        mid=mesh.faces[selected].mean(axis=1)
        ids.extend(selected[np.argsort(mid[:,1 if side<3 else 0])])
    ids=np.asarray(ids)
    xy=mesh.faces[ids].mean(axis=1)
    area=mesh.length[ids]
    owner=mesh.owner[ids]
    distance=np.linalg.norm(mesh.xy[owner]-xy,axis=1)
    return dict(ids=ids,xy=xy,area=area,owner=owner,distance=distance,
                tau=area/distance,n=len(ids)//4)


def interface_faces(mesh):
    internal=np.flatnonzero(mesh.neighbour>=0)
    own,nei=mesh.owner[internal],mesh.neighbour[internal]
    ids=internal[mesh.inside[own]!=mesh.inside[nei]]
    mid=mesh.faces[ids].mean(axis=1)
    order=[]
    for dim,value,coord in ((0,.4,1),(0,1.2,1),(1,.4,0),(1,1.2,0)):
        selected=np.flatnonzero(np.isclose(mid[:,dim],value,atol=1e-10))
        order.extend(selected[np.argsort(mid[selected,coord])])
    ids=ids[np.asarray(order)]
    own,nei=mesh.owner[ids],mesh.neighbour[ids]
    inward=np.where(mesh.inside[own],own,nei)
    outward=np.where(mesh.inside[own],nei,own)
    mid=mesh.faces[ids].mean(axis=1)
    area=mesh.length[ids]
    di=np.linalg.norm(mesh.xy[inward]-mid,axis=1)
    do=np.linalg.norm(mesh.xy[outward]-mid,axis=1)
    return dict(ids=ids,xy=mid,inside=inward,outside=outward,area=area,di=di,do=do,n=len(ids)//4)


def local_response(mesh,ports=32):
    """Exact basis labels; these are never installed as neural model buffers."""
    from .nofvm_hole import boundary_prolongation
    ids=np.flatnonzero(mesh.inside)
    mapping=np.full(len(mesh.xy),-1,dtype=int);mapping[ids]=np.arange(len(ids))
    valid=mesh.neighbour>=0
    a,b=mesh.owner[valid],mesh.neighbour[valid]
    length=mesh.length[valid]
    keep=mesh.inside[a]&mesh.inside[b]
    tau=length[keep]/np.linalg.norm(mesh.xy[a[keep]]-mesh.xy[b[keep]],axis=1)
    face=interface_faces(mesh)
    t=face['area']/face['di']
    owner=mapping[face['inside']]
    P=boundary_prolongation(face['n'],ports)
    lift=sparse.csc_matrix((t,(owner,np.arange(len(t)))),shape=(len(ids),len(t)))
    B=(lift@P).tocsc()
    T=(P.T@sparse.diags(t)@P).toarray()
    A=graph_matrix(len(ids),[(mapping[a[keep]],mapping[b[keep]],tau)],
                   np.bincount(owner,weights=t,minlength=len(ids)))
    H=splu(A,permc_spec='COLAMD').solve(B.toarray())
    S=T-B.T@H
    return dict(A=A,B=B,T=T,H=H,S=S,xy=mesh.xy[ids]-.4,volume=mesh.volume[ids],ids=ids,ports=ports)


class TiledFV:
    """Repeated real polygon tiles with shared seam fluxes.

    In hybrid mode each central region is replaced by the supplied response;
    all exterior cells and the tile-to-tile fluxes remain classical FV.
    """
    def __init__(self,mesh,p=1,response=None):
        self.mesh,self.p,self.response=mesh,p,response
        self.offsets=np.array([[1.6*x,1.6*y] for y in range(p) for x in range(p)])
        keep=~mesh.inside if response is not None else np.ones(len(mesh.xy),dtype=bool)
        self.ids=np.flatnonzero(keep);self.nordinary=len(self.ids)
        mapping=np.full(len(mesh.xy),-1,dtype=int);mapping[self.ids]=np.arange(len(self.ids))
        self.m=response['S'].shape[0] if response is not None else 0
        self.n=self.nordinary+self.m
        valid=mesh.neighbour>=0
        a,b=mesh.owner[valid],mesh.neighbour[valid]
        k=keep[a]&keep[b]
        t=mesh.length[valid][k]/np.linalg.norm(mesh.xy[a[k]]-mesh.xy[b[k]],axis=1)
        edges=[(mapping[a[k]],mapping[b[k]],t)]
        if response is not None:
            face=interface_faces(mesh)
            if len(face['ids'])!=self.m:
                raise ValueError('Coarse exterior faces must match the retained trace ports')
            self.interface=face
            self.interface_owner=mapping[face['outside']]
            self.interface_tau=face['area']/face['do']
            edges.append((self.interface_owner,self.nordinary+np.arange(self.m),self.interface_tau))
            self.tile_xy=np.concatenate((mesh.xy[self.ids],face['xy']))
        else:
            self.tile_xy=mesh.xy
        tile_matrix=graph_matrix(self.n,edges,np.zeros(self.n))
        if response is not None:
            tile_matrix+=sparse.block_diag((sparse.csc_matrix((self.nordinary,self.nordinary)),
                                           sparse.csc_matrix(response['S'])),format='csc')
        self.matrix=sparse.kron(sparse.eye(p*p,format='csc'),tile_matrix,format='csc')
        outer=outer_faces(mesh);self.outer=outer
        own=mapping[outer['owner']];nside=outer['n']
        owners=own.reshape(4,nside)
        lengths=outer['area'].reshape(4,nside)
        dist=outer['distance'].reshape(4,nside)
        seam_a=[];seam_b=[];seam_tau=[]
        outer_owner=[];outer_xy=[];outer_tau=[];outer_tag=[]
        for bid,offset in enumerate(self.offsets):
            iy,ix=divmod(bid,p)
            for other,sa,sb in ((bid+1,1,0),(bid+p,3,2)):
                if (sa==1 and ix+1<p) or (sa==3 and iy+1<p):
                    seam_a.extend(bid*self.n+owners[sa]);seam_b.extend(other*self.n+owners[sb])
                    seam_tau.extend(lengths[sa]/(dist[sa]+dist[sb]))
            for side,visible in enumerate((ix==0,ix==p-1,iy==0,iy==p-1)):
                if visible:
                    sl=slice(side*nside,(side+1)*nside)
                    outer_owner.extend(bid*self.n+own[sl])
                    outer_xy.extend(outer['xy'][sl]+offset)
                    outer_tau.extend(outer['tau'][sl]);outer_tag.extend([side]*nside)
        seam_a,seam_b,seam_tau=map(np.asarray,(seam_a,seam_b,seam_tau))
        seam_a,seam_b=seam_a.astype(int),seam_b.astype(int)
        self.boundary_owner=np.asarray(outer_owner)
        self.boundary_xy=np.asarray(outer_xy);self.boundary_tau=np.asarray(outer_tau)
        self.boundary_tag=np.asarray(outer_tag);self.dirichlet=self.boundary_tag<2
        diagonal=np.bincount(self.boundary_owner,weights=self.boundary_tau*self.dirichlet,
                             minlength=self.matrix.shape[0])
        self.matrix+=graph_matrix(self.matrix.shape[0],[(seam_a,seam_b,seam_tau)],diagonal)
        # Geometry-only least-squares reconstruction, prepared with assembly.
        ga=[];gb=[]
        for bid in range(p*p):
            for a,b,_ in edges:
                ga.extend(bid*self.n+a);gb.extend(bid*self.n+b)
        ga.extend(seam_a);gb.extend(seam_b)
        self.reconstruction_edges=(np.asarray(ga,dtype=int),np.asarray(gb,dtype=int))
        self.global_xy=(self.offsets[:,None,:]+self.tile_xy[None,:,:]).reshape(-1,2)
        self.factor=None

    def factorize(self):
        self.factor=splu(self.matrix,permc_spec='COLAMD')
        return self.factor

    def solve(self,boundary):
        prescribed=boundary(self.boundary_xy)
        rhs=np.zeros((self.matrix.shape[0],prescribed.shape[1]))
        np.add.at(rhs,self.boundary_owner,self.boundary_tau[:,None]*self.dirichlet[:,None]*prescribed)
        if self.factor is None:self.factorize()
        u=self.factor.solve(rhs)
        physical=u.reshape(self.p*self.p,self.n,-1)
        flux_outer=self.boundary_tau[:,None]*self.dirichlet[:,None]*(u[self.boundary_owner]-prescribed)
        result=dict(solution=u,ordinary=physical[:,:self.nordinary],
                    linear_residual=float(abs(self.matrix@u-rhs).max()),
                    global_balance=float(abs(flux_outer.sum(axis=0)).max()))
        if self.response is not None:
            trace=physical[:,self.nordinary:]
            flux=-np.einsum('ij,bjk->bik',self.response['S'],trace,optimize=True)
            exterior_flux=self.interface_tau[None,:,None]*(physical[:,self.interface_owner]-trace)
            result.update(trace=trace,flux=flux,
                          local_balance=float(abs(flux.sum(axis=1)).max()),
                          interface_balance=float(abs(flux+exterior_flux).max()))
        else:
            face=interface_faces(self.mesh)
            q=face['area']/(face['di']+face['do'])
            result['flux']=q[None,:,None]*(physical[:,face['inside']]-physical[:,face['outside']])
            result['local_balance']=float(abs(result['flux'].sum(axis=1)).max())
        return result

    def gradients(self,result,boundary):
        u=result['solution'];xy=self.global_xy
        a,b=self.reconstruction_edges
        d=xy[b]-xy[a];w=1/np.sum(d*d,axis=1)
        terms=d[:,:,None]*d[:,None,:]*w[:,None,None]
        rhs=d[:,:,None]*(u[b]-u[a])[:,None,:]*w[:,None,None]
        lhs=np.zeros((len(u),2,2));numer=np.zeros((len(u),2,u.shape[1]))
        for index in (a,b):np.add.at(lhs,index,terms);np.add.at(numer,index,rhs)
        own=self.boundary_owner
        d=self.boundary_xy-xy[own];w=1/np.sum(d*d,axis=1)
        np.add.at(lhs,own,d[:,:,None]*d[:,None,:]*w[:,None,None])
        delta=self.dirichlet[:,None]*(boundary(self.boundary_xy)-u[own])
        np.add.at(numer,own,d[:,:,None]*delta[:,None,:]*w[:,None,None])
        # Trace rows need no reconstruction and can be rank-deficient.
        ordinary=(np.arange(len(u))%self.n)<self.nordinary
        gradient=np.zeros_like(numer)
        gradient[ordinary]=np.linalg.solve(lhs[ordinary],numer[ordinary])
        return gradient.reshape(self.p*self.p,self.n,2,-1)

    def reconstruct(self,result,target,boundary,local_xy=None):
        nload=result['solution'].shape[1]
        out=np.empty((self.p*self.p,len(target.xy),nload))
        gradient=self.gradients(result,boundary)
        if self.response is None:
            tree=cKDTree(self.mesh.xy)
            _,parents=tree.query(target.xy)
            delta=target.xy-self.mesh.xy[parents]
            out[:]=result['ordinary'][:,parents]+np.einsum('qi,bqik->bqk',delta,gradient[:,parents],optimize=True)
        else:
            exterior=~target.inside
            _,parents=cKDTree(self.mesh.xy[self.ids]).query(target.xy[exterior])
            delta=target.xy[exterior]-self.mesh.xy[self.ids][parents]
            out[:,exterior]=result['ordinary'][:,parents]+np.einsum('qi,bqik->bqk',delta,gradient[:,parents],optimize=True)
            if local_xy is None or not np.allclose(target.xy[target.inside]-.4,local_xy,atol=1e-12):
                raise ValueError('Complete output must match the decoder coordinates')
            # Batch all tiles/loads into one GEMM, preserving tile-major order.
            traces=result['trace'].transpose(1,0,2).reshape(self.m,-1)
            local=self.response['H']@traces
            out[:,target.inside]=local.reshape(-1,self.p*self.p,nload).transpose(1,0,2)
        return out


class ExactTileReference:
    """Algebraically exact full-FV reference using complete-tile condensation.

    This is an accuracy reference, not the direct-FV timing baseline. It removes
    no approximation and is checked against direct full FV for small layouts.
    """
    def __init__(self,mesh):
        from .nofvm_polygon_mesh import fv_system
        self.mesh=mesh;self.outer=outer_faces(mesh)
        A,_,_,_,_=fv_system(mesh,all_dirichlet=True)
        outer=self.outer
        self.B=sparse.csc_matrix((outer['tau'],(outer['owner'],np.arange(len(outer['owner'])))),
                                shape=(len(mesh.xy),len(outer['owner'])))
        self.H=splu(A,permc_spec='COLAMD').solve(self.B.toarray())
        self.S=np.diag(outer['tau'])-self.B.T@self.H

    def solve(self,p,boundary):
        offsets=np.array([[1.6*x,1.6*y] for y in range(p) for x in range(p)])
        coordinates=offsets[:,None,:]+self.outer['xy'][None,:,:]
        xy,inverse=np.unique(np.round(coordinates.reshape(-1,2),11),axis=0,return_inverse=True)
        ports=inverse.reshape(p*p,-1)
        k=self.S.shape[0]
        a=np.repeat(ports,k,axis=1).ravel();b=np.tile(ports,(1,k)).ravel()
        values=np.tile(self.S.ravel(),p*p)
        A=sparse.coo_matrix((values,(a,b)),shape=(len(xy),len(xy))).tocsc()
        fixed=np.isclose(xy[:,0],0)|np.isclose(xy[:,0],p*1.6)
        free=~fixed
        trace=np.zeros((len(xy),boundary(xy[:1]).shape[1]))
        trace[fixed]=boundary(xy[fixed])
        trace[free]=splu(A[free][:,free],permc_spec='COLAMD').solve(-A[free][:,fixed]@trace[fixed])
        tile_trace=trace[ports]
        u=self.H@tile_trace.transpose(1,0,2).reshape(k,-1)
        u=u.reshape(len(self.mesh.xy),p*p,-1).transpose(1,0,2)
        face=interface_faces(self.mesh)
        flux=(face['area']/(face['di']+face['do']))[None,:,None]*(u[:,face['inside']]-u[:,face['outside']])
        return dict(field=u,flux=flux,trace=tile_trace,
                    free_trace_residual=float(abs((A@trace)[free]).max()),
                    local_balance=float(abs(flux.sum(axis=1)).max()))
