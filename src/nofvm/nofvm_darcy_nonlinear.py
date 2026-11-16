"""D4: Kirchhoff-consistent FV and nonlinear conservative trace coupling.

All face fluxes are integrated. The full-grid reference and condensed solver
use independent assembly paths. The known transform is specific to K(u).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .nofvm_2d import boundary_geometry
from .nofvm_darcy import DarcyHybrid


def phi(u):
    u = np.asarray(u)
    return np.sign(u)*np.log1p(np.abs(u)/.1)


def inverse_phi(w):
    w = np.asarray(w)
    return np.sign(w)*.1*np.expm1(np.abs(w))


def conductivity(u):
    return 1/(np.abs(u)+.1)


def mixed_face(inner, outer):
    """Solve lambda+Phi(lambda)=u_outer+Phi(u_inner) monotonically.

    Bracketed Newton handles both positive and sign-changing data. Derivatives
    returned are of the inner-outward flux wrt inner and outer pressures.
    """
    target = outer+phi(inner)
    lower, upper = np.minimum(inner, outer).copy(), np.maximum(inner, outer).copy()
    value = .5*(lower+upper)
    for _ in range(60):
        residual = value+phi(value)-target
        if np.max(abs(residual), initial=0.) < 2e-14:
            break
        upper = np.where(residual > 0, value, upper)
        lower = np.where(residual < 0, value, lower)
        candidate = value-residual/(1+conductivity(value))
        value = np.where((candidate >= lower) & (candidate <= upper), candidate, .5*(lower+upper))
    if np.max(abs(value+phi(value)-target), initial=0.) > 1e-11:
        raise RuntimeError('Mixed-material face pressure did not converge')
    denominator = 1+conductivity(value)
    return (2*(phi(inner)-phi(value)), 2*conductivity(inner)/denominator,
            -2*conductivity(value)/denominator, value)


def newton(equations, initial, maxiter=60, tolerance=2e-11):
    value = np.array(initial, dtype=float, copy=True)
    history = []
    for iteration in range(maxiter+1):
        residual, jacobian = equations(value, True)
        norm = float(np.max(abs(residual), initial=0.))
        history.append(dict(iteration=iteration, residual_inf=norm))
        if norm <= tolerance:
            return value, history
        if iteration == maxiter:
            break
        update = (splu(jacobian, permc_spec='MMD_AT_PLUS_A').solve(-residual)
                  if sparse.issparse(jacobian) else np.linalg.solve(jacobian, -residual))
        merit = np.linalg.norm(residual)
        for trial in range(30):
            step = 2.**(-trial)
            candidate = value+step*update
            next_residual, _ = equations(candidate, False)
            if np.all(np.isfinite(next_residual)) and np.linalg.norm(next_residual) <= (1-1e-4*step)*merit:
                value = candidate
                history[-1]['accepted_step'] = step
                break
        else:
            raise RuntimeError(f'Newton line search failed at residual {norm}')
    raise RuntimeError(f'Newton failed: {history[-1]}')


class FullNonlinearFV:
    def __init__(self, n=64, m=32, all_nonlinear=False):
        if n%2 or n%m:
            raise ValueError('Even n must resolve the interface ports')
        self.n, self.m, self.size = n, m, 2*n
        size = self.size
        ids = np.arange(size*size).reshape(size, size)
        self.nonlinear = np.full((size, size), all_nonlinear, dtype=bool)
        self.nonlinear[n//2:3*n//2, n//2:3*n//2] = True
        flag = self.nonlinear.ravel()
        a = np.r_[ids[:, :-1].ravel(), ids[:-1].ravel()]
        b = np.r_[ids[:, 1:].ravel(), ids[1:].ravel()]
        same = flag[a] == flag[b]
        self.same_a, self.same_b, self.same_flag = a[same], b[same], flag[a[same]]
        self.inner = np.where(flag[a[~same]], a[~same], b[~same])
        self.outer = np.where(flag[a[~same]], b[~same], a[~same])
        self.boundary = boundary_geometry(size, size, (2., 2.))
        self.dirichlet = np.repeat([True, True, False, False], size)
        self.bc_ids = self.boundary.owner[self.dirichlet]
        self.bc_flag = flag[self.bc_ids]
        left, right = n//2, 3*n//2
        self.interface_inner = np.r_[ids[left:right, left], ids[left:right, right-1],
                                      ids[left, left:right], ids[right-1, left:right]]
        self.interface_outer = np.r_[ids[left:right, left-1], ids[left:right, right],
                                      ids[left-1, left:right], ids[right, left:right]]

    def equations(self, u, g, with_jacobian=True):
        a, b, flag = self.same_a, self.same_b, self.same_flag
        pu, ku = phi(u), conductivity(u)
        q = np.where(flag, pu[a]-pu[b], u[a]-u[b])
        da, db = np.where(flag, ku[a], 1.), -np.where(flag, ku[b], 1.)
        mixed, di, do, _ = mixed_face(u[self.inner], u[self.outer])
        a, b = np.r_[a, self.inner], np.r_[b, self.outer]
        q, da, db = np.r_[q, mixed], np.r_[da, di], np.r_[db, do]
        exterior = 2*np.where(self.bc_flag, pu[self.bc_ids]-phi(g[self.dirichlet]),
                              u[self.bc_ids]-g[self.dirichlet])
        r = np.bincount(a, weights=q, minlength=len(u))-np.bincount(b, weights=q, minlength=len(u))
        r += np.bincount(self.bc_ids, weights=exterior, minlength=len(u))
        if not with_jacobian:
            return r, None
        diag = 2*np.where(self.bc_flag, ku[self.bc_ids], 1.)
        jac = sparse.coo_matrix((np.r_[da, db, -da, -db, diag],
              (np.r_[a, a, b, b, self.bc_ids], np.r_[a, b, a, b, self.bc_ids])), shape=(len(u), len(u))).tocsc()
        return r, jac

    def solve(self, boundary, initial=None):
        g = boundary(self.boundary.xy)
        if g.ndim == 1:
            g = g[:, None]
        fields, fluxes, histories, residuals, exterior_balances = [], [], [], [], []
        for j in range(g.shape[1]):
            start = np.full(self.size**2, np.mean(g[self.dirichlet, j])) if initial is None else np.broadcast_to(initial, (self.size**2,)).copy()
            u, history = newton(lambda v,jac:self.equations(v,g[:,j],jac), start)
            q, _, _, _ = mixed_face(u[self.interface_inner], u[self.interface_outer])
            if np.all(self.nonlinear):
                q = phi(u[self.interface_inner])-phi(u[self.interface_outer])
            fluxes.append(q.reshape(4*self.m, self.n//self.m).sum(axis=1))
            fields.append(u)
            histories.append(history)
            residuals.append(history[-1]['residual_inf'])
            exterior = 2*np.where(self.bc_flag, phi(u[self.bc_ids])-phi(g[self.dirichlet,j]),
                                  u[self.bc_ids]-g[self.dirichlet,j])
            exterior_balances.append(float(abs(exterior.sum())))
        field, flux = np.array(fields).T, np.array(fluxes).T
        return dict(field=field, flux=flux, boundary=g, histories=histories,
                    nonlinear_residual=max(residuals), region_balance=float(abs(flux.sum(axis=0)).max()),
                    exterior_balance=max(exterior_balances))

    def reconstruct(self, result, target_n=64):
        """Piecewise FV reconstruction; nonlinear-region slopes act on Phi(u)."""
        if target_n == self.n:
            return result['field']
        if target_n%self.n:
            raise ValueError('Target must be an integer refinement')
        # Slopes use equal/opposite actual face fluxes, not a pressure-only stencil.
        field = result['field']
        h, size = 1/self.n, self.size
        ids = np.arange(size*size).reshape(size,size)
        flag = self.nonlinear.ravel()
        transformed = np.where(flag[:,None], phi(field), field)
        slopes = np.zeros((size*size, 2, field.shape[1]))
        counts = np.zeros((size*size, 2))
        for dim, (a,b) in enumerate(((ids[:,:-1],ids[:,1:]), (ids[:-1],ids[1:]))):
            a,b=a.ravel(),b.ravel()
            q = transformed[b]-transformed[a]
            mixed = flag[a] != flag[b]
            inner, outer = np.where(flag[a[mixed]],a[mixed],b[mixed]), np.where(flag[a[mixed]],b[mixed],a[mixed])
            flux,_,_,_=mixed_face(field[inner],field[outer])
            q[mixed] = np.where(flag[a[mixed],None],-flux,flux)
            for owner in (a,b):
                slopes[owner,dim] += q/h
                counts[owner,dim] += 1
        for side,(dim,sign) in enumerate(((0,-1),(0,1),(1,-1),(1,1))):
            sl=slice(side*size,(side+1)*size)
            owner=self.boundary.owner[sl]
            bc=np.where(flag[owner,None],phi(result['boundary'][sl]),result['boundary'][sl])
            slopes[owner,dim] += sign*(bc-transformed[owner])/(h/2) if side<2 else 0.
            counts[owner,dim] += 1
        slopes /= counts[:,:,None]
        x,y=np.meshgrid((np.arange(2*target_n)+.5)/target_n,(np.arange(2*target_n)+.5)/target_n)
        xy=np.column_stack((x.ravel(),y.ravel()))
        ij=np.minimum((xy/h).astype(int),size-1)
        owner=ij[:,1]*size+ij[:,0]
        v=transformed[owner]+np.einsum('nd,ndb->nb',xy-(ij+.5)*h,slopes[owner])
        return np.where(flag[owner,None],inverse_phi(v),v)


class NonlinearHybrid(DarcyHybrid):
    def __init__(self, response, n=64, m=32):
        super().__init__(response,n,m)
        # Remove linear local S used by the parent; keep only ordinary FV.
        aoo=self.matrix[:self.count,:self.count].tocsc()
        self.ordinary_factor=splu(aoo,permc_spec='MMD_AT_PLUS_A')
        self.C=self.matrix[:self.count,self.count:].tocsc()
        self.lift=-self.ordinary_factor.solve(self.C.toarray())
        self.M=2*np.eye(4*m)+self.C.T@self.lift

    def trace_equations(self, trace, c, with_jacobian=True):
        residual=self.M@trace+self.response['S']@phi(trace)-c
        jacobian=self.M+self.response['S']*conductivity(trace)[None,:] if with_jacobian else None
        return residual,jacobian

    def solve(self,boundary,initial=None):
        g=boundary(self.exterior.xy)
        if g.ndim==1:g=g[:,None]
        rhs=np.zeros((self.count,g.shape[1]))
        np.add.at(rhs,self.exterior_owner,2*self.dirichlet[:,None]*g)
        particular=self.ordinary_factor.solve(rhs)
        c=-self.C.T@particular
        traces,histories=[],[]
        for j in range(g.shape[1]):
            start=np.full(4*self.m,np.mean(g[self.dirichlet,j])) if initial is None else np.broadcast_to(initial,(4*self.m,)).copy()
            value,history=newton(lambda v,jac:self.trace_equations(v,c[:,j],jac),start)
            traces.append(value);histories.append(history)
        trace=np.array(traces).T
        ordinary=particular+self.lift@trace
        local=inverse_phi(self.response['H']@phi(trace))
        flux=-self.response['S']@phi(trace)
        outer_flux=2*(ordinary[self.interface_owner]-trace)
        exterior=2*self.dirichlet[:,None]*(ordinary[self.exterior_owner]-g)
        return dict(ordinary=ordinary,trace=trace,local_field=local,flux=flux,boundary=g,histories=histories,
            nonlinear_residual=max(h[-1]['residual_inf'] for h in histories),
            interface_balance=float(abs(flux+outer_flux).max()),region_balance=float(abs(flux.sum(axis=0)).max()),
            exterior_balance=float(abs(exterior.sum(axis=0)).max()))
