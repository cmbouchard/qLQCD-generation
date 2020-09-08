import numba
import numpy as np
import math

### v0:  module with tools useful for generating gauge field configurations with spacetime deformation
###      * derivs in defn of, e.g., Christoffel symbols are wrt flat spacetime coords
###      * gauge field is:  U[t, x, y, z, mu, :, :]


### v1: * correct calculation of metric
###     * speedup of factor 30 by more intelligent placement within loops and use of njit

### do not use direct calc of metric from Delta, according to weak field expansion at O(D), as the resulting
### h is only O(D) and the curvature derived from h will vanish, ie. R is O(D^2)


### function that returns list of unique points within specified number of steps from reference point, subject to periodic bcs.  Receives following args
### *        X:  reference point, as a list [Xt, Xx, Xy, Xz]
### *        N:  spacetime volume, as a list [Nt, Nx, Ny, Nz]
### *   nsteps:  number of steps from X, an integer
#@numba.njit
def fn_points(X, N, nsteps):
    Xc = np.ascontiguousarray( [ X[0], X[1], X[2], X[3] ] )
    Xlist = []
    Xlist.append( np.ascontiguousarray( [ Xc[0], Xc[1], Xc[2], Xc[3] ] ) )
    for jt in range(-nsteps, nsteps + 1):
        Xc[0] = (Xc[0] + jt) % N[0]
        for jx in range(-nsteps + abs(jt), nsteps + 1 - abs(jt)):
            Xc[1] = (Xc[1] + jx) % N[1]
            for jy in range(-nsteps + abs(jt) + abs(jx), nsteps + 1 - abs(jt) - abs(jx)):
                Xc[2] = (Xc[2] + jy) % N[2]
                for JayZ in range(-nsteps + abs(jt) + abs(jx) + abs(jy), nsteps + 1 - abs(jt) - abs(jx) - abs(jy)):
                    Xc[3] = (Xc[3] + JayZ) % N[3]
                    Xlist.append( np.ascontiguousarray( [Xc[0], Xc[1], Xc[2], Xc[3]] ) )
                    ulist = np.array( [list(el) for el in set(tuple(el) for el in Xlist)] )
    return ulist


### function that returns lattice spacing in fm, given beta
### * from Necco and Sommer, NPB 622, 328-346 (2002), hep-lat/0108008
### * valid for 5.7 <= beta <= 6.92
### * valid for Nf = 0 (quenched, ie. no quarks)
def fn_a(beta):
    return 0.5 * np.exp(-1.6804 - 1.7331 * (beta - 6.) + 0.7849 * (beta - 6.)**2 - 0.4428 * (beta - 6.)**3)

@numba.njit
def fnx_Jinv_fwd(D, P):
    N = np.array( [len(D), len(D[0]), len(D[0, 0]), len(D[0, 0, 0]) ])
    Pp1 = np.array( [ P[0], P[1], P[2], P[3] ] )
    Jinv = np.array( [[0. for i in range(4)] for j in range(4)] ) 
    #Jinv = np.array( [[np.float64(0) for i in range(4)] for j in range(4)] )
    ### c is curved space coordinate
    ### f is flat space coordinate
    for c in range(4):
        for f in range(4):
            Pp1[f] = (P[f] + 1) % N[f]
            Jinv[f, c] = fn_kdelta(f, c) + D[Pp1[0], Pp1[1], Pp1[2], Pp1[3], c] - D[P[0], P[1], P[2], P[3], c]
    return Jinv


@numba.njit
def fnx_Jacobian_fwd(D, P):
    Jinv = fnx_Jinv_fwd(D, P)
    return np.linalg.inv( Jinv )


### Kronecker delta
@numba.njit
def fn_kdelta(i, j):
    if i == j:
        return 1.
    else:
        return 0.

@numba.njit
def fnx_metric_DD_fwd(D, P):
    J = np.ascontiguousarray( fnx_Jacobian_fwd(D, P) )
    ### this gives identical results to np.matmul(np.transpose(J), J), for flatg=(1,1,1,1)
    #ans = np.zeros((4, 4))
    #for m in range(4):
    #    for a in range(4):
    #    #    for n in range(4):
    #        for b in range(4):
    #            ans[a, b] += J[a, m] * J[m, b]
    #return ans
    return np.transpose(J) @ J

#@numba.njit
#def fnx_metric_DD_fwd(D, X):
#    N = np.array( [len(D), len(D[0]), len(D[0, 0]), len(D[0, 0, 0]) ])
#    ans = np.eye(4)
#    for m in range(4):
#        Xpm = np.array( [X[0], X[1], X[2], X[3]] )
#        Xpm[m] = (Xpm[m] + 1) % N[m]
#        for n in range(4):
#            Xpn = np.array( [X[0], X[1], X[2], X[3]] )
#            Xpn[n] = (Xpn[n] + 1) % N[n]
#            ans[m, n] += -( D[Xpm[0], Xpm[1], Xpm[2], Xpm[3], n] - D[X[0], X[1], X[2], X[3], n]) 
#            ans[m, n] += -( D[Xpn[0], Xpn[1], Xpn[2], Xpn[3], m] - D[X[0], X[1], X[2], X[3], m])
#    return ans


@numba.njit
def fnx_h_DD_fwd(D, X):
    #g_DD = fnx_metric_DD_fwd(D, P)
    #ans = np.copy( g_DD )
    #for m in range(4):
    #    ans[m, m] -= 1.
    return fnx_metrix_DD_fwd(D, P) - np.eye(4)

### metric with two contravariant (up) indices
### use fwd difference
@numba.njit
def fnx_metric_UU_fwd(D, P):
    g_DD = fnx_metric_DD_fwd(D, P)
    return np.linalg.inv( g_DD )


### calculate the metric perturbation with two up indices
@numba.njit
def fnx_h_UU_fwd(D, P):
    g_UU = fnx_metric_UU_fwd(D, P)
    ans = np.copy( g_UU )
    for m in range(4):
        ans[m, m] -= 1.
    return ans


### calculate the Ricci scalar from the metric perturbation
#@numba.njit
def fnx_R_fwd(D, P):
    N = np.array( [len(D), len(D[0]), len(D[0, 0]), len(D[0, 0, 0])] )
    R = 0.
    for m in range(4):
        Ppm = np.copy( P )
        Ppm[m] = (Ppm[m] + 1) % N[m]

        Ppmpm = np.copy( P )
        Ppmpm[m] = (Ppmpm[m] + 2) % N[m]

        for n in range(4):
            Ppn = np.copy( P )
            #Ppn = np.array([int(P[0]), int(P[1]), int(P[2]), int(P[3])])
            Ppn[n] = (Ppn[n] + 1) % N[n]

            Ppnpm = np.copy( Ppm )
            Ppnpm[n] = (Ppnpm[n] + 1) % N[n]

            #print('D= ', D)
            #print('Ppnpm= ', Ppnpm)
            #if fnx_h_UU_fwd(D, Ppnpm)[m, n] != 0.:
            #    print('h(X+n+m)= ', fnx_h_UU_fwd(D, Ppnpm))
            #if fnx_h_UU_fwd(D, P)[n, n] != 0.:
            #    print('h(X)= ', fnx_h_UU_fwd(D, P)[n, n])

            R += fnx_h_UU_fwd(D, Ppnpm)[m, n] \
                - fnx_h_UU_fwd(D, Ppm)[m, n] \
                - fnx_h_UU_fwd(D, Ppn)[m, n] \
                + fnx_h_UU_fwd(D, P)[m, n] \
                - fnx_h_UU_fwd(D, Ppmpm)[n, n] \
                + 2.0 * fnx_h_UU_fwd(D, Ppm)[n, n] \
                - fnx_h_UU_fwd(D, P)[n, n]
    return R


### calculate derivative of Ricci scalar 
### * from metric perturbation
### * use forward difference
### * in version of code now lost, I verified that coding the difference of fnx_R_fwd() is equivalent to coding up dR
#@numba.njit
def fnx_dR_fwd(D, P):
    N = np.array( [len(D), len(D[0]), len(D[0, 0]), len(D[0, 0, 0])] )
    dR = np.array( [0., 0., 0., 0.] )
    for m in range(4):
        Pp1 = np.copy( P )
        Pp1[m] = (Pp1[m] + 1) % N[m]
        dR[m] += fnx_R_fwd(D, Pp1) - fnx_R_fwd(D, P)
    return dR


### here !!!!!!!!!!!!


"""

### derivative of the Ricci scalar, an array of length 4 holding derivatives in the 4 spacetime directions
### * use fwd difference
### * in terms of metric perturbation
@numba.njit
def fnx_dR_fwd(D, P):
    N = np.array( [len(D), len(D[0]), len(D[0, 0]), len(D[0, 0, 0])] )
    dR = np.zeros(( 4 ))

    for r in range(4):
        for n in range(4):
            for m in range(4):

                ### 1. shift from x by 1 unit in the n direction
                #dR[r] += shift(fnx_h_DD_fwd(D, P)[m, n], D, P, 1, n)

                ### 2. shift from x+n by 1 unit in the m direction
                Ppn = (P[n] + 1) % N[n]
                #shift(obj, D, Ppn, 1, m)
                #shift(shift(fnx_h_DD_fwd(D, P)[m, n], D, P, 1, n), D, Ppn, 1, m)

                ### 3. shift from x+n+m by 1 unit in the r direction
                Ppnpm = (Ppn[m] + 1) % N[m]
                #shift(obj, D, Ppnpm, 1, r)
                dR[r] += shift(shift(shift(fnx_h_DD_fwd(D, P)[m, n], D, P, 1, n), D, Ppn, 1, m), D, Ppnpm, 1, r)

    return dR
"""

"""
### Calculate the Christoffel symbol, with all lower indices.
### use fwd finite difference
def fnx_Christoffel_DDD_fwd(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    Gamma = np.zeros((4, 4, 4))
    J = fnx_Jacobian_fwd(D, x, y, z, t)
    g_DD_x = fnx_metric_DD_fwd(D, x, y, z, t)
    g_DD_xp0 = fnx_metric_DD_fwd(D, (x + 1)%Nx, y, z, t)
    g_DD_xp1 = fnx_metric_DD_fwd(D, x, (y + 1)%Ny, z, t)
    g_DD_xp2 = fnx_metric_DD_fwd(D, x, y, (z + 1)%Nz, t)
    g_DD_xp3 = fnx_metric_DD_fwd(D, x, y, z, (t + 1)%Nt)
    for m in range(4):
        for a in range(4):
            for b in range(4):
                #term1 = 0.
                #term2 = 0.
                #term3 = 0.

                if b == 0:
                    term1 = g_DD_xp0[m, a] - g_DD_x[m, a]
                elif b == 1:
                    term1 = g_DD_xp1[m, a] - g_DD_x[m, a]
                elif b == 2:
                    term1 = g_DD_xp2[m, a] - g_DD_x[m, a]
                elif b == 3:
                    term1 = g_DD_xp3[m, a] - g_DD_x[m, a]

                if a == 0:
                    term2 = g_DD_xp0[m, b] - g_DD_x[m, b]
                elif a == 1:
                    term2 = g_DD_xp1[m, b] - g_DD_x[m, b]
                elif a == 2:
                    term2 = g_DD_xp2[m, b] - g_DD_x[m, b]
                elif a == 3:
                    term2 = g_DD_xp3[m, b] - g_DD_x[m, b]

                if m == 0:
                    term3 = g_DD_xp0[a, b] - g_DD_x[a, b]
                elif m == 1:
                    term3 = g_DD_xp1[a, b] - g_DD_x[a, b]
                elif m == 2:
                    term3 = g_DD_xp2[a, b] - g_DD_x[a, b]
                elif m == 3:
                    term3 = g_DD_xp3[a, b] - g_DD_x[a, b]
                
                Gamma[m, a, b] = 0.5 * (term1 + term2 - term3)
    return Gamma

### use sym finite difference
def fnx_Christoffel_DDD_sym(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    Gamma = np.zeros((4, 4, 4))
    J = fnx_Jacobian_sym(D, x, y, z, t)
    g_DD_xp1 = fnx_metric_DD_sym(D, (x + 1)%Nx, y, z, t)
    g_DD_xm1 = fnx_metric_DD_sym(D, (x - 1)%Nx, y, z, t)
    g_DD_yp1 = fnx_metric_DD_sym(D, x, (y + 1)%Ny, z, t)
    g_DD_ym1 = fnx_metric_DD_sym(D, x, (y - 1)%Ny, z, t)
    g_DD_zp1 = fnx_metric_DD_sym(D, x, y, (z + 1)%Nz, t)
    g_DD_zm1 = fnx_metric_DD_sym(D, x, y, (z - 1)%Nz, t)
    g_DD_tp1 = fnx_metric_DD_sym(D, x, y, z, (t + 1)%Nt)
    g_DD_tm1 = fnx_metric_DD_sym(D, x, y, z, (t - 1)%Nt)
    for m in range(4):
        for a in range(4):
            for b in range(4):
                #term1 = 0.
                #term2 = 0.
                #term3 = 0.
                
                if b == 0:
                    term1 = 0.5 * (g_DD_xp1[m, a] - g_DD_xm1[m, a])
                elif b == 1:
                    term1 = 0.5 * (g_DD_yp1[m, a] - g_DD_ym1[m, a])
                elif b == 2:
                    term1 = 0.5 * (g_DD_zp1[m, a] - g_DD_zm1[m, a])
                elif b == 3:
                    term1 = 0.5 * (g_DD_tp1[m, a] - g_DD_tm1[m, a])

                if a == 0:
                    term2 = 0.5 * (g_DD_xp1[m, b] - g_DD_xm1[m, b])
                elif a == 1:
                    term2 = 0.5 * (g_DD_yp1[m, b] - g_DD_ym1[m, b])
                elif a == 2:
                    term2 = 0.5 * (g_DD_zp1[m, b] - g_DD_zm1[m, b])
                elif a == 3:
                    term2 = 0.5 * (g_DD_tp1[m, b] - g_DD_tm1[m, b])

                if m == 0:
                    term3 = 0.5 * (g_DD_xp1[a, b] - g_DD_xm1[a, b])
                elif m == 1:
                    term3 = 0.5 * (g_DD_yp1[a, b] - g_DD_ym1[a, b])
                elif m == 2:
                    term3 = 0.5 * (g_DD_zp1[a, b] - g_DD_zm1[a, b])
                elif m == 3:
                    term3 = 0.5 * (g_DD_tp1[a, b] - g_DD_tm1[a, b])
                
                Gamma[m, a, b] = 0.5 * (term1 + term2 - term3)
    return Gamma


                
### Christoffel symbol with 1 up and 2 down indices
### use fwd difference
def fnx_Christoffel_UDD_fwd(D, x, y, z, t):
    Chris_UDD = np.zeros((4, 4, 4))
    Chris_DDD = fnx_Christoffel_DDD_fwd(D, x, y, z, t)
    g_UU = fnx_metric_UU_fwd(D, x, y, z, t)
    for b in range(4):
        for a in range(4):
            for m in range(4):
                for n in range(4):
                    Chris_UDD[m, a, b] += g_UU[m, n] * Chris_DDD[n, a, b]

    return Chris_UDD

### use sym difference
def fnx_Christoffel_UDD_sym(D, x, y, z, t):
    Chris_UDD = np.zeros((4, 4, 4))
    Chris_DDD = fnx_Christoffel_DDD_sym(D, x, y, z, t)
    g_UU = fnx_metric_UU_sym(D, x, y, z, t)
    for b in range(4):
        for a in range(4):
            for m in range(4):
                for n in range(4):
                    Chris_UDD[m, a, b] += g_UU[m, n] * Chris_DDD[n, a, b]

    return Chris_UDD


    
### Riemann tensor with 1 up and 3 lower indices
### use fwd difference
def fnx_Riemann_UDDD_fwd(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    
    Riemann = np.zeros((4, 4, 4, 4))
    Gamma_x = fnx_Christoffel_UDD_fwd(D, x, y, z, t)
    J = fnx_Jacobian_fwd(D, x, y, z, t)
    Gamma_x_p0 = fnx_Christoffel_UDD_fwd(D, (x + 1)%Nx, y, z, t)
    Gamma_x_p1 = fnx_Christoffel_UDD_fwd(D, x, (y + 1)%Ny, z, t)
    Gamma_x_p2 = fnx_Christoffel_UDD_fwd(D, x, y, (z + 1)%Nz, t)
    Gamma_x_p3 = fnx_Christoffel_UDD_fwd(D, x, y, z, (t + 1)%Nt)

    for a in range(4):
        for b in range(4):
            for m in range(4):
                for n in range(4):
                    ### manual sum over repeated index in Jacobian
                    #term1 = 0.
                    #term2 = 0.
                    term3 = 0.
                    term4 = 0.

                    if m == 0:
                        term1 = Gamma_x_p0[a, b, n] - Gamma_x[a, b, n]
                    elif m == 1:
                        term1 = Gamma_x_p1[a, b, n] - Gamma_x[a, b, n]
                    elif m == 2:
                        term1 = Gamma_x_p2[a, b, n] - Gamma_x[a, b, n]
                    elif m == 3:
                        term1 = Gamma_x_p3[a, b, n] - Gamma_x[a, b, n]

                    if n == 0:
                        term2 = Gamma_x_p0[a, b, m] - Gamma_x[a, b, m]
                    elif n == 1:
                        term2 = Gamma_x_p1[a, b, m] - Gamma_x[a, b, m]
                    elif n == 2:
                        term2 = Gamma_x_p2[a, b, m] - Gamma_x[a, b, m]
                    elif n == 3:
                        term2 = Gamma_x_p3[a, b, m] - Gamma_x[a, b, m]

                    for s in range(4):
                        term3 += Gamma_x[a, s, m] * Gamma_x[s, b, n]
                        term4 += Gamma_x[a, s, n] * Gamma_x[s, b, m]
                    
                    Riemann[a, b, m , n] = term1 - term2 + term3 - term4
    return Riemann

### use symmetric difference
def fnx_Riemann_UDDD_sym(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    
    Riemann = np.zeros((4, 4, 4, 4))
    Gamma_x = fnx_Christoffel_UDD_sym(D, x, y, z, t)
    J = fnx_Jacobian_sym(D, x, y, z, t)
    Gamma_x_p0 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, y, z, t)
    Gamma_x_m0 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, y, z, t)
    Gamma_x_p1 = fnx_Christoffel_UDD_sym(D, x, (y + 1)%Ny, z, t)
    Gamma_x_m1 = fnx_Christoffel_UDD_sym(D, x, (y - 1)%Ny, z, t)
    Gamma_x_p2 = fnx_Christoffel_UDD_sym(D, x, y, (z + 1)%Nz, t)
    Gamma_x_m2 = fnx_Christoffel_UDD_sym(D, x, y, (z - 1)%Nz, t)
    Gamma_x_p3 = fnx_Christoffel_UDD_sym(D, x, y, z, (t + 1)%Nt)
    Gamma_x_m3 = fnx_Christoffel_UDD_sym(D, x, y, z, (t - 1)%Nt)

    for a in range(4):
        for b in range(4):
            for m in range(4):
                for n in range(4):
                    ### manual sum over repeated index in Jacobian
                    #term1 = 0.
                    #term2 = 0.
                    term3 = 0.
                    term4 = 0.

                    if m == 0:
                        term1 = 0.5 * (Gamma_x_p0[a, b, n] - Gamma_x_m0[a, b, n])
                    elif m == 1:
                        term1 = 0.5 * (Gamma_x_p1[a, b, n] - Gamma_x_m1[a, b, n])
                    elif m == 2:
                        term1 = 0.5 * (Gamma_x_p2[a, b, n] - Gamma_x_m2[a, b, n])
                    elif m == 3:
                        term1 = 0.5 * (Gamma_x_p3[a, b, n] - Gamma_x_m3[a, b, n])
                    
                    if n == 0:
                        term2 = 0.5 * (Gamma_x_p0[a, b, m] - Gamma_x_m0[a, b, m])
                    elif n == 1:
                        term2 = 0.5 * (Gamma_x_p1[a, b, m] - Gamma_x_m1[a, b, m])
                    elif n == 2:
                        term2 = 0.5 * (Gamma_x_p2[a, b, m] - Gamma_x_m2[a, b, m])
                    elif n == 3:
                        term2 = 0.5 * (Gamma_x_p3[a, b, m] - Gamma_x_m3[a, b, m])
                    
                    for s in range(4):
                        term3 += Gamma_x[a, s, m] * Gamma_x[s, b, n]
                        term4 += Gamma_x[a, s, n] * Gamma_x[s, b, m]
                    
                    Riemann[a, b, m , n] = term1 - term2 + term3 - term4
    return Riemann



### Riemann tensor with all lower indices
### use fwd difference
def fnx_Riemann_DDDD_fwd(D, x, y, z, t):
    R_uddd = fnx_Riemann_UDDD_fwd(D, x, y, z, t)
    g_DD = fnx_metric_DD_fwd(D, x, y, z, t)
    R_dddd = np.zeros((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    for e in range(4):
                        R_dddd[a, c, d, e] += g_DD[a, b] * R_uddd[b, c, d, e]
    return R_dddd

### use symmetric difference
def fnx_Riemann_DDDD_sym(D, x, y, z, t):
    R_uddd = fnx_Riemann_UDDD_sym(D, x, y, z, t)
    g_DD = fnx_metric_DD_sym(D, x, y, z, t)
    R_dddd = np.zeros((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    for e in range(4):
                        R_dddd[a, c, d, e] += g_DD[a, b] * R_uddd[b, c, d, e]
    return R_dddd


### Ricci tensor
### use fwd difference
def fnx_RicciT_fwd(D, x, y, z, t):
    RicciT = np.zeros((4, 4))
    #Riemann = fnx_Riemann_UDDD_fwd(D, x, y, z, t)
    Riemann = fnx_Riemann_DDDD_fwd(D, x, y, z, t)
    g_UU = fnx_metric_UU_fwd(D, x, y, z, t)
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                #RicciT[mu, nu] += Riemann[alpha, mu, alpha, nu]
                for rho in range(4):
                    RicciT[mu, nu] += g_UU[alpha, rho] * Riemann[alpha, mu, rho, nu]
    return RicciT

### use symmetric difference
def fnx_RicciT_sym(D, x, y, z, t):
    RicciT = np.zeros((4, 4))
    #Riemann = fnx_Riemann_UDDD_sym(D, x, y, z, t)
    Riemann = fnx_Riemann_DDDD_sym(D, x, y, z, t)
    g_UU = fnx_metric_UU_sym(D, x, y, z, t)
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                #RicciT[mu, nu] += Riemann[alpha, mu, alpha, nu]
                for rho in range(4):
                    RicciT[mu, nu] += g_UU[alpha, rho] * Riemann[alpha, mu, rho, nu]
    return RicciT


### Ricci scalar
### use fwd difference
def fnx_Ricci_fwd(D, x, y, z, t):
    Ricci = 0.0
    RicciT = fnx_RicciT_fwd(D, x, y, z, t)
    g_UU =  fnx_metric_UU_fwd(D, x, y, z, t)
    for mu in range(4):
        for nu in range(4):
            Ricci += g_UU[mu, nu] * RicciT[nu, mu]
    return Ricci

### use symmetric difference
def fnx_Ricci_sym(D, x, y, z, t):
    Ricci = 0.0
    RicciT = fnx_RicciT_sym(D, x, y, z, t)
    g_UU =  fnx_metric_UU_sym(D, x, y, z, t)
    for mu in range(4):
        for nu in range(4):
            Ricci += g_UU[mu, nu] * RicciT[nu, mu]
    return Ricci


### simple deriviative of Ricci scalar
def fnx_dR_fwd_simple(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    dR = np.zeros(( 4 ))
    dR[0] = fnx_Ricci_fwd(D, (x + 1)%Nx, y, z, t) - fnx_Ricci_fwd(D, x, y, z, t)
    dR[1] = fnx_Ricci_fwd(D, x, (y + 1)%Ny, z, t) - fnx_Ricci_fwd(D, x, y, z, t)
    dR[2] = fnx_Ricci_fwd(D, x, y, (z + 1)%Nz, t) - fnx_Ricci_fwd(D, x, y, z, t)
    dR[3] = fnx_Ricci_fwd(D, x, y, z, (t + 1)%Nt) - fnx_Ricci_fwd(D, x, y, z, t)
    return dR

### simple deriviative of Ricci scalar
def fnx_dR_sym_simple(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    dR = np.zeros(( 4 ))
    dR[0] = 0.5 * ( fnx_Ricci_sym(D, (x + 1)%Nx, y, z, t) - fnx_Ricci_sym(D, (x - 1)%Nx, y, z, t) )
    dR[1] = 0.5 * ( fnx_Ricci_sym(D, x, (y + 1)%Ny, z, t) - fnx_Ricci_sym(D, x, (y - 1)%Ny, z, t) )
    dR[2] = 0.5 * ( fnx_Ricci_sym(D, x, y, (z + 1)%Nz, t) - fnx_Ricci_sym(D, x, y, (z - 1)%Nz, t) )
    dR[3] = 0.5 * ( fnx_Ricci_sym(D, x, y, z, (t + 1)%Nt) - fnx_Ricci_sym(D, x, y, z, (t - 1)%Nt) )
    return dR

### derivative of the Ricci scalar, an array of length 4 holding derivatives in the 4 spacetime directions
### use fwd difference
def fnx_dR_fwd(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    dR = np.zeros(( 4 ))

    g_UU_x = fnx_metric_UU_fwd(D, x, y, z, t)
    dg_UU_p0 = fnx_metric_UU_fwd(D, (x + 1)%Nx, y, z, t) - g_UU_x
    dg_UU_p1 = fnx_metric_UU_fwd(D, x, (y + 1)%Ny, z, t) - g_UU_x
    dg_UU_p2 = fnx_metric_UU_fwd(D, x, y, (z + 1)%Nz, t) - g_UU_x
    dg_UU_p3 = fnx_metric_UU_fwd(D, x, y, z, (t + 1)%Nt) - g_UU_x
    dg_UU = np.array([ dg_UU_p0, dg_UU_p1, dg_UU_p2, dg_UU_p3 ])

    G_UDD_x = fnx_Christoffel_UDD_fwd(D, x, y, z, t)
    G_UDD_x_p0 = fnx_Christoffel_UDD_fwd(D, (x + 1)%Nx, y, z, t)
    G_UDD_x_p1 = fnx_Christoffel_UDD_fwd(D, x, (y + 1)%Ny, z, t)
    G_UDD_x_p2 = fnx_Christoffel_UDD_fwd(D, x, y, (z + 1)%Nz, t)
    G_UDD_x_p3 = fnx_Christoffel_UDD_fwd(D, x, y, z, (t + 1)%Nt)
    dG_UDD = np.array([ G_UDD_x_p0 - G_UDD_x, G_UDD_x_p1 - G_UDD_x, G_UDD_x_p2 - G_UDD_x, G_UDD_x_p3 - G_UDD_x ])
    
    ddG_UDD_p0p0 = fnx_Christoffel_UDD_fwd(D, (x + 2)%Nx, y, z, t) - 2. * G_UDD_x_p0 + G_UDD_x
    ddG_UDD_p0p1 = fnx_Christoffel_UDD_fwd(D, (x + 1)%Nx, (y + 1)%Ny, z, t) - G_UDD_x_p0 - G_UDD_x_p1 + G_UDD_x
    ddG_UDD_p0p2 = fnx_Christoffel_UDD_fwd(D, (x + 1)%Nx, y, (z + 1)%Nz, t) - G_UDD_x_p0 - G_UDD_x_p2 + G_UDD_x
    ddG_UDD_p0p3 = fnx_Christoffel_UDD_fwd(D, (x + 1)%Nx, y, z, (t + 1)%Nt) - G_UDD_x_p0 - G_UDD_x_p3 + G_UDD_x
    ddG_UDD_p1p1 = fnx_Christoffel_UDD_fwd(D, x, (y + 2)%Ny, z, t) - 2. * G_UDD_x_p1 + G_UDD_x
    ddG_UDD_p1p2 = fnx_Christoffel_UDD_fwd(D, x, (y + 1)%Ny, (z + 1)%Nz, t) - G_UDD_x_p1 - G_UDD_x_p2 + G_UDD_x
    ddG_UDD_p1p3 = fnx_Christoffel_UDD_fwd(D, x, (y + 1)%Ny, z, (t + 1)%Nt) - G_UDD_x_p1 - G_UDD_x_p3 + G_UDD_x
    ddG_UDD_p2p2 = fnx_Christoffel_UDD_fwd(D, x, y, (z + 2)%Nz, t) - 2. * G_UDD_x_p2 + G_UDD_x
    ddG_UDD_p2p3 = fnx_Christoffel_UDD_fwd(D, x, y, (z + 1)%Nz, (t + 1)%Nt) - G_UDD_x_p2 - G_UDD_x_p3 + G_UDD_x
    ddG_UDD_p3p3 = fnx_Christoffel_UDD_fwd(D, x, y, z, (t + 2)%Nt) - 2. * G_UDD_x_p3 + G_UDD_x
    ddG_UDD = np.array( [[ddG_UDD_p0p0, ddG_UDD_p0p1, ddG_UDD_p0p2, ddG_UDD_p0p3 ],
                         [ddG_UDD_p0p1, ddG_UDD_p1p1, ddG_UDD_p1p2, ddG_UDD_p1p3 ],
                         [ddG_UDD_p0p2, ddG_UDD_p1p2, ddG_UDD_p2p2, ddG_UDD_p2p3 ],
                         [ddG_UDD_p0p3, ddG_UDD_p1p3, ddG_UDD_p2p3, ddG_UDD_p3p3 ]] )

    for r in range(4):
        for m in range(4):
            for n in range(4):
                for a in range(4):
                    dR[r] += dg_UU[r, m, n] * ( dG_UDD[a, a, m, n] - dG_UDD[n, a, m, a] )
                    dR[r] += fn_kdelta(m, n) * ( ddG_UDD[r, a, a, m, n] - ddG_UDD[r, n, a, m, a] )
    return dR


### use symmetric difference
def fnx_dR_sym(D, x, y, z, t):
    Nx = len(D)
    Ny = len(D[0])
    Nz = len(D[0, 0])
    Nt = len(D[0, 0, 0])
    dR = np.zeros((4))

    g_UU_x = fnx_metric_UU_sym(D, x, y, z, t)
    dg_UU_p0 = fnx_metric_UU_sym(D, (x + 1)%Nx, y, z, t) - g_UU_x
    dg_UU_p1 = fnx_metric_UU_sym(D, x, (y + 1)%Ny, z, t) - g_UU_x
    dg_UU_p2 = fnx_metric_UU_sym(D, x, y, (z + 1)%Nz, t) - g_UU_x
    dg_UU_p3 = fnx_metric_UU_sym(D, x, y, z, (t + 1)%Nt) - g_UU_x
    dg_UU = np.array([ dg_UU_p0, dg_UU_p1, dg_UU_p2, dg_UU_p3 ])

    G_UDD_x = fnx_Christoffel_UDD_sym(D, x, y, z, t)
    G_UDD_x_p0 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, y, z, t)
    G_UDD_x_p1 = fnx_Christoffel_UDD_sym(D, x, (y + 1)%Ny, z, t)
    G_UDD_x_p2 = fnx_Christoffel_UDD_sym(D, x, y, (z + 1)%Nz, t)
    G_UDD_x_p3 = fnx_Christoffel_UDD_sym(D, x, y, z, (t + 1)%Nt)
    G_UDD_x_m0 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, y, z, t)
    G_UDD_x_m1 = fnx_Christoffel_UDD_sym(D, x, (y - 1)%Ny, z, t)
    G_UDD_x_m2 = fnx_Christoffel_UDD_sym(D, x, y, (z - 1)%Nz, t)
    G_UDD_x_m3 = fnx_Christoffel_UDD_sym(D, x, y, z, (t - 1)%Nt)
    dG_UDD = 0.5 * np.array([ G_UDD_x_p0 - G_UDD_x_m0, G_UDD_x_p1 - G_UDD_x_m1, G_UDD_x_p2 - G_UDD_x_m2, G_UDD_x_p3 - G_UDD_x_m3 ])

    G_UDD_p0p0 = fnx_Christoffel_UDD_sym(D, (x + 2)%Nx, y, z, t)
    G_UDD_m0m0 = fnx_Christoffel_UDD_sym(D, (x - 2)%Nx, y, z, t)
    G_UDD_p0p1 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, (y + 1)%Ny, z, t)
    G_UDD_m0p1 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, (y + 1)%Ny, z, t)
    G_UDD_p0m1 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, (y - 1)%Ny, z, t)
    G_UDD_m0m1 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, (y - 1)%Ny, z, t)
    G_UDD_p0p2 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, y, (z + 1)%Nz, t)
    G_UDD_m0p2 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, y, (z + 1)%Nz, t)
    G_UDD_p0m2 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, y, (z - 1)%Nz, t)
    G_UDD_m0m2 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, y, (z - 1)%Nz, t)
    G_UDD_p0p3 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, y, z, (t + 1)%Nt)
    G_UDD_m0p3 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, y, z, (t + 1)%Nt)
    G_UDD_p0m3 = fnx_Christoffel_UDD_sym(D, (x + 1)%Nx, y, z, (t - 1)%Nt)
    G_UDD_m0m3 = fnx_Christoffel_UDD_sym(D, (x - 1)%Nx, y, z, (t - 1)%Nt)
    G_UDD_p1p1 = fnx_Christoffel_UDD_sym(D, x, (y + 2)%Ny, z, t)
    G_UDD_m1m1 = fnx_Christoffel_UDD_sym(D, x, (y - 2)%Ny, z, t)
    G_UDD_p1p2 = fnx_Christoffel_UDD_sym(D, x, (y + 1)%Ny, (z + 1)%Nz, t)
    G_UDD_m1p2 = fnx_Christoffel_UDD_sym(D, x, (y - 1)%Ny, (z + 1)%Nz, t)
    G_UDD_p1m2 = fnx_Christoffel_UDD_sym(D, x, (y + 1)%Ny, (z - 1)%Nz, t)
    G_UDD_m1m2 = fnx_Christoffel_UDD_sym(D, x, (y - 1)%Ny, (z - 1)%Nz, t)
    G_UDD_p1p3 = fnx_Christoffel_UDD_sym(D, x, (y + 1)%Ny, z, (t + 1)%Nt)
    G_UDD_m1p3 = fnx_Christoffel_UDD_sym(D, x, (y - 1)%Ny, z, (t + 1)%Nt)
    G_UDD_p1m3 = fnx_Christoffel_UDD_sym(D, x, (y + 1)%Ny, z, (t - 1)%Nt)
    G_UDD_m1m3 = fnx_Christoffel_UDD_sym(D, x, (y - 1)%Ny, z, (t - 1)%Nt)
    G_UDD_p2p2 = fnx_Christoffel_UDD_sym(D, x, y, (z + 2)%Nz, t)
    G_UDD_m2m2 = fnx_Christoffel_UDD_sym(D, x, y, (z - 2)%Nz, t)
    G_UDD_p2p3 = fnx_Christoffel_UDD_sym(D, x, y, (z + 1)%Nz, (t + 1)%Nt)
    G_UDD_m2p3 = fnx_Christoffel_UDD_sym(D, x, y, (z - 1)%Nz, (t + 1)%Nt)
    G_UDD_p2m3 = fnx_Christoffel_UDD_sym(D, x, y, (z + 1)%Nz, (t - 1)%Nt)
    G_UDD_m2m3 = fnx_Christoffel_UDD_sym(D, x, y, (z - 1)%Nz, (t - 1)%Nt)
    G_UDD_p3p3 = fnx_Christoffel_UDD_sym(D, x, y, z, (t + 2)%Nt)
    G_UDD_m3m3 = fnx_Christoffel_UDD_sym(D, x, y, z, (t - 2)%Nt)
    
    ddG_UDD_00 = 0.25 * ( G_UDD_p0p0 + G_UDD_m0m0 - 2.0 * G_UDD_x )
    ddG_UDD_01 = 0.25 * ( G_UDD_p0p1 + G_UDD_m0m1 - G_UDD_m0p1 - G_UDD_p0m1 )
    ddG_UDD_02 = 0.25 * ( G_UDD_p0p2 + G_UDD_m0m2 - G_UDD_m0p2 - G_UDD_p0m2 )
    ddG_UDD_03 = 0.25 * ( G_UDD_p0p3 + G_UDD_m0m3 - G_UDD_m0p3 - G_UDD_p0m3 )
    ddG_UDD_11 = 0.25 * ( G_UDD_p1p1 + G_UDD_m1m1 - 2.0 * G_UDD_x )
    ddG_UDD_12 = 0.25 * ( G_UDD_p1p2 + G_UDD_m1m2 - G_UDD_m1p2 - G_UDD_p1m2 )
    ddG_UDD_13 = 0.25 * ( G_UDD_p1p3 + G_UDD_m1m3 - G_UDD_m1p3 - G_UDD_p1m3 )
    ddG_UDD_22 = 0.25 * ( G_UDD_p2p2 + G_UDD_m2m2 - 2.0 * G_UDD_x )
    ddG_UDD_23 = 0.25 * ( G_UDD_p2p3 + G_UDD_m2m3 - G_UDD_m2p3 - G_UDD_p2m3 )
    ddG_UDD_33 = 0.25 * ( G_UDD_p3p3 + G_UDD_m3m3 - 2.0 * G_UDD_x )
    
    ddG_UDD = np.array( [[ddG_UDD_00, ddG_UDD_01, ddG_UDD_02, ddG_UDD_03 ],
                         [ddG_UDD_01, ddG_UDD_11, ddG_UDD_12, ddG_UDD_13 ],
                         [ddG_UDD_02, ddG_UDD_12, ddG_UDD_22, ddG_UDD_23 ],
                         [ddG_UDD_03, ddG_UDD_13, ddG_UDD_23, ddG_UDD_33 ]] )

    for r in range(4):
        for m in range(4):
            for n in range(4):
                for a in range(4):
                    dR[r] += dg_UU[r, m, n] * ( dG_UDD[a, a, m, n] - dG_UDD[n, a, m, a] )
                    dR[r] += fn_kdelta(m, n) * ( ddG_UDD[r, a, a, m, n] - ddG_UDD[r, n, a, m, a] )
    return dR



### function that takes an invertible 4x4 matrix M and returns M^{-1}
### * doing this by hand might be faster than calling an builtin inversion function
def fn_inv_4x4(M):

    M = np.array(M)
    
    det = M[0,0]*M[1,1]*M[2,2]*M[3,3] - M[0,0]*M[1,1]*M[2,3]*M[3,2] - M[0,0]*M[1,2]*M[2,1]*M[3,3] + M[0,0]*M[1,2]*M[2,3]*M[3,1] + M[0,0]*M[1,3]*M[2,1]*M[3,2] - M[0,0]*M[1,3]*M[2,2]*M[3,1] - M[0,1]*M[1,0]*M[2,2]*M[3,3] + M[0,1]*M[1,0]*M[2,3]*M[3,2] + M[0,1]*M[1,2]*M[2,0]*M[3,3] - M[0,1]*M[1,2]*M[2,3]*M[3,0] - M[0,1]*M[1,3]*M[2,0]*M[3,2] + M[0,1]*M[1,3]*M[2,2]*M[3,0] + M[0,2]*M[1,0]*M[2,1]*M[3,3] - M[0,2]*M[1,0]*M[2,3]*M[3,1] - M[0,2]*M[1,1]*M[2,0]*M[3,3] + M[0,2]*M[1,1]*M[2,3]*M[3,0] + M[0,2]*M[1,3]*M[2,0]*M[3,1] - M[0,2]*M[1,3]*M[2,1]*M[3,0] - M[0,3]*M[1,0]*M[2,1]*M[3,2] + M[0,3]*M[1,0]*M[2,2]*M[3,1] + M[0,3]*M[1,1]*M[2,0]*M[3,2] - M[0,3]*M[1,1]*M[2,2]*M[3,0] - M[0,3]*M[1,2]*M[2,0]*M[3,1] + M[0,3]*M[1,2]*M[2,1]*M[3,0]

    ans00 =  (M[1,1]*M[2,2]*M[3,3] - M[1,1]*M[2,3]*M[3,2] - M[1,2]*M[2,1]*M[3,3] + M[1,2]*M[2,3]*M[3,1] + M[1,3]*M[2,1]*M[3,2] - M[1,3]*M[2,2]*M[3,1])
    ans01 = -(M[0,1]*M[2,2]*M[3,3] - M[0,1]*M[2,3]*M[3,2] - M[0,2]*M[2,1]*M[3,3] + M[0,2]*M[2,3]*M[3,1] + M[0,3]*M[2,1]*M[3,2] - M[0,3]*M[2,2]*M[3,1])
    ans02 =  (M[0,1]*M[1,2]*M[3,3] - M[0,1]*M[1,3]*M[3,2] - M[0,2]*M[1,1]*M[3,3] + M[0,2]*M[1,3]*M[3,1] + M[0,3]*M[1,1]*M[3,2] - M[0,3]*M[1,2]*M[3,1])
    ans03 = -(M[0,1]*M[1,2]*M[2,3] - M[0,1]*M[1,3]*M[2,2] - M[0,2]*M[1,1]*M[2,3] + M[0,2]*M[1,3]*M[2,1] + M[0,3]*M[1,1]*M[2,2] - M[0,3]*M[1,2]*M[2,1])
    ans10 = -(M[1,0]*M[2,2]*M[3,3] - M[1,0]*M[2,3]*M[3,2] - M[1,2]*M[2,0]*M[3,3] + M[1,2]*M[2,3]*M[3,0] + M[1,3]*M[2,0]*M[3,2] - M[1,3]*M[2,2]*M[3,0])
    ans11 =  (M[0,0]*M[2,2]*M[3,3] - M[0,0]*M[2,3]*M[3,2] - M[0,2]*M[2,0]*M[3,3] + M[0,2]*M[2,3]*M[3,0] + M[0,3]*M[2,0]*M[3,2] - M[0,3]*M[2,2]*M[3,0])
    ans12 = -(M[0,0]*M[1,2]*M[3,3] - M[0,0]*M[1,3]*M[3,2] - M[0,2]*M[1,0]*M[3,3] + M[0,2]*M[1,3]*M[3,0] + M[0,3]*M[1,0]*M[3,2] - M[0,3]*M[1,2]*M[3,0])
    ans13 =  (M[0,0]*M[1,2]*M[2,3] - M[0,0]*M[1,3]*M[2,2] - M[0,2]*M[1,0]*M[2,3] + M[0,2]*M[1,3]*M[2,0] + M[0,3]*M[1,0]*M[2,2] - M[0,3]*M[1,2]*M[2,0])
    ans20 =  (M[1,0]*M[2,1]*M[3,3] - M[1,0]*M[2,3]*M[3,1] - M[1,1]*M[2,0]*M[3,3] + M[1,1]*M[2,3]*M[3,0] + M[1,3]*M[2,0]*M[3,1] - M[1,3]*M[2,1]*M[3,0])
    ans21 = -(M[0,0]*M[2,1]*M[3,3] - M[0,0]*M[2,3]*M[3,1] - M[0,1]*M[2,0]*M[3,3] + M[0,1]*M[2,3]*M[3,0] + M[0,3]*M[2,0]*M[3,1] - M[0,3]*M[2,1]*M[3,0])
    ans22 =  (M[0,0]*M[1,1]*M[3,3] - M[0,0]*M[1,3]*M[3,1] - M[0,1]*M[1,0]*M[3,3] + M[0,1]*M[1,3]*M[3,0] + M[0,3]*M[1,0]*M[3,1] - M[0,3]*M[1,1]*M[3,0])
    ans23 = -(M[0,0]*M[1,1]*M[2,3] - M[0,0]*M[1,3]*M[2,1] - M[0,1]*M[1,0]*M[2,3] + M[0,1]*M[1,3]*M[2,0] + M[0,3]*M[1,0]*M[2,1] - M[0,3]*M[1,1]*M[2,0])
    ans30 = -(M[1,0]*M[2,1]*M[3,2] - M[1,0]*M[2,2]*M[3,1] - M[1,1]*M[2,0]*M[3,2] + M[1,1]*M[2,2]*M[3,0] + M[1,2]*M[2,0]*M[3,1] - M[1,2]*M[2,1]*M[3,0])
    ans31 =  (M[0,0]*M[2,1]*M[3,2] - M[0,0]*M[2,2]*M[3,1] - M[0,1]*M[2,0]*M[3,2] + M[0,1]*M[2,2]*M[3,0] + M[0,2]*M[2,0]*M[3,1] - M[0,2]*M[2,1]*M[3,0])
    ans32 = -(M[0,0]*M[1,1]*M[3,2] - M[0,0]*M[1,2]*M[3,1] - M[0,1]*M[1,0]*M[3,2] + M[0,1]*M[1,2]*M[3,0] + M[0,2]*M[1,0]*M[3,1] - M[0,2]*M[1,1]*M[3,0])
    ans33 =  (M[0,0]*M[1,1]*M[2,2] - M[0,0]*M[1,2]*M[2,1] - M[0,1]*M[1,0]*M[2,2] + M[0,1]*M[1,2]*M[2,0] + M[0,2]*M[1,0]*M[2,1] - M[0,2]*M[1,1]*M[2,0])

    ans = [[ans00, ans01, ans02, ans03],
           [ans10, ans11, ans12, ans13],
           [ans20, ans21, ans22, ans23],
           [ans30, ans31, ans32, ans33]]
    
    return ans / det




### run the test case for comparison
def testx(x, y, z, t):
    ### set the lattice of deformations
    D = fn_test_D()  #D[x,y,z,t,mu]
    D_e = fn_FRW_D(x, y, z, t)
    print('D(x=', x, ',y=', y, ',z=', z, ',t=', t, ')= ', D[x, y, z, t])
    print('D(exact)= ', D_e)


    Jinv_exa = fn_FRW_Jinv(x, y, z, t)
    Jinv_fwd = fnx_Jinv_fwd(D, x, y, z, t)
    Jinv_sym = fnx_Jinv_sym(D, x, y, z, t)
    print('Jinv_fwd= ', Jinv_fwd)
    print('Jinv_sym= ', Jinv_sym)
    print('Jinv_e= ', Jinv_exa)
    
    J_fwd = fnx_Jacobian_fwd(D, x, y, z, t)
    J_sym = fnx_Jacobian_sym(D, x, y, z, t)
    print('J_fwd= ', J_fwd)
    print('J_sym= ', J_sym)
    print('J_e=', fn_FRW_J(x, y, z, y))
    
    g_DD_fwd = fnx_metric_DD_fwd(D, x, y, z, t)
    g_DD_sym = fnx_metric_DD_sym(D, x, y, z, t)
    print('g_DD_fwd= ', g_DD_fwd)
    print('g_DD_sym= ', g_DD_sym)
    print('g_DD_e= ', fn_FRW_g_DD(x, y, z, t))
    
    print('g_UU_fwd= ', fnx_metric_UU_fwd(D, x, y, z, t))
    print('g_UU_sym= ', fnx_metric_UU_sym(D, x, y, z, t))
    print('g_UU_e= ', fn_FRW_g_UU(x, y, z, t))
    

    G_UDD_fwd = fnx_Christoffel_UDD_fwd(D, x, y, z, t)
    G_UDD_sym = fnx_Christoffel_UDD_sym(D, x, y, z, t)
    print('G[mu,nu,rho]_fwd= ', G_UDD_fwd)
    print('G[mu,nu,rho]_sym= ', G_UDD_sym)
    print('G[mu,nu,rho]_e= ', FRW_Christoffel_UDD_ex(x, y, z, t))
    """




"""
    
    #diff_G_UDD = np.zeros((4, 4, 4))
    #for a in range(4):
    #    for b in range(4):
    #        for c in range(4):
    #            diff_G_UDD[a, b, c] = G_UDD[a, b, c] - G_UDD[a, c, b]
    #print('(G^m_{nr} - G^m_{rn})_d = ', diff_G_UDD)

    
    R_UDDD_fwd = fnx_Riemann_UDDD_fwd(D, x, y, z, t)
    R_UDDD_sym = fnx_Riemann_UDDD_sym(D, x, y, z, t)
    #R_DDDD = fnx_Riemann_DDDD(D, x, y, z, t)
    print('Riemann^0_{1,mu,nu}_fwd= ', R_UDDD_fwd[0,1,:,:])
    print('Riemann^0_{1,mu,nu}_sym= ', R_UDDD_sym[0,1,:,:])
    print('Riemann^0_{1,mu,nu}_e= ', FRW_Riemann_UDDD_ex(x, y, z, t)[0,1,:,:])
    
    
    #diff_R1 = np.zeros((4, 4, 4, 4)) #skew symmetry R_abcd + R_bacd = 0
    #diff_R2 = np.zeros((4, 4, 4, 4)) #skew symmetry R_abcd + R_abdc = 0
    #diff_R3 = np.zeros((4, 4, 4, 4)) #interchange symmetry R_abcd - R_cdab = 0
    #diff_R4 = np.zeros((4, 4, 4, 4)) #1st Bianchi id R_abcd + R_acdb + R_adbc = 0
    #for a in range(4):
    #    for b in range(4):
    #        for c in range(4):
    #            for d in range(4):
    #                diff_R1[a, b, c, d] += R_DDDD[a, b, c, d] + R_DDDD[b, a, c, d]
    #                diff_R2[a, b, c, d] += R_DDDD[a, b, c, d] + R_DDDD[a, b, d, c]
    #                diff_R3[a, b, c, d] += R_DDDD[a, b, c, d] - R_DDDD[c, d, a, b]
    #                diff_R4[a, b, c, d] += R_DDDD[a, b, c, d] + R_DDDD[a, c, d, b] + R_DDDD[a, d, b, c]
    #                
    #print('check identities: R_abcd + R_bacd = 0')
    #print('diff1 = ', diff_R1)
    #print('diff2 = ', diff_R2)
    #print('diff3 = ', diff_R3)
    #print('diff4 = ', diff_R4)
    
    
    RT_fwd = fnx_RicciT_fwd(D, x, y, z, t)
    RT_sym = fnx_RicciT_sym(D, x, y, z, t)
    print('RicciT_fwd= ', RT_fwd)
    print('RicciT_sym= ', RT_sym)
    print('RicciT_e= ', FRW_Ricci_DD_ex(x, y, z, t))
    
    R_fwd = fnx_Ricci_fwd(D, x, y, z, t)
    R_sym = fnx_Ricci_sym(D, x, y, z, t)
    print('R_fwd= ', R_fwd)
    print('R_sym= ', R_sym)
    print('R_e= ', FRW_R_ex(x, y, z, t))

    dR_fwd = fnx_dR_fwd(D, x, y, z, t)
    dR_sym = fnx_dR_sym(D, x, y, z, t)
    print('dR_fwd= ', dR_fwd)
    print('dR_sym= ', dR_sym)

    dR_fwd_simp = fnx_dR_fwd_simple(D, x, y, z, t)
    dR_sym_simp = fnx_dR_sym_simple(D, x, y, z, t)
    print('dR_fwd(simple)= ', dR_fwd_simp)
    print('dR_sym(simple)= ', dR_sym_simp)
    
    print('dR_e= ', FRW_dR_ex(x, y, z, t))



### sweep through the lattice
### Pass:
###   * spacetime point x,y,z,t and mu specifying the update location
###   * current lattice of deformationsm D
###   * size of deformation to allow, delta
###   * current lattice gauge links, U
###   * size of link jiggles to allow, eps
def fnx_sweep(delta, eps):
    
    ### jiggle D^\mu(x,y,z,t)
    D_x_mu = D[x, y, z, t, mu]
    D_x_mu_p = D_x_mu + delta * r.uniform(-1., 1.)


    ### jiggle U^\mu(x,y,z,t)

    
    
### evaluate the relevant part of the EH action, the part associated with point x,y,z,t and in direction mu
### Pass:
###   * spacetime point x,y,z,t and mu specifying the update location
###   * current lattice of deformations D
###   * current lattice gauge links, U
def fnx_evalS(x, y, z, t, mu, D, U):

    dR = fnx_dR(D, y, y, z, t)[mu]
 
    
"""
