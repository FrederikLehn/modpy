import numpy as np
import numpy.linalg as la
from modpy._exceptions import InfeasibleProblemError


class Presolver:
    def __init__(self, g, A, b, C, d, lb, ub, H=None):
        """
        Parameters
        ----------
        g : array_like, shape (n,)
            System vector with coefficients of the linear terms.
        A : array_like, shape (me, n)
            System matrix with coefficients for the equality constraints.
        b : array_like, shape (me,)
            Results vector of the equality constraints.
        C : array_like, shape (mi, n)
            System matrix with coefficients for the inequality constraints.
        d : array_like, shape (mi,)
            Upper bound vector of the inequality constraints.
        lb : array_like, shape (n,)
            Lower bound.
        ub : array_like, shape (n,)
            Upper bound.
        H : array_like, shape (n, n), optional
            System matrix with coefficients of the quadratic terms.
        """

        # problem vectors/matrices (will change during presolve procedure)
        self.H = None   # quadratic coefficient matrix (for QP)
        self.g = g      # linear coefficient vector
        self.A = A      # equality constraint coefficients
        self.b = b      # equality constraint solutions
        self.C = C      # inequality constraint coefficients
        self.d = d      # inequality constraint upper bounds
        self.lb = lb    # variable lower bound
        self.ub = ub    # variable upper bound

        # most QP literature assumes Cx >= d, but the pre-solver
        # assumes Cx <= d to share methods between LP and QP.
        # the inequality constraints are reversed to comply.
        if H is not None:
            self.H = H
            self.C = -self.C
            self.d = -self.d

        # original problem dimensions
        self.n = g.size
        self.me = b.size
        self.mi = d.size

        # presolved solution
        self.x = np.zeros((self.n,))    # presolved solution to the primal problem
        self.k = 0.                     # constant to be added to the optimal function value.

        # Lagrangian multiplier bounds
        self.yl = np.zeros((self.me,))  # lower bound on the equality Lagrangian multiplier
        self.yu = np.zeros((self.me,))  # Upper bound on the equality Lagrangian multiplier
        self.zl = np.zeros((self.mi,))  # lower bound on the inequality Lagrangian multiplier
        self.zu = np.zeros((self.mi,))  # Upper bound on the inequality Lagrangian multiplier

        # index trackers of remaining equations and variables
        self.idx_x = np.arange(self.n)     # index of variables remaining in the reduced LP
        self.idx_e = np.arange(self.me)    # index of eq. constraints remaining in the reduced LP
        self.idx_i = np.arange(self.mi)    # index of ineq. constraints remaining in the reduced LP

        # internal parameters
        self._tol = 1e-13       # tolerance below which values are assumed 0.
        self._reduced = True    # used in pre-solver loops
        self._scale = 1.        # scaling factor

    def get_LP(self):
        return self.g, self.A, self.b, self.C, self.d, self.lb, self.ub

    def get_QP(self):
        return self.H, self.g, self.A, self.b, -self.C, -self.d, self.lb, self.ub

    def presolve_LP(self):
        """
        Pre-solves a linear programming problem of the form::

            min  g'x
            s.t. Ax = b
            s.t. Cx <= d
            s.t. lb <= x <= ub

        References
        ----------
        [1] Andersen, E. D., Andersen, K. D. (1993). Presolving in Linear Programming. Mathematical Programming 71.
            Page: 235
        [2] Shawwa, N. E., Peshko, O., Olvera, A., Li, J. Preprocessing Techniques.
        [3] Linear Programming Algorithms. MathWorks.
            Link: https://se.mathworks.com/help/optim/ug/linear-programming-algorithms.html
        """

        # remove fixed variables
        self._remove_fixed_variables()

        # iteratively reduce system
        while self._reduced:

            self._reduced = False

            # remove singleton rows
            self._remove_singleton_rows_eq()
            self._remove_singleton_rows_iq()

            # remove forcing constraints
            self._remove_forcing_constraints()     # equality
            self._tighten_forcing_constraints()    # inequality

            # remove duplicate rows
            self.A, self.b, self.idx_e = self._remove_duplicate_rows(self.A, self.b, self.idx_e, eq=True)
            self.C, self.d, self.idx_i = self._remove_duplicate_rows(self.C, self.d, self.idx_i, eq=False)

        # remove zero rows
        self.A, self.b, self.idx_e = self._remove_zero_rows(self.A, self.b, self.idx_e, eq=True)
        self.C, self.d, self.idx_i = self._remove_zero_rows(self.C, self.d, self.idx_i, eq=False)

        # remove zero columns
        self._remove_zero_columns_LP()

        # shift bounds
        self._shift_lower_bounds()

        # scale system
        self._scaling_equilibration()

    def presolve_QP(self):
        """
        Pre-solves a quadratic programming problem of the form::

            min  1/2 x'Hx + g'x
            s.t. Ax = b
            s.t. Cx <= d
            s.t. lb <= x <= ub

        Notice the inequality constraints are reversed relative to
        most QP algorithms in literature. This is to re-use pre-solver
        methods from LP.

        References
        ----------
        [1] Andersen, E. D., Andersen, K. D. (1993). Presolving in Linear Programming. Mathematical Programming 71.
            Page: 235
        [2] Shawwa, N. E., Peshko, O., Olvera, A., Li, J. Preprocessing Techniques.
        [3] Quadratic Programming Algorithms. MathWorks.
            Link: https://se.mathworks.com/help/optim/ug/quadratic-programming-algorithms.html
        """

        # remove fixed variables
        self._remove_fixed_variables()

        # iteratively reduce system
        while self._reduced:

            self._reduced = False

            # remove singleton rows
            self._remove_singleton_rows_eq()
            self._remove_singleton_rows_iq()

            # remove forcing constraints
            self._remove_forcing_constraints()      # equality
            self._tighten_forcing_constraints()     # inequality

            # remove duplicate rows
            self.A, self.b, self.idx_e = self._remove_duplicate_rows(self.A, self.b, self.idx_e, eq=True)
            self.C, self.d, self.idx_i = self._remove_duplicate_rows(self.C, self.d, self.idx_i, eq=False)

        # remove zero rows
        self.A, self.b, self.idx_e = self._remove_zero_rows(self.A, self.b, self.idx_e, eq=True)
        self.C, self.d, self.idx_i = self._remove_zero_rows(self.C, self.d, self.idx_i, eq=False)

        # remove zero columns
        self._remove_zero_columns_QP()

        # shift bounds
        self._shift_lower_bounds()

        # scale system
        self._scaling_equilibration()

    def postsolve(self, x):
        """
        Post-solves a linear programming problem of the form::

            min  g'x
            s.t. Ax = b
            s.t. Cx <= d
            s.t. lb <= x <= ub

        References
        ----------
        [1] Andersen, E. D., Andersen, K. D. (1993). Presolving in Linear Programming. Mathematical Programming 71.
            Page: 235
        [2] Shawwa, N. E., Peshko, O., Olvera, A., Li, J. Preprocessing Techniques.

        Parameters
        ----------
        x : array_like, shape (n-r,)
            Solution to the reduced LP.

        Returns
        -------
        x : array_like, shape (n,)
            Solution to the full LP.
        f : float
            Optimal function value of the full LP.
        """

        # scale the solved solution back to the original problem domain
        x *= self._scale

        # shift the solved solution back to the original problem domain
        x = self._shift_x_by_bound(x)

        # merge pre-solved and solved solution
        self.x[self.idx_x] = x

        # calculate optimal function value
        f = self.g.T @ x + self.k

        if self.H is not None:
            f += x.T @ self.H @ self.x

        return x, f

    def is_solved(self):
        return not self.idx_x.size

    def _shift_lower_bounds(self):
        """
        Shifts the lower bounds of variables to 0 prior to solving LP.
        """

        mask = self.lb != -np.inf

        if np.any(mask):
            self.b -= self.A[:, mask] @ self.lb[mask]
            self.d -= self.C[:, mask] @ self.lb[mask]
            self.ub[mask] -= self.lb[mask]

    def _shift_x_by_bound(self, x):
        """
        Shifts the resulting LP solution by the lower bounds, if lower bounds were shifted during presolve.

        Parameters
        ----------
        x : array_like, shape (n-r,)
            Solution to the reduced LP.

        Returns
        -------
        x : array_like, shape (n-r,)
            Solution to the reduced LP shifted back to original the domain w.r.t. lower bounds.
        """

        mask = self.lb != -np.inf

        if np.any(mask):
            x[mask] += self.lb[mask]

        return x

    def _scaling_equilibration(self):
        """
        Performs an equilibration scaling of the programming problem to improve numerical stability.
        The equilibration does not use the max(A), but rather the sqrt(max(A)) similar to [1].

        References
        ----------
        [1] Zhang, Y. (1996). Solving Large-Scale Linear Programs by Interior-Point Methods under the MATLAB Environment.
        """

        # scaling is only done if the system is poorly scaled.
        absnz = np.abs(np.block([self.A[np.nonzero(self.A)], self.C[np.nonzero(self.C)]]))
        max_scale = np.amin(absnz) / np.amax(absnz) if absnz.size else np.inf

        if max_scale >= 1e-4:
            return

        # calculate column scaling
        A_max = np.amax(np.abs(self.A), axis=0) if self.A.size else 0.
        C_max = np.amax(np.abs(self.C), axis=0) if self.C.size else 0.
        col_scale = np.sqrt(np.maximum(A_max, C_max))
        self._scale = 1. / col_scale

        # calculate row scaling
        A_row_scale = np.sqrt(np.amax(np.abs(self.A), axis=1))
        C_row_scale = np.sqrt(np.amax(np.abs(self.C), axis=1))

        # scale columns
        self.ub *= col_scale
        self.g /= col_scale
        col_scale = np.diag(1. / col_scale)
        self.A = self.A @ col_scale
        self.C = self.C @ col_scale
        del col_scale

        # scale rows
        self.b /= A_row_scale
        self.d /= C_row_scale
        self.A = np.diag(1. / A_row_scale) @ self.A
        self.C = np.diag(1. / C_row_scale) @ self.C

        # what is the logic of the following?
        norm = la.norm(np.block([self.b, self.d]))
        if norm > 0.:

            q = np.median([1., la.norm(self.g) / norm, 1e8])

            if q > 10.:
                self.A *= q
                self.C *= q
                self.b *= q
                self.d *= q

    def _remove_zero_rows(self, A, b, idx, eq=True):
        """
        Removes zero-rows from the constraints. If the corresponding results vector is non-empty,
        the system is checked primal-infeasibility and an error is raised.

        Parameters
        ----------
        A : array_like, shape (m, n)
            System matrix with coefficients for the equality/inequality constraints.
        b : array_like, shape (m,)
            Results vector of the equality/inequality constraints.
        idx : array_like, shape (m,)
            Index tracker of remaining equality/inequality constraints
        eq : bool
            True if equality constraints, False if inequality constraints.

        Returns
        -------
        A : array_like, shape (m-r, n)
            Reduced system matrix with coefficients for the equality/inequality constraints.
        b : array_like, shape (m-r,)
            Reduced results vector of the equality/inequality constraints.
        idx : array_like, shape (m-r,)
            Reduced index tracker of remaining equality/inequality constraints
        """

        mask = _non_zero_rows(A, tol=self._tol)
        A = A[mask, :]

        if eq:
            if not (np.allclose(b[~mask], 0, atol=self._tol)):
                raise InfeasibleProblemError(-2, 'LP is primal-infeasible due to zero-row in `A`'
                                                 ' with corresponding non-zero row in `b`.')

        else:
            if np.any(b[~mask] < -self._tol):
                raise InfeasibleProblemError(-2, 'LP is primal-infeasible due to zero-row in `C`'
                                                 ' with corresponding negative row in `d`.')

        b = b[mask]
        idx = idx[mask]

        return A, b, idx

    def _remove_zero_columns_LP(self):
        """
        Removes redundant columns from an LP. If the corresponding variable is unbounded,
        then an error is raised due to dual-infeasibility of the system.
        """

        mask = self._non_zero_columns_mask()
        x = self._assign_variables_empty(mask)
        self._remove_variables(x, mask)

    def _remove_zero_columns_QP(self):
        """
        Removes redundant columns from a QP. If the corresponding variable is unbounded,
        then an error is raised due to dual-infeasibility of the system.
        """

        mask = self._non_zero_columns_mask()
        me = np.full_like(mask, False)
        ms = np.full_like(mask, False)

        # the structure of H has to be investigated for QPs, prior to removing columns.
        # H is per definition symmetric, so only have to check row or column, not both
        for i in np.nonzero(~mask):

            # check if all coefficients are zero
            if np.allclose(self.H[i, :], 0, atol=self._tol):

                me[i] = True

            # check if all coefficients are zero except for H[i, i]
            elif np.abs(self.H[i, i]) > self._tol:

                nz = np.abs(self.H[i, :]) < self._tol
                nz = np.delete(nz, i)

                if np.all(nz):

                    ms[i] = True

        # reduce the programming problem
        if np.any(me):
            xe = self._assign_variables_empty(me)
            self._remove_variables(xe, me)

        if np.any(ms):
            xs = self._assign_variables_single(ms)
            self._remove_variables(xs, ms)

    def _remove_fixed_variables(self):
        """
        Removes fixed variables from an LP.
        """

        # mask of non-fixed variables
        mask = np.abs(self.ub - self.lb) > self._tol
        self._remove_variables(self.lb[~mask], mask)

    def _remove_forcing_constraints(self):
        """
        Remove forcing constraints.

        References
        ----------
        [1] Andersen, E. D., Andersen, K. D. (1993). Presolving in Linear Programming. Mathematical Programming 71.
            Page: 226
        """

        m, n = self.A.shape

        # calculate the lower and upper constraint bounds
        g, h, P, M = self._calculate_constraint_bounds(self.A)

        # test for inequality: h < b or b > g
        if np.any((h < self.b) | (g > self.b)):
            raise InfeasibleProblemError(-2, 'LP is primal-infeasible due to '
                                             'an infeasible forcing constraint '
                                             'of the equality system.')

        else:

            g_mask = np.abs(g - self.b) < self._tol
            h_mask = np.abs(h - self.b) < self._tol
            idx = []

            # lower forcing constraint
            if np.any(g_mask):
                ig = np.argwhere(g_mask).flatten()
                for i in ig:

                    for j in P[i]:
                        self.x[j] = self.lb[j]
                        idx.append(j)

                    for j in M[i]:
                        self.x[j] = self.ub[j]
                        idx.append(j)

            # upper forcing constraint
            if np.any(h_mask):
                ih = np.argwhere(h_mask).flatten()
                for i in ih:

                    for j in P[i]:
                        self.x[j] = self.ub[j]
                        idx.append(j)

                    for j in M[i]:
                        self.x[j] = self.lb[j]
                        idx.append(j)

            if np.any(idx):
                # ensure index list is unique
                idx = list(set(idx))

                mask = _idx2mask(idx, n)
                self._remove_variables(self.x[idx], mask)

                self._reduced = True

    def _tighten_forcing_constraints(self):
        """
        Checks for infeasibility of the inequality constraints and tighten them if possible.
        """

        # calculate the lower and upper constraint bounds
        g, h, _, _ = self._calculate_constraint_bounds(self.C)

        # if lower constraint bound is larger than 'd', then the problem is infeasible.
        if np.any(g > self.d):
            raise InfeasibleProblemError(-2, 'LP is primal-infeasible due to '
                                             'an infeasible forcing constraint '
                                             'of the inequality system.')

        # if upper constraint bound is more restrictive than 'd', then tighten 'd'.
        h_mask = h < self.d
        self.d[h_mask] = h[h_mask]

        if np.any(h_mask):
            self._reduced = True

    def _remove_singleton_rows_eq(self):
        """
        Removes singleton rows from equality constraints.
        """

        x, i, j = self._get_singleton_rows(self.A, self.b)

        if np.any((x < self.lb[j]) | (x > self.ub[j])):
            raise InfeasibleProblemError(-2, 'LP is primal-infeasible due to '
                                             'an infeasible singleton row.')

        # reduce the problem dimension
        mask = _idx2mask(j, self.g.size)
        if np.any(~mask):
            self._remove_variables(x, ~mask)

        # remove constraints
        if np.any(i):
            self.A = self.A[~i, :]
            self.b = self.b[~i]
            self.idx_e = self.idx_e[~i]

            self._reduced = True

    def _remove_singleton_rows_iq(self):
        """
        Change singleton rows from inequality constraints to upper bounds
        or remove constraint if the existing upper bound is more restrictive.
        """

        x, ii, jj = self._get_singleton_rows(self.C, self.d)

        for x_, i, j in zip(x, *np.nonzero(ii), jj):

            if self.lb[j] <= x_:

                if x_ <= self.ub[j]:  # tighten the bound

                    if self.C[i, j] > 0.:

                        self.ub[j] = x_

                    else:

                        self.lb[j] = x_

            else:

                raise InfeasibleProblemError(-2, 'LP is primal-infeasible due to '
                                                 'an infeasible singleton row.')

        # remove constraints
        if np.any(ii):
            self.C = self.C[~ii, :]
            self.d = self.d[~ii]
            self.idx_i = self.idx_i[~ii]

            self._reduced = True

    def _remove_duplicate_rows(self, A, b, idx, eq=True):
        """
        Removes redundant rows from the constraints. If the corresponding results vector is non-empty,
        then an error is raised due to primal-infeasibility of the system.

        Parameters
        ----------
        A : array_like, shape (m, n)
            System matrix with coefficients for the equality/inequality constraints.
        b : array_like, shape (m,)
            Results vector of the equality/inequality constraints.
        idx : array_like, shape (m,)
            Index tracker of remaining equality/inequality constraints
        eq : bool
            True if equality constraints, False if inequality constraints.

        Returns
        -------
        A : array_like, shape (m-r, n)
            Reduced system matrix with coefficients for the equality/inequality constraints.
        b : array_like, shape (m-r,)
            Reduced results vector of the equality/inequality constraints.
        idx : array_like, shape (m-r,)
            Reduced index tracker of remaining equality/inequality constraints
        """

        nz = _non_zero_count(A, axis=1, tol=self._tol)
        id_ = np.arange(nz.size)[nz > 1]

        # split row indices into lists with same number of non-zeroes
        snz = {}  # split_non_zero
        for i in id_:
            key = nz[i]

            if key in snz:
                snz[key].append(i)
            else:
                snz[key] = [i]

        # find rows with similar sparsity pattern
        sp = []  # sparsity_pattern
        for rows in snz.values():

            for i, ri in enumerate(rows):

                spi = np.nonzero(A[ri])

                for rk in rows[(i+1):]:

                    spk = np.nonzero(A[rk])

                    if np.array_equal(spi, spk):

                        sp.append((ri, rk))

        # check if the rows with similar sparsity pattern are duplicates
        dup = []
        for ri, rk in sp:

            nu = A[ri] / A[rk]

            if np.all(nu == nu[0]):

                ratio = b[ri] / b[rk]

                if eq:  # equality constraints

                    if ratio == nu[0]:

                        dup.append(rk)

                    else:

                        raise InfeasibleProblemError(-2, 'Problem is primal-infeasible due to '
                                                         'duplicate rows with varying `b`.')

                else:  # inequality constraints

                    if ratio == nu[0]:

                        dup.append(ri if (np.abs(ratio) < 1.) else rk)

        if dup:
            A = np.delete(A, dup, axis=0)
            b = np.delete(b, dup)
            idx = np.delete(idx, dup)

        return A, b, idx

    def _remove_variables(self, x, mask):
        """
        Removes a variable from the programming problem.

        Parameters
        ----------
        x : array_like, shape (n-r,)
            Vector of removed variables.
        mask : array_like, shape (n,)
            Mask of variables to keep in the programming problem.
        """

        if np.any(~mask):
            # update constant and constraint equations
            self.k += self.g[~mask] @ x
            self.b -= self.A[:, ~mask] @ x
            self.d -= self.C[:, ~mask] @ x

            # update pre-solved solution
            self.x[~mask] = x
            self.idx_x = self.idx_x[mask]

            # update programming problem
            self.g = self.g[mask]
            self.A = self.A[:, mask]
            self.C = self.C[:, mask]
            self.lb = self.lb[mask]
            self.ub = self.ub[mask]

            if self.H is not None:
                self.k += self.H[~mask, ~mask] @ (x ** 2.)
                self.g += 2. * (self.H[~mask][mask] @ x)

                self.H = self.H[mask, :]
                self.H = self.H[:, mask]

    def _get_singleton_rows(self, A, b):
        i = _non_zero_count(A, axis=1, tol=self._tol) == 1
        j = [int(np.argwhere(np.abs(a) > self._tol)) for a in A[i, :]]

        x = b[i] / A[i, j]

        return x, i, j

    def _calculate_constraint_bounds(self, A):
        """
        Calculate the upper and lower bounds of each constraint.

        Parameters
        ----------
        A : array_like, shape (m, n)
            Coefficient matrix of equality or inequality system.
        """

        m, n = A.shape

        # define the sets P and M
        js = np.arange(n)
        P = [js[A[i, :] > 0.] for i in range(m)]
        M = [js[A[i, :] < 0.] for i in range(m)]

        # calculate lower and upper term of constraint bounds
        r = range(m)
        gp = np.array([np.sum([A[i, j] * self.lb[j] for j in P[i]]) for i in r])
        gm = np.array([np.sum([A[i, j] * self.ub[j] for j in M[i]]) for i in r])
        hp = np.array([np.sum([A[i, j] * self.ub[j] for j in P[i]]) for i in r])
        hm = np.array([np.sum([A[i, j] * self.lb[j] for j in M[i]]) for i in r])

        # calculate lower and upper constraint bounds
        g = gp + gm
        h = hm + hp

        return g, h, P, M

    def _non_zero_columns_mask(self):
        # if there are no constraints, then return
        n = self.g.size
        me = self.b.size
        mi = self.d.size

        # find variables that are unbounded both in eq. and ineq.
        maskA = _non_zero_columns(self.A, tol=self._tol) if me else np.full((n,), True, dtype=bool)
        maskC = _non_zero_columns(self.C, tol=self._tol) if mi else np.full((n,), True, dtype=bool)

        return maskA & maskC

    def _assign_variables_empty(self, mask):
        """
        Determines the values variables which are not included in any of the constraints (zero-columns in A and C),
        and no included in the quadratic matrix (if QP). The values are determined based on the variables bounds.

        Parameters
        ----------
        mask : array_like, shape (n,)
            Mask of variables to keep in the programming problem.

        Returns
        -------
        x : array_like, shape (r,)
            The values to assign to the new fixed variables.
        """

        x = np.zeros_like(mask, dtype=np.float64)

        # if c[~mask] == 0, the variable can be set to an arbitrary value
        # within its bounds and removed from the problem.
        g0m = (np.abs(self.g) < self._tol) & ~mask
        x[g0m] = (self.lb[g0m] + self.ub[g0m]) / 2.

        # if c[~mask] > 0, and lb[~mask] == -np.inf then the problem
        # is dual infeasible. If lb[~mask] is finite, then the columns
        # can be removed and x[mask] set to lb[mask]
        glm = (self.g > self._tol) & ~mask

        if np.any(self.lb[glm] == -np.inf):
            raise InfeasibleProblemError(-3, 'LP is dual-infeasible due to zero-column in `A` or `C`'
                                             ' with corresponding row in `g`>0 and lb=-np.inf.')

        x[glm] = self.lb[glm]

        # if c[~mask] < 0, and ub[~mask] == np.inf then the problem
        # is dual infeasible. If ub[~mask] is finite, then the columns
        # can be removed and x[mask] set to ub[mask]
        gum = (self.g < -self._tol) & ~mask

        if np.any(self.ub[gum] == np.inf):
            raise InfeasibleProblemError(-3, 'LP is dual-infeasible due to zero-column in `A`` or `C`'
                                             ' with corresponding row in `g`<0 and ub=np.inf.')

        x[gum] = self.ub[gum]

        return x[~mask]

    def _assign_variables_single(self, mask):
        """
        Determines the values variables which are not included in any of the constraints (zero-columns in A and C),
        and which include a single quadratic coefficient in H[i, i].

        Parameters
        ----------
        mask : array_like, shape (n,)
            Mask of variables to keep in the programming problem.

        Returns
        -------
        x : array_like, shape (r,)
            The values to assign to the new fixed variables.
        """

        lb = self.lb[mask]
        ub = self.ub[mask]
        c = -self.g[mask] / self.H[mask, mask]
        x = np.where((lb <= c) & (c >= ub), c, np.where(c < lb, lb, ub))
        return x


def _non_zero_count(A, axis=1, tol=1e-13):
    """
    Count all non-zero elements along the rows (axis=1) or columns (axis=0) in `A`.

    References
    ----------
    [1] https://github.com/scipy/scipy/blob/master/scipy/optimize/_remove_redundancy.py

    Parameters
    ----------
    A : array_like, shape (m, n)
        System matrix with coefficients for the constraints.
    axis : int, optional
        Axis along which to check for all zeros, row=1, column=0.
    tol : float, optional
        Tolerance below which values are assumed 0.
    Returns
    -------
    count : array_like, shape (m,)
        Count of all non-zero rows or columns in `A`.
    """

    return np.array((np.abs(A) > tol).sum(axis=axis)).flatten()


def _non_zero_mask(A, axis=1, tol=1e-13):
    """
    Creates a mask of all non-zero rows (axis=1) or columns (axis=0) in `A`.

    Parameters
    ----------
    A : array_like, shape (m, n)
        System matrix with coefficients for the constraints.
    axis : int, optional
        Axis along which to check for all zeros, row=1, column=0.
    tol : float, optional
        Tolerance below which values are assumed 0.
    Returns
    -------
    mask : array_like, shape (m,)
        Mask of all non-zero rows or columns in `A`.
    """

    return _non_zero_count(A, axis=axis, tol=tol) > 0.


def _non_zero_rows(A, tol=1e-13):
    return _non_zero_mask(A, axis=1, tol=tol)


def _non_zero_columns(A, tol=1e-13):
    return _non_zero_mask(A, axis=0, tol=tol)


def _mask2idx(mask):
    return np.nonzero(mask)


def _idx2mask(idx, n):
    mask = np.arange(n)
    return mask[idx]
