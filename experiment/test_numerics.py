"""Small independent analytic checks, including the draft's Appendix B fixture."""
import unittest
import numpy as np
from src.maze import Maze, make_maze
from src.harmonic import HarmonicBasis, build_basis
from src.ctmc import CTMCBuilder
from src.probabilities import evaluate
from src.simulate import simulate_ctmc
from src.adjoint import backward_adjoint
from src.wall_safety import wall_matrices,audit_wall,repulsive_constraint
from src.wall_safety import interpolated_face_normals
from src.simulate import DriftInterpolator,simulate_sde


class NumericalChecks(unittest.TestCase):
    def appendix_b(self):
        occupied = np.zeros((5,5),bool)
        occupied[[0,-1],:] = True
        occupied[:,[0,-1]] = True
        goal = np.zeros_like(occupied); goal[1:4,0] = True; occupied[goal] = False
        maze = Maze('appendix_b',occupied,goal,(2,2),4.)
        values = maze.x[...,None]
        gradients = np.zeros((5,5,2,1)); gradients[:,:,0,0] = 1.
        basis = HarmonicBasis(maze,values,gradients,np.zeros_like(values),0.,0.)
        builder = CTMCBuilder(basis,sigma=np.sqrt(2.),gain=1.)
        return builder, builder.build(np.array([1.]))

    def test_paper_appendix_b_exact_probabilities(self):
        builder,chain = self.appendix_b()
        result = evaluate(chain,builder.maze.start,6.)
        self.assertAlmostEqual(result['p'],0.47720364741641336,places=12)
        self.assertAlmostEqual(result['q'],0.5227402075697434,places=12)
        np.testing.assert_allclose(chain.Q@np.ones(9)+chain.wall+chain.goal,0,atol=1e-13)
        np.testing.assert_allclose(chain.rates,[ [1.,2.,1.,1.] ]*9,atol=1e-13)

    def test_independent_ctmc_monte_carlo(self):
        builder,chain = self.appendix_b()
        mc = simulate_ctmc(chain,builder.maze.start,6.,n=30000,seed=9182)
        self.assertLess(abs(mc['collision_rate']-.4772036474),.012)
        self.assertLess(abs(mc['arrival_rate']-.5227402076),.012)

    def test_harmonic_partition_and_time_monotonicity(self):
        maze = make_maze('u_maze',24)
        basis = build_basis(maze,8)
        self.assertLess(basis.residual,1e-12)
        self.assertTrue(np.all(basis.values[maze.interior]>=-1e-12))
        self.assertTrue(np.all(basis.values[maze.interior].sum(axis=1)<=1+1e-12))
        chain = CTMCBuilder(basis).build(np.arange(1.,9.))
        short = evaluate(chain,maze.start,7.)
        long = evaluate(chain,maze.start,15.)
        self.assertLessEqual(short['q'],long['q']+1e-12)
        self.assertLessEqual(long['q']+long['p'],1.+1e-12)

    def test_adjoint_against_independent_finite_differences(self):
        maze=make_maze('open',18)
        builder=CTMCBuilder(build_basis(maze,12,'local'))
        theta=np.linspace(1.,6.,12)
        value=backward_adjoint(builder,theta,7.,steps=32)
        for index in (0,4,9):
            plus=theta.copy(); plus[index]+=1e-5
            minus=theta.copy(); minus[index]-=1e-5
            vp=backward_adjoint(builder,plus,7.,steps=32)
            vm=backward_adjoint(builder,minus,7.,steps=32)
            self.assertAlmostEqual(value['gp'][index],(vp['p']-vm['p'])/2e-5,places=6)
            self.assertAlmostEqual(value['gq'][index],(vp['q']-vm['q'])/2e-5,places=6)

    def test_wall_constraints_enforce_actual_velocity_direction(self):
        basis=build_basis(make_maze('u_maze',24),50,'local')
        theta=np.full(50,8.)
        constraint=repulsive_constraint(basis)
        self.assertGreaterEqual(float(np.min(constraint.A@theta-constraint.lb)),-1e-10)
        self.assertTrue(audit_wall(basis,theta)['repulsive_at_grid_faces'])
        matrix=wall_matrices(basis)
        v,u=basis.field(theta)
        r,c,wr,wc=matrix['faces'].T
        dc=wc-c; dr=wr-r
        np.testing.assert_allclose(matrix['jump']@theta,v[wr,wc]-v[r,c],atol=1e-12)
        np.testing.assert_allclose(matrix['inward']@theta,-u[r,c,0]*dc-u[r,c,1]*dr,atol=1e-12)

    def test_wall_interpolation_excludes_solid_interior_gradients(self):
        basis=build_basis(make_maze('u_maze',24),50,'contour')
        theta=np.linspace(1.,6.,50)
        _,drift=basis.field(theta)
        rows,positions,normals=interpolated_face_normals(basis)
        correct=DriftInterpolator(basis.maze,drift)(positions)
        corrupt=drift.copy(); corrupt[~basis.maze.interior]=[1e6,-1e6]
        np.testing.assert_allclose(DriftInterpolator(basis.maze,corrupt)(positions),correct,atol=1e-12)
        np.testing.assert_allclose(rows@theta,-np.sum(correct*normals,axis=1),atol=1e-12)

    def test_contour_basis_partition_and_independent_wall_components(self):
        basis=build_basis(make_maze('u_maze',32),80,'contour')
        data=basis.boundary_metadata
        self.assertEqual(sum(data['knots_per_contour']),80)
        self.assertEqual(len(data['knots_per_contour']),2)
        self.assertLess(data['contour_partition_error'],1e-14)
        self.assertLess(basis.residual,1e-12)
        wall=np.any(basis.boundary_weights>0,axis=2)
        self.assertEqual(np.linalg.matrix_rank(basis.boundary_weights[wall]),80)

    def test_integer_deadline_does_not_truncate_fractional_exit_times(self):
        builder,_=self.appendix_b()
        metrics,paths=simulate_sde(builder.basis,np.array([1.]),0.,1.,deadline=5,
                                    horizon=10,n=1,dt=.01,save_paths=0)
        self.assertTrue(np.issubdtype(paths['exit_times'].dtype,np.floating))
        self.assertAlmostEqual(paths['exit_times'][0],1.5,delta=.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
