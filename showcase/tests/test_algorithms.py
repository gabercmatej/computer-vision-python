"""Analytical tests with known answers, independent of portfolio images."""
import unittest
import numpy as np
from cvportfolio import processing, filters, edges, features, geometry, pca

class AlgorithmTests(unittest.TestCase):
    def test_otsu_separates_bimodal_image_deterministically(self):
        image=np.array([[.1,.1,.9,.9],[.1,.1,.9,.9]])
        t=processing.otsu_threshold(image)
        self.assertGreaterEqual(t,.1); self.assertLess(t,.9)
        self.assertEqual(t,processing.otsu_threshold(image))

    def test_ransac_rejects_outliers_and_is_repeatable(self):
        rng=np.random.default_rng(17)
        a=rng.uniform(-100,100,(30,2));b=a+np.array([12.,-8.])
        b[-8:]=rng.uniform(-100,100,(8,2))
        matches=np.c_[np.arange(30),np.arange(30)]
        H,inliers=features.ransac_homography(a,b,matches,threshold=.1,k=150)
        H2,inliers2=features.ransac_homography(a,b,matches,threshold=.1,k=150)
        self.assertEqual(len(inliers),22)
        np.testing.assert_allclose(H,[[1,0,12],[0,1,-8],[0,0,1]],atol=1e-8)
        np.testing.assert_allclose(H,H2);np.testing.assert_array_equal(inliers,inliers2)

    def test_histogram_includes_white_and_normalizes(self):
        np.testing.assert_allclose(processing.myhist(np.array([0.,.25,.5,1.]),4), [.25,.25,.25,.25])

    def test_convolution_matches_numpy_away_from_padding(self):
        x=np.arange(12,dtype=float)**2; k=np.array([.2,.5,.3])
        np.testing.assert_allclose(filters.simple_convolution_fix(x,k)[1:-1],np.convolve(x,k,'same')[1:-1])

    def test_hysteresis_keeps_connected_weak_edges_only(self):
        a=np.zeros((5,7)); a[2,1]=1; a[2,2:4]=.4; a[0,6]=.4
        result=edges.hysteresis(a,.3,.8)
        self.assertEqual(int(result.sum()),3)

    def test_nonmax_suppression_follows_positive_diagonal(self):
        m=np.zeros((5,5)); m[2,2]=2; m[1,1]=3
        angles=np.full((5,5),np.pi/4)
        self.assertEqual(edges.non_max_suppression(m,angles)[2,2],0)

    def test_homography_recovers_known_transform(self):
        a=np.array([[0,0],[1,0],[1,1],[0,1],[.3,.7],[2,3]],float)
        H=np.array([[1.2,.1,10],[.05,.9,-3],[.002,.001,1]])
        q=(H@np.c_[a,np.ones(len(a))].T).T; b=q[:,:2]/q[:,2:]
        fitted=features.estimate_homography(np.c_[a,b])
        np.testing.assert_allclose(fitted,H,atol=1e-9)

    def test_homography_rejects_degenerate_points(self):
        with self.assertRaises(ValueError): features.estimate_homography(np.ones((4,4)))

    def test_empty_feature_matches_are_valid(self):
        matches,a,b=features.find_matches(np.zeros((40,40)),np.zeros((40,40)))
        self.assertEqual(matches.shape,(0,2))

    def test_triangulation_recovers_known_3d(self):
        P1=np.c_[np.eye(3),np.zeros(3)]; P2=np.c_[np.eye(3),np.array([-1,0,0])]
        X=np.array([[.1,.2,3],[1,-.5,5],[-.4,.3,7]])
        def project(P):
            q=(P@np.c_[X,np.ones(len(X))].T).T
            return q[:,:2]/q[:,2:]
        np.testing.assert_allclose(geometry.triangulate(project(P1),project(P2),P1,P2),X,atol=1e-10)

    def test_fundamental_matrix_epipolar_constraint(self):
        rng=np.random.default_rng(5); X=rng.normal(size=(20,3)); X[:,2]+=5
        a=X[:,:2]/X[:,2:]; shifted=X+np.array([1,.1,0]); b=shifted[:,:2]/shifted[:,2:]
        F=geometry.fundamental_matrix(a,b)
        self.assertLess(np.linalg.svd(F)[1][-1],1e-10)
        self.assertLess(geometry.reprojection_errors(F,a,b).max(),1e-8)

    def test_pca_reconstruction_and_orthonormality(self):
        rng=np.random.default_rng(7); X=rng.normal(size=(20,6))
        U,S,mean=pca.dual_pca(X)
        np.testing.assert_allclose(U.T@U,np.eye(U.shape[1]),atol=1e-8)
        np.testing.assert_allclose(U@(U.T@(X-mean))+mean,X,atol=1e-8)

    def test_pca_constant_data_has_no_components(self):
        U,S,mean=pca.dual_pca(np.ones((10,4)))
        self.assertEqual(U.shape,(10,0)); self.assertEqual(len(S),0)

    def test_feature_suppression_uses_vertical_neighbors(self):
        response=np.zeros((7,7)); response[3,3]=2; response[2,3]=3
        self.assertTrue(hasattr(features, 'spatial_nonmax'))
        selected=features.spatial_nonmax(response)
        self.assertEqual(selected[3,3],0)
        self.assertEqual(selected[2,3],3)

    def test_gaussian_filter_preserves_constant(self):
        np.testing.assert_allclose(filters.gaussfilter(np.ones((30,30)),2),1,atol=1e-10)

    def test_histogram_distance_known_extremes(self):
        a=np.array([1.,0]); b=np.array([0.,1])
        self.assertAlmostEqual(filters.compare_histograms(a,b,'Hellinger'),1)
        self.assertAlmostEqual(filters.compare_histograms(a,a,'Hellinger'),0)

if __name__=='__main__': unittest.main()
