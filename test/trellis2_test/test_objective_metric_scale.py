import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from metric_scale import _trimmed_scaled_icp, _pcd, VOXEL
from test_roundtrip import make_meshes, unitize, simulate_view

def setup(mesh_metric, seed):
    r = np.random.default_rng(seed)
    canon = unitize(mesh_metric)
    s_true = mesh_metric.extents.max()
    R_true = Rotation.random(random_state=seed).as_matrix()
    t_true = np.array([0.0, 0.0, r.uniform(0.45, 0.7)])
    placed = canon.copy()
    T = np.eye(4); T[:3, :3] = s_true * R_true; T[:3, 3] = t_true
    placed.apply_transform(T)
    P_obs = simulate_view(placed)
    P_obs_ds = np.asarray(_pcd(P_obs, VOXEL).points)
    P_mesh = np.asarray(canon.sample(30_000))
    return P_obs_ds, P_mesh, s_true, R_true, t_true

print("A) ICP seeded EXACTLY at ground truth -- does it stay?")
print(f"{'mesh':12s} {'s_true':>8s} {'s_after':>9s} {'drift':>9s} {'rmse':>8s}")
for name, m in make_meshes().items():
    P_obs, P_mesh, s_t, R_t, t_t = setup(m, 1)
    s, R, t, rmse = _trimmed_scaled_icp(P_obs, P_mesh, s_t, R_t, t_t)
    print(f"{name:12s} {s_t*1000:8.1f} {s*1000:9.1f} "
          f"{100*(s/s_t-1):+8.2f}% {rmse*1000:7.2f}mm")

print("\nB) cost(s) sweep with R,t re-solved -- is the objective monotone?")
def one_sided_cost(P_obs, P_mesh, s, R, t, trim=0.75):
    Q = (P_obs - t) @ R / s
    d, _ = cKDTree(P_mesh).query(Q, k=1)
    d = d * s
    return np.sqrt((d[d <= np.quantile(d, trim)] ** 2).mean())

P_obs, P_mesh, s_t, R_t, t_t = setup(make_meshes()["mug"], 1)
print(f"{'s/s_true':>9s} {'trimmed rmse (one-sided)':>26s}")
for k in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
    s = k * s_t
    # re-center so the observed patch sits on the (rescaled) mesh
    t = t_t + (s_t - s) * (R_t @ P_mesh.mean(0))
    c = one_sided_cost(P_obs, P_mesh, s, R_t, t)
    print(f"{k:9.2f} {c*1000:22.2f}mm")

print("\nC) how good is the naive RMS-ratio seed on its own?")
print(f"{'mesh':12s} {'s_true':>8s} {'r_obs/r_msh':>12s} {'err':>9s}")
for name, m in make_meshes().items():
    P_obs, P_mesh, s_t, _, _ = setup(m, 1)
    r_o = np.sqrt(P_obs.var(0).sum()); r_m = np.sqrt(P_mesh.var(0).sum())
    print(f"{name:12s} {s_t*1000:8.1f} {r_o/r_m*1000:12.1f} "
          f"{100*((r_o/r_m)/s_t-1):+8.2f}%")