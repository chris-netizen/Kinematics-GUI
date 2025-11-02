# core/kinematics.py
from collections import defaultdict
import math
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import mplstereonet

PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI
STRIKE_TOLERANCE = 20.0
DAYLIGHT_TOLERANCE = 90.0


class KinematicAnalyzer:
    def __init__(self, data=None, file_path=None, n_clusters=3):
        self.df = None
        self.results = None
        self.clusters = {}
        self.cluster_means = {}
        self.slope_dip_dir = None
        self.slope_dip = None
        self.friction = None
        self.n_clusters = n_clusters
        self.cluster_labels = None
        if data is not None:
            self.df = data.copy()
        if file_path:
            self.load_data(file_path)

    def load_data(self, path):
        df = pd.read_csv(path)
        self.df = df
        self._validate_data()

    def _validate_data(self):
        if self.df is None:
            raise ValueError("No data loaded")
        required = ['Dip_Direction', 'Dip_Angle']
        if not all(c in self.df.columns for c in required):
            raise ValueError(f"Missing required columns: {required}")
        # numeric & bounds
        if not (self.df['Dip_Direction'].between(0, 360).all() and self.df['Dip_Angle'].between(0, 90).all()):
            raise ValueError("Angles out of expected bounds")

    def set_parameters(self, slope_dip_dir, slope_dip, friction, n_clusters=3):
        self.slope_dip_dir = float(max(0, min(360, slope_dip_dir)))
        self.slope_dip = float(max(0, min(90, slope_dip)))
        self.friction = float(max(0, min(45, friction)))
        self.n_clusters = int(n_clusters)

    def _clamp_angle(self, val, max_val=360):
        return max(0, min(max_val, val))

    def azimuth_diff(self, a1, a2):
        diff = abs(a1 - a2) % 360
        return min(diff, 360 - diff)

    def plane_to_normal(self, dip_dir, dip):
        pole_plunge = 90.0 - dip
        theta = (90.0 - pole_plunge) * DEG_TO_RAD
        phi = dip_dir * DEG_TO_RAD
        nx = math.sin(theta) * math.cos(phi)
        ny = math.sin(theta) * math.sin(phi)
        nz = math.cos(theta)
        norm = math.sqrt(nx**2 + ny**2 + nz**2)
        if norm == 0:
            return np.array([0.0, 0.0, 1.0])
        return np.array([nx, ny, nz]) / norm

    def line_from_vector(self, vec):
        vx, vy, vz = vec
        horiz = math.sqrt(vx**2 + vy**2)
        if horiz == 0:
            return 0.0, 90.0 if vz < 0 else 0.0
        plunge = math.degrees(math.atan2(-vz, horiz))
        trend = math.degrees(math.atan2(vy, vx)) % 360
        return trend, plunge

    def compute_intersection(self, plane1, plane2):
        n1 = self.plane_to_normal(plane1['dip_dir'], plane1['dip'])
        n2 = self.plane_to_normal(plane2['dip_dir'], plane2['dip'])
        inter_vec = np.cross(n1, n2)
        if np.linalg.norm(inter_vec) == 0:
            return None, None
        trend, plunge = self.line_from_vector(inter_vec)
        if plunge > 90:
            trend = (trend + 180) % 360
            plunge = 180 - plunge
        return trend, plunge

    def check_planar_failure(self, plane):
        dip_dir, dip = plane['dip_dir'], plane['dip']
        if self.slope_dip_dir is None or self.slope_dip is None or self.friction is None:
            return False, 0.0
        if self.azimuth_diff(dip_dir, self.slope_dip_dir) > STRIKE_TOLERANCE:
            return False, 0.0
        if dip >= self.slope_dip:
            return False, 0.0
        cone_angle = self.slope_dip - self.friction
        if dip > cone_angle:
            fs = math.tan(math.radians(self.friction)) / \
                math.tan(math.radians(dip))
            return fs < 1.0, fs
        return False, 0.0

    def check_wedge_failure(self, inter_trend, inter_plunge):
        if self.slope_dip_dir is None or self.slope_dip is None or self.friction is None:
            return False, 0.0
        if inter_plunge >= self.slope_dip:
            return False, 0.0
        if self.azimuth_diff(inter_trend, self.slope_dip_dir) > DAYLIGHT_TOLERANCE:
            return False, 0.0
        cone_plunge = self.slope_dip - self.friction
        if inter_plunge > cone_plunge:
            fs = math.tan(math.radians(self.friction)) / \
                math.tan(math.radians(inter_plunge))
            return fs < 1.0, fs
        return False, 0.0

    def check_toppling_failure(self, plane):
        dip_dir, dip = plane['dip_dir'], plane['dip']
        if self.slope_dip_dir is None or self.slope_dip is None or self.friction is None:
            return False, 0.0
        if dip < 70:
            return False, 0.0
        if self.azimuth_diff(dip_dir + 180, self.slope_dip_dir) > 20:
            return False, 0.0
        if dip > self.slope_dip + self.friction:
            fs = (self.slope_dip + self.friction - dip) / 10
            return fs < 1.0, fs
        return False, 0.0

    def perform_clustering(self):
        if self.df is None or len(self.df) == 0:
            self.cluster_labels = None
            self.clusters = {}
            self.cluster_means = {}
            return
        # use pole coordinates (azimuth rad, plunge rad) as features
        poles = []
        for _, row in self.df.iterrows():
            pole_az = (row['Dip_Direction'] + 180) % 360
            pole_pl = 90 - row['Dip_Angle']
            poles.append([math.radians(pole_az), math.radians(pole_pl)])
        X = np.array(poles)
        n_clust = min(self.n_clusters, len(X))
        if n_clust < 2:
            self.cluster_labels = np.zeros(len(X), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init='auto')
            self.cluster_labels = kmeans.fit_predict(X)

        self.clusters = {}
        self.cluster_means = {}
        for cid in np.unique(self.cluster_labels):
            mask = self.cluster_labels == cid
            subset = self.df[mask].copy()
            if len(subset) == 0:
                continue
            mean_dir = subset['Dip_Direction'].mean()
            mean_dip = subset['Dip_Angle'].mean()
            subset['Cluster'] = cid
            self.clusters[int(cid)] = subset
            self.cluster_means[int(cid)] = (float(mean_dir), float(mean_dip))

    def analyze(self):
        if self.df is None:
            raise ValueError("Load data first.")
        self.perform_clustering()
        df_processed = self.df.copy().rename(
            columns={'Dip_Direction': 'dip_dir', 'Dip_Angle': 'dip'})
        total = len(df_processed)
        planar_potential = []
        wedge_potential = []
        toppling_potential = []

        for idx, row in df_processed.iterrows():
            p = {'dip_dir': float(row['dip_dir']), 'dip': float(row['dip'])}
            pf, fs = self.check_planar_failure(p)
            if pf:
                planar_potential.append((idx, fs))
            tf, fs2 = self.check_toppling_failure(p)
            if tf:
                toppling_potential.append((idx, fs2))

        for i, j in combinations(range(total), 2):
            plane1 = df_processed.iloc[i].to_dict()
            plane2 = df_processed.iloc[j].to_dict()
            inter_trend, inter_plunge = self.compute_intersection(
                plane1, plane2)
            if inter_trend is None:
                continue
            wf, fsw = self.check_wedge_failure(inter_trend, inter_plunge)
            if wf:
                wedge_potential.append((i, j, inter_trend, inter_plunge, fsw))

        cluster_results = {}
        for cid, subset in self.clusters.items():
            sub_df = subset.rename(
                columns={'Dip_Direction': 'dip_dir', 'Dip_Angle': 'dip'})
            sub_planar = [(idx, self.check_planar_failure(row)[1])
                          for idx, row in sub_df.iterrows() if self.check_planar_failure(row)[0]]
            cluster_results[cid] = {
                'planar': len(sub_planar),
                'mean_dir': self.cluster_means[cid][0],
                'mean_dip': self.cluster_means[cid][1],
                'n': len(subset)
            }

        self.results = {
            'overall': {
                'planar_potential': planar_potential,
                'wedge_potential': wedge_potential,
                'toppling_potential': toppling_potential,
                'counts': {'planar': len(planar_potential), 'wedge': len(wedge_potential), 'toppling': len(toppling_potential), 'total_planes': total}
            },
            'clusters': cluster_results,
            'df_processed': df_processed,
            'cluster_labels': self.cluster_labels
        }
        return self.results

    def get_summary(self):
        if not self.results:
            return "Run analysis first."
        r = self.results['overall']
        s = f"Total Planes: {r['counts']['total_planes']}\nPlanar: {r['counts']['planar']}\nWedge: {r['counts']['wedge']}\nToppling: {r['counts']['toppling']}\n"
        s += "\nClusters:\n"
        for cid, c in self.results['clusters'].items():
            s += f"Cluster {cid}: n={c['n']}, Mean {c['mean_dir']:.1f}/{c['mean_dip']:.1f}, PlanarRisk={c['planar']}\n"
        return s

    def get_cluster_stats(self):
        if not self.clusters:
            return "Run clustering first."
        out = []
        for cid, subset in self.clusters.items():
            out.append(
                f"Cluster {cid}: n={len(subset)}, Mean Dir {self.cluster_means[cid][0]:.1f}, Mean Dip {self.cluster_means[cid][1]:.1f}")
        return "\n".join(out)

    def plot_stereonet(self, projection='equal_area', save_path=None, show_contours=True, show_clusters=True, show_friction=True):
        if not self.results:
            raise ValueError("Run analysis first.")
        r = self.results
        proj_param = 'stereonet' if projection == 'equal_area' else 'wulff'
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=proj_param)
        ax.grid(True)
        hover_data = defaultdict(list)

        # plot poles
        pole_azs = []
        pole_pls = []
        colors = ['blue'] * len(r['df_processed'])
        cluster_colors = None
        if show_clusters and r.get('cluster_labels') is not None:
            n_colors = len(
                self.cluster_means) if self.cluster_means else self.n_clusters
            cluster_colors = cm.Set1(np.linspace(0, 1, n_colors))
            for i, label in enumerate(r['cluster_labels']):
                colors[i] = cluster_colors[label % len(cluster_colors)]

        for i, (idx, row) in enumerate(r['df_processed'].iterrows()):
            pole_az = (row['dip_dir'] + 180) % 360
            pole_pl = 90 - row['dip']
            pole_azs.append(pole_az)
            pole_pls.append(pole_pl)
            color = 'red' if idx in [
                p[0] for p in r['overall']['planar_potential']] else colors[i]
            ax.pole(pole_az, pole_pl, color=color, marker='o', markersize=5)

        # contours (simple KDE in pole angle space)
        if show_contours and len(pole_azs) > 20:
            coords = np.column_stack(
                [np.radians(pole_azs), np.radians(pole_pls)])
            kde = gaussian_kde(coords.T)
            xx, yy = np.mgrid[-np.pi:np.pi:100j, 0:np.pi:100j]
            grid = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(grid).reshape(xx.shape)
            ax.contour(xx, yy, z, colors='fuchsia', alpha=0.6)

        # slope normal
        if self.slope_dip_dir is not None and self.slope_dip is not None:
            slope_pole_az = (self.slope_dip_dir + 180) % 360
            slope_pole_plunge = 90 - self.slope_dip
            ax.pole(slope_pole_az, slope_pole_plunge, marker='s',
                    color='green', markersize=10, label='Slope Normal')
            ax.plane(self.slope_dip_dir, self.slope_dip, 'g--', linewidth=1)

        # friction cone
        if show_friction and self.slope_dip is not None and self.friction is not None:
            for offset in [-self.friction, self.friction]:
                cone_dip = self.slope_dip + offset
                if 0 <= cone_dip <= 90:
                    ax.plane(self.slope_dip_dir, cone_dip, color='gray',
                             linestyle=':', linewidth=1, alpha=0.7)

        # wedges
        for idx1, idx2, inter_trend, inter_plunge, fs in r['overall']['wedge_potential']:
            ax.line(inter_trend, inter_plunge, color='orange', linewidth=2)

        ax.legend(loc='upper right')
        title = f"Stereonet | Planar {r['counts']['planar'] if 'counts' in r['overall'] else r['overall']['counts']['planar']}"
        ax.set_title(title)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
