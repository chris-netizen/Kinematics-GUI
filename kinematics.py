from collections import defaultdict
import sys
import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import matplotlib.cm as cm  # For cluster colors

PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI
STRIKE_TOLERANCE = 20.0
DAYLIGHT_TOLERANCE = 90.0


class KinematicAnalyzer:
    """
    Enhanced class for kinematic slope stability analysis.
    Now includes clustering, toppling, friction cone, contours.
    Supports flexible column selection.
    """

    def __init__(self, data=None, file_path=None, n_clusters=3):
        self.df = None
        self.dip_dir_col = 'Dip_Direction'
        self.dip_col = 'Dip_Angle'
        self.results = None
        self.clusters = None  # Dict: cluster_id -> df_subset
        self.cluster_means = None  # Dict: cluster_id -> mean_dip_dir, mean_dip
        self.slope_dip_dir = None
        self.slope_dip = None
        self.friction = None
        self.n_clusters = n_clusters
        self.cluster_labels = None  # Store labels separately for access
        if file_path:
            self.load_data(file_path)
        elif data is not None:
            self.df = data.copy()
            self._validate_data()

    def load_data(self, file_path):
        """Load CSV or Excel files without strict validation."""
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.df = pd.read_excel(file_path)
            else:
                self.df = pd.read_csv(file_path)
            # No strict validation here; defer to user selection
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def set_dip_columns(self, dip_dir_col, dip_col):
        """Set the column names for dip direction and dip angle."""
        if dip_dir_col not in self.df.columns or dip_col not in self.df.columns:
            raise ValueError("Selected columns do not exist.")
        if dip_dir_col == dip_col:
            raise ValueError(
                "Dip direction and dip columns must be different.")
        self.dip_dir_col = dip_dir_col
        self.dip_col = dip_col
        self._validate_data()

    def _validate_data(self):
        """Check selected columns and angle bounds."""
        if self.df is None or self.dip_dir_col not in self.df.columns or self.dip_col not in self.df.columns:
            raise ValueError(
                f"DataFrame missing required columns: {self.dip_dir_col}, {self.dip_col}")
        dip_dir_data = self.df[self.dip_dir_col]
        dip_data = self.df[self.dip_col]
        if not (dip_dir_data.between(0, 360).all() and dip_data.between(0, 90).all()):
            raise ValueError(
                f"Angles must be: {self.dip_dir_col} 0-360°, {self.dip_col} 0-90°")

    def set_parameters(self, slope_dip_dir, slope_dip, friction, n_clusters=3):
        """Set params and validate."""
        self.slope_dip_dir = self._clamp_angle(slope_dip_dir)
        self.slope_dip = self._clamp_angle(slope_dip, max_val=90)
        self.friction = self._clamp_angle(friction, max_val=45)
        self.n_clusters = n_clusters

    def _clamp_angle(self, val, max_val=360):
        """Clamp angle to 0-max_val."""
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
        if dip < 70:
            return False, 0.0
        if self.azimuth_diff(dip_dir + 180, self.slope_dip_dir) > 20:
            return False, 0.0
        if dip > self.slope_dip + self.friction:
            fs = (self.slope_dip + self.friction - dip) / 10  # Placeholder
            return fs < 1.0, fs
        return False, 0.0

    def perform_clustering(self):
        """K-means on pole coordinates using selected columns."""
        if self.df is None:
            return
        poles = []
        for _, row in self.df.iterrows():
            pole_az = (row[self.dip_dir_col] + 180) % 360
            pole_pl = 90 - row[self.dip_col]
            poles.append([math.radians(pole_az), math.radians(pole_pl)])
        poles = np.array(poles)

        n_clust = min(self.n_clusters, len(poles))
        if n_clust < 2:
            self.cluster_labels = np.zeros(len(poles), dtype=int)
            return

        kmeans = KMeans(n_clusters=n_clust, random_state=42)
        self.cluster_labels = kmeans.fit_predict(poles)

        self.clusters = {}
        self.cluster_means = {}
        for cid in range(n_clust):
            mask = self.cluster_labels == cid
            if mask.sum() > 0:
                subset = self.df[mask].copy()
                subset['Cluster'] = cid
                mean_dir = subset[self.dip_dir_col].mean()
                mean_dip = subset[self.dip_col].mean()
                self.clusters[cid] = subset
                self.cluster_means[cid] = (mean_dir, mean_dip)

    def analyze(self):
        """Main analysis: Cluster first, then analyze each/all using selected columns."""
        if self.df is None:
            raise ValueError("Load data first.")
        self.perform_clustering()
        df_processed = self.df[[self.dip_dir_col, self.dip_col]].copy()
        df_processed.columns = ['dip_dir', 'dip']

        total_planes = len(df_processed)
        planar_potential = []
        wedge_potential = []
        toppling_potential = []
        for idx, row in df_processed.iterrows():
            is_failure, fs = self.check_planar_failure(row)
            if is_failure:
                planar_potential.append((idx, fs))
            is_topple, fs_top = self.check_toppling_failure(row)
            if is_topple:
                toppling_potential.append((idx, fs_top))

        for idx1, idx2 in combinations(range(total_planes), 2):
            plane1 = df_processed.iloc[idx1].to_dict()
            plane2 = df_processed.iloc[idx2].to_dict()
            inter_trend, inter_plunge = self.compute_intersection(
                plane1, plane2)
            if inter_trend is None:
                continue
            is_failure, fs = self.check_wedge_failure(
                inter_trend, inter_plunge)
            if is_failure:
                wedge_potential.append(
                    (idx1, idx2, inter_trend, inter_plunge, fs))

        cluster_results = {}
        for cid, subset in self.clusters.items():
            sub_df = subset[[self.dip_dir_col, self.dip_col]].copy()
            sub_df.columns = ['dip_dir', 'dip']
            sub_planar = [(local_idx, self.check_planar_failure(row)[
                           1]) for local_idx, row in sub_df.iterrows() if self.check_planar_failure(row)[0]]
            cluster_results[cid] = {'planar': len(
                sub_planar), 'mean_dir': self.cluster_means[cid][0], 'mean_dip': self.cluster_means[cid][1]}

        self.results = {
            'overall': {
                'planar_potential': planar_potential,
                'wedge_potential': wedge_potential,
                'toppling_potential': toppling_potential,
                'counts': {'planar': len(planar_potential), 'wedge': len(wedge_potential), 'toppling': len(toppling_potential), 'total_planes': total_planes}
            },
            'clusters': cluster_results,
            'df_processed': df_processed,
            'cluster_labels': self.cluster_labels
        }
        return self.results

    def get_summary(self):
        """Enhanced summary with clusters."""
        if not self.results:
            return "Run analysis first."
        r = self.results['overall']
        summary = f"=== OVERALL SUMMARY ===\nTotal Planes: {r['counts']['total_planes']}\n"
        summary += f"Potential Planar (FS<1): {r['counts']['planar']}\n"
        summary += f"Potential Wedge (FS<1): {r['counts']['wedge']}\n"
        summary += f"Potential Toppling (FS<1): {r['counts']['toppling']}\n"
        if r['counts']['planar'] + r['counts']['wedge'] + r['counts']['toppling'] == 0:
            summary += "Stable geometrically.\n"
        else:
            summary += "WARNING: Proceed to detailed FS.\n"

        summary += "\n=== CLUSTER SUMMARY ===\n"
        for cid, cdata in self.results['clusters'].items():
            summary += f"Cluster {cid}: {len(self.clusters[cid])} planes | Mean: {cdata['mean_dir']:.1f}°/{cdata['mean_dip']:.1f}° | Planar Risk: {cdata['planar']}\n"
        return summary

    def get_cluster_stats(self):
        """Stats summary for clusters."""
        if not self.clusters:
            return "Run clustering first."
        stats = []
        for cid, subset in self.clusters.items():
            mean_dir = subset[self.dip_dir_col].mean()
            std_dir = subset[self.dip_dir_col].std()
            mean_dip = subset[self.dip_col].mean()
            std_dip = subset[self.dip_col].std()
            stats.append(
                f"Cluster {cid}: n={len(subset)}, Mean Dir/Dip: {mean_dir:.1f}±{std_dir:.1f}/{mean_dip:.1f}±{std_dip:.1f}")
        return "\n".join(stats)

    def save_results(self, file_path):
        """Save enhanced CSV with clusters, types, FS, and all original columns."""
        if not self.results:
            raise ValueError("Run analysis first.")
        r = self.results['overall']
        if self.cluster_labels is not None:
            self.df['Cluster'] = self.cluster_labels
        planar_df = pd.DataFrame(r['planar_potential'], columns=[
                                 'Plane_Index', 'FS'])
        planar_df['Type'] = 'Planar'
        wedge_df = pd.DataFrame(r['wedge_potential'], columns=[
                                'Plane1_Index', 'Plane2_Index', 'Inter_Trend', 'Inter_Plunge', 'FS'])
        wedge_df['Type'] = 'Wedge'
        toppling_df = pd.DataFrame(
            r['toppling_potential'], columns=['Plane_Index', 'FS'])
        toppling_df['Type'] = 'Toppling'
        combined = pd.concat(
            [planar_df, wedge_df, toppling_df], ignore_index=True)
        combined = combined.rename(columns={'Plane1_Index': 'Plane_Index'})
        if 'Cluster' in self.df.columns:
            combined = pd.merge(combined, self.df[[
                                'Cluster']], left_on='Plane_Index', right_index=True, how='left')
        # Merge with original df for all columns
        combined = pd.merge(
            combined, self.df, left_on='Plane_Index', right_index=True, how='left')
        combined.to_csv(file_path, index=False)

    def plot_stereonet(self, projection='equal_area', save_path=None, show_contours=True, show_clusters=True, show_friction=True):
        """Enhanced plot: Clusters (colored), contours (fuchsia), friction cone (gray arcs, toggleable)."""
        if not self.results:
            raise ValueError("Run analysis first.")
        r = self.results
        proj_param = 'stereonet' if projection == 'equal_area' else 'wulff'

        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=proj_param)
        ax.grid(True)

        hover_data = defaultdict(list)

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
            x = math.radians(pole_az)
            if projection == 'equal_area':
                r_val = 2 * math.asin(math.sin(math.radians(pole_pl) / 2))
            else:
                r_val = math.radians(pole_pl)
            ax.pole(pole_az, pole_pl, color=color, marker='o', markersize=6)
            hover_data[(x, r_val)].append({
                'type': 'pole', 'index': idx, 'dip_dir': row['dip_dir'], 'dip': row['dip'],
                'fs': next((fs for p, fs in r['overall']['planar_potential'] if p == idx), None),
                'cluster': r['cluster_labels'][i] if r.get('cluster_labels') is not None else None
            })

        if show_contours and len(pole_azs) > 20:
            coords = np.column_stack(
                [np.radians(pole_azs), np.radians(pole_pls)])
            if projection == 'equal_area':
                rho = 2 * np.arcsin(np.sin(np.radians(pole_pls)) / 2)
                coords[:, 1] = rho
            kde = gaussian_kde(coords.T)
            levels = np.linspace(0, kde(coords.T).max(), 5)
            xx, yy = np.mgrid[-np.pi:np.pi:100j, 0:np.pi:100j]
            contour_grid = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(contour_grid).reshape(xx.shape)
            ax.contour(
                xx, yy, z, levels=levels[1:], colors='fuchsia', linestyles='--', alpha=0.6)

        slope_pole_az = (self.slope_dip_dir + 180) % 360
        slope_pole_plunge = 90 - self.slope_dip
        x_slope = math.radians(slope_pole_az)
        if projection == 'equal_area':
            r_slope = 2 * \
                math.asin(math.sin(math.radians(slope_pole_plunge) / 2))
        else:
            r_slope = math.radians(slope_pole_plunge)
        ax.pole(slope_pole_az, slope_pole_plunge, marker='s',
                color='green', markersize=12, label='Slope Normal')
        hover_data[(x_slope, r_slope)].append(
            {'type': 'slope_normal', 'dir': self.slope_dip_dir, 'dip': self.slope_dip})
        ax.plane(self.slope_dip_dir, self.slope_dip, 'g--',
                 linewidth=2, label='Slope Great Circle')
        gc_x = math.radians((self.slope_dip_dir + 180) % 360)
        gc_r = math.pi / 2
        hover_data[(gc_x, gc_r)].append(
            {'type': 'great_circle', 'dir': self.slope_dip_dir, 'dip': self.slope_dip})

        if show_friction:
            for offset in [-self.friction, self.friction]:
                cone_dip = self.slope_dip + offset
                if 0 <= cone_dip <= 90:
                    ax.plane(self.slope_dip_dir, cone_dip, color='red', linestyle='--', linewidth=2,
                             alpha=0.7, label='Friction Cone' if offset == -self.friction else "")

        for idx1, idx2, inter_trend, inter_plunge, fs in r['overall']['wedge_potential']:
            end_x = math.radians(inter_trend)
            if projection == 'equal_area':
                end_r = 2 * math.asin(math.sin(math.radians(inter_plunge) / 2))
            else:
                end_r = math.radians(inter_plunge)
            ax.line(inter_trend, inter_plunge, color='orange', linewidth=2)
            hover_data[(end_x, end_r)].append({'type': 'wedge', 'planes': (
                idx1, idx2), 'trend': inter_trend, 'plunge': inter_plunge, 'fs': fs})

        if show_clusters and self.cluster_means:
            for cid, (mean_dir, mean_dip) in self.cluster_means.items():
                color = cluster_colors[cid] if cluster_colors is not None else 'purple'

                # Plot mean pole (triangle)
                mean_pole_az = (mean_dir + 180) % 360
                mean_pole_pl = 90 - mean_dip
                ax.pole(mean_pole_az, mean_pole_pl, marker='^', color=color, markersize=12,
                        label=f'Cluster {cid} Mean Pole')

                # PLOT THE MEAN PLANE GREAT CIRCLE
                ax.plane(mean_dir, mean_dip, color=color, linewidth=2.5, linestyle='-', alpha=0.8,
                         label=f'Cluster {cid} Mean Plane')

                # Optional: Add label on the great circle
                # (approximate position at dip direction)
                label_az = math.radians(mean_dir)
                label_r = 0.75 * (math.pi / 2)  # 75% out on radius
                if projection == 'equal_area':
                    label_r = 2 * math.asin(math.sin(label_r / 2))
                ax.text(label_az, label_r, f'C{cid}', color=color, fontsize=10, fontweight='bold',
                        ha='center', va='center', transform=ax.transData)

        ax.legend(loc='upper right')
        title = f'Stereonet ({projection}, Lower Hem.) | Planar: {r["overall"]["counts"]["planar"]} | Wedge: {r["overall"]["counts"]["wedge"]} | Toppling: {r["overall"]["counts"]["toppling"]}'
        ax.set_title(title)

        fig.hover_data = hover_data
        fig.hover_tolerance = 0.05

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_scatter(self, x_col, y_col):
        """Generate a scatter plot for two selected columns."""
        if self.df is None or x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError("Invalid columns for scatter plot.")
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df.plot.scatter(x=x_col, y=y_col, ax=ax, alpha=0.6)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        return fig

    def plot_bar(self, col):
        """Generate a bar chart for a selected column's value counts."""
        if self.df is None or col not in self.df.columns:
            raise ValueError("Invalid column for bar chart.")
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Bar Chart: {col} Distribution')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        return fig
