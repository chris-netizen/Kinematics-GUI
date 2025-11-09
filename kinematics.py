from collections import defaultdict
import sys
import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

import mplstereonet
from mplstereonet import kinematic_analysis

PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# MODIFIED: Renamed and clarified tolerances.
# These are now set via set_parameters in the GUI.
# DEFAULT_PLANAR_LIMIT = 20.0
# DEFAULT_TOPPLE_LIMIT = 30.0
# DEFAULT_WEDGE_LIMIT = 90.0
DEFAULT_TOPPLING_DIP_MIN = 60.0


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
        self.clusters = None
        self.cluster_means = None
        self.slope_dip_dir = 0.0
        self.slope_dip = 0.0
        self.friction = 0.0
        self.n_clusters = n_clusters
        self.cluster_labels = None

        # NEW: Parameters for kinematic tests
        self.lateral_limit_planar = 20.0
        self.lateral_limit_toppling = 30.0
        self.daylight_limit_wedge = 90.0

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
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def set_dip_columns(self, dip_dir_col, dip_col):
        """Set the column names for dip direction and dip angle."""
        if self.df is None:
            raise ValueError("Load data before mapping columns.")
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

        # Ensure data is numeric and handle NaNs
        self.df[self.dip_dir_col] = pd.to_numeric(
            self.df[self.dip_dir_col], errors='coerce')
        self.df[self.dip_col] = pd.to_numeric(
            self.df[self.dip_col], errors='coerce')
        self.df = self.df.dropna(subset=[self.dip_dir_col, self.dip_col])

        dip_dir_data = self.df[self.dip_dir_col]
        dip_data = self.df[self.dip_col]
        if not (dip_dir_data.between(0, 360, inclusive='left').all() and dip_data.between(0, 90).all()):
            raise ValueError(
                f"Angles must be: {self.dip_dir_col} 0-360°, {self.dip_col} 0-90°")

    # MODIFIED: Added lateral limit params
    def set_parameters(self, slope_dip_dir, slope_dip, friction, n_clusters=3, lateral_limit_planar=20.0, lateral_limit_toppling=30.0):
        """Set params and validate."""
        self.slope_dip_dir = self._clamp_angle(slope_dip_dir)
        self.slope_dip = self._clamp_angle(slope_dip, max_val=90)
        self.friction = self._clamp_angle(
            friction, max_val=90)  # Friction can be > 45
        self.n_clusters = n_clusters
        self.lateral_limit_planar = lateral_limit_planar
        self.lateral_limit_toppling = lateral_limit_toppling

    def _clamp_angle(self, val, max_val=360):
        """Clamp angle to 0-max_val."""
        return max(0, min(max_val, val))

    def azimuth_diff(self, a1, a2):
        diff = abs(a1 - a2) % 360
        return min(diff, 360 - diff)

    # ... (plane_to_normal, line_from_vector, compute_intersection remain the same) ...
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

        # Ensure plunge is always positive (lower hemisphere)
        if vz > 0:
            vec = -vec
            vx, vy, vz = vec

        plunge = math.degrees(math.atan2(-vz, horiz))
        trend = math.degrees(math.atan2(vy, vx)) % 360
        return trend, plunge

    def compute_intersection(self, plane1, plane2):
        n1 = self.plane_to_normal(plane1['dip_dir'], plane1['dip'])
        n2 = self.plane_to_normal(plane2['dip_dir'], plane2['dip'])
        inter_vec = np.cross(n1, n2)
        if np.linalg.norm(inter_vec) == 0:
            return None, None

        # MODIFIED: Ensure intersection is lower hemisphere
        if inter_vec[2] > 0:  # Z is positive, pointing up
            inter_vec = -inter_vec  # Flip to point down

        trend, plunge = self.line_from_vector(inter_vec)
        return trend, plunge

    # MODIFIED: Corrected kinematic logic
    def check_planar_failure(self, plane):
        """
        Check for planar failure potential.
        Based on Dips Tutorial 9. [cite: 726, 727, 728, 729, 746, 747]
        1. Lateral Limits: Plane dip_dir must be within ±limit of slope_dip_dir.
        2. Friction: Plane dip must be > friction angle (pole outside friction cone).
        3. Daylighting: Plane dip must be < slope dip (pole inside daylight envelope).
        """
        dip_dir, dip = plane['dip_dir'], plane['dip']

        # 1. Check Lateral Limits
        if self.azimuth_diff(dip_dir, self.slope_dip_dir) > self.lateral_limit_planar:
            return False, 0.0

        # 2. Check Friction
        if dip <= self.friction:
            return False, 0.0

        # 3. Check Daylighting
        if dip >= self.slope_dip:
            return False, 0.0

        # If all pass, it's critical. Calculate FS.
        fs = math.tan(math.radians(self.friction)) / \
            math.tan(math.radians(dip))
        return True, fs

    # MODIFIED: Corrected kinematic logic
    def check_wedge_failure(self, inter_trend, inter_plunge):
        """
        Check for wedge failure potential.
        Based on Dips Tutorial 10. [cite: 130, 146, 148, 151, 152]
        1. Daylighting: Intersection must "daylight" (plunge < slope_dip).
        2. Friction: Intersection must be steep enough to slide (plunge > friction).
        3. Plunge Direction: Intersection must slide "out" of the face (trend within ±90° of slope_dip_dir).
        """
        # 1. Check Daylighting (Plunge)
        if inter_plunge >= self.slope_dip:
            return False, 0.0

        # 2. Check Friction
        if inter_plunge <= self.friction:
            return False, 0.0

        # 3. Check Daylighting (Trend)
        if self.azimuth_diff(inter_trend, self.slope_dip_dir) > self.daylight_limit_wedge:
            return False, 0.0

        # If all pass, it's critical. Calculate FS.
        fs = math.tan(math.radians(self.friction)) / \
            math.tan(math.radians(inter_plunge))
        return True, fs

    # MODIFIED: Corrected kinematic logic for FLEXURAL toppling
    def check_toppling_failure(self, plane):
        """
        Check for flexural toppling potential.
        Based on Dips Tutorial 8. [cite: 502, 514, 524]
        1. Lateral Limits: Plane must dip *into* the slope (dip_dir within ±limit of slope_dip_dir + 180°).
        2. Slip Limit: Plane dip must be > (slope_dip - friction).
        3. Steepness: Plane must be reasonably steep (e.g., > 60°).
        """
        dip_dir, dip = plane['dip_dir'], plane['dip']

        # 1. Check Lateral Limits (dipping into slope)
        topple_dir = (self.slope_dip_dir + 180) % 360
        if self.azimuth_diff(dip_dir, topple_dir) > self.lateral_limit_toppling:
            return False, 0.0  # Not toppling

        # 2. Check Slip Limit
        slip_limit_dip = self.slope_dip - self.friction
        if dip <= slip_limit_dip:
            return False, 0.0  # Pole is inside slip limit, too stable

        # 3. Check Steepness
        if dip < DEFAULT_TOPPLING_DIP_MIN:
            return False, 0.0  # Not steep enough to form toppling slabs

        # If all pass, it's critical. No simple FS.
        return True, 0.0  # FS calculation is complex, return 0.0

    def perform_clustering(self):
        """K-means on pole coordinates using selected columns."""
        if self.df is None:
            return

        # Ensure data is valid before clustering
        self._validate_data()

        poles_cartesian = []  # Use 3D cartesian coordinates for poles
        for _, row in self.df.iterrows():
            pole_az = (row[self.dip_dir_col] + 180) % 360
            pole_pl = 90 - row[self.dip_col]

            # Convert polar (az, pl) to cartesian (x, y, z)
            az_rad = np.radians(pole_az)
            pl_rad = np.radians(pole_pl)

            x = np.cos(az_rad) * np.cos(pl_rad)
            y = np.sin(az_rad) * np.cos(pl_rad)
            z = np.sin(pl_rad)
            poles_cartesian.append([x, y, z])

        if not poles_cartesian:
            self.cluster_labels = []
            return

        poles_cartesian = np.array(poles_cartesian)

        n_clust = min(self.n_clusters, len(poles_cartesian))
        if n_clust < 1:
            self.cluster_labels = np.zeros(len(poles_cartesian), dtype=int)
            return

        if n_clust == 1:
            self.cluster_labels = np.zeros(len(poles_cartesian), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clust, random_state=42,
                            n_init=10)  # NEW: added n_init
            self.cluster_labels = kmeans.fit_predict(poles_cartesian)

        self.clusters = {}
        self.cluster_means = {}
        for cid in range(n_clust):
            mask = self.cluster_labels == cid
            if mask.sum() > 0:
                subset = self.df[mask].copy()
                subset['Cluster'] = cid

                # MODIFIED: Mean calculation should be vectorial, but this is a good approximation
                mean_dir = subset[self.dip_dir_col].mean()
                mean_dip = subset[self.dip_col].mean()

                self.clusters[cid] = subset
                self.cluster_means[cid] = (mean_dir, mean_dip)

    def analyze(self):
        """Main analysis: Cluster first, then analyze each/all using selected columns."""
        if self.df is None:
            raise ValueError("Load data first.")

        # Ensure data is valid
        self._validate_data()

        self.perform_clustering()
        df_processed = self.df[[self.dip_dir_col, self.dip_col]].copy()
        df_processed.columns = ['dip_dir', 'dip']

        total_planes = len(df_processed)
        planar_potential = []
        wedge_potential = []
        toppling_potential = []

        # Store all intersections for wedge plot
        all_intersections = []

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

            all_intersections.append((idx1, idx2, inter_trend, inter_plunge))

            is_failure, fs = self.check_wedge_failure(
                inter_trend, inter_plunge)
            if is_failure:
                wedge_potential.append(
                    (idx1, idx2, inter_trend, inter_plunge, fs))

        cluster_results = {}
        if self.clusters:  # Check if clusters were formed
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
                'all_intersections': all_intersections,  # NEW: Store all intersections
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
        summary = f"=== KINEMATIC ANALYSIS SUMMARY ===\n"
        summary += f"Slope: {self.slope_dip_dir:.1f}° / {self.slope_dip:.1f}° | Friction: {self.friction:.1f}°\n"
        summary += f"Total Planes: {r['counts']['total_planes']}\n"
        summary += f"\n--- Potential Failures ---\n"
        summary += f"Planar (Limit: ±{self.lateral_limit_planar}°): {r['counts']['planar']} critical poles\n"
        summary += f"Wedge: {r['counts']['wedge']} critical intersections\n"
        summary += f"Toppling (Limit: ±{self.lateral_limit_toppling}°): {r['counts']['toppling']} critical poles\n"

        if r['counts']['planar'] + r['counts']['wedge'] + r['counts']['toppling'] == 0:
            summary += "\nSlope appears kinematically stable.\n"
        else:
            summary += "\nWARNING: Potential kinematic failures identified.\n"

        if self.results['clusters']:
            summary += "\n=== CLUSTER SUMMARY ===\n"
            for cid, cdata in self.results['clusters'].items():
                summary += f"Cluster {cid} (n={len(self.clusters[cid])}): Mean: {cdata['mean_dir']:.1f}°/{cdata['mean_dip']:.1f}° | Planar Risk: {cdata['planar']}\n"
        return summary

    def get_cluster_stats(self):
        """Stats summary for clusters."""
        if not self.clusters:
            return "Run clustering first (or load more data)."
        stats = []
        for cid, subset in self.clusters.items():
            mean_dir = subset[self.dip_dir_col].mean()
            std_dir = subset[self.dip_dir_col].std()
            mean_dip = subset[self.dip_col].mean()
            std_dip = subset[self.dip_col].std()
            stats.append(
                f"Cluster {cid}: n={len(subset)}, Mean Dir/Dip: {mean_dir:.1f}±{std_dir:.1f} / {mean_dip:.1f}±{std_dip:.1f}")
        return "\n".join(stats)

    def save_results(self, file_path):
        """Save enhanced CSV with clusters, types, FS, and all original columns."""
        if not self.results:
            raise ValueError("Run analysis first.")
        r = self.results['overall']

        # Create a copy to avoid modifying the original dataframe
        output_df = self.df.copy()

        if self.cluster_labels is not None and len(self.cluster_labels) == len(output_df):
            output_df['Cluster'] = self.cluster_labels

        # Add flags for critical planes
        output_df['Is_Planar_Critical'] = False
        output_df['Is_Topple_Critical'] = False

        planar_indices = [p[0] for p in r['planar_potential']]
        output_df.loc[planar_indices, 'Is_Planar_Critical'] = True

        topple_indices = [t[0] for t in r['toppling_potential']]
        output_df.loc[topple_indices, 'Is_Topple_Critical'] = True

        # Save main data
        output_df.to_csv(file_path, index=False)

        # Save wedge data separately (it's many-to-many)
        if r['wedge_potential']:
            wedge_df = pd.DataFrame(r['wedge_potential'], columns=[
                                    'Plane1_Index', 'Plane2_Index', 'Inter_Trend', 'Inter_Plunge', 'FS'])
            wedge_path = file_path.replace('.csv', '_wedge_failures.csv')
            wedge_df.to_csv(wedge_path, index=False)

    # =========================================================================
    # NEW AND IMPROVED PLOTTING FUNCTION
    # This is the key to the "Dips-like" plot.
    # =========================================================================
    # =========================================================================
    # REFACTORED PLOTTING FUNCTION
    # This now uses the built-in mplstereonet.kinematic_analysis
    # functions to draw the correct shaded failure zones.
    # =========================================================================
    def plot_stereonet(self, projection='equal_area', failure_mode='all_poles', show_contours=True, show_clusters=True):
        """
        Enhanced plot: Draws shaded kinematic zones just like Dips
        using mplstereonet's kinematic_analysis module.
        """
        if not self.results:
            raise ValueError("Run analysis first.")

        # Map GUI projection to mplstereonet string
        mpl_proj = 'stereonet' if projection == 'equal_area' else 'equal_angle'

        # Use mplstereonet.subplots() for convenience
        fig, ax = mplstereonet.subplots(projection=mpl_proj, figsize=(10, 10))
        ax.grid(True)

        r = self.results
        pole_azs, pole_pls = [], []

        # 1. --- Get Cluster Colors ---
        cluster_colors_map = {}
        if show_clusters and r.get('cluster_labels') is not None:
            n_colors = len(
                self.cluster_means) if self.cluster_means else self.n_clusters
            if n_colors > 0:
                cluster_colors_list = cm.Set1(np.linspace(0, 1, n_colors))
                cluster_colors_map = {
                    i: cluster_colors_list[i % n_colors] for i in range(n_colors)}

        # 2. --- Plot All Poles ---
        default_color = 'blue'
        for i, (idx, row) in enumerate(r['df_processed'].iterrows()):
            pole_az = (row['dip_dir'] + 180) % 360
            pole_pl = 90 - row['dip']
            pole_azs.append(pole_az)
            pole_pls.append(pole_pl)

            color = default_color
            if show_clusters and cluster_colors_map:
                label = r['cluster_labels'][i]
                color = cluster_colors_map.get(label, default_color)

            # Plot poles, but only add one label for the legend
            ax.pole(pole_az, pole_pl, color=color,
                    marker='o', markersize=4, alpha=0.6,
                    label='Poles' if i == 0 else "")

        # 3. --- Plot Contours ---
        if show_contours and len(pole_azs) > 20:
            try:
                ax.density_contour(pole_azs, pole_pls,
                                   cmap='viridis', zorder=0, alpha=0.5)
            except Exception as e:
                print(f"Contour plotting failed: {e}")  # Non-fatal

        # 4. --- Plot Kinematic Zones & Critical Features ---

        # Always plot the slope plane
        ax.plane(self.slope_dip_dir, self.slope_dip,
                 'g--', linewidth=2, label='Slope Plane')

        title = f'Stereonet ({projection.replace("_", " ")})'

        if failure_mode == 'planar':
            title = 'Planar Sliding Analysis'

            # -----------------------------------------------------------------
            # NEW: Use mplstereonet's built-in function to get failure patches
            # This draws the daylight envelope, lateral limits, and friction cone zone
            planar_patches = kinematic_analysis.planar_sliding_patches(
                self.slope_dip,
                self.slope_dip_dir,
                self.friction,
                self.lateral_limit_planar
            )
            for patch in planar_patches:
                ax.add_patch(patch)
            # -----------------------------------------------------------------

            # Highlight critical poles (your code for this was good)
            has_planar_label = False
            for idx, fs in r['overall']['planar_potential']:
                row = r['df_processed'].loc[idx]
                ax.pole((row['dip_dir'] + 180) % 360, 90 - row['dip'], 'rs', markersize=8,
                        label='Critical Pole' if not has_planar_label else "")
                has_planar_label = True

        elif failure_mode == 'wedge':
            title = 'Wedge Sliding Analysis'

            # -----------------------------------------------------------------
            # NEW: Use mplstereonet's built-in function for wedge failure
            # This draws the friction cone and the critical lune
            wedge_patches = kinematic_analysis.wedge_sliding_patches(
                self.slope_dip,
                self.slope_dip_dir,
                self.friction
            )
            for patch in wedge_patches:
                ax.add_patch(patch)
            # -----------------------------------------------------------------

            # Highlight critical intersections (your code for this was good)
            has_wedge_label = False
            for i, (idx1, idx2, trend, plunge, fs) in enumerate(r['overall']['wedge_potential']):
                ax.line(trend, plunge, 'r*', markersize=8,
                        label='Critical Intersection' if not has_wedge_label else "")
                has_wedge_label = True

        elif failure_mode == 'flexural_toppling':
            title = 'Flexural Toppling Analysis'

            # -----------------------------------------------------------------
            # NEW: Use mplstereonet's built-in function for toppling
            # This draws the slip limit, lateral limits, and critical toppling zone
            toppling_patches = kinematic_analysis.flexural_toppling_patches(
                self.slope_dip,
                self.slope_dip_dir,
                self.friction,
                self.lateral_limit_toppling
            )
            for patch in toppling_patches:
                ax.add_patch(patch)
            # -----------------------------------------------------------------

            # Highlight critical poles (your code for this was good)
            has_topple_label = False
            for idx, fs in r['overall']['toppling_potential']:
                row = r['df_processed'].loc[idx]
                ax.pole((row['dip_dir'] + 180) % 360, 90 - row['dip'], 'cs', markersize=8,
                        label='Critical Pole' if not has_topple_label else "")
                has_topple_label = True

        # 5. --- Plot Cluster Means (Only on 'all_poles' view) ---
        if show_clusters and self.cluster_means and failure_mode == 'all_poles':
            for cid, (mean_dir, mean_dip) in self.cluster_means.items():
                color = cluster_colors_map.get(cid, 'purple')
                # Plot mean pole
                ax.pole((mean_dir + 180) % 360, 90 - mean_dip, marker='^', color=color, markersize=12,
                        label=f'Cluster {cid} Mean Pole')
                # Plot mean plane
                ax.plane(mean_dir, mean_dip, color=color, linewidth=2.5, linestyle='-', alpha=0.8,
                         label=f'Cluster {cid} Mean Plane')

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.set_title(title)
        fig.tight_layout()

        return fig

    # ... (plot_scatter and plot_bar remain the same) ...
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
